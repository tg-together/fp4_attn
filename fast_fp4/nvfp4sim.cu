#include <iostream>
#include <cassert>
#include <array>
#include <vector>
#include <utility>
#include <initializer_list>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cooperative_groups.h>
#include <cudaTypedefs.h>

#include <mma.h>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_profiler_api.h>

#include <torch/types.h>
#include <torch/extension.h>

#include <cuda/barrier>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
using namespace cooperative_groups;

using namespace torch::indexing;
using namespace nvcuda;

#define FULL_MASK 0xffffffff
#define HALF_MASK 0x0000ffff

#define CHECK_CUDA(x)           TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)     TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) 	        do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while(false)
#define gpuErrchk(ans)          do { gpuAssert((ans), __FILE__, __LINE__); } while (false)

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T result, char const *const func, const char *const file,
           int64_t const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line,
            static_cast<unsigned int>(result), func);
    exit(EXIT_FAILURE);
  }
}

void f32_to_nvf4(
    torch::Tensor XQout,
    torch::Tensor XSout,
    torch::Tensor Xin
);

__device__ uint32_t f32_as_u32(float x) {
    union { float f; uint32_t u; } z;
    z.f = x;
    return z.u;
}

__device__ float u32_as_f32(uint32_t x) {
    union { float f; uint32_t u; } z;
    z.u = x;
    return z.f;
}

__device__ uint8_t f32_to_f4(float x, float y) {
    const float SCALE = 1.1754943508222875e-38f;
    x = fminf(fmaxf(-6.0,x),6.0) * SCALE;
    y = fminf(fmaxf(-6.0,y),6.0) * SCALE;
    uint32_t ux = f32_as_u32(x) + (1 << 21);
    uint32_t uy = f32_as_u32(y) + (1 << 21);
    ux = ((ux >> 22) & 7) | ((ux >> 28) & 8);
    uy = ((uy >> 22) & 7) | ((uy >> 28) & 8);
    return (uint8_t)(ux | (uy << 4));
}

__device__ float2 f4_to_f32(uint8_t z) {
    const float ISCALE = 8.507059173023462e+37f;
    uint32_t ux = z & 15;
    uint32_t uy = (z >> 4) & 15;
    ux = ((ux & 7) << 22) | ((ux & 8) << 28);
    uy = ((uy & 7) << 22) | ((uy & 8) << 28);
    return make_float2(u32_as_f32(ux) * ISCALE, u32_as_f32(uy) * ISCALE);
}

__global__ void f32_to_nvf4_kernel(
    uint64_t* pxqout,
    uint8_t* pxsout,
    const float* pxin
) {
    // each warp handles 2 groups of 16 scalars each
    int64_t groupIdx = ((int64_t)blockIdx.x) * 2 + threadIdx.x / 16;

    // load all 16 weights to all threads in the half-warp
    float x[16];
    float max_abs_x = 1e-10;
    {
        float my_x = pxin[((int64_t)blockIdx.x) * 32 + threadIdx.x];
        #pragma unroll
        for (int64_t i = 0; i < 16; i++) {
            x[i] = __shfl_sync(FULL_MASK, my_x, i, 16);
            max_abs_x = fmaxf(max_abs_x, fabsf(x[i]));   // work repeated on all threads
        }
    }

    __nv_fp8_storage_t xscale_f8 = __nv_cvt_float_to_fp8(max_abs_x * (1.0 / 6.0), __NV_SATFINITE, __NV_E4M3);
    xscale_f8 = xscale_f8 + ((threadIdx.x & 15) - 7); // adjust scale
    float xscale = __half2float(__nv_cvt_fp8_to_halfraw(xscale_f8, __NV_E4M3));
    float xscale_inv = 1.0 / xscale;

    float loss = 0.0;
    union {
        uint64_t u64;
        uint8_t u8[8];
    } xqs;
    #pragma unroll
    for (int64_t i = 0; i < 8; i++) {
        xqs.u8[i] = f32_to_f4(x[2*i+0] * xscale_inv, x[2*i+1] * xscale_inv);
        float2 xhat = f4_to_f32(xqs.u8[i]);
        loss += (xhat.x * xscale - x[2*i+0]) * (xhat.x * xscale - x[2*i+0]);
        loss += (xhat.y * xscale - x[2*i+1]) * (xhat.y * xscale - x[2*i+1]);
    }

    // tag the loss with my own thread ID to avoid collision
    {
        union { uint32_t u; float f; } l;
        l.f = loss;
        l.u = (l.u & (~(15))) | (threadIdx.x & 15);
        loss = l.f;
    }

    // Use XOR mode to perform butterfly reduction
    float min_loss = loss;
    #pragma unroll
    for (int64_t i=8; i>=1; i/=2) {
        min_loss = fminf(min_loss, __shfl_xor_sync(0xffffffff, min_loss, i));
    }

    if (loss == min_loss) {
        // we have the best scale! write it out
        pxqout[groupIdx] = xqs.u64;
        pxsout[groupIdx] = xscale_f8;
    }
}

void f32_to_nvf4(
    torch::Tensor XQout,
    torch::Tensor XSout,
    torch::Tensor Xin
) {
    const int64_t GROUP_SIZE = 16;

    CHECK_INPUT(XQout);
    CHECK_INPUT(XSout);
    CHECK_INPUT(Xin);

    assert(XQout.dim() == 1);
    assert(XSout.dim() == 1);
    assert(Xin.dim() == 2);
    assert(Xin.sizes()[1] == GROUP_SIZE);

    int64_t N = Xin.sizes()[0];
    assert(XQout.sizes()[0] == N);
    assert(XSout.sizes()[0] == N);
    assert(N % 2 == 0);
    
    assert(XQout.dtype() == torch::kInt64);
    assert(XSout.dtype() == torch::kFloat8_e4m3fn);
    assert(Xin.dtype() == torch::kFloat32);

    int64_t BLOCKS = N / 2;
    int64_t THREADS = 32;


    auto stream = at::cuda::getCurrentCUDAStream().stream();

    f32_to_nvf4_kernel<<<BLOCKS, THREADS, 0, stream>>>(
        (uint64_t*)XQout.data_ptr<int64_t>(),
        (uint8_t*)XSout.data_ptr<at::Float8_e4m3fn>(),
        (const float*)Xin.data_ptr<float>()
    );
}

__global__ void f16_to_nvf4_kernel(
    uint64_t* pxqout,
    uint8_t* pxsout,
    const __half* pxin
) {
    // each warp handles 2 groups of 16 scalars each
    int64_t groupIdx = ((int64_t)blockIdx.x) * 2 + threadIdx.x / 16;

    // load all 16 weights to all threads in the half-warp
    float x[16];
    float max_abs_x = 1e-10;
    {
        float my_x = __half2float(pxin[((int64_t)blockIdx.x) * 32 + threadIdx.x]);
        #pragma unroll
        for (int64_t i = 0; i < 16; i++) {
            x[i] = __shfl_sync(FULL_MASK, my_x, i, 16);
            max_abs_x = fmaxf(max_abs_x, fabsf(x[i]));   // work repeated on all threads
        }
    }

    __nv_fp8_storage_t xscale_f8 = __nv_cvt_float_to_fp8(max_abs_x * (1.0 / 6.0), __NV_SATFINITE, __NV_E4M3);
    xscale_f8 = xscale_f8 + ((threadIdx.x & 15) - 7); // adjust scale
    float xscale = __half2float(__nv_cvt_fp8_to_halfraw(xscale_f8, __NV_E4M3));
    float xscale_inv = 1.0 / xscale;

    float loss = 0.0;
    union {
        uint64_t u64;
        uint8_t u8[8];
    } xqs;
    #pragma unroll
    for (int64_t i = 0; i < 8; i++) {
        xqs.u8[i] = f32_to_f4(x[2*i+0] * xscale_inv, x[2*i+1] * xscale_inv);
        float2 xhat = f4_to_f32(xqs.u8[i]);
        loss += (xhat.x * xscale - x[2*i+0]) * (xhat.x * xscale - x[2*i+0]);
        loss += (xhat.y * xscale - x[2*i+1]) * (xhat.y * xscale - x[2*i+1]);
    }

    // tag the loss with my own thread ID to avoid collision
    {
        union { uint32_t u; float f; } l;
        l.f = loss;
        l.u = (l.u & (~(15))) | (threadIdx.x & 15);
        loss = l.f;
    }

    // Use XOR mode to perform butterfly reduction
    float min_loss = loss;
    #pragma unroll
    for (int64_t i=8; i>=1; i/=2) {
        min_loss = fminf(min_loss, __shfl_xor_sync(0xffffffff, min_loss, i));
    }

    if (loss == min_loss) {
        // we have the best scale! write it out
        pxqout[groupIdx] = xqs.u64;
        pxsout[groupIdx] = xscale_f8;
    }
}

void f16_to_nvf4(
    torch::Tensor XQout,
    torch::Tensor XSout,
    torch::Tensor Xin
) {
    const int64_t GROUP_SIZE = 16;

    CHECK_INPUT(XQout);
    CHECK_INPUT(XSout);
    CHECK_INPUT(Xin);

    assert(XQout.dim() == 1);
    assert(XSout.dim() == 1);
    assert(Xin.dim() == 2);
    assert(Xin.sizes()[1] == GROUP_SIZE);

    int64_t N = Xin.sizes()[0];
    assert(XQout.sizes()[0] == N);
    assert(XSout.sizes()[0] == N);
    assert(N % 2 == 0);
    
    assert(XQout.dtype() == torch::kInt64);
    assert(XSout.dtype() == torch::kFloat8_e4m3fn);
    assert(Xin.dtype() == torch::kFloat16);

    int64_t BLOCKS = N / 2;
    int64_t THREADS = 32;

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    f16_to_nvf4_kernel<<<BLOCKS, THREADS, 0, stream>>>(
        (uint64_t*)XQout.data_ptr<int64_t>(),
        (uint8_t*)XSout.data_ptr<at::Float8_e4m3fn>(),
        (const __half*)Xin.data_ptr<at::Half>()
    );
}


__global__ void nvf4_to_f32_kernel(
    float* pxout,
    const uint64_t* pxqin,
    const uint8_t* pxsin
) {
    // each warp handles 2 groups of 16 scalars each
    int64_t groupIdx = ((int64_t)blockIdx.x) * 2 + threadIdx.x / 16;

    uint64_t xq = pxqin[groupIdx];
    uint8_t xs = pxsin[groupIdx];

    float xscale = __half2float(__nv_cvt_fp8_to_halfraw(xs, __NV_E4M3));
    float x = f4_to_f32((uint8_t)(xq >> (4 * (threadIdx.x & 15)))).x;

    pxout[((int64_t)blockIdx.x) * 32 + threadIdx.x] = x * xscale;
}

void nvf4_to_f32(
    torch::Tensor Xout,
    torch::Tensor XQin,
    torch::Tensor XSin
) {
    const int64_t GROUP_SIZE = 16;

    CHECK_INPUT(Xout);
    CHECK_INPUT(XQin);
    CHECK_INPUT(XSin);

    assert(XQin.dim() == 1);
    assert(XSin.dim() == 1);
    assert(Xout.dim() == 2);
    assert(Xout.sizes()[1] == GROUP_SIZE);

    int64_t N = Xout.sizes()[0];
    assert(XQin.sizes()[0] == N);
    assert(XSin.sizes()[0] == N);
    assert(N % 2 == 0);
    
    assert(XQin.dtype() == torch::kInt64);
    assert(XSin.dtype() == torch::kFloat8_e4m3fn);
    assert(Xout.dtype() == torch::kFloat32);

    int64_t BLOCKS = N / 2;
    int64_t THREADS = 32;

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    nvf4_to_f32_kernel<<<BLOCKS, THREADS, 0, stream>>>(
        (float*)Xout.data_ptr<float>(),
        (const uint64_t*)XQin.data_ptr<int64_t>(),
        (const uint8_t*)XSin.data_ptr<at::Float8_e4m3fn>()
    );
}


__global__ void nvf4_to_f16_kernel(
    __half* pxout,
    const uint64_t* pxqin,
    const uint8_t* pxsin
) {
    // each warp handles 2 groups of 16 scalars each
    int64_t groupIdx = ((int64_t)blockIdx.x) * 2 + threadIdx.x / 16;

    uint64_t xq = pxqin[groupIdx];
    uint8_t xs = pxsin[groupIdx];

    float xscale = __half2float(__nv_cvt_fp8_to_halfraw(xs, __NV_E4M3));
    float x = f4_to_f32((uint8_t)(xq >> (4 * (threadIdx.x & 15)))).x;

    pxout[((int64_t)blockIdx.x) * 32 + threadIdx.x] = __float2half_rn(x * xscale);
}


void nvf4_to_f16(
    torch::Tensor Xout,
    torch::Tensor XQin,
    torch::Tensor XSin
) {
    const int64_t GROUP_SIZE = 16;

    CHECK_INPUT(Xout);
    CHECK_INPUT(XQin);
    CHECK_INPUT(XSin);

    assert(XQin.dim() == 1);
    assert(XSin.dim() == 1);
    assert(Xout.dim() == 2);
    assert(Xout.sizes()[1] == GROUP_SIZE);

    int64_t N = Xout.sizes()[0];
    assert(XQin.sizes()[0] == N);
    assert(XSin.sizes()[0] == N);
    assert(N % 2 == 0);
    
    assert(XQin.dtype() == torch::kInt64);
    assert(XSin.dtype() == torch::kFloat8_e4m3fn);
    assert(Xout.dtype() == torch::kFloat16);

    int64_t BLOCKS = N / 2;
    int64_t THREADS = 32;

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    nvf4_to_f16_kernel<<<BLOCKS, THREADS, 0, stream>>>(
        (__half*)Xout.data_ptr<at::Half>(),
        (const uint64_t*)XQin.data_ptr<int64_t>(),
        (const uint8_t*)XSin.data_ptr<at::Float8_e4m3fn>()
    );
}


__global__ void f32_to_nvf4_nosearch_kernel(
    uint64_t* pxqout,
    uint8_t* pxsout,
    const float* pxin
) {
    // each warp handles 2 groups of 16 scalars each
    int64_t groupIdx = ((int64_t)blockIdx.x) * 2 + threadIdx.x / 16;

    // load all 16 weights to all threads in the half-warp
    float x[16];
    float max_abs_x = 1e-10;
    {
        float my_x = pxin[((int64_t)blockIdx.x) * 32 + threadIdx.x];
        #pragma unroll
        for (int64_t i = 0; i < 16; i++) {
            x[i] = __shfl_sync(FULL_MASK, my_x, i, 16);
            max_abs_x = fmaxf(max_abs_x, fabsf(x[i]));   // work repeated on all threads
        }
    }

    __nv_fp8_storage_t xscale_f8 = __nv_cvt_float_to_fp8(max_abs_x * (1.0 / 6.0), __NV_SATFINITE, __NV_E4M3);
    xscale_f8 = xscale_f8; // don't adjust scale
    float xscale = __half2float(__nv_cvt_fp8_to_halfraw(xscale_f8, __NV_E4M3));
    float xscale_inv = 1.0 / xscale;

    float loss = 0.0;
    union {
        uint64_t u64;
        uint8_t u8[8];
    } xqs;
    #pragma unroll
    for (int64_t i = 0; i < 8; i++) {
        xqs.u8[i] = f32_to_f4(x[2*i+0] * xscale_inv, x[2*i+1] * xscale_inv);
        float2 xhat = f4_to_f32(xqs.u8[i]);
        loss += (xhat.x * xscale - x[2*i+0]) * (xhat.x * xscale - x[2*i+0]);
        loss += (xhat.y * xscale - x[2*i+1]) * (xhat.y * xscale - x[2*i+1]);
    }

    // tag the loss with my own thread ID to avoid collision
    {
        union { uint32_t u; float f; } l;
        l.f = loss;
        l.u = (l.u & (~(15))) | (threadIdx.x & 15);
        loss = l.f;
    }

    // Use XOR mode to perform butterfly reduction
    float min_loss = loss;
    #pragma unroll
    for (int64_t i=8; i>=1; i/=2) {
        min_loss = fminf(min_loss, __shfl_xor_sync(0xffffffff, min_loss, i));
    }

    if (loss == min_loss) {
        // we have the best scale! write it out
        pxqout[groupIdx] = xqs.u64;
        pxsout[groupIdx] = xscale_f8;
    }
}

void f32_to_nvf4_nosearch(
    torch::Tensor XQout,
    torch::Tensor XSout,
    torch::Tensor Xin
) {
    const int64_t GROUP_SIZE = 16;

    CHECK_INPUT(XQout);
    CHECK_INPUT(XSout);
    CHECK_INPUT(Xin);

    assert(XQout.dim() == 1);
    assert(XSout.dim() == 1);
    assert(Xin.dim() == 2);
    assert(Xin.sizes()[1] == GROUP_SIZE);

    int64_t N = Xin.sizes()[0];
    assert(XQout.sizes()[0] == N);
    assert(XSout.sizes()[0] == N);
    assert(N % 2 == 0);
    
    assert(XQout.dtype() == torch::kInt64);
    assert(XSout.dtype() == torch::kFloat8_e4m3fn);
    assert(Xin.dtype() == torch::kFloat32);

    int64_t BLOCKS = N / 2;
    int64_t THREADS = 32;   


    auto stream = at::cuda::getCurrentCUDAStream().stream();

    f32_to_nvf4_nosearch_kernel<<<BLOCKS, THREADS, 0, stream>>>(
        (uint64_t*)XQout.data_ptr<int64_t>(),
        (uint8_t*)XSout.data_ptr<at::Float8_e4m3fn>(),
        (const float*)Xin.data_ptr<float>()
    );
}
