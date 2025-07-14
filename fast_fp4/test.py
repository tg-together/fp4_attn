import torch
import nvfp4sim  



def single_nvfp4_qd_searched(x):

    

    x=x.to(torch.float32)

    m_orig, n_orig = x.shape
    n=n_orig
    m=m_orig

    if n_orig % 16 != 0:
        # Pad to make it divisible
        pad_size = 16 - (n_orig % 16)
        x = torch.nn.functional.pad(x, (0, pad_size), value=0.0)
        n = x.shape[1]
    if m_orig % 2 != 0:

        x = torch.nn.functional.pad(x, (0, 0,0,1), value=0.0)
        m= x.shape[0]

    input_f32 = x.reshape(m * (n // 16), 16).contiguous()
    print(input_f32.shape)

    M = input_f32.shape[0]  # total number of groups

    quantized_data = torch.empty(M, dtype=torch.int64, device=input_f32.device)
    scales = torch.empty(M, dtype=torch.float8_e4m3fn, device=input_f32.device)

    
    print("Input32:",input_f32.shape)
    print(f"Python side debugging:")
    print(f"  input_f32.data_ptr(): {hex(input_f32.data_ptr())}")
    print(f"  input_f32.element_size(): {input_f32.element_size()}")
    print(f"  input_f32.storage_offset(): {input_f32.storage_offset()}")
    print(f"  input_f32.stride(): {input_f32.stride()}")
    print(f"  input_f32.is_contiguous(): {input_f32.is_contiguous()}")
    print(f"  input_f32.nbytes: {input_f32.nbytes}")
    print(f"  Expected end address: {hex(input_f32.data_ptr() + input_f32.nbytes)}")
    
    # Try to access some elements from Python side
    try:
        print(f"  input_f32[0, 0]: {input_f32[0, 0].item()}")
        print(f"  input_f32[16, 0]: {input_f32[16, 0].item()}")  # Element 256
        print(f"  input_f32[-1, -1]: {input_f32[-1, -1].item()}")
    except Exception as e:
        print(f"  Error accessing elements: {e}")


    nvfp4sim.f32_to_nvf4(quantized_data, scales, input_f32)
    
    reconstructed_f32 = torch.empty(m * (n // 16), 16, dtype=torch.float32, device=input_f32.device)
    nvfp4sim.nvf4_to_f32(reconstructed_f32, quantized_data, scales)


    reconstructed_f32 = reconstructed_f32.view(m, n)

    if n != n_orig:
        reconstructed_f32 = reconstructed_f32[:, :n_orig]

    if m != m_orig:
        reconstructed_f32 = reconstructed_f32[:m_orig, :]


    return reconstructed_f32.to(torch.bfloat16)

def demonstrate_fp4_kernels():
    """Demonstrate usage of FP4 CUDA kernels with device selection"""
    
    # Device selection - kernels will run on the device where input tensors are located
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # Use first GPU
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        raise RuntimeError("CUDA not available - these kernels require GPU")
    
    # Input parameters
    m = 1024
    n = 4053
    n_orig = n

    group_size = 16  # Fixed group size for FP4 quantization


    
    # Create input data on GPU
    x_orig = torch.randn(m, n, device=device, dtype=torch.bfloat16).to(torch.float32)

    if n % group_size != 0:
    # Pad to make it divisible
        pad_size = group_size - (n % group_size)
        x = torch.nn.functional.pad(x_orig, (0, pad_size), value=0.0)
        n = x.shape[1]


    input_f32 = x.reshape(m * (n // group_size), group_size).contiguous()

    M = input_f32.shape[0]  # total number of groups

    quantized_data = torch.empty(M, dtype=torch.int64, device=input_f32.device)
    scales = torch.empty(M, dtype=torch.float8_e4m3fn, device=input_f32.device)


    
    print(f"Input shape: {input_f32.shape}")
    print(f"Input range: [{input_f32.min():.3f}, {input_f32.max():.3f}]")
    
    # # Prepare output tensors for quantization
    # quantized_data = torch.empty(batch_size, device=device, dtype=torch.int64)
    # scales = torch.empty(batch_size, device=device, dtype=torch.float8_e4m3fn)
    
    # === FP32 to FP4 Quantization (with scale search) ===
    print("\n=== FP32 to FP4 Quantization ===")
    nvfp4sim.f32_to_nvf4(quantized_data, scales, input_f32)
    
    # Dequantize back to FP32
    reconstructed_f32 = torch.empty_like(input_f32)
    nvfp4sim.nvf4_to_f32(reconstructed_f32, quantized_data, scales)

    reconstructed_f32 = reconstructed_f32.view(m, n)
    if n != n_orig:
        reconstructed_f32 = reconstructed_f32[:, :n_orig]

    
    # Calculate reconstruction error
    mse_f32 = torch.mean((x_orig - reconstructed_f32) ** 2)
    print(f"MSE (FP32): {mse_f32:.6f}")
    

    # === FP32 to FP4 (No Scale Search) ===
    print("\n=== FP32 to FP4 (No Scale Search) ===")
    nvfp4sim.f32_to_nvf4_nosearch(quantized_data, scales, input_f32)
    
    reconstructed_nosearch = torch.empty_like(input_f32)
    nvfp4sim.nvf4_to_f32(reconstructed_nosearch, quantized_data, scales)

    reconstructed_nosearch = reconstructed_nosearch.view(m, n)
    if n != n_orig:
        reconstructed_nosearch = reconstructed_nosearch[:, :n_orig]
    
    mse_nosearch = torch.mean((x_orig - reconstructed_nosearch) ** 2)
    print(f"MSE (No Search): {mse_nosearch:.6f}")
    
    print(f"\nScale search improvement: {(mse_nosearch - mse_f32)/mse_nosearch*100:.1f}%")

if __name__ == "__main__":
    # demonstrate_fp4_kernels()
    x = torch.randn(2952, 4096, device=torch.device('cuda:0'), dtype=torch.bfloat16)

    x_=single_nvfp4_qd_searched(x)

    mse_f32 = torch.mean((x - x_) ** 2)
    print(f"MSE (FP32): {mse_f32:.6f}")