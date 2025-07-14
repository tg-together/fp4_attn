


#include <torch/extension.h>

#include <iostream>
#include <cassert>


void f32_to_nvf4(
    torch::Tensor XQout,
    torch::Tensor XSout,
    torch::Tensor Xin
);

void f32_to_nvf4_nosearch(
    torch::Tensor XQout,
    torch::Tensor XSout,
    torch::Tensor Xin
);

void f16_to_nvf4(
    torch::Tensor XQout,
    torch::Tensor XSout,
    torch::Tensor Xin
);

void nvf4_to_f32(
    torch::Tensor Xout,
    torch::Tensor XQin,
    torch::Tensor XSin
);

void nvf4_to_f16(
    torch::Tensor Xout,
    torch::Tensor XQin,
    torch::Tensor XSin
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("f32_to_nvf4", &f32_to_nvf4, "f32_to_nvf4");
  m.def("f32_to_nvf4_nosearch", &f32_to_nvf4_nosearch, "f32_to_nvf4_nosearch");
  m.def("nvf4_to_f32", &nvf4_to_f32, "nvf4_to_f32");
  m.def("f16_to_nvf4", &f16_to_nvf4, "f16_to_nvf4");
  m.def("nvf4_to_f16", &nvf4_to_f16, "nvf4_to_f16");
}