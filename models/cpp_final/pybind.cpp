#include <torch/extension.h>

#include "order.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess_order", &preprocess_order, "preprocess_order");
}
