#pragma once

// Wrap tensor operation outputs as PyObject*

#include <ATen/ATen.h>
#include <Python.h>
#include <tuple>

#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/python_numbers.h"

namespace torch { namespace autograd { namespace utils {

inline PyObject* wrap(at::Tensor tensor) {
  if (tensor.defined() && tensor.dim() == 0) {
    // don't expose 0-dim tensors to Variable API.
    Variable(tensor).data().as_strided_({1}, {1});
  }
  return THPVariable_Wrap(Variable(std::move(tensor)));
}

inline PyObject* wrap(std::tuple<at::Tensor, at::Tensor> tensors) {
  auto r = THPObjectPtr{PyTuple_New(2)};
  if (!r) throw python_error();
  PyTuple_SET_ITEM(r.get(), 0, wrap(std::get<0>(tensors)));
  PyTuple_SET_ITEM(r.get(), 1, wrap(std::get<1>(tensors)));
  return r.release();
}

inline PyObject* wrap(std::tuple<at::Tensor, at::Tensor, at::Tensor> tensors) {
  auto r = THPObjectPtr{PyTuple_New(3)};
  if (!r) throw python_error();
  PyTuple_SET_ITEM(r.get(), 0, wrap(std::move(std::get<0>(tensors))));
  PyTuple_SET_ITEM(r.get(), 1, wrap(std::move(std::get<1>(tensors))));
  PyTuple_SET_ITEM(r.get(), 2, wrap(std::move(std::get<2>(tensors))));
  return r.release();
}

inline PyObject* wrap(at::TensorList tl) {
  auto r = THPObjectPtr{PyTuple_New(tl.size())};
  if (!r) throw python_error();
  for (size_t i = 0; i < tl.size(); ++i) {
    PyTuple_SET_ITEM(r.get(), i, wrap(tl[i]));
  }
  return r.release();
}

inline PyObject* wrap(bool value) {
  if (value) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

inline PyObject* wrap(int64_t value) {
  return THPUtils_packInt64(value);
}

inline PyObject* wrap(at::Scalar scalar) {
  return wrap(scalar.toTensor());
}


}}} // namespace torch::autograd::utils
