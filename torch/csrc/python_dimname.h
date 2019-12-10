#pragma once
#include <torch/csrc/python_headers.h>
#include <ATen/Dimname.h>
#include <ATen/core/EnableNamedTensor.h>

at::Dimname THPDimname_parse(PyObject* obj);
bool THPUtils_checkDimname(PyObject* obj);
bool THPUtils_checkDimnameList(PyObject* obj);

