#include "python_tensor.h"

#include <structmember.h>
#include <pybind11/pybind11.h>

#include "torch/csrc/Dtype.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/Layout.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/utils/cuda_enabled.h"
#include "torch/csrc/utils/cuda_lazy_init.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/tensor_new.h"
#include "torch/csrc/utils/tensor_types.h"
#include "torch/csrc/variable_tensor_functions.h"

#include <ATen/ATen.h>

#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace torch { namespace tensors {

using namespace at;
using namespace torch::autograd;

struct PyTensorType {
  PyTypeObject py_type;
  at::Type* aten_type_;
  THPDtype* dtype;
  THPLayout* layout;
  bool is_cuda;
  char name[64];
  int backend;
  int scalar_type;

  // Precondition: Access to this struct is protected by the GIL
  at::Type* aten_type() {
    if (!aten_type_) {
      if (is_cuda) {
        torch::utils::cuda_lazy_init();
      }
      auto* baseType = globalContext().getNonVariableTypeOpt(static_cast<at::Backend>(backend), static_cast<at::ScalarType>(scalar_type));
      aten_type_ = baseType ? torch::autograd::VariableType::getVariableTypeFromBaseType(*baseType) : nullptr;
    }
    return aten_type_;
  }
};

static_assert(std::is_standard_layout<PyTensorType>::value, "PyTensorType must be standard layout");

// This is always an instance of VariableType
static at::Type* default_tensor_type;

static void py_bind_tensor_types(const std::vector<PyTensorType>& tensor_types);

static TypeError unavailable_type(const PyTensorType& type) {
  const char* cuda_msg = torch::utils::cuda_enabled() ? ". Torch not compiled with CUDA enabled." : "";
  return TypeError("type %s not available%s", type.name, cuda_msg);
}

static PyObject* Tensor_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  HANDLE_TH_ERRORS
  auto& tensor_type = *((PyTensorType*)type);
  auto aten_type = tensor_type.aten_type();
  if (!aten_type) {
    throw unavailable_type(tensor_type);
  }
  return THPVariable_Wrap(torch::utils::legacy_tensor_ctor(*aten_type, args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject* Tensor_instancecheck(PyTensorType* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (THPVariable_Check(arg)) {
    auto& var = ((THPVariable*)arg)->cdata;
    // NB: This is a little unfortunate, in that if I do an isinstance check
    // against torch.cuda.FloatTensor, this will immediately initialize CUDA.
    // I originally thought that it would not be possible for aten_type_ to
    // be nullptr if you had a tensor of some type, in which case you can
    // skip initializign aten_type(), but TestAutograd.test_type_conversions
    // seems to violate this property (for whatever reason.)
    if (&var.type() == self->aten_type()) {
      Py_RETURN_TRUE;
    }
  }
  Py_RETURN_FALSE;
  END_HANDLE_TH_ERRORS
}

PyObject *Tensor_dtype(PyTensorType* self) {
  return torch::autograd::utils::wrap(self->dtype);
}

PyObject *Tensor_layout(PyTensorType* self) {
  return torch::autograd::utils::wrap(self->layout);
}

PyObject *Tensor_is_cuda(PyTensorType* self) {
  if (self->is_cuda) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

PyObject *Tensor_is_sparse(PyTensorType *self) {
  if (self->layout->layout == at::Layout::Strided) {
    Py_RETURN_FALSE;
  } else {
    Py_RETURN_TRUE;
  }
}

static struct PyMethodDef metaclass_methods[] = {
  {"__instancecheck__", (PyCFunction)Tensor_instancecheck, METH_O, nullptr},
  {nullptr}
};

typedef PyObject *(*getter)(PyObject *, void *);

static struct PyGetSetDef metaclass_properties[] = {
  {"dtype",        (getter)Tensor_dtype, nullptr, nullptr, nullptr},
  {"layout",       (getter)Tensor_layout, nullptr, nullptr, nullptr},
  {"is_cuda",      (getter)Tensor_is_cuda, nullptr, nullptr, nullptr},
  {"is_sparse",    (getter)Tensor_is_sparse, nullptr, nullptr, nullptr},
  {nullptr}
};

static PyTypeObject metaclass;

static void py_initialize_metaclass(PyTypeObject& metaclass) {
  ((PyObject*)&metaclass)->ob_refcnt = 1;
  metaclass.tp_basicsize = sizeof(PyTypeObject);
  metaclass.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  metaclass.tp_methods = metaclass_methods;
  metaclass.tp_getset = metaclass_properties;
  metaclass.tp_name = "torch.tensortype";
  metaclass.tp_base = &PyType_Type;
  if (PyType_Ready(&metaclass) < 0) {
    throw python_error();
  }
}

static void py_initialize_tensor_type(PyTypeObject& type, const char* name, PyObject* tp_dict) {
  // NOTE: we don't use the typical static declaration of PyTypeObject because
  // we need to initialize as many types as there are VariableType instances.
  // The typical PyVarObject_HEAD_INIT(nullptr, 0) is described in the Python
  // documentation: it initializes the refcnt to 1 and the other object header
  // fields to zero.
  memset(&type, 0, sizeof(PyTypeObject));
  ((PyObject*)&type)->ob_refcnt = 1;
  ((PyObject*)&type)->ob_type = &metaclass;
  type.tp_basicsize = sizeof(PyTensorType);
  type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  type.tp_name = name;
  type.tp_new = Tensor_new;
  if (PyType_Ready(&type) < 0) {
    throw python_error();
  }
  if (PyDict_Merge(type.tp_dict, tp_dict, 0) < 0) {
    throw python_error();
  }
}

static const char* get_module(Backend backend) {
  switch (backend) {
    case Backend::CPU: return "torch";
    case Backend::CUDA: return "torch.cuda";
    case Backend::SparseCPU: return "torch.sparse";
    case Backend::SparseCUDA: return "torch.cuda.sparse";
    default: AT_ERROR("invalid backend: ", toString(backend));
  }
}

static std::string get_name(Backend backend, ScalarType scalarType) {
  std::ostringstream ss;
  ss << get_module(backend) << "." << at::toString(scalarType) << "Tensor";
  return ss.str();
}

static THPObjectPtr get_storage_obj(const Type& type) {
  auto module_name = get_module(type.backend());
  auto module_obj = THPObjectPtr(PyImport_ImportModule(module_name));
  if (!module_obj) throw python_error();

  auto storage_name = std::string(at::toString(type.scalarType())) + "Storage";
  THPObjectPtr storage(PyObject_GetAttrString(module_obj.get(), storage_name.c_str()));
  if (!storage.get()) {
    throw TypeError("couldn't find storage object %s", storage_name.c_str());
  }
  return storage;
}

static void set_type(PyTensorType& type_obj, Backend backend, ScalarType scalarType) {
  // This field is lazily initialized from backend and scalar_type
  type_obj.aten_type_ = nullptr;
  type_obj.backend = static_cast<int>(backend);
  type_obj.scalar_type = static_cast<int>(scalarType);
  type_obj.layout = torch::getLayout(backend);
  type_obj.dtype = torch::getDtype(scalarType);
  type_obj.is_cuda = (backend == at::Backend::CUDA || backend == at::Backend::SparseCUDA);
}

static void set_name(PyTensorType& type_obj, const std::string& name) {
  size_t n = sizeof(type_obj.name);
  strncpy(type_obj.name, name.c_str(), n);
  type_obj.name[n - 1] = '\0';
}

static THPObjectPtr get_tensor_dict() {
  auto torch = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch) throw python_error();

  auto tensor_class = THPObjectPtr(PyObject_GetAttrString(torch, "Tensor"));
  if (!tensor_class) throw python_error();

  auto tensor_type = (PyTypeObject*)tensor_class.get();
  AT_CHECK(tensor_type->tp_base, "missing base type for Tensor");

  auto res = THPObjectPtr(PyDict_New());
  if (!res) throw python_error();

  if (PyDict_Merge(res.get(), tensor_type->tp_dict, 0) < 0) {
    throw python_error();
  }
  if (PyDict_Merge(res.get(), tensor_type->tp_base->tp_dict, 0) < 0) {
    throw python_error();
  }

  return res;
}

static std::vector<PyTensorType> tensor_types;

static void initialize_aten_types(std::vector<PyTensorType>& tensor_types) {
  // includes CUDA types even when PyTorch is not built with CUDA
  auto declared_types = torch::utils::all_declared_types();
  tensor_types.resize(declared_types.size());

  for (size_t i = 0, end = declared_types.size(); i != end; i++) {
    auto& tensor_type = tensor_types[i];
    Backend backend = declared_types[i].first;
    ScalarType scalar_type = declared_types[i].second;
    set_type(tensor_type, backend, scalar_type);
    set_name(tensor_type, get_name(backend, scalar_type));
  }
}

void initialize_python_bindings() {
  // Initialize the at::Type* pointers, name, and properties of the PyTensorType
  // vector. After this call, the vector must not be resized.
  initialize_aten_types(tensor_types);

  // Initialize the Python metaclass for the torch.FloatTensor, etc. types.
  // The metaclass handles __instancecheck__ checks and binds the dtype property
  // on the type objects.
  py_initialize_metaclass(metaclass);

  // Get the tp_dict of the Variable class. We copy function definitions
  // onto each Tensor type object so that they can be accessed via e.g.
  // `torch.FloatTensor.add`.
  auto tensor_dict = get_tensor_dict();

  // Initialize each Python type object torch.FloatTensor, torch.DoubleTensor, etc.
  for (auto& tensor_type : tensor_types) {
    py_initialize_tensor_type(tensor_type.py_type, tensor_type.name, tensor_dict.get());
  }

  // Add the type objects to their corresponding modules. e.g. torch.FloatTensor
  // is added to the `torch` module as `FloatTensor`. Also add all the type
  // objects to the set torch._tensor_classes.
  py_bind_tensor_types(tensor_types);

  // Use torch.float32 as the default tensor type
  set_default_tensor_type(torch::CPU(kFloat));
}

static void py_bind_tensor_types(const std::vector<PyTensorType>& tensor_types) {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) throw python_error();

  auto tensor_classes = THPObjectPtr(PyObject_GetAttrString(torch_module.get(), "_tensor_classes"));
  if (!tensor_classes) throw python_error();

  for (auto& tensor_type : tensor_types) {
    auto name = std::string(tensor_type.name);
    auto idx = name.rfind(".");
    auto type_name = name.substr(idx + 1);
    auto module_name = name.substr(0, idx);

    auto module_obj = THPObjectPtr(PyImport_ImportModule(module_name.c_str()));
    if (!module_obj) throw python_error();

    PyObject* type_obj = (PyObject*)&tensor_type;
    Py_INCREF(type_obj);
    if (PyModule_AddObject(module_obj.get(), type_name.c_str(), type_obj) < 0) {
      throw python_error();
    }
    if (PySet_Add(tensor_classes.get(), type_obj) < 0) {
      throw python_error();
    }
  }
}

static bool PyTensorType_Check(PyObject* obj) {
  auto it = std::find_if(tensor_types.begin(), tensor_types.end(),
    [obj](const PyTensorType& x) {
      return (PyObject*)&x == obj;
    });
  return it != tensor_types.end();
}

static PyTensorType& get_tensor_type(THPDtype *dtype, THPLayout *layout, bool is_cuda) {
  auto it = std::find_if(tensor_types.begin(), tensor_types.end(),
    [dtype, layout, is_cuda](const PyTensorType& x) {
      return x.dtype == dtype && x.layout == layout && x.is_cuda == is_cuda;
    });
  if (it == tensor_types.end()) {
    throw TypeError("invalid dtype object");
  }
  return *it;
}

void py_set_default_tensor_type(PyObject* obj) {
  PyTensorType *type;
  if (PyTensorType_Check(obj)) {
    type = (PyTensorType*)obj;
  } else {
    throw TypeError("invalid type object");
  }
  auto aten_type = type->aten_type();
  if (!aten_type) {
    throw unavailable_type(*type);
  }
  set_default_tensor_type(*aten_type);
}

void py_set_default_dtype(PyObject* obj) {
  PyTensorType *type;
  if (THPDtype_Check(obj)) {
    auto &current_default = get_default_tensor_type();
    type = &get_tensor_type((THPDtype*)obj, torch::getLayout(current_default.backend()),
                            torch::getDeviceType(current_default) == at::Device::Type::CUDA);
  } else {
    throw TypeError("invalid type object");
  }
  auto aten_type = type->aten_type();
  if (!aten_type) {
    throw unavailable_type(*type);
  }
  set_default_tensor_type(*aten_type);
}

void set_default_tensor_type(const at::Type& type) {
  if (!at::isFloatingType(type.scalarType())) {
    throw TypeError("only floating-point types are supported as the default type");
  }
  if (!type.is_variable() && !type.is_undefined()) {
    throw TypeError("only variable types are supported");
  }
  if (type.is_sparse()) {
    throw TypeError("only dense types are supported as the default type");
  }

  // get the storage first, so if it doesn't exist we don't change the default tensor type
  THPObjectPtr storage = get_storage_obj(type);
  default_tensor_type = const_cast<Type*>(&type);

  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) throw python_error();

  if (PyObject_SetAttrString(torch_module.get(), "Storage", storage) != 0) {
    // technically, we should undo the change of default tensor type.
    throw python_error();
  }
}

at::Type& get_default_tensor_type() {
  AT_ASSERT(default_tensor_type);
  return *default_tensor_type;
}

Device getDevice(const at::Tensor& tensor) {
  if (tensor.is_cuda()) {
    return at::Device(at::DeviceType::CUDA, tensor.get_device());
  }
  return at::Device(at::DeviceType::CPU);
}
}} // namespace torch::tensors
