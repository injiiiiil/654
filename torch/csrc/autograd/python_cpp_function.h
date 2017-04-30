#pragma once

#include <Python.h>
#include <memory>
#include <typeinfo>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/Exceptions.h"

namespace torch { namespace autograd {

struct THPCppFunction {
  PyObject_HEAD
  std::shared_ptr<Function> cdata;
};

template<typename Ctor>
PyObject* CppFunction_pynew(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  THPObjectPtr obj = type->tp_alloc(type, 0);
  if (!obj) return NULL;
  THPCppFunction* f = (THPCppFunction*)obj.get();
  HANDLE_TH_ERRORS
  new (&f->cdata) std::shared_ptr<Function>(Ctor()(args));
  END_HANDLE_TH_ERRORS
  if (!f->cdata) {
    return NULL;
  }
  return obj.release();
}

PyObject* next_functions(THPCppFunction* self, PyObject* hook);
PyObject* register_hook_dict(PyObject* self, PyObject* _var);
PyObject* register_hook(PyObject* self, PyObject* hook);

static struct PyMethodDef default_methods[] = {
  {(char*)"_register_hook_dict", (PyCFunction)register_hook_dict, METH_O, NULL},
  {(char*)"register_hook", (PyCFunction)register_hook, METH_O, NULL},
  {NULL}
};

static struct PyGetSetDef default_properties[] = {
  {(char*)"next_functions", (getter)next_functions, NULL, NULL, NULL},
  {NULL}
};

PyTypeObject* _initFunctionPyTypeObject(PyTypeObject& type, const char* name,
  PyGetSetDef* function_properties, PyMethodDef* function_methods);

PyObject* registerFunctionHook(Function& fn, PyObject* hook);

template<typename Ctor>
PyTypeObject* createForwardFunctionPyTypeObject(PyTypeObject& type, const char* name,
  PyGetSetDef* function_properties=NULL, PyMethodDef* function_methods=NULL)
{
  type.tp_new = &CppFunction_pynew<Ctor>;
  return _initFunctionPyTypeObject(type, name, function_properties, function_methods);
}

void registerCppFunction(const std::type_info& type, PyTypeObject* pytype);
PyObject* functionToPyObject(std::shared_ptr<Function> cdata);

}} // namespace torch::autograd
