#include <torch/csrc/distributed/rpc/request_callback_impl.h>

#include <c10/util/C++17.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/context/context.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rref_backward_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rref_backward_resp.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/profiler/server_process_global_profiler.h>
#include <torch/csrc/distributed/rpc/python_call.h>
#include <torch/csrc/distributed/rpc/python_remote_call.h>
#include <torch/csrc/distributed/rpc/python_resp.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_impl.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/distributed/rpc/script_resp.h>
#include <torch/csrc/distributed/rpc/unpickled_python_call.h>
#include <torch/csrc/distributed/rpc/unpickled_python_remote_call.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/frontend/code_template.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace distributed {
namespace rpc {

using namespace torch::distributed::autograd;

namespace {

std::unique_ptr<RpcCommandBase> deserializePythonRpcCommandReference(
    RpcCommandBase& rpc,
    const MessageType& messageType) {
  switch (messageType) {
    case MessageType::PYTHON_CALL: {
      auto& pc = static_cast<PythonCall&>(rpc);
      return std::make_unique<UnpickledPythonCall>(
          pc.serializedPyObj(), pc.isAsyncExecution());
    }
    case MessageType::PYTHON_REMOTE_CALL: {
      auto& prc = static_cast<PythonRemoteCall&>(rpc);
      return std::make_unique<UnpickledPythonRemoteCall>(
          prc.serializedPyObj(),
          prc.retRRefId(),
          prc.retForkId(),
          prc.isAsyncExecution());
    }
    case MessageType::FORWARD_AUTOGRAD_REQ: {
      // Deserialize the wrapped RPC if it contains Python UDF
      auto& rwa = static_cast<RpcWithAutograd&>(rpc);
      auto& wrappedRpc = rwa.wrappedRpc();
      auto pythonRpc = deserializePythonRpcCommandReference(
          wrappedRpc, rwa.wrappedMessageType());
      if (pythonRpc) {
        rwa.setWrappedRpc(std::move(pythonRpc));
      }
      return nullptr;
    }
    case MessageType::RUN_WITH_PROFILING_REQ: {
      // Deserialize wrapped RPC if it contains python call
      auto& rpcWithProfilingReq = static_cast<RpcWithProfilingReq&>(rpc);
      auto& wrappedRpc = rpcWithProfilingReq.wrappedRpc();
      auto pythonRpc = deserializePythonRpcCommandReference(
          wrappedRpc, rpcWithProfilingReq.wrappedMessageType());
      if (pythonRpc) {
        rpcWithProfilingReq.setWrappedRpc(std::move(pythonRpc));
      }
      return nullptr;
    }
    default: {
      return nullptr;
    }
  }
}

void processPythonExecution(
    const py::object& pyFn,
    const c10::intrusive_ptr<JitFuture>& responseFuture,
    bool isAsyncExecution,
    std::function<void(
        const py::object&,
        PythonRpcHandler&,
        const c10::intrusive_ptr<JitFuture>&)> postProcessing) {
  std::shared_ptr<jit::PythonFutureWrapper> pyFuture;
  auto& pythonRpcHandler = PythonRpcHandler::getInstance();
  {
    py::gil_scoped_acquire acquire;
    auto result = pythonRpcHandler.runPythonUdf(pyFn);

    if (pythonRpcHandler.isRemoteException(result) || !isAsyncExecution) {
      // Hit exception when running the user function or there is no async
      // execution. Not releasing GIL before serialize to avoid an additional
      // context switch.
      postProcessing(result, pythonRpcHandler, responseFuture);
      return;
    }

    try {
      pyFuture = result.cast<std::shared_ptr<jit::PythonFutureWrapper>>();
    } catch (const py::cast_error& e) {
      auto type = result.get_type();
      auto errMsg = c10::str(
          e.what(),
          ". Functions decorated with @rpc.async_function must return a "
          "torch.futures.Future object, but got ",
          type.attr("__module__").cast<std::string>(),
          ".",
          type.attr("__qualname__").cast<std::string>());
      throw std::runtime_error(errMsg);
    }
  }

  pyFuture->fut->addCallback([responseFuture,
                              postProcessing{std::move(postProcessing)},
                              &pythonRpcHandler](JitFuture& jitFuture) {
    py::gil_scoped_acquire acquire;
    postProcessing(
        jit::toPyObject(jitFuture.value()), pythonRpcHandler, responseFuture);
  });
}

} // anonymous namespace

std::unique_ptr<RpcCommandBase> RequestCallbackImpl::
    deserializePythonRpcCommand(
        std::unique_ptr<RpcCommandBase> rpc,
        const MessageType& messageType) const {
  auto pythonRpc = deserializePythonRpcCommandReference(*rpc, messageType);
  return pythonRpc ? std::move(pythonRpc) : std::move(rpc);
}

void RequestCallbackImpl::processScriptCall(
    RpcCommandBase& rpc,
    const std::function<void(Message)>& markComplete,
    const c10::intrusive_ptr<JitFuture>& responseFuture) const {
  auto& scriptCall = static_cast<ScriptCall&>(rpc);
  auto& stack = scriptCall.stackRef();
  if (scriptCall.hasOp()) {
    processScriptCallOp(scriptCall, markComplete, stack);
    return;
  }

  // runAsync() starts in the calling thread, but may return an uncompleted
  // future (though for non-async code, it will typically be completed).
  // If it was async, our callback will typically be invoked by the
  // continuation on an at::launch() thread.
  auto jitFuture = PythonRpcHandler::getInstance()
                       .jitCompilationUnit()
                       ->get_function(scriptCall.qualifiedName())
                       .runAsync(stack);

  // Fastpath: avoid callbacks if not neeeded.
  if (jitFuture->completed() && !scriptCall.isAsyncExecution()) {
    markComplete(std::move(ScriptResp(jitFuture->value())).toMessage());
    return;
  }

  jitFuture->addCallback([responseFuture,
                          isAsyncExecution = scriptCall.isAsyncExecution(),
                          markComplete](JitFuture& jitFutureCaptured) {
    try {
      JitFuture& jitFuture = isAsyncExecution
          ? *jitFutureCaptured.value().toFuture()
          : jitFutureCaptured;

      // Setup response callback appropriately.
      auto responseCb = [responseFuture](JitFuture& jitFuture) {
        try {
          Message m = ScriptResp(jitFuture.value()).toMessage();
          responseFuture->markCompleted(
              IValue(c10::make_intrusive<Message>(std::move(m))));
        } catch (const std::exception& /* unused */) {
          responseFuture->setError(std::current_exception());
        }
      };

      // Call inline if we don't have async execution.
      isAsyncExecution ? jitFuture.addCallback(responseCb)
                       : responseCb(jitFuture);
    } catch (const std::exception& /* unused */) {
      responseFuture->setError(std::current_exception());
    }
  });
}

void RequestCallbackImpl::processPythonCall(
    RpcCommandBase& rpc,
    const std::function<void(Message)>& markComplete,
    const c10::intrusive_ptr<JitFuture>& responseFuture) const {
  auto& upc = static_cast<UnpickledPythonCall&>(rpc);
  try {
    processPythonExecution(
        upc.pythonUdf(),
        responseFuture,
        upc.isAsyncExecution(),
        [](const py::object& result,
           PythonRpcHandler& pythonRpcHandler,
           const c10::intrusive_ptr<JitFuture>& responseFuture) {
          // Check we have GIL.
          DCHECK(PyGILState_Check());

          auto serializedPyObj = pythonRpcHandler.serialize(result);
          py::gil_scoped_release release;
          auto m =
              std::move(PythonResp(std::move(serializedPyObj))).toMessage();
          responseFuture->markCompleted(
              IValue(c10::make_intrusive<Message>(std::move(m))));
        });
  } catch (std::exception& e) {
    // Pass a dummy message ID since it will be overwritten anyways.
    responseFuture->markCompleted(IValue(
        c10::make_intrusive<Message>(createExceptionResponse(e.what(), -1))));
  }
}

TypePtr RequestCallbackImpl::getScriptRemoteCallType(
    ScriptRemoteCall& scriptRemoteCall) const {
  TypePtr returnType;
  if (scriptRemoteCall.hasOp()) {
    returnType = scriptRemoteCall.op()->schema().returns()[0].type();
  } else {
    returnType = PythonRpcHandler::getInstance()
                     .jitCompilationUnit()
                     ->get_function(scriptRemoteCall.qualifiedName())
                     .getSchema()
                     .returns()
                     .at(0)
                     .type();
  }
  return returnType;
}

void RequestCallbackImpl::processScriptRemoteCall(
    ScriptRemoteCall& scriptRemoteCall,
    const std::function<void(void)>& postProcessing,
    std::vector<at::IValue>& stack,
    const c10::intrusive_ptr<OwnerRRef>& ownerRRef) const {
  if (scriptRemoteCall.hasOp()) {
    processScriptRemoteCallOp(
        scriptRemoteCall, postProcessing, stack, ownerRRef);
    return;
  }

  auto isAsyncExecution = scriptRemoteCall.isAsyncExecution();
  auto asyncPostProcessing = [ownerRRef, postProcessing, isAsyncExecution](
                                 c10::ivalue::Future& jitFuture) mutable {
    // The user function will return a JIT future, install
    // setRRefValue and postProcessing to that valueFuture
    try {
      JitFuture& valueJitFuture =
          isAsyncExecution ? *jitFuture.value().toFuture() : jitFuture;

      // Setup callback.
      auto setRRefValue = [ownerRRef,
                           postProcessing](JitFuture& valueJitFuture) mutable {
        try {
          ownerRRef->setValue(valueJitFuture.value());
        } catch (const std::exception& e) {
          ownerRRef->setError(std::current_exception());
        }
        postProcessing();
      };

      // Call inline if not async execution.
      isAsyncExecution ? valueJitFuture.addCallback(setRRefValue)
                       : setRRefValue(valueJitFuture);
    } catch (std::exception& e) {
      ownerRRef->setError(std::current_exception());
      postProcessing();
    }
  };

  c10::intrusive_ptr<c10::ivalue::Future> jitFuture;
  try {
    jitFuture = PythonRpcHandler::getInstance()
                    .jitCompilationUnit()
                    ->get_function(scriptRemoteCall.qualifiedName())
                    .runAsync(stack);
    if (jitFuture->completed()) { // short-cut.
      asyncPostProcessing(*jitFuture);
      return;
    }
  } catch (const std::exception& e) {
    asyncPostProcessing(*jitFuture);
    return;
  }
  jitFuture->addCallback(
      [asyncPostProcessing{std::move(asyncPostProcessing)}](
          JitFuture& jitFuture) mutable { asyncPostProcessing(jitFuture); });
}

void RequestCallbackImpl::processPythonRemoteCall(
    RpcCommandBase& rpc,
    const std::function<void(Message)>& markComplete,
    const c10::intrusive_ptr<JitFuture>& responseFuture,
    std::shared_ptr<LazyStreamContext> lsctx) const {
  auto& uprc = static_cast<UnpickledPythonRemoteCall&>(rpc);

  const auto& rrefId = uprc.rrefId();
  const auto& forkId = uprc.forkId();
  auto& ctx = RRefContext::getInstance();

  c10::intrusive_ptr<OwnerRRef> ownerRRef;
  if (rrefId == forkId) {
    // Creating an owner RRef on self, should already exist in owners map
    ownerRRef =
        fromRRefInterface(ctx.getOwnerRRef(rrefId, /* forceCreated */ true)
                              ->constValue()
                              .toRRef());
  } else {
    ownerRRef = ctx.getOrCreateOwnerRRef(rrefId, PyObjectType::get());
  }
  // NOLINTNEXTLINE(clang-diagnostic-unused-variable)
  auto& pythonRpcHandler = PythonRpcHandler::getInstance();

  if (rrefId != forkId) {
    // Caller is a user and callee is the owner, add fork
    //
    // NB: rrefId == forkId is true if and only if calling remote to self.
    // In that case both the caller and the callee will access the
    // OwnerRRef. Hence, on the callee side (here), it should not call
    // addForkOfOwner as it is not a fork. To allow callee to distinguish
    // when this request is sent to self, the caller will set forkId using
    // rrefId (OwnerRRef does not have a forkId anyway).
    ctx.addForkOfOwner(rrefId, forkId);
  }

  try {
    processPythonExecution(
        uprc.pythonUdf(),
        responseFuture,
        uprc.isAsyncExecution(),
        [ownerRRef, rrefId, forkId, markComplete, lsctx = std::move(lsctx)](
            const py::object& result,
            PythonRpcHandler& /* unused */,
            const c10::intrusive_ptr<JitFuture>& responseFuture) {
          // Check we have GIL.
          DCHECK(PyGILState_Check());

          IValue py_ivalue = jit::toIValue(result, PyObjectType::get());

          py::gil_scoped_release release;
          ownerRRef->recordAllStreams(lsctx);
          ownerRRef->setValue(std::move(py_ivalue));
          auto m = RemoteRet(rrefId, forkId).toMessage();
          responseFuture->markCompleted(
              IValue(c10::make_intrusive<Message>(std::move(m))));
        });
  } catch (py::error_already_set& e) {
    // py::error_already_set requires GIL to destruct, take special care.
    ownerRRef->setError(std::current_exception());
    py::gil_scoped_acquire acquire;
    e.restore();
    PyErr_Clear();
  } catch (std::exception& e) {
    ownerRRef->setError(std::current_exception());
    markComplete(RemoteRet(rrefId, forkId).toMessage());
  }
}

void RequestCallbackImpl::processPythonRRefFetchCall(
    RpcCommandBase& rpc,
    const c10::intrusive_ptr<JitFuture>& responseFuture,
    std::shared_ptr<LazyStreamContext> lsctx) const {
  // Making this lambda mutable to allow move-capture it in callbacks
  auto postProcessing = [responseFuture, lsctx = std::move(lsctx)](
                            const c10::intrusive_ptr<OwnerRRef>& rref) mutable {
    auto whenValueSet = rref->getFuture();
    if (whenValueSet->hasError()) {
      responseFuture->setError(whenValueSet->exception_ptr());
      return;
    }
    try {
      auto& pythonRpcHandler = PythonRpcHandler::getInstance();
      std::shared_ptr<SerializedPyObj> result;
      {
        // Need this GIL to guard jit::toPyObj and destruct its returned
        // py::object
        py::gil_scoped_acquire acquire;
        result = std::make_shared<SerializedPyObj>(
            pythonRpcHandler.serialize(jit::toPyObject(rref->getValue())));
      }
      Message m =
          PythonRRefFetchRet(std::move(*result).toIValues()).toMessage();
      rref->blockAllStreams(lsctx);
      responseFuture->markCompleted(
          IValue(c10::make_intrusive<Message>(std::move(m))));
    } catch (py::error_already_set& e) {
      // py::error_already_set requires GIL to destruct, take special care.
      responseFuture->setError(
          std::make_exception_ptr(std::runtime_error(e.what())));
      py::gil_scoped_acquire acquire;
      e.restore();
      PyErr_Clear();
    } catch (const std::exception& /* unused */) {
      responseFuture->setError(std::current_exception());
    }
  };

  auto& prf = static_cast<PythonRRefFetchCall&>(rpc);
  auto& ctx = RRefContext::getInstance();

  auto futureOwner = ctx.getOwnerRRef(prf.rrefId());
  if (futureOwner->completed()) {
    auto rref = fromRRefInterface(futureOwner->constValue().toRRef());
    if (rref->hasValue()) {
      // optional fast-path, the OwnerRRef has been created
      postProcessing(rref);
      return;
    }
  }

  futureOwner->addCallback([postProcessing{std::move(postProcessing)}](
                               JitFuture& futureOwner) mutable {
    const auto& rref = fromRRefInterface(futureOwner.constValue().toRRef());

    // Our response is satisfied when the the rpc.remote() request
    // finishes executing on the owner.
    rref->getFuture()->addCallback(
        [rref, postProcessing{std::move(postProcessing)}](
            JitFuture& /* unused */) mutable { postProcessing(rref); });
  });
}

void RequestCallbackImpl::handleRRefDelete(
    c10::intrusive_ptr<RRef>& rref) const {
  if (rref && rref->isPyObj()) {
    py::gil_scoped_acquire acquire;
    rref.reset();
  }
}

c10::intrusive_ptr<JitFuture> RequestCallbackImpl::processRpcWithErrors(
    RpcCommandBase& rpc,
    const MessageType& messageType,
    std::shared_ptr<LazyStreamContext> ctx) const {
  try {
    return processRpc(rpc, messageType, std::move(ctx));
  } catch (py::error_already_set& e) {
    // Pass a dummy message ID since it will be overwritten anyways.
    auto future = asFuture(handleError(e, messageType, -1));
    // There are request callback impls in Python, where Python
    // exceptions could be thrown. For releasing Python exception
    // py::objects, GIL must be held.
    py::gil_scoped_acquire acquire;
    e.restore(); // Release ownership on py::objects and also restore
                 // Python Error Indicator.
    PyErr_Clear(); // Clear the Python Error Indicator as we has
                   // recorded the exception in the response message.
    return future;
  } catch (std::exception& e) {
    // Pass a dummy message ID since it will be overwritten anyways.
    return asFuture(handleError(e, messageType, -1));
  }
}

bool RequestCallbackImpl::cudaAvailable() const {
#ifdef USE_CUDA
  return true;
#else
  return false;
#endif
}

void RequestCallbackImpl::processRRefBackward(
    RpcCommandBase& rpc,
    const c10::intrusive_ptr<JitFuture>& responseFuture) const {
  auto& rrefBackwardReq = static_cast<RRefBackwardReq&>(rpc);

  // Get all fields
  const auto& rrefId = rrefBackwardReq.getRRefId();
  const auto& autogradContextId = rrefBackwardReq.getAutogradContextId();
  const auto& retainGraph = rrefBackwardReq.retainGraph();

  auto futureOwner = RRefContext::getInstance().getOwnerRRef(rrefId);
  futureOwner->addCallback(
      [responseFuture, autogradContextId, retainGraph](JitFuture& futureOwner) {
        const auto& rref = fromRRefInterface(futureOwner.constValue().toRRef());
        auto whenValueSet = rref->getFuture();

        whenValueSet->addCallback(
            [responseFuture, rref, autogradContextId, retainGraph](
                JitFuture& whenValueSet) {
              if (whenValueSet.hasError()) {
                responseFuture->setError(whenValueSet.exception_ptr());
                return;
              }

              try {
                // Run backward (TODO: make this async?).
                PyRRef::backward(autogradContextId, retainGraph, rref);

                // Return the response.
                Message m = RRefBackwardResp().toMessage();
                responseFuture->markCompleted(
                    IValue(c10::make_intrusive<Message>(std::move(m))));
              } catch (const std::exception& /* unused */) {
                responseFuture->setError(std::current_exception());
              }
            });
      });
}

} // namespace rpc
} // namespace distributed
} // namespace torch
