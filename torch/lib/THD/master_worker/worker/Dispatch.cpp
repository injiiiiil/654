#include <TH/THStorage.h>
#include <cstdint>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "../../base/TensorTraits.hpp"
#include "../../base/tensors/THTensor.hpp"
#include "../common/Functions.hpp"
#include "../common/RPC.hpp"
#include "../master/Master.hpp"
#include "Worker.hpp"

namespace thd {
namespace worker {

namespace detail {

static std::unique_ptr<Tensor> createTensor(Type type) {
  if (type == Type::UCHAR)
    return std::unique_ptr<Tensor>(new THTensor<unsigned char>());
  else if (type == Type::CHAR)
    return std::unique_ptr<Tensor>(new THTensor<char>());
  else if (type == Type::SHORT)
    return std::unique_ptr<Tensor>(new THTensor<short>());
  else if (type == Type::INT)
    return std::unique_ptr<Tensor>(new THTensor<int>());
  else if (type == Type::LONG)
    return std::unique_ptr<Tensor>(new THTensor<long>());
  else if (type == Type::FLOAT)
    return std::unique_ptr<Tensor>(new THTensor<float>());
  else if (type == Type::DOUBLE)
    return std::unique_ptr<Tensor>(new THTensor<double>());
  throw std::invalid_argument("passed characted doesn't represent a tensor type");
}

static void construct(rpc::RPCMessage& raw_message) {
  // TODO: assert_empty(raw_message)
  Type type = rpc::unpackType(raw_message);
  thd::tensor_id_type id = rpc::unpackTensorAsId(raw_message);
  workerTensors.insert(std::make_pair(
    id,
    createTensor(type)
  ));
}

static void constructWithSize(rpc::RPCMessage& raw_message) {
  // TODO: assert_empty(raw_message)
  Type type = rpc::unpackType(raw_message);
  tensor_id_type id = rpc::unpackTensorAsId(raw_message);
  THLongStorage *sizes = rpc::unpackTHLongStorage(raw_message);
  THLongStorage *strides = rpc::unpackTHLongStorage(raw_message);
}

static void add(rpc::RPCMessage& raw_message) {
//THTensor& result = parse_tensor(raw_message);
  //THTensor& source = parse_tensor(raw_message);
  //double x = parse_scalar(raw_message);
  //assert_end(raw_message);
  //result.add(source, x);
}

static void free(rpc::RPCMessage& raw_message) {
  unsigned long long tensor_id = unpackInteger(raw_message);
  (void)workerTensors.erase(tensor_id);
}

using dispatch_fn = void (*)(rpc::RPCMessage&);
using Functions = thd::Functions;


static const std::unordered_map<std::uint16_t, dispatch_fn> functions {
    {Functions::construct, construct},
    {Functions::constructWithSize, constructWithSize},
    {Functions::add, add},
    {Functions::free, free}
};

} // namespace detail

std::string execute(std::unique_ptr<rpc::RPCMessage> raw_message_ptr) {
  try {
    // TODO: unify the function id type (it's in rpc:: now)
    auto &raw_message = *raw_message_ptr;
    uint16_t fid = rpc::unpackFunctionId(raw_message);
    auto iter = detail::functions.find(fid);
    if (iter != detail::functions.end())
      (*iter->second)(raw_message);
    else
      throw std::invalid_argument("invalid function id");
    return std::string();
  } catch(std::exception& e) {
    return std::string(e.what());
  }
}

} // namespace worker
} // namespace thd
