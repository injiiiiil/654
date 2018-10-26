#pragma once

#include <unordered_map>
#include <stack>

#include <ATen/core/Allocator.h>

#include "caffe2/core/common.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/proto/torch_pb.h"
#include "caffe2/serialize/inline_container.h"

namespace at {
namespace serialize {

// multiple tensor may point to the data
// this class contains one unique pointer to the real data
class SharedData {
 public:
  explicit SharedData(uint64_t record_id, at::DataPtr&& data_ptr)
    : recordId_(record_id), dataPtr_(std::move(data_ptr)){}

  // getters
  void* rawData() {
    return dataPtr_->get();
  }

  uint64_t recordId() const {
    return recordId_;
  }

  // setters
  void setDataPtr(at::DataPtr&& data_ptr) {
    dataPtr_ = std::move(data_ptr);
  }

  void setRecordId(uint64_t record_id) {
    recordId_ = record_id;
  }

 private:
  uint64_t recordId_;
  at::DataPtr dataPtr_;
  uint64_t size_;
}

struct IntermediateDeviceOption {
  int32_t deviceType = 0;
  bool hasDeviceId = false;
  int32_t deviceId;
};

class IntermediateTensor final {
 public:
  // constructor
  IntermediateTensor() = default;

  // update functions
  void update(caffe2::TensorProto* tensor_proto,
      std::unordered_map<uint64_t, std::shared_ptr<SharedData>>* id_tensor) {
    AT_ASSERTM(tensor_proto->has_data_type(), "no data_type in TensorProto!");
    dataType_ = tensor_proto->data_type();
    for (int i = 0; i < tensor_proto->dims_size()) {
      dims_.push_back(tensor_proto->dims(i));
    }
    AT_ASSERTM(tensor_proto->has_name(), "no name in TensorProto!");
    name_ = tensor_proto->name();
    if (tensor_proto->has_device_detail()) {
      const auto& device_detail = tensor_proto->device_detail();
      deviceDetail_.deviceType = device_detail.device_type;
      if (device_detail.has_device_id()) {
        deviceDetail_.hasDeviceId = true;
        deviceDetail_.devicedId = device_detail.device_id();
      }
      if (device_detail.has_random_seed()) {
        deviceDetail_.hasRandomSeed = true;
        deviceDetail_.randomSeed = device_detail.random_seed();
      }
      if (device_detail.has_node_name()) {
        deviceDetail_.hasNodeName = true;
        deviceDetail_.nodeName = device_detail.nodeName();
      }
      for (int i = 0; i < device_detail.extra_info_size(); ++i) {
        deviceDetail_.extraInfo.emplace_back(device_detail.extra_info(i));
      }
    }
    AT_ASSERTM(tensor_proto->has_storage_type(), "no storage_type in TensorProto!");
    int64_t storage_type = tensor_proto->storage_type();
    switch (storage_type) {
      case TensorProto_StorageType_TYPED:
        // TODO
        AT_ERROR("Not implemented!");
      case TensorProto_StorageType_RAW:
        // TODO
        AT_ERROR("Not implemented!");
      case TensorProto_StorageType_EXTERNAL:
        AT_ERROR(tensor_proto->has_external_data(), "storage type is EXTERNAL, "
            "but no external_data in TensorProto!");
        const auto& external_data = tensor_proto->external_data();
        if (external_data->has_offset()) {
          offset_ = external_data->offset();
        }
        for (int i = 0; i < external_data.strides_size(); ++i) {
          strides_->push_back(external_data.strides(i));
        }
        AT_ASSERTM(external_data->has_source_type(), "no source_type in TensorProto!");
        int64_t source_type = external_data->source_type();
        if (source_type == ExternalDataProto_SourceType::ExternalDataProto_SourceType_INLINE_CONTAINER) {
          AT_ASSERTM(external_data->has_record_id(), "no record_id in ExternalDataProto!");
          int64_t record_id = external_data->record_id();
          auto it = id_tensor.find(record_id);
          if (it == id_tensor.end()) {
            AT_ERROR("Tensor's data is missing, tensor name is ", name_);
          }
          dataPtr_ = std::move(it->second);
        } else if (source_type == ExternalDataProto_SourceType::ExternalDataProto_SourceType_SIMPLE_FILE) {
          // TODO
          AT_ERROR("Not implemented!");
        } else {
          // TODO
          AT_ERROR("Not implemented!");
        }
        break;
      case TensorProto_StorageType_NO_CONTENT:
        noContent_ = true;
        break;
      default:
        AT_ERROR("Uknown type %lld", storage_type);
    }
  }

  void updateData(std::shared_ptr<SharedData> data) {
    this->data_ = data;
  }

  void dump(caffe2::TensorProto* tensor_proto) {
    for (auto dim : dims_) {
      tensor_proto->add_dims(dim);
    }
    tensor_proto->set_data_type(dataType_);
    tensor_proto->set_storage_type(caffe2::TensorProto_StorageType::TensorProto_StorageType_EXTERNAL);
    caffe2::ExternalDataProto* data_proto = tensor_proto->mutable_exteranl_data();
    data_proto->set_source_type(caffe2::ExternalDataProto_SourceType_INLINE_CONTAINER);
    data_proto->set_record_id(std::to_string(data_->recordId));
    data_proto->set_offset(offset_);
    for (auto stride : strides_) {
      data_proto->add_strides(stride);
    }
    caffe2::DeviceOption* device_detail = data_proto->mutable_device_detail();
    device_detail->set_device_type(deviceDetail_.deviceType);
    if (deviceDetail_.hasDeviceId) {
      device_detail->set_device_id(deviceDetail_.deviceId);
    }
    // TODO maybe later to support RAW
  }

  // getters/setters
  std::shared_ptr<SharedData> data() {
    return data_;
  }

  const IntermediateDeviceOption& deviceDetail() const {
    return deviceDetail_;
  }

  std::vector<int64_t>* mutableDims() {
    return &dims_;
  }

  std::vector<int64_t>* mutableStrides() {
    return &strides_;
  }

  IntermediateDeviceOption* mutableDeviceDetail() {
    return &deviceDetail_;
  }

  void setDataType(int64_t data_type) {
    dataType_ = data_type;
  }

  void setData(std::shared_ptr<SharedData> data) {
    data_ = data;
  }

 private:
  std::string name_;
  int64_t dataType_;
  std::vector<int64_t> dims_;
  int64_t offset_;
  std::vector<int64_t> strides_;
  // TODO: since we still have 2 different Tensor classes in Caffe2 and PyTorch
  // right now, let's just store the data pointer, and create Tensors
  // while converting IntermediateModel to the JitScriptModule/Predictor/etc.
  std::shared_ptr<SharedData> data_;
  IntermediateDeviceOption deviceDetail_;
  bool noContent_ = false;
};

class IntermediateParameter final {
 public:
  // constructor
  IntermediateParameter() = default;

  explicit IntermediateParameter(torch::ParameterDef* param_def,
      std::unordered_map<uint64_t, std::shared_ptr<SharedData>>* id_data) {
    isBuffer_ = param_def->is_buffer();
    requireGradient_ = param_def->require_gradient();
    if (param_def->has_tensor()) {
      tensor_.update(param_def->mutable_tensor(), id_tensor);
    } else {
      // TODO
      AT_ERROR("A ParameterDef does not contain any tensor!");
    }
  }

  void dump(torch::ParameterDef* param_def) {
    param_def->set_name(name_);
    param_def->set_is_buffer(isBuffer_);
    param_def->set_is_gradient(isGradient_);
    caffe2::TensorProto* tensor_def = param_def->mutable_tensor();
    tensor_.dump(tensor_def);
  }

  // getters/setters
  const std::string& name() const {
    return name_;
  }

  bool isBuffer() const {
    return isBuffer_;
  }

  bool requireGradient() const {
    return requireGradient_;
  }

  IntermediateTensor* mutableTensor() {
    return &tensor_;
  }

  void setIsBuffer(bool is_buffer) {
    isBuffer_ = is_buffer;
  }

  void setRequireGradient(bool require_gradient) {
    requireGradient_ = require_gradient;
  }

 private:
  bool isBuffer_ = false;
  bool requireGradient_ = false;
  IntermediateTensor tensor_;
  std::string name_;
};

class IntermediateMethod final {
 public:
  explicit IntermediateMethod(torch::MethodDef* method_def) {
    if (method_def->has_name()) {
      name_ = method_def->name();
    } else {
      // TODO throw exception
    }
    if (method_def->has_torch_script()) {
      torchScript_ = method_def->torch_script();
    } else if (method_def->has_graph()) {
      graph_ = caffe2::make_unique(release_graph());
    } else {
      // TODO throw exception
    }
  }

  void dump(torch::MethodDef* method_def) {
    // TODO
    AT_ERROR("Not implemented yet!");
  }

 private:
  std::string name_;
  std::unique_ptr<caffe2::NetDef> graph_;
  std::string torchScript_;
};

class IntermediateModule final {
 public:
  // constructor
  IntermediateModule() = default;

  // update functions
  void update(torch::ModuleDef* module_def,
      std::unordered_map<uint64_t, std::shared_ptr<SharedData>>* id_data) {
    if (module_def->has_name()) {
      name_ = module_def->name();
    } else {
      // TODO throw exception
    }
    for (int i = 0; i < module_def->parameters_size(); ++i) {
      auto* param_def = module_def->mutable_parameters(i);
      parameters_.emplace_back(param_def, id_tensor);
    }

    for (int i = 0; i < module_def->submodules_size(); ++i) {
      auto* sub_def = module_def->mutable_submodules(i);
      submodules_.emplace_back(sub_def, id_tensor);
    }

    for (int i = 0; i < module_def->methods_size(); ++i) {
      auto* method_def = module_def->mutable_methods(i);
      methods_.emplace_back(method_def);
    }
  }

  void dump(torch::ModuleDef* module_def) {
    module_def->set_name(name_);

    for (int i = 0; i < parameters_.size(); ++i) {
      module_def->add_parameters();
      torch::ParameterDef* param_def = module_def->mutable_parameters(module_def->parameters_size() - 1);
      parameters_.at(i)->dump(param_def);
    }

    for (int i = 0; i < submodules_.size(); ++i) {
      module_def->add_submodules();
      torch::ModuleDef* sub_def = module_def->mutable_submodules(module_def->submodules_size() - 1);
      submodules_.at(i)->dump(sub_def);
    }

    for (int i = 0; i < methods_.size(); ++i) {
      module_def->add_methods();
      torch::MethodDef* method_def = module_def->mutable_methods(module_def->methods_size() - 1);
      methods_.at(i)->dump(method_def);
    }
  }

  // getters/setters
  std::vector<IntermediateParameter>* mutableParameters() {
    return &parameters_;
  }

  std::vector<IntermediateModule>* mutableSubmodules() {
    return &submodules_;
  }

  std::vector<IntermediateMethod>* mutableMethods() {
    return &methods_;
  }

  void setName(const std::string& name) {
    name_ = name;
  }

 private:
  std::string name_;
  std::vector<IntermediateParameter> parameters_;
  std::vector<IntermediateModule> submodules_;
  std::vector<IntermediateMethod> methods_;
};

class IntermediateModel final {
 public:
  // constructor
  IntermediateModel() = default;

  // update functions
  void update(torch::ModelDef* model_def,
      std::unordered_map<uint64_t, std::shared_ptr<SharedData>* id_data) {
    if (model_def->has_name()) {
      name_ = model_def->name();
    } else {
      AT_ERROR("ModelDef does not have name.");
    }
    if (model_def->has_producer_name()) {
      producerName_ = model_def->producer_name();
    } else {
      AT_ERROR("ModelDef does not have producer_name.");
    }
    producerVersion_ = model_def->producer_version();
    if (model_def->has_proto_version()) {
      protoVersion_ = model_def->proto_version();
    } else {
      AT_ERROR("ModelDef does not have proto_version.");
    }
    if (model_def->has_main_module()) {
      mainModule.update(model_def->mutable_main_module(), id_data);
    } else {
      AT_ERROR("ModelDef does not have main_module.");
    }
  }

  void dump(torch::Model* model_def) {
    // TODO
  }

  // getters
  const std::string& name() const {
    return name_;
  }

  const std::string& producerName() const {
    return producerName_;
  }

  const std::string& producerVersion() const {
    return producerVersion_;
  }

  int64_t producerVersion() const {
    return protoVersion_;
  }

  const IntermediateModule& mainModule() const {
    return mainModule_;
  }

  // setters, most for test purposes
  void setName(const std::string& name) {
    name_ = name;
  }

  void setProducerName(const std::string& producer_name) {
    producerName_ = producer_name;
  }

  void setProducerVersion(const std::string& producer_version) {
    producerVersion_ = producer_version;
  }

  void setProtoVersion(int64_t proto_version) {
    protoVersion_ = proto_version;
  }

  IntermediateModule* mutableMainModule() {
    return &mainModule_;
  }

 private:
  std::string name_;
  std::string producerName_;
  std::string producerVersion_;
  int64_t protoVersion_;
  IntermediateModule mainModule_;

};

void serializeIntermediateModel(IntermediateModel* imodel,
    torch::jit::PyTorchFileWriter* writer) {
  std::unordered_map<void*, uint64_t> data_id;
  // TODO
  std::stack<IntermediateModule*> imodules;
  imodules.push(&(imodel->mainModule()));
  while (!modules.empty()) {
    IntermediateModule* m = imodules.pop();
    auto* params = m->mutableParameters();
    for (int i = 0; i < params->size(); ++i) {
      std::shared_ptr<SharedData>* t = params->at(i)->mutableTensor()->mutableData();
      void* data = t->rawData();
      size_t size = t->size();
      auto it = data_id.find(data);
      if (it != data_id.end()) {
        t->setRecordId(it->second);
      } else {
        uint64_t id = writer->writeRecord(data, size);
        t->setRecordId(id);
        data_id[data] = id;
      }
    }

    auto* subms = m->mutableSubmodules();
    for (int i = 0; i < subms->size(); ++i) {
      imodules.push(subms->at(i));
    }
  }

  // TODO
  torch::ModelDef model_def;
  imodel->dump(&model_def);
  size_t proto_size = model_def.ByteSizeLong();
  void* buffer = malloc(proto_size);
  model_def.SerializeToArray(buffer, size);
  writer->writeRecord(buffer, proto_size);
  free(proto_size);
}

// serialize functions
void serializeIntermediateModel(const std::string& filename) {
  PyTorchFileWriter writer(filename);
  serializeIntermediateModel(&writer);
}

void deserializeIntermediateModel(torch::jit::PyTorchFileReader* reader) {
  std::unordered_map<uint64_t, std::shared_ptr<SharedData> id_data;
  while (reader->hasNextRecord()) {
    at::DataPtr data_ptr;
    size_t data_key;
    size_t data_size;
    std::tie(data_ptr, data_key, data_size) = reader->getNextRecord();
    if (!reader->hasNextRecord()) {
      auto it = id_data.find(data_key);
      if (it != id_data.end()) {
        id_data->second->updateData(std::move(data_ptr));
      } else {
        id_data[data_key] = make_shared<SharedData>(
            data_key, std::move(data_ptr));
      }
      continue;
    }
    torch::ModelDef model_def = torch::ModelDef();
    AT_ASSERTM(model_def.ParsePartialFromArray(proto_ptr.get(),
          proto_size), "Parsing proto from array failedl");
    update(&model_def);
  }
}

void deserializeIntermediateModel(const std::string& filename) {
  PyTorchFileReader reader(filename);
  deserializeIntermediateModel(&reader);
}

}  // namespace serialize
}  // namespace at
