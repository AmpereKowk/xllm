/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if defined(USE_NPU)

#include "core/framework/model/ep_model.h"

#include <glog/logging.h>

#include "common/macros.h"

namespace xllm {

namespace {

std::mutex& GetGeMutex() {
  static std::mutex ge_mutex;
  return ge_mutex;
}

bool& GetGeInitializedFlag() {
  static bool ge_initialized = false;
  return ge_initialized;
}

bool InitializeGE() {
  bool& ge_initialized = GetGeInitializedFlag();
  if (ge_initialized) {
    return true;
  }

  std::lock_guard<std::mutex> lock(GetGeMutex());
  if (ge_initialized) {
    return true;
  }

  // TODO: Call GEInitializeV2() from GE V2 library
  // if (ge::GEInitializeV2() != ge::SUCCESS) {
  //   LOG(ERROR) << "Failed to initialize GE";
  //   return false;
  // }

  ge_initialized = true;
  LOG(INFO) << "GE initialized successfully";
  return true;
}

}  // namespace

EpModel::EpModel(const torch::TensorOptions& options) : options_(options) {
  device_id_ = options_.device().index();
}

EpModel::~EpModel() {}

void EpModel::load_model(std::unique_ptr<ModelLoader> loader) {
  if (!InitializeGE()) {
    LOG(ERROR) << "Failed to initialize GE";
    return;
  }

  std::string model_path = loader->model_weights_path();
  epair_path_ = model_path + "/model.epair";

  LOG(INFO) << "Loading epair model from: " << epair_path_;

  // TODO: Load epair model using EpairModelLoader
  // td::EpairModelLoader epair_loader;
  // if (epair_loader.Load(epair_path_) != td::SUCCESS) {
  //   LOG(ERROR) << "Failed to load epair: " << epair_path_;
  //   return;
  // }
  //
  // if (epair_loader.CompileAndLoad(device_id_, nullptr) != td::SUCCESS) {
  //   LOG(ERROR) << "Failed to compile and load model";
  //   return;
  // }
  //
  // input_names_ = epair_loader.GetInputNames();
  // output_names_ = epair_loader.GetOutputNames();

  initialized_ = true;
  LOG(INFO) << "EpModel loaded successfully";
}

ModelOutput EpModel::forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& params) {
  if (!initialized_) {
    LOG(ERROR) << "EpModel not initialized";
    return ModelOutput();
  }

  // TODO: Build graph inputs from input_tensor_map
  // std::vector<gert::Tensor> graph_inputs;
  // for (const auto& name : input_names_) {
  //   auto it = params.input_tensor_map.find(name);
  //   if (it == params.input_tensor_map.end()) {
  //     LOG(ERROR) << "Missing input tensor: " << name;
  //     return ModelOutput();
  //   }
  //   graph_inputs.push_back(it->second);
  // }

  // TODO: Execute graph using RunGraphAsyncWithStream
  // aclrtStream stream = c10_npu::getCurrentNPUStream(device_id_).stream();
  // std::vector<gert::Tensor> device_outputs;
  // if (epair_loader_.RunGraphAsyncWithStream(stream, graph_inputs,
  //                                            device_outputs) != td::SUCCESS) {
  //   LOG(ERROR) << "RunGraphAsyncWithStream failed";
  //   return ModelOutput();
  // }
  //
  // if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
  //   LOG(ERROR) << "aclrtSynchronizeStream failed";
  //   return ModelOutput();
  // }

  // TODO: Convert outputs to ModelOutput
  ModelOutput result;

  LOG(WARNING) << "EpModel::forward() not fully implemented - returning empty output";
  return result;
}

torch::Tensor EpModel::logits(const torch::Tensor& hidden_states,
                              const torch::Tensor& seleted_idxes) {
  NOT_IMPLEMENTED();
  return torch::Tensor();
}

torch::Device EpModel::device() const { return options_.device(); }

const torch::TensorOptions& EpModel::options() const { return options_; }

void EpModel::prepare_expert_weight(int32_t layer_id,
                                    const std::vector<int32_t>& expert_ids) {
  NOT_IMPLEMENTED();
}

void EpModel::update_expert_weight(int32_t layer_id) { NOT_IMPLEMENTED(); }

}  // namespace xllm

#endif  // USE_NPU