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

#include "core/runtime/ge_graph_executor_impl.h"

#include <glog/logging.h>

#include "common/metrics.h"
#include "core/framework/model/ep_model.h"

namespace xllm::npu {

GeGraphExecutorImpl::GeGraphExecutorImpl(CausalLM* model,
                                         const ModelArgs& args,
                                         const torch::Device& device,
                                         const runtime::Options& options)
    : model_(model),
      args_(args),
      device_(device),
      options_(options),
      initialized_(false) {
  if (model_ == nullptr) {
    LOG(ERROR) << "GeGraphExecutorImpl requires model to be EpModel";
    return;
  }

  EpModel* ep_model = dynamic_cast<EpModel*>(model_);
  if (ep_model == nullptr) {
    LOG(ERROR) << "GeGraphExecutorImpl requires model to be EpModel type";
    return;
  }

  if (!ep_model->IsInitialized()) {
    LOG(ERROR) << "EpModel not initialized";
    return;
  }

  device_id_ = device_.index();
  initialized_ = true;

  LOG(INFO) << "GeGraphExecutorImpl initialized successfully on device "
            << device_id_;
}

ForwardInput GeGraphExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

ModelOutput GeGraphExecutorImpl::run(const torch::Tensor& tokens,
                                     const torch::Tensor& positions,
                                     std::vector<KVCache>& kv_caches,
                                     const ModelInputParams& params) {
  if (!initialized_) {
    LOG(ERROR) << "GeGraphExecutorImpl not initialized";
    return ModelOutput();
  }

  COUNTER_INC(num_model_execution_total_ge_graph);
  return model_->forward(tokens, positions, kv_caches, params);
}

}  // namespace xllm::npu

#endif  // USE_NPU