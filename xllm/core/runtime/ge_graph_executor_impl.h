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

#pragma once

#if defined(USE_NPU)

#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "common/macros.h"
#include "framework/batch/batch.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_input_params.h"
#include "framework/model/model_output.h"
#include "runtime/executor_impl.h"
#include "runtime/executor_impl_factory.h"
#include "runtime/options.h"

namespace xllm::npu {

class GeGraphExecutorImpl final : public ExecutorImpl {
 public:
  GeGraphExecutorImpl(CausalLM* model,
                      const ModelArgs& args,
                      const torch::Device& device,
                      const runtime::Options& options);

  ~GeGraphExecutorImpl() override = default;

  ForwardInput prepare_inputs(Batch& batch) override;

  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelInputParams& params) override;

 private:
  CausalLM* model_;
  int64_t device_id_ = 0;
  bool initialized_ = false;

  ModelArgs args_;
  torch::Device device_;
  runtime::Options options_;
};

REGISTER_EXECUTOR("ge", GeGraphExecutorImpl);

}  // namespace xllm::npu

#endif  // USE_NPU