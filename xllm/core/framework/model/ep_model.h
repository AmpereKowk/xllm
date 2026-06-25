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

#include <memory>
#include <string>
#include <vector>

#include "core/framework/model/ep_causal_lm.h"
#include "core/framework/model_loader.h"

namespace xllm {

class EpModel final : public EpCausalLM {
 public:
  explicit EpModel(const torch::TensorOptions& options);
  ~EpModel() override;

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& params) override;

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override;

  void load_model(std::unique_ptr<ModelLoader> loader) override;

  torch::Device device() const override;

  const torch::TensorOptions& options() const override;

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) override;
  void update_expert_weight(int32_t layer_id) override;

  bool IsInitialized() const { return initialized_; }

 private:
  torch::TensorOptions options_;
  int64_t device_id_ = 0;
  bool initialized_ = false;

  std::string epair_path_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};

}  // namespace xllm

#endif  // USE_NPU