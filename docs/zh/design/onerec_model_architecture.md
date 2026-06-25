# OneRecForConditionalGeneration 代码流程与网络结构分析

## 1. 整体架构

OneRec 是一个 **Encoder-Decoder** 架构的生成式推荐模型（类似 T5），采用两阶段推理。其类继承关系如下：

```
CausalLM (抽象基类)
  └── RecCausalLM
        └── RecCausalLMImpl<OneRecForConditionalGeneration>   // 最外层调度代理
              └── OneRecForConditionalGenerationImpl           // 持有 model + lm_head
                    ├── OneRecModelImpl                        // Encoder-Decoder 主体
                    │     ├── shared_: WordEmbedding           // 共享词嵌入
                    │     ├── encoder_: OneRecStack            // 编码器
                    │     └── decoder_: OneRecStack            // 解码器
                    └── lm_head_: LmHead                       // 输出投影层
```

### 1.1 关键源文件

| 文件 | 说明 |
|------|------|
| `xllm/models/rec/npu/onerec.h` | NPU 版 OneRec 模型定义（含 Encoder/Decoder） |
| `xllm/models/rec/onerec.h` | 通用版 OneRec 模型定义（无 Encoder/Decoder Stack） |
| `xllm/models/rec/npu/onerec_npu_impl.h` | `OneRecStack` 实现（NPU 侧 Encoder/Decoder 前向逻辑） |
| `xllm/models/rec/rec_model_base.h` | `RecForCausalLMImplBase` 基类（forward/logits/pooler） |
| `xllm/core/framework/model/rec_causal_lm.h` | `RecCausalLMImpl` 模板（最外层调度代理） |
| `xllm/core/layers/npu/npu_onerec_block_layer_impl.h` | `NpuOneRecBlockLayer`（单个 Transformer Block） |
| `xllm/core/layers/onerec_block_layer.h` | `OneRecBlockLayer` 抽象接口 |

## 2. 类层次详解

### 2.1 RecCausalLMImpl（最外层调度代理）

`RecCausalLMImpl<OneRecForConditionalGeneration>` 是框架调度的入口，实现了 `RecCausalLM` 接口：

- `forward()` → 委托给内部 `model_`（即 `OneRecForConditionalGeneration`）
- `logits()` → 委托给 `model_->logits()`
- `pooler()` → 委托给 `model_->pooler()`
- `load_model()` → 委托给 `model_->load_model()`

### 2.2 RecForCausalLMImplBase（基类）

`RecForCausalLMImplBase<OneRecModel>` 提供通用 CausalLM 骨架：

```cpp
// rec_model_base.h:36
template <typename ModelType>
class RecForCausalLMImplBase : public torch::nn::Module {
  ModelType model_;           // OneRecModel
  layer::LmHead lm_head_;     // 输出投影
  float scale_factor_;        // 1/sqrt(hidden_size)，tie embeddings 时使用
  bool tie_word_embeddings_;  // 是否共享嵌入与 LM Head
};
```

- `forward()` 默认委托给 `model_->forward()`
- `logits()` 在 `tie_word_embeddings_=true` 时对 hidden_states 乘以 `scale_factor_`，再用 `lm_head_` 投影
- `pooler()` 按 `selected_idxes` 选取 hidden_states

### 2.3 OneRecForConditionalGenerationImpl

继承 `RecForCausalLMImplBase<OneRecModel>`，核心职责是 **权重加载**：

```cpp
// npu/onerec.h:256
class OneRecForConditionalGenerationImpl final
    : public RecForCausalLMImplBase<OneRecModel> {
  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override;
};
```

权重加载流程：

```
load_model()
  ├── 尝试 prefix "module.module3.t5_model."（兼容旧 checkpoint）
  ├── 回退到 "model." prefix
  ├── model_->load_state_dict()
  │     ├── shared（WordEmbedding）
  │     ├── encoder（OneRecStack）
  │     └── decoder（OneRecStack）
  ├── lm_head 加载
  │     ├── tie=true  → 复用 shared 权重
  │     └── tie=false → 加载独立 lm_head 权重
  ├── verify_loaded_weights()   // 校验所有层权重已加载
  └── merge_loaded_weights()    // 合并 MoE 专家权重
```

### 2.4 OneRecModelImpl（Encoder-Decoder 主体）

```cpp
// npu/onerec.h:35
class OneRecModelImpl final : public torch::nn::Module {
  layer::WordEmbedding shared_;       // 共享词嵌入
  OneRecStack encoder_;               // 编码器
  OneRecStack decoder_;               // 解码器
  torch::Tensor encoder_output_;      // 缓存的编码器输出
  std::mutex encoder_output_mutex_;   // 缓存互斥锁
};
```

## 3. 两阶段前向流程

### 3.1 阶段一：Encoder Forward（`is_encoder_forward = true`）

```
输入 tokens
    │
    ▼
WordEmbedding (shared_)
    │
    ▼
OneRecStack (encoder)
    ├── Embedding 查找
    ├── 计算 position bias / attention mask
    ├── N × NpuOneRecBlockLayer
    │     ├── Self-Attention
    │     └── FFN
    └── RMSNorm (final_layer_norm)
    │
    ▼
pad_encoder_output()  →  缓存到 encoder_output_
    │
    ▼
返回 ModelOutput
```

编码器输出会被 pad 到 `encoder_max_seq_len` 长度，并缓存在 `encoder_output_` 中供解码阶段使用。

### 3.2 阶段二：Decoder Forward（`is_encoder_forward = false`）

```
输入 tokens + decoder_context_embedding
    │
    ▼
WordEmbedding + context 融合
    │  (将 token embedding 写入 context 的指定位置)
    ▼
OneRecStack (decoder)
    ├── 计算 position bias / attention mask
    ├── N × NpuOneRecBlockLayer
    │     ├── Self-Attention（causal，带 KVCache）
    │     ├── Cross-Attention（使用 encoder_output）
    │     └── FFN（可选 MoE）
    └── RMSNorm (final_layer_norm)
    │
    ▼
hidden_states
    │
    ▼
RecForCausalLMImplBase::logits()
    ├── tie=true → hidden_states * scale_factor_
    ├── selected_idxes 索引选取
    └── LmHead 投影
    │
    ▼
返回 logits
```

### 3.3 forward() 分发逻辑

`OneRecModelImpl::forward()` 根据 `onerec_params` 决定执行路径：

```cpp
if (onerec_params->is_encoder_forward) {
    // 阶段一：编码器
    encoder_output = encoder_(tokens, positions, encoder_kv_caches, input_params);
    cached_encoder_output = pad_encoder_output(encoder_output, input_params);
    encoder_output_ = cached_encoder_output;  // 缓存
    return ModelOutput(cached_encoder_output);
} else {
    // 阶段二：解码器
    cached_encoder_output = encoder_output_;  // 读取缓存
    decoder_output = decoder_(tokens, positions, kv_caches, input_params,
                              cached_encoder_output);
    return ModelOutput(decoder_output);
}
```

## 4. OneRecStack 详解

`OneRecStack`（`onerec_npu_impl.h:152`）是 Encoder/Decoder 的核心执行单元：

```cpp
class OneRecStackImpl : public torch::nn::Module {
  layer::WordEmbedding embed_tokens_;            // 词嵌入（共享引用）
  layer::WordEmbedding position_bias_embedding_; // 相对位置编码嵌入
  layer::RMSNorm norm_;                          // 最终 LayerNorm
  std::vector<layer::NpuOneRecBlockLayer> layers_; // Transformer Block 列表
  torch::nn::ModuleList blocks_;                 // Module 注册容器
};
```

### 4.1 OneRecStack::forward() 流程

```
1. 构建 hidden_states (h)
   ├── hybrid_mode + encoder → 直接使用 tokens（已是 embedding）
   ├── decoder_context_embedding 存在 → 融合 token embedding 与 context
   └── 其他 → embed_tokens_(tokens)

2. 计算 attention mask / position bias
   ├── use_absolute_position_embedding=true  → create_moe_attention_mask()
   └── use_absolute_position_embedding=false → compute_position_bias_mask()
       └── compute_onerec_position_bias()  // T5 风格相对位置编码

3. 预处理 attention mask（设备对齐、格式转换）

4. 逐层执行 Transformer Block
   for (i = 0; i < layers_.size(); ++i) {
       layers_[i]->forward(h, attn_mask, kv_cache, input_params,
                           encoder_output, layer_id, event, event_flag,
                           expert_array);
   }

5. 最终 RMSNorm
   h = norm_->forward(h)

6. 返回 h
```

### 4.2 Decoder Context 融合逻辑

当 `decoder_context_embedding` 存在时，`OneRecStack` 会将 token embedding 写入 context tensor 的指定位置：

```
context_emb: [bs, group_width, context_total_tokens, hidden_size]
                                    ├── seq_len2 (context 部分)
                                    └── seq_len1 (token 部分，被覆盖)

h = embed_tokens_(tokens)  →  reshape  →  写入 context_emb 的 [seq_len2:] 位置
h = context_emb.view({-1, hidden_size})
```

## 5. NpuOneRecBlockLayer（单个 Transformer Block）

`NpuOneRecBlockLayer`（`npu_onerec_block_layer_impl.h:41`）是底层算子执行层，通过 ATB（Ascend Transformer Boost）算子库实现：

### 5.1 Block 类型

| Block 类型 | 组成 |
|------------|------|
| Encoder Block | Self-Attention + FFN |
| Decoder Block | Self-Attention (causal, KVCache) + Cross-Attention (encoder_output) + FFN |
| Decoder MoE Block | Self-Attention (causal, KVCache) + Cross-Attention + MoE FFN |

### 5.2 内部 Node 结构

每个 Block Layer 内部维护多个 ATB Node，区分 prefill 和 decode 阶段：

```cpp
atb_speed::Model::Node prefill_node_;              // 编码器 prefill
atb_speed::Model::Node decode_node_;               // 解码器 decode
atb_speed::Model::Node decoder_prefill_only_decode_node_;  // 解码器 prefill
```

## 6. 模型配置参数

来自 `REGISTER_MODEL_ARGS(onerec, ...)`：

| 参数 | 配置键 | 默认值 | 说明 |
|------|--------|--------|------|
| `hidden_size` | `d_model` | 128 | 隐藏层维度 |
| `intermediate_size` | `d_ff` | 256 | FFN 中间层维度 |
| `n_encoder_layers` | `num_layers` | 12 | 编码器层数 |
| `n_layers` | `num_decoder_layers` | 4 | 解码器层数 |
| `n_heads` | `num_heads` | 4 | 编码器注意力头数 |
| `head_dim` | `d_kv` | 32 | 每头维度 |
| `decoder_n_heads` | `decoder_num_heads` | 同 `n_heads` | 解码器注意力头数 |
| `decoder_head_dim` | `decoder_d_kv` | 同 `head_dim` | 解码器每头维度 |
| `vocab_size` | `vocab_size` | 8200 | 词表大小 |
| `rms_norm_eps` | `layer_norm_epsilon` | 1e-6 | RMSNorm epsilon |
| `max_position_embeddings` | `max_length` | 500 | 最大序列长度 |
| `tie_word_embeddings` | `tie_word_embeddings` | true | 嵌入与 LM Head 共享权重 |
| `use_moe` | `use_moe` | false | 解码器是否使用 MoE |
| `n_routed_experts` | `moe_num_experts` | 8 | MoE 专家数 |
| `num_experts_per_tok` | `moe_topk` | 2 | 每 token 激活专家数 |
| `relative_attention_num_buckets` | - | 32 | 相对位置编码 bucket 数 |
| `relative_attention_max_distance` | - | 128 | 相对位置编码最大距离 |

## 7. 相对位置编码

OneRec 使用 T5 风格的相对位置编码（`compute_onerec_position_bias`）：

```
1. 计算相对位置矩阵: relative_position = memory_pos - context_pos
2. 分桶:
   ├── Encoder: 双向，正负各 num_buckets/2
   └── Decoder: 单向（只看历史），全部 num_buckets 用于负方向
3. 小距离直接映射，大距离对数映射
4. 通过 position_bias_embedding 查找得到 bias tensor
5. 形状: [num_heads, query_length, key_length]
```

## 8. 调用链路总结

```
外部请求
  → RecMaster
  → FixedStepsScheduler
  → RecEngine
  → RecWorkerImpl
  → RecCausalLMImpl::forward()
    → OneRecForConditionalGenerationImpl::forward()  (继承自 RecForCausalLMImplBase)
      → OneRecModelImpl::forward()
        ├── [阶段一] encoder_() → OneRecStack::forward()
        │     ├── embed_tokens_(tokens)
        │     ├── compute_position_bias_mask()
        │     ├── N × NpuOneRecBlockLayer::forward()  (Self-Attn + FFN)
        │     └── RMSNorm → pad → 缓存 encoder_output_
        │
        └── [阶段二] decoder_() → OneRecStack::forward()
              ├── embed_tokens_(tokens) + context 融合
              ├── compute_position_bias_mask()
              ├── N × NpuOneRecBlockLayer::forward()
              │     ├── Self-Attention (causal, KVCache)
              │     ├── Cross-Attention (encoder_output)
              │     └── FFN / MoE FFN
              └── RMSNorm → hidden_states
    → RecForCausalLMImplBase::logits()
      ├── scale (if tie_word_embeddings)
      ├── index_select (selected_idxes)
      └── LmHead → logits
```
