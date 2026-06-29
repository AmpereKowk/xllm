# RecWorkPipeline 各实例 ForwardInput 消费对比分析

## 1. ForwardInput 字段清单

**文件**: `xllm/core/runtime/forward_params.h:572-608`

| 字段 | 类型 | 说明 |
|------|------|------|
| `token_ids` | `Tensor` | 展平的 token ID |
| `positions` | `Tensor` | 展平的 position ID |
| `token_ids_host` | `Tensor` | CPU 侧 token_ids 视图 |
| `positions_host` | `Tensor` | CPU 侧 positions 视图 |
| `input_params` | `ModelInputParams` | 注意力元数据、embedding、meta、rec_params、parallel、expert、multimodal 等 |
| `sampling_params` | `SamplingParameters` | Prefill/首轮采样配置 |
| `decoder_sampling_params` | `SamplingParameters` | Decode/多轮采样配置 |
| `step_decode` | `optional<StepDecodeMeta>` | Step 级 decode 元数据（batch_size, beam_width, total_round 等） |
| `skip_sampling_for_logits_only` | `bool` | 跳过 sampler，只保留 logits |
| `transfer_kv_infos` | `vector<TransferKVInfo>` | 分离式 prefill/decode KV 传输信息 |
| `input_host_buffer` | `Tensor` | Host 连续输入 buffer |
| `device_input_buffer` | `Tensor` | Device 连续输入 buffer |
| `device_tensors_ready` | `bool` | 跳过 H2D |
| `cp_partitioned` | `bool` | CP 分区标记 |
| `metadata_ready_event` | `StreamEventPtr` | 依赖事件 |

### StepDecodeMeta

**文件**: `xllm/core/runtime/forward_params.h:404-414`

```cpp
struct StepDecodeMeta {
  int32_t batch_size = 0;
  int32_t beam_width = 1;
  int32_t current_round = 0;
  int32_t total_round = 0;
  std::vector<int64_t> full_kv_shape;
  std::vector<int32_t> decode_positions_vec;
};
```

---

## 2. RecPipelineRuntime 结构

**文件**: `xllm/core/runtime/rec_worker_impl.h:70-88`

```cpp
struct RecPipelineRuntime {
  std::unique_ptr<Stream> stream;
  std::unique_ptr<CausalLM> model;
  std::unique_ptr<Executor> executor;
  std::unique_ptr<ModelContext> context;
  std::unique_ptr<EplbExecutor> eplb_executor;
  torch::Tensor expert_load_data;
  RecWorkerImpl& worker;
};
```

每个 Pipeline 实例拥有独立的 stream、model、executor、context，支持并发执行。

---

## 3. 各实例 ForwardInput 字段消费详情

### 3.1 Base RecWorkPipeline

**文件**: `rec_worker_impl.cpp:403-604`

#### prepare_work_before_execute() (lines 403-457)

| 操作 | 行号 | 说明 |
|------|------|------|
| `inputs.to(device, dtype)` | 424-425 | 全量 H2D 传输所有字段到 `processed_inputs` |
| `apply_kv_block_swaps(input_params)` | 427 | KV block 交换 |
| DP/EP padding | 430-455 | 读取 `batch_forward_type` 和 `dp_global_token_nums` |

#### step() (lines 463-604)

| ForwardInput 字段 | 使用方式 | 行号 |
|-------------------|---------|------|
| `token_ids` | 传入 `executor->forward()` 第一个参数 | 500 |
| `positions` | 传入 `executor->forward()` 第二个参数 | 501 |
| `input_params` | 传入 `executor->forward()` 第四个参数 | 503 |
| `input_params.expert.eplb_info` | 传入 `eplb_executor->eplb_execute()` | 495 |
| `input_params.meta.batch_forward_type` | 检查 `is_decode()` 用于投机解码路径 | 574 |
| `sampling_params` | 整体引用 | 466 |
| `sampling_params.selected_token_idxes` | 传入 `model->logits()` 和 `sampler->forward()` | 509, 548 |
| `sampling_params.do_sample` | 复制到 `output.do_sample` | 566 |
| `sampling_params.logprobs` | 复制到 `output.logprobs` | 567 |
| `sampling_params.max_top_logprobs` | 复制到 `output.max_top_logprobs` | 568 |
| `sampling_params.use_beam_search` | 门控 beam search kernel | 554 |
| `sampling_params.acc_logprob` | 传入 `beam_searcher_->forward()` | 555-556 |
| `transfer_kv_infos` | KV cache push 门控和等待 | 471, 529, 587 |
| `decoder_sampling_params` | **忽略** | - |
| `step_decode` | **忽略** | - |
| `skip_sampling_for_logits_only` | **忽略** | - |

#### executor->forward() 调用

```cpp
// line 500-503
runtime_.executor->forward(input.token_ids,
                           input.positions,
                           runtime_.worker.kv_caches_,
                           input.input_params);
```

**调用次数**: 1 次/step

#### Sampler

```cpp
// line 549
runtime_.worker.sampler_->forward(logits, sampling_params);
```

使用通用 LLM `sampler_`，无 filter_mask。

---

### 3.2 LlmRecWorkPipeline

**文件**: `rec_worker_impl.h:109-116`, `rec_worker_impl.cpp:606-612`

#### 与 Base 的差异

- **step()**: 不覆盖，继承 Base `RecWorkPipeline::step()`
- **prepare_work_before_execute()**: 覆盖，在 Base 基础上增加多模态数据处理

```cpp
void LlmRecWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs, ForwardInput& processed_inputs) {
  RecWorkPipeline::prepare_work_before_execute(inputs, processed_inputs);
  runtime_.worker.prepare_multi_modal_data(processed_inputs);
}
```

#### 额外消费的字段

| 字段 | 行号 | 说明 |
|------|------|------|
| `input_params.multimodal.mm_data` | 3105-3144 | 读取 `MULTI_MODAL_VALUES` 和 `MULTI_MODAL_INDICES` Tensor |
| `token_ids` | 3130/3134 | 计算 word embeddings 用于多模态数据合并 |
| 设置 `input_params.embedding.input_embedding` | - | 作为多模态处理的结果 |

#### 忽略的字段

与 Base 相同：`decoder_sampling_params`、`step_decode`、`skip_sampling_for_logits_only`

---

### 3.3 OneRecWorkPipeline

**文件**: `rec_worker_impl.h:118-138`, `rec_worker_impl.cpp:614-860`

#### 覆盖情况

- **step()**: 覆盖 (lines 715-860)
- **prepare_inputs()**: 覆盖 (lines 645-656)
- **prepare_work_before_execute()**: 覆盖 (lines 658-675)

#### prepare_work_before_execute() (lines 658-675)

在 Base 基础上增加 `decoder_context_embedding` dtype 检查和转换：

```cpp
RecWorkPipeline::prepare_work_before_execute(inputs, processed_inputs);
auto& onerec_params = processed_inputs.input_params.mutable_onerec_params();
// 如果 decoder_context_embedding dtype 不匹配 worker dtype，进行转换
```

#### step() (lines 715-860)

| ForwardInput 字段 | 使用方式 | 行号 |
|-------------------|---------|------|
| `token_ids` | Decoder token ID，传入 `run_onerec_forward` | 762, 797, 811 |
| `positions` | Decoder positions，传入 `run_onerec_forward` | 762, 797, 811 |
| `input_params` | 传入 `run_onerec_forward`，**原地修改** flag | 745 |
| `sampling_params` | 整体引用 | 721 |
| `sampling_params.selected_token_idxes` | filter mask 门控、logits、sampling | 751, 833, 840 |
| `sampling_params.do_sample` | 复制到 output | 849 |
| `sampling_params.logprobs` | 复制到 output | 850 |
| `sampling_params.max_top_logprobs` | 复制到 output | 851 |
| `onerec_params.rec_stage` | PREFILL vs DECODE 分支 | 756 |
| `onerec_params.is_first_prefill` | 首次 prefill vs 后续 prefill 分支 | 757 |
| `onerec_params.has_encoder_output` | 是否有 encoder 输出门控 | 732-733 |
| `onerec_params.encoder_sparse_embedding` | 作为 encoder tokens（优先） | 771, 781-782 |
| `onerec_params.encoder_token_ids` | 作为 encoder tokens（备选） | 772, 784 |
| `onerec_params.encoder_positions` | Encoder positions | 772, 789 |
| `onerec_params.decoder_context_embedding` | 是否有 decoder context 门控 | 729-730 |
| `onerec_params.generated_tokens` | 传入 `prepare_filter_mask_async()` 用于约束解码 | 752 |
| `decoder_sampling_params` | **忽略** | - |
| `step_decode` | **忽略** | - |
| `transfer_kv_infos` | **忽略** | - |
| `skip_sampling_for_logits_only` | **忽略** | - |

#### executor->forward() 调用

通过 `run_onerec_forward` lambda (lines 734-746)：

```cpp
auto run_onerec_forward = [&](token_ids, positions, is_encoder_forward,
                              forward_has_encoder_output, is_hybrid_mode) {
  mutable_onerec_params.is_encoder_forward = is_encoder_forward;
  mutable_onerec_params.has_encoder_output = forward_has_encoder_output;
  mutable_onerec_params.is_hybrid_mode = is_hybrid_mode;
  return runtime_.executor->forward(token_ids, positions,
                                    runtime_.worker.kv_caches_,
                                    mutable_input.input_params);
};
```

**调用次数**:
- First prefill: **2 次**（encoder forward + decoder forward）
- 其他情况: **1 次**

Encoder forward 使用 `onerec_params.encoder_token_ids` 或 `encoder_sparse_embedding` 作为 token 输入。

#### Sampler

```cpp
// line 846
rec_sampler_->forward(logits, sampling_params, filter_mask);
```

使用 `RecSampler`，带 `filter_mask`（约束解码）。

---

### 3.4 OneRecXAttentionWorkPipeline

**文件**: `rec_worker_impl.h:140-203`, `rec_worker_impl.cpp:862-1948`

#### 覆盖情况

- **step()**: 覆盖 (lines 1345-1948)
- **prepare_inputs()**: 覆盖 (lines 1149-1161)
- **prepare_work_before_execute()**: 覆盖 (lines 1163-1303)

#### prepare_work_before_execute() (lines 1163-1303)

在 Base 基础上进行大量额外处理：

| 操作 | 行号 | 说明 |
|------|------|------|
| Debug H2D roundtrip 验证 | 1198, 1202 | 验证 `sampling_params` 和 `decoder_sampling_params` |
| 分配 `shared_k_caches` / `shared_v_caches` | 1240-1255 | 每层分配共享 KV cache |
| `prepare_unshared_kv_caches_for_input()` | 1256 | 读取 `step_meta()->beam_width` 或 `onerec_params.group_width` 和 `bs` |
| 设置 `block_tables` | 1257-1261 | 从 KV 状态构建 |
| 创建 `beam_width_tensor` / `current_round_tensor` | 1271-1273 | 从 `step_meta()->beam_width` 构建 |
| `decoder_context_embedding` dtype 转换 | 1294-1301 | 与 OneRec 相同 |

#### step() (lines 1345-1948)

| ForwardInput 字段 | 使用方式 | 行号 |
|-------------------|---------|------|
| `token_ids` | Decoder tokens，**多轮覆盖** | 1500, 1551, 1571, 1690 |
| `positions` | Decoder positions，**多轮覆盖** | 1501, 1552, 1572, 1717 |
| `input_params` | **拷贝为 encoder_params/decoder_params**，decode 轮次清除 embedding/attn_metadata | 1503, 1539, 1554, 1574 |
| `sampling_params` | Round 0 采样参数 | 1740, 1794 |
| `sampling_params.selected_token_idxes` | logits 和 sampling 门控 | 1433, 1441, 1588 |
| `sampling_params.do_sample` | 复制到 output | 1760, 1831 |
| `sampling_params.logprobs` | 复制到 output；output_logprobs 门控 | 1761, 1800 |
| `sampling_params.max_top_logprobs` | 复制到 output | 1762, 1801 |
| `sampling_params.num_return_sequences` | 复制到 `decoder_sampling_params` | 1721 |
| `decoder_sampling_params` | **多轮门控**；round > 0 采样参数 | 1718, 1733, 1795 |
| `decoder_sampling_params.selected_token_idxes` | 多轮门控；每轮覆盖 | 1718-1719, 1733 |
| `step_decode` (via `step_meta()`) | `batch_size`, `beam_width`, `total_round`, `decode_positions_vec` | 1729-1733, 1771-1773, 1787 |
| `onerec_xattention_params.decoder_context_embedding` | encoder context 门控；decode 轮次清除 | 1381-1382, 1696 |
| `onerec_xattention_params.has_encoder_output` | forward flag | 1384, 1498, 1549, 1569 |
| `onerec_xattention_params.encoder_sparse_embedding` | Encoder tokens | 1509-1510, 1529 |
| `onerec_xattention_params.encoder_token_ids` | Encoder tokens 备选 | 1512, 1532 |
| `onerec_xattention_params.encoder_positions` | Encoder positions | 1513, 1537 |
| `onerec_xattention_params.generated_tokens` | 约束解码 filter mask | 1436 |
| `onerec_xattention_params.rec_stage` | PREFILL vs DECODE 分支 | 1487 |
| `onerec_xattention_params.is_first_prefill` | 首次 prefill 分支 | 1488 |
| `onerec_xattention_params.bs` | 每轮设置 batch size | 1697 |
| `onerec_xattention_params.group_width` | 每轮设置 beam width | 1698 |
| `onerec_xattention_params.seq_len` | decode 轮次设为 1 | 1699 |
| `input_params.embedding.input_embedding` | decode 轮次清除 | 1725 |
| `input_params.meta.batch_forward_type` | decode 轮次设为 DECODE | 1722 |
| `input_params.meta.num_sequences` | decode 轮次设为 `batch_size * beam_width` | 1724 |
| `input_params.attn_metadata` | decode 轮次设为 nullptr | 1726 |
| `transfer_kv_infos` | **忽略** | - |
| `skip_sampling_for_logits_only` | **忽略** | - |

#### executor->forward() 调用

**First prefill** (2 次):

```cpp
// Encoder forward (line 1535-1539)
runtime_.executor->forward(encoder_tokens,
                           round_params->encoder_positions,
                           runtime_.worker.kv_caches_,
                           encoder_params);  // input_params 的拷贝，is_encoder_forward=true

// Decoder forward (line 1550-1554)
runtime_.executor->forward(mutable_input.token_ids,
                           mutable_input.positions,
                           runtime_.worker.kv_caches_,
                           decoder_params);  // input_params 的拷贝，is_encoder_forward=false
```

**后续 prefill 或 decode** (每轮 1 次，共 N 轮):

```cpp
runtime_.executor->forward(mutable_input.token_ids,
                           mutable_input.positions,
                           runtime_.worker.kv_caches_,
                           decoder_params);
```

**关键差异**: 与 OneRecWorkPipeline 不同，XAttention **拷贝** input_params 为独立的 encoder_params 和 decoder_params，而非原地修改。

#### Sampler

```cpp
// line 1662-1663
rec_sampler_->forward(logits, sampling_params, filter_mask, &sampling_context);
```

使用 `RecSampler`，带 `filter_mask` 和 `RecSamplingContext`（含 `sequence_group`, `current_step`, `beam_width`）。支持 NPU 设备约束采样（`rec_constrained_topk`）。

---

### 3.5 LlmRecMultiRoundPipeline

**文件**: `rec_worker_impl.h:214-369`, `rec_worker_impl.cpp:1954-2349`

#### 覆盖情况

- **step()**: 覆盖 (lines 2212-2349)
- **prepare_inputs()**: 覆盖 (lines 1970-1982)
- **prepare_work_before_execute()**: 覆盖 (lines 1984-1994)

#### prepare_work_before_execute() (lines 1984-1994)

```cpp
RecWorkPipeline::prepare_work_before_execute(inputs, processed_inputs);
runtime_.worker.prepare_multi_modal_data(processed_inputs);
prepare_kv_caches_related_for_input(inputs, processed_inputs);
```

`prepare_kv_caches_related_for_input()` (lines 2072-2210) 消费：

| 字段 | 行号 | 说明 |
|------|------|------|
| `step_meta()->batch_size` | 2083 | 每 batch 大小 |
| `step_meta()->beam_width` | 2084 | 每 beam 大小 |
| `step_meta()->total_round` | 2085 | 轮次数 |
| `step_meta()->full_kv_shape` | 2089-2093 | KV cache shape `[full_kv_len, num_kv_heads, head_dim]` |
| `step_meta()->decode_positions_vec` | 2192-2208 | 构建每轮 position tensors |

设置到 `processed_inputs.input_params.mutable_llmrec_params()`：
- `batch_size`, `beam_width`, `total_round`
- `full_k_caches`, `full_v_caches`, `unshared_k_caches`, `unshared_v_caches`, `shared_k_caches`, `shared_v_caches`
- `decode_positions_tensor_list`
- `block_tables` (lines 2185-2190)

#### step() (lines 2212-2349)

| ForwardInput 字段 | 使用方式 | 行号 |
|-------------------|---------|------|
| `token_ids` | 传入 `executor->forward()`，**每轮覆盖** | 2275, 2645, 2648 |
| `positions` | 传入 `executor->forward()`，**每轮覆盖** | 2276, 2655, 2703 |
| `input_params` (含 `llmrec_params`) | 传入 `executor->forward()`，**每轮修改** | 2278, 2632-2661 |
| `input_params.mutable_llmrec_params()` | 大量每轮修改 | 2632-2661 等 |
| `input_params.attention.device.kv_seq_lens` | 异步下一轮计算 | 2854 |
| `input_params.embedding.input_embedding` | decode 轮次清除 | 2660, 2708 |
| `input_params.meta.batch_forward_type` | decode 轮次设为 DECODE | 2659, 2707 |
| `input_params.attn_metadata` | decode 轮次设为 nullptr | 2640, 2711 |
| `sampling_params` | Round 0 采样参数 | 2247 |
| `sampling_params.selected_token_idxes` | logits 和 sampling 门控 | 2284 |
| `sampling_params.do_sample` | 复制到 output (via build_final_output) | 2501 |
| `sampling_params.logprobs` | output logprobs 门控 | 2252, 2339 |
| `sampling_params.max_top_logprobs` | output max_top_logprobs | 2253, 2339 |
| `sampling_params.num_return_sequences` | `get_requested_beam_result_width()` | 2251 |
| `decoder_sampling_params` | Round > 0 采样参数 | 2246 |
| `decoder_sampling_params.selected_token_idxes` | 通过 sampling_params 引用使用 | - |
| `step_decode` (via `step_meta()`) | `total_round`, `batch_size`, `beam_width`, `max_decode_step` | 2220-2226 |
| `transfer_kv_infos` | **忽略** | - |
| `skip_sampling_for_logits_only` | **忽略** | - |

#### executor->forward() 调用

```cpp
// line 2275-2278
runtime_.executor->forward(mutable_input.token_ids,
                           mutable_input.positions,
                           runtime_.worker.kv_caches_,
                           mutable_input.input_params);
```

**调用次数**: 每轮 1 次，共 `total_rounds` 次循环。

**每轮输入准备**:
- **NPU** (`prepare_round_input_for_npu`, lines 2627-2662): 设置 `beam_width_tensor`/`current_round_tensor`，从 beam search 输出替换 `token_ids`，从 `decode_positions_tensor_list` 替换 `positions`，清除 `input_embedding`，设置 `batch_forward_type` 为 DECODE，置空 `attn_metadata`
- **CUDA** (`prepare_round_input_and_schedule_next`, lines 2664-2712): 同上，额外处理 paged KV attention tensors

#### Sampler

```cpp
// line 2287
rec_sampler_->forward(logits, round_sampling_params);
```

使用 `RecSampler`，**无** filter_mask，**无** sampling_context。

---

## 4. 总对比表

### 4.1 ForwardInput 字段消费矩阵

| 字段 | Base/LlmRec | OneRec | OneRecXAttention | LlmRecMultiRound |
|------|:-----------:|:------:|:----------------:|:----------------:|
| `token_ids` | READ | READ | READ+**多轮覆盖** | READ+**多轮覆盖** |
| `positions` | READ | READ | READ+**多轮覆盖** | READ+**多轮覆盖** |
| `input_params` | READ | READ+**改 flag** | READ+**拷贝+改 flag** | READ+**每轮改** |
| `sampling_params` | READ | READ | READ (round 0) | READ (round 0) |
| `sampling_params.selected_token_idxes` | READ | READ | READ | READ |
| `sampling_params.do_sample` | READ | READ | READ | READ |
| `sampling_params.logprobs` | READ | READ | READ | READ |
| `sampling_params.max_top_logprobs` | READ | READ | READ | READ |
| `sampling_params.use_beam_search` | READ | **忽略** | **忽略** | **忽略** |
| `sampling_params.acc_logprob` | READ | **忽略** | **忽略** | **忽略** |
| `sampling_params.num_return_sequences` | **忽略** | **忽略** | READ | READ |
| `decoder_sampling_params` | **忽略** | **忽略** | READ (round > 0) | READ (round > 0) |
| `step_decode` / `step_meta()` | **忽略** | **忽略** | READ | READ |
| `transfer_kv_infos` | READ | **忽略** | **忽略** | **忽略** |
| `skip_sampling_for_logits_only` | **忽略** | **忽略** | **忽略** | **忽略** |
| `input_params.multimodal.mm_data` | **忽略** | **忽略** | **忽略** | READ |
| `input_params.expert.eplb_info` | READ | **忽略** | **忽略** | **忽略** |
| `onerec_params.*` | - | READ | - | - |
| `onerec_xattention_params.*` | - | - | READ | - |
| `llmrec_params.*` | - | - | - | READ |

### 4.2 executor->forward() 调用对比

| Pipeline | tokens 参数 | positions 参数 | kv_caches 参数 | params 参数 | 每步调用次数 |
|----------|------------|---------------|---------------|------------|:-----------:|
| **Base/LlmRec** | `input.token_ids` | `input.positions` | `worker.kv_caches_` | `input.input_params` | 1 |
| **OneRec** | `token_ids` 或 `encoder_tokens` | `positions` 或 `encoder_positions` | `worker.kv_caches_` | `input_params`（原地改 flag） | 1~2 |
| **OneRecXAttention** | `token_ids` 或 `encoder_tokens`，decode 轮次**重写** | `positions` 或 `encoder_positions`，decode 轮次**重写** | `worker.kv_caches_` | input_params **拷贝**（encoder/decoder 独立） | 每轮 1~2，N 轮 |
| **LlmRecMultiRound** | `token_ids`，decode 轮次**重写** | `positions`，decode 轮次**重写** | `worker.kv_caches_` | `input_params`（每轮**原地修改**） | 每轮 1，N 轮 |

### 4.3 Sampler 对比

| Pipeline | Sampler 类型 | 签名 |
|----------|-------------|------|
| **Base/LlmRec** | `worker.sampler_`（通用 LLM） | `sampler_->forward(logits, sampling_params)` |
| **OneRec** | `rec_sampler_`（Rec 专用） | `rec_sampler_->forward(logits, sampling_params, filter_mask)` |
| **OneRecXAttention** | `rec_sampler_`（Rec 专用） | `rec_sampler_->forward(logits, sampling_params, filter_mask, &sampling_context)` |
| **LlmRecMultiRound** | `rec_sampler_`（Rec 专用） | `rec_sampler_->forward(logits, round_sampling_params)` |

### 4.4 prepare_work_before_execute() 对比

| Pipeline | 额外处理（在 Base 之上） |
|----------|------------------------|
| **Base** | H2D 传输 + KV block swap + DP padding |
| **LlmRec** | 同上 + 多模态数据处理 (`prepare_multi_modal_data`) |
| **OneRec** | 同上 + `decoder_context_embedding` dtype 转换 |
| **OneRecXAttention** | 同上 + **分配 shared/unshared KV caches** + 构建 `beam_width_tensor` / `current_round_tensor` + `decoder_context_embedding` dtype 转换 |
| **LlmRecMultiRound** | 同上 + 多模态 + **分配 full/unshared/shared KV caches** + 构建 `decode_positions_tensor_list` + 设置 `llmrec_params` |

---

## 5. 对 GE Pipeline 接入的启示

如果 GE 图将 beam search 全部内化，新的 WorkerPipeline 的特征：

| 维度 | GE Pipeline 预期 |
|------|-----------------|
| `token_ids` | READ（不需要多轮覆盖） |
| `positions` | READ（不需要多轮覆盖） |
| `input_params` | READ（不需要每轮修改） |
| `sampling_params` | READ（用于输出格式对齐） |
| `decoder_sampling_params` | **忽略**（多轮在图内） |
| `step_decode` | **忽略**（多轮在图内） |
| `onerec_params` / `onerec_xattention_params` / `llmrec_params` | **忽略**（GE 图不需要） |
| `transfer_kv_infos` | **忽略** |
| executor->forward() 调用次数 | **1 次** |
| Sampler | 不需要（图内完成采样和 beam search） |
| prepare_work_before_execute() | H2D + KV block swap（Base 逻辑即可） |

**结论**: GE WorkerPipeline 的消费模式最接近 **Base RecWorkPipeline**（单次 forward，无多轮逻辑），主要差异在于：
1. 输入准备可能需要 GE 特有的字段（如 `input_tensor_map`）
2. 输出需要组装为 `beam_sequence_group` 格式以对接 EnginePipeline
3. 不需要 sampler、beam_search kernel、filter_mask 等组件
