# RecEnginePipeline::step 输入组装与输出处理详解

## 1. 总览

`RecEnginePipeline::step()` 是推荐引擎的核心执行入口，负责：
1. 将 `Batch` 中的序列数据组装成 Worker 计算所需的 `ForwardInput`
2. 将 `ForwardInput` 发送给 Worker（本地或远程）执行推理
3. 接收 Worker 返回的输出并回写到 `Batch` 的序列中

本文档按四种 Pipeline 分别说明其 step 的输入组装和输出处理流程。

---

## 2. 核心数据结构

### 2.1 ForwardInput (Worker 输入)

```
ForwardInput
├── token_ids: Tensor [total_tokens]          — 所有序列的 token 拼接成一维
├── positions: Tensor [total_tokens]          — 每个 token 的位置 ID
├── token_ids_host / positions_host           — CPU 侧视图
├── input_params: ModelInputParams
│   ├── meta: BatchMetadata
│   │   ├── batch_forward_type               — PREFILL / DECODE / MIXED
│   │   ├── num_sequences                    — batch 中序列数
│   │   ├── kv_max_seq_len                   — 最大 KV 序列长度
│   │   └── q_max_seq_len                    — 最大 Query 序列长度
│   ├── attention: AttentionMetadata
│   │   ├── kv_seq_lens: Tensor [num_seqs]   — 每条序列的 KV 长度
│   │   ├── kv_cache_tokens_nums             — 已缓存 token 数
│   │   ├── q_seq_lens / q_cu_seq_lens       — Query 长度及累积长度
│   │   ├── new_cache_slots                  — 新 token 写入的 KV slot ID
│   │   ├── block_tables: Tensor [num_seqs, max_blocks] — 每条序列的 block ID 表
│   │   ├── paged_kv_indptr                  — FlashInfer 分页索引
│   │   ├── paged_kv_indices
│   │   └── paged_kv_last_page_len
│   ├── multimodal: mm_data                  — 多模态数据
│   └── embedding: input_embedding           — 外部输入 embedding
├── sampling_params: SamplingParameters
│   ├── frequency_penalties / presence_penalties / repetition_penalties
│   ├── temperatures / top_p / top_k
│   ├── selected_token_idxes                 — 需要采样的 token 在 flatten 中的位置
│   ├── sample_idxes                         — 采样参数索引
│   ├── unique_token_ids / counts            — 用于 repetition penalty
│   └── do_sample                            — 是否需要采样
├── step_decode: StepDecodeMeta              — step 级 decode 元数据
└── transfer_kv_infos                        — 分离式 prefill/decode KV 传输信息
```

### 2.2 ForwardOutput / RawForwardOutput (Worker 输出)

**ForwardOutput** (本地 Worker 直接返回):
```
ForwardOutput
├── sample_output: SampleOutput
│   ├── next_tokens: Tensor [num_seqs]       — 生成的下一个 token ID
│   ├── logprobs: Tensor [num_seqs]          — token 对数概率
│   ├── top_logprobs / top_tokens            — top-k 对数概率和 token
│   ├── embeddings: Tensor [num_seqs, dim]   — 隐藏状态 embedding
│   └── mm_embeddings                        — 多模态 embedding
├── beam_search_output: BeamSearchOutput
│   ├── src_seq_idxes: Tensor [num_seqs]     — beam search 源序列索引
│   ├── out_tokens: Tensor [num_seqs]        — beam 选中的 token
│   └── out_logprobs: Tensor [num_seqs]      — beam 累积对数概率
├── beam_sequence_group: Tensor [groups, beam_width, rounds] — 多轮 beam 结果
├── logits / embedding                       — 原始 logits / embedding
└── expert_load_data                         — MoE 专家负载数据
```

**RawForwardOutput** (远程 Worker 通过 RPC/SHM 返回):
```
RawForwardOutput
├── outputs: vector<RawSampleOutput>
│   └── tokens: vector<RawToken>
│       ├── id: int64                        — token ID
│       ├── logprob: optional<float>         — 对数概率
│       ├── top_tokens / top_logprobs        — top-k 信息
│       └── embeddings: vector<float>        — 隐藏状态
├── src_seq_idxes: vector<int32_t>           — beam search 源序列索引
├── out_tokens: vector<int32_t>              — beam 选中的 token
├── out_logprobs: vector<float>              — beam 累积对数概率
├── beam_sequence_group: vector<int32_t>     — 多轮 beam 结果 (展平)
├── mm_embeddings: vector<Tensor>            — 多模态 embedding
└── expert_load_data / prepared_layer_id     — MoE 负载
```

---

## 3. 输入组装流程

### 3.1 LlmRecEnginePipeline::prepare_inputs

**调用链**:
```
LlmRecEnginePipeline::step(batches)
  └─ prepare_inputs(batches)
       ├─ for each dp_rank:
       │    ├─ batch[dp_rank].refresh_forward_type()
       │    └─ batch[dp_rank].prepare_forward_input(args, threadpool)
       │         └─ BatchInputBuilder::build_forward_input()
       │              ├─ process_sequences()
       │              │    └─ for each sequence:
       │              │         ├─ extract_tokens_and_positions()
       │              │         ├─ setup_kv_cache_info()
       │              │         └─ handle_sampling_parameters()
       │              ├─ padding_decode_batch_size()
       │              └─ state_to_forward_input()
       │
       └─ 同步 DP 元数据:
            ├─ dp_global_token_nums[dp_rank] = 各 rank 的 token 总数
            ├─ dp_is_decode[dp_rank] = 是否全部为 decode
            └─ 空 batch 继承第一个非空 batch 的 forward_type
```

**详细步骤**:

#### Step 1: refresh_forward_type

重置 `batch_forward_type_` 为 EMPTY，遍历所有 sequence 重新推导：

```
sequence.stage() → PREFILL / CHUNKED_PREFILL / DECODE
多条 sequence 合并 → PREFILL / DECODE / CHUNKED_PREFILL / MIXED
```

#### Step 2: BatchInputBuilder::process_sequences

对每条 sequence 调用 `process_single_sequence`，核心三步：

**2a. extract_tokens_and_positions**:
```
对于 sequence 中 [n_kv_cache_tokens, seq_len) 范围的每个 token:
  flatten_tokens_vec.push_back(token_ids[j])
  flatten_positions_vec.push_back(j)

结果: 所有序列的 token 拼接成一个一维数组
例: seq0=[A,B,C], seq1=[D,E] → flatten=[A,B,C,D,E], positions=[0,1,2,0,1]
```

**2b. setup_kv_cache_info**:
```
对于每条 sequence:
  - 从 sequence->kv_state().kv_blocks() 获取已分配的 KV cache block ID
  - 计算新 token 需要写入的 slot ID → new_token_slot_ids
  - 构建 block_tables (每条序列的 block ID 列表，padding 到等长)
  - 构建 FlashInfer 分页注意力元数据:
    - paged_kv_indptr: 累积 block 数量索引
    - paged_kv_indices: 所有 block ID 展平
    - paged_kv_last_page_len: 最后一个 page 的有效长度
  - 记录 kv_seq_lens, q_seq_lens, kv_cache_tokens_nums
```

**2c. handle_sampling_parameters**:
```
记录需要采样的 token 在 flatten_tokens_vec 中的位置 → selected_token_idxes
收集该 sequence 的采样超参 (temperature, top_p, top_k, penalties)
收集已生成的 unique tokens 用于 repetition penalty
```

#### Step 3: state_to_forward_input

将 Builder 累积的 vector 转换为 Tensor：
```cpp
token_ids     = torch::tensor(flatten_tokens_vec, kInt)
positions     = torch::tensor(flatten_positions_vec, kInt)
block_tables  = create_2d_tensor(block_tables_vec, kInt)
kv_seq_lens   = torch::tensor(seq_lens, kInt)
...
sampling_params.init(sampling_params_vec, selected_token_idxes, ...)
```

#### Step 4: DP 元数据同步

```cpp
// 每个 dp_rank 的 ForwardInput 需要知道全局信息:
dp_global_token_nums[dp_rank] = token_ids.numel()  // 各 rank token 数
dp_is_decode[dp_rank] = (forward_type == DECODE && q_max_seq_len == 1)

// 空 batch 继承第一个非空 batch 的 forward_type
// 确保所有 DP rank 的 batch_forward_type 一致
```

### 3.2 OneRecLocalEnginePipeline (OneRecPrefillOnly / XAttention)

**调用链**:
```
OneRecPrefillOnlyEnginePipeline::step(batches)
  └─ workers_[0]->prepare_inputs(batches[0])
       └─ WorkerImpl::prepare_inputs(batch)
            └─ model_executor_->prepare_inputs(batch)
                 └─ batch.prepare_forward_input(num_decoding_tokens, 0, args)
                      └─ batch.prepare_rec_forward_input(...)
                           └─ RecBatchInputBuilder::create(rec_type, ...)
                                └─ builder->build_rec_forward_input(...)
```

与 LlmRec 的主要区别：
- **不经过 DP 拆分**，直接对整个 batch 调用 `prepare_inputs`
- 使用 `RecBatchInputBuilder` 子类（`OneRecBatchInputBuilder` / `OneRecXAttentionBatchInputBuilder`），针对 OneRec 模型的特殊输入格式
- `prepare_rec_forward_input` 会先 `refresh_sequences_from_groups()` 同步 beam search 后的序列状态

### 3.3 RecMultiRoundEnginePipeline

与 OneRecLocal 类似，通过 `workers_[0]->prepare_inputs(batches[0])` 调用，使用 `RecMultiRoundBatchInputBuilder`。多轮 decode 逻辑在 worker/device 内部完成，engine 层只准备一次输入。

---

## 4. Worker 执行与数据传输

### 4.1 LlmRec: 远程 Worker (step_remote_async)

```
LlmRecEnginePipeline::step()
  │
  ├─ for each worker_rank in [0, worker_clients_num_):
  │    dp_rank = worker_rank / dp_local_tp_size_
  │    worker_clients_[worker_rank]->step_remote_async(forward_inputs[dp_rank])
  │
  └─ folly::collectAll(futures).get()  // 等待所有 worker 完成
```

**远程执行链路**:
```
WorkerClient::step_remote_async(ForwardInput)
  └─ RemoteWorker::step_remote_async(ForwardInput)
       └─ threadpool_.schedule([channel, input, promise]() {
            channel_->execute_model_async(input, promise)
          })

channel 有两种传输方式:

[1] ShmChannel (共享内存, 优先):
    input_shm_manager_->input_write(input)   // 写入 SHM
    output_shm_manager_->raw_output_read(raw_output)  // 从 SHM 读结果
    promise.setValue(raw_output)

[2] CommChannel (brpc, 回退):
    forward_input_to_packed_proto(input, pb_input)  // 序列化为 protobuf
    stub_->ExecuteModel(pb_input, pb_output, closure)  // 异步 RPC
    // RPC 返回时:
    ExecuteModelClosure::Run():
      proto_to_forward_output(pb_output, raw_output)  // 反序列化
      promise.setValue(raw_output)
```

### 4.2 OneRec / MultiRound: 本地 Worker (step_async)

```
OneRecPrefillOnlyEnginePipeline::get_model_output(forward_inputs)
  │
  ├─ for each worker in workers_:
  │    worker->step_async(model_inputs)
  │      └─ 本地进程内直接执行模型推理
  │
  └─ folly::collectAll(futures).get()
       └─ 取 results.front() (rank 0 的输出)
       └─ D2H: sample_output → CPU
       └─ Device::synchronize_default_stream()
```

---

## 5. 输出处理流程

### 5.1 LlmRecEnginePipeline::step 输出处理

```
results = folly::collectAll(futures).get()

for each dp_rank (worker_rank 按 dp_local_tp_size_ 步进):
  result = results[worker_rank].value()  // RawForwardOutput

  if result.src_seq_idxes.empty():
    // 普通采样输出
    batches[dp_rank].process_sample_output(result, false)
  else:
    // beam search 输出
    batches[dp_rank].process_beam_search_output(result, false)
    // 处理 KV cache block 转移
    for each seq in batch:
      if seq->check_beam_search() && !src_blocks.empty():
        seq->kv_state().process_beam_search(nullopt)

  batches[dp_rank].refresh_sequences_from_groups()
```

#### process_sample_output (RawForwardOutput 版本)

```
for each output_target (对应一条 sequence):
  if sequence 已完成 → 跳过
  if sequence 状态需要更新 → update_sequence_state()

  raw_sample_output = result.outputs[output_idx]
  for each token in raw_sample_output.tokens:
    append_token_for_sequence(seq, token)  // 将生成的 token 追加到序列
    if token 包含 embeddings:
      seq->update_embeddings(embeddings)   // 更新隐藏状态
    if seq 已完成 → break
```

#### process_beam_search_output

```
beam_width = sequences[0]->sampling_param()->beam_width

for each sequence_group:
  // 保存所有 beam 的当前状态 (tokens, logprobs, acc_logprob)
  src_token_ids[i] = sequences[src_seq_idx].tokens()
  src_logprobs[i] = sequences[src_seq_idx].logprobs()
  src_acc_logprob[i] = sequences[src_seq_idx].get_acc_logprob()

  for each beam i in [0, beam_width):
    task_id = group_id * beam_width + i
    src_seq_idx = raw_output.src_seq_idxes[task_id]

    // 1. 用源序列的历史 token 替换当前序列
    for token_idx in [num_prompt_tokens, num_tokens):
      base_seq->update_token(token_idx, src_token_ids[i][token_idx])

    // 2. 追加新生成的 token
    new_token = raw_output.out_tokens[task_id]
    new_token.logprob = raw_output.out_logprobs[task_id] - src_acc_logprob[i]
    append_token_for_sequence(base_seq, new_token)

    // 3. 更新累积 logprob
    base_seq->set_acc_logprob(raw_output.out_logprobs[task_id])

    // 4. 复制源序列的 KV cache blocks 到当前序列
    base_seq->kv_state().set_src_blocks(src_seq.kv_blocks(), need_swap)
```

### 5.2 OneRecPrefillOnlyEnginePipeline::step 输出处理

```
// Prefill 阶段
prefill_output = get_model_output(forward_inputs)
batches[0].process_sample_output(prefill_output.sample_output, false)

// Decode 阶段 (kRecDecodeSteps 次)
for i in [0, kRecDecodeSteps):
  forward_inputs = workers_[0]->prepare_inputs(batches[0])
  decode_output = get_model_output(forward_inputs)
  batches[0].process_sample_output(
    decode_output.sample_output,
    false,
    force_requested_beam_result_size = (i + 1 == kRecDecodeSteps)
  )

batches[0].finish()
```

**get_model_output 中的 D2H 搬运**:
```
sample_output.embeddings → CPU (float32)
sample_output.next_tokens → CPU
sample_output.logprobs → CPU
sample_output.top_tokens → CPU
sample_output.top_logprobs → CPU
Device::synchronize_default_stream()  // 等待 D2H 完成
```

### 5.3 OneRecXAttentionEnginePipeline::step 输出处理

```
output = get_model_output(forward_inputs)

if output.beam_sequence_group 非空:
  // beam search 结果 (多轮 decode 在 device 内完成)
  batches[0].process_beam_sequence_group(output)
else:
  // 普通采样结果
  batches[0].process_sample_output(output.sample_output, false)

batches[0].finish()
```

#### process_beam_sequence_group

```
beam_width = sequences[0]->sampling_param()->beam_width
result_width = output.beam_sequence_group.size(1)
total_rounds = output.beam_sequence_group.size(2)

for each group g:
  for each beam b in [0, result_width):
    for each round r in [0, total_rounds):
      row_tokens[r] = beam_sequence_group[g][b][r]
    group_flat2d[b] = row_tokens
    last_logprobs[b] = output.beam_search_output.out_logprobs[g * result_width + b]

  seq->set_beam_result(result_width, total_rounds, group_flat2d, last_logprobs)
```

将多轮 beam search 结果以 `[result_width, total_rounds]` 的二维 token 矩阵形式写入 sequence，供上层读取。

### 5.4 RecMultiRoundEnginePipeline::step 输出处理

与 XAttention 类似：
```
output = get_model_output(forward_inputs)
batches[0].process_beam_sequence_group(output)
batches[0].finish()
```

D2H 搬运 `beam_sequence_group` 和 `beam_search_output.out_logprobs`。

---

## 6. 完整数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                    RecEnginePipeline::step()                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────── 输入组装 ──────────────┐                      │
│  │                                        │                      │
│  │  Batch (sequences)                     │                      │
│  │    │                                   │                      │
│  │    ├─ refresh_forward_type()           │                      │
│  │    │   → PREFILL / DECODE / MIXED      │                      │
│  │    │                                   │                      │
│  │    ├─ prepare_forward_input()          │                      │
│  │    │   │                               │                      │
│  │    │   ├─ extract_tokens_and_positions │                      │
│  │    │   │   flatten_tokens_vec          │                      │
│  │    │   │   flatten_positions_vec       │                      │
│  │    │   │                               │                      │
│  │    │   ├─ setup_kv_cache_info          │                      │
│  │    │   │   block_tables                │                      │
│  │    │   │   new_cache_slots             │                      │
│  │    │   │   paged_kv_indptr/indices     │                      │
│  │    │   │   kv_seq_lens / q_seq_lens    │                      │
│  │    │   │                               │                      │
│  │    │   ├─ handle_sampling_parameters   │                      │
│  │    │   │   selected_token_idxes        │                      │
│  │    │   │   temperature/top_p/top_k     │                      │
│  │    │   │   penalties                   │                      │
│  │    │   │                               │                      │
│  │    │   └─ state_to_forward_input()     │                      │
│  │    │       → ForwardInput              │                      │
│  │    │                                   │                      │
│  │    └─ (LlmRec) DP 元数据同步           │                      │
│  │        dp_global_token_nums            │                      │
│  │        dp_is_decode                    │                      │
│  │                                        │                      │
│  └────────────────────────────────────────┘                      │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────── Worker 执行 ───────────┐                      │
│  │                                        │                      │
│  │  [LlmRec] 远程 Worker:                │                      │
│  │    step_remote_async(ForwardInput)     │                      │
│  │      → SHM / brpc 传输                │                      │
│  │      → Worker 端模型推理               │                      │
│  │      → 返回 RawForwardOutput           │                      │
│  │                                        │                      │
│  │  [OneRec/MultiRound] 本地 Worker:     │                      │
│  │    step_async(ForwardInput)            │                      │
│  │      → 进程内直接推理                  │                      │
│  │      → 返回 ForwardOutput              │                      │
│  │                                        │                      │
│  └────────────────────────────────────────┘                      │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────── 输出处理 ──────────────┐                      │
│  │                                        │                      │
│  │  [LlmRec]                              │                      │
│  │    src_seq_idxes 为空?                 │                      │
│  │      ├─ Yes → process_sample_output    │                      │
│  │      │   逐 seq 追加 token             │                      │
│  │      │   更新 embeddings               │                      │
│  │      │                                 │                      │
│  │      └─ No → process_beam_search_output│                      │
│  │          按 beam_width 分组            │                      │
│  │          用 src_seq_idx 复制历史 token  │                      │
│  │          追加新 token                  │                      │
│  │          转移 KV cache blocks          │                      │
│  │          更新 acc_logprob              │                      │
│  │    refresh_sequences_from_groups()     │                      │
│  │                                        │                      │
│  │  [OneRecPrefillOnly]                   │                      │
│  │    1次 prefill + N次 decode            │                      │
│  │    每次: process_sample_output         │                      │
│  │    D2H: GPU→CPU                        │                      │
│  │                                        │                      │
│  │  [XAttention / MultiRound]             │                      │
│  │    beam_sequence_group 非空?           │                      │
│  │      ├─ Yes → process_beam_sequence_group                     │
│  │      │   写入 [beam_width, rounds] token 矩阵                 │
│  │      └─ No → process_sample_output     │                      │
│  │                                        │                      │
│  │  batch.finish()                        │                      │
│  │    → 标记所有 sequence 为 FINISHED     │                      │
│  │                                        │                      │
│  └────────────────────────────────────────┘                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 各 Pipeline step 对比

| 特性 | LlmRec | OneRecPrefillOnly | OneRecXAttention | MultiRound |
|------|--------|-------------------|------------------|------------|
| Worker 类型 | 远程 (DistManager) | 本地 | 本地 | 本地 |
| 传输方式 | SHM / brpc | 进程内 | 进程内 | 进程内 |
| 返回类型 | RawForwardOutput | ForwardOutput | ForwardOutput | ForwardOutput |
| 输入准备 | prepare_forward_input (DP) | prepare_inputs (worker) | prepare_inputs (worker) | prepare_inputs (worker) |
| 推理步数 | 动态 (max_tokens) | 1 prefill + N decode | 1 次 | 1 次 |
| 多轮 decode | engine 层循环 | engine 层循环 | device 内部 | device 内部 |
| 采样输出 | process_sample_output | process_sample_output | process_sample_output | - |
| beam 输出 | process_beam_search_output | - | process_beam_sequence_group | process_beam_sequence_group |
| D2H 搬运 | 远程已完成 | get_model_output 中 | get_model_output 中 | get_model_output 中 |
| DP 支持 | 是 | 否 | 否 | 否 |
