# RecEnginePipeline 技术文档

## 1. 概述

`RecEnginePipeline` 是 `RecEngine` 内部的**策略模式抽象基类**，位于 `xllm/core/distributed_runtime/rec_engine.h`。它将推荐引擎的模型推理执行逻辑按不同模型类型（LlmRec / OneRec）和推理模式（单步 / 多轮 / xattention）解耦为独立的 Pipeline 实现，由工厂方法在运行时选择。

## 2. 类继承体系

```
RecEnginePipeline (抽象基类)
│
├── LlmRecEnginePipeline
│       qwen2/qwen3 系列，通过 DistManager 远程 Worker 执行
│       支持 DP + TP 分布式并行
│
├── OneRecLocalEnginePipeline (中间层，共享本地 Worker 逻辑)
│   ├── OneRecPrefillOnlyEnginePipeline
│   │       传统 OneRec prefill + decode 模式
│   │
│   └── OneRecXAttentionEnginePipeline
│           OneRec xattention 模式，多轮 decode 在 device 内完成
│
└── RecMultiRoundEnginePipeline
        LlmRec 多轮推理，多轮 decode 在 device/worker 内完成
```

## 3. 抽象接口

`RecEnginePipeline` 定义了 6 个核心纯虚函数，覆盖推理引擎的完整生命周期：

| 接口 | 签名 | 功能 |
|------|------|------|
| `setup_workers()` | `void setup_workers()` | 初始化 Worker 连接（远程 DistManager 或本地 Worker） |
| `process_group_test()` | `void process_group_test()` | 测试分布式通信组连通性（HCCL/NCCL） |
| `init_model_workers()` | `bool init_model_workers(const string& path)` | 加载模型权重到各 Worker |
| `estimate_min_available_memory()` | `int64_t estimate_min_available_memory()` | 估算所有 Worker 中最小的可用 KV Cache 显存 |
| `allocate_kv_cache()` | `bool allocate_kv_cache(const KVCacheShape&)` | 在各 Worker 上分配 KV Cache |
| `step()` | `ForwardOutput step(vector<Batch>&)` | 执行一次推理前向（可能包含多步 decode） |

辅助虚函数：

| 接口 | 默认值 | 功能 |
|------|--------|------|
| `minimal_kv_cache_blocks()` | 返回 0 | OneRec 模式需要预分配最小 metadata KV blocks |
| `get_active_activation_memory()` | 纯虚 | 获取各 Worker 当前激活内存占用 |
| `num_workers()` | 纯虚 | 返回 Worker 数量 |

## 4. Pipeline 实现详解

### 4.1 LlmRecEnginePipeline

**适用模型**: qwen2 / qwen3 / qwen3_moe

**Worker 管理**:
- 通过 `DistManager` 获取远程 `WorkerClient` 列表
- 支持 DP（数据并行）+ TP（张量并行）混合并行
- `dp_size_` 控制数据并行度，`dp_local_tp_size_` = worker 总数 / dp_size

**step 流程**:

```
┌─────────────────────────────────────────────────┐
│ 1. prepare_inputs(batches)                       │
│    - 按 dp_rank 拆分 batch                       │
│    - 每个 dp_rank 独立 prepare_forward_input      │
│    - 同步 dp_global_token_nums / dp_is_decode     │
├─────────────────────────────────────────────────┤
│ 2. 异步发送到所有 worker (step_remote_async)      │
│    - worker_rank / dp_local_tp_size_ = dp_rank   │
│    - 同一 dp_rank 的 TP workers 共享同一输入      │
├─────────────────────────────────────────────────┤
│ 3. folly::collectAll 收集所有 worker 结果         │
├─────────────────────────────────────────────────┤
│ 4. 处理输出 (每个 dp_rank 独立)                   │
│    - 有 src_seq_idxes → process_beam_search_output│
│    - 否则 → process_sample_output                 │
│    - beam search 时执行 process_beam_search 转移  │
│      src_blocks_ → blocks_                       │
│    - refresh_sequences_from_groups()              │
├─────────────────────────────────────────────────┤
│ 5. 循环 max_steps 步 (由 batch 中 max_tokens 决定)│
│ 6. batch.finish()                                │
└─────────────────────────────────────────────────┘
```

**多步控制**: `get_max_steps_from_batch()` 遍历所有 sequence 的 `stopping_checker`，取最大 `max_generated_tokens` 作为循环上限；无 stopping_checker 时回退到 `kRecDecodeSteps`。

**性能埋点**: 记录 first/second/third token latency、prepare_input latency、sampling latency。

### 4.2 OneRecLocalEnginePipeline (中间层)

**职责**: 封装 OneRec 系列 Pipeline 共享的本地 Worker 管理逻辑。

**Worker 初始化**:
- 单卡: 创建 rank=0 的轻量 `ProcessGroup`，不建立通信后端
- 多卡 NPU: 通过 `parallel_state::create_npu_process_groups()` 创建 HCCL 通信组
- 多卡非 NPU: 通过 `create_local_process_groups()` 创建本地通信组
- 每个 rank 创建一个本地 `Worker` 对象 (WorkerType::REC)

**共享功能**: `process_group_test`、`init_model_workers`、`estimate_min_available_memory`、`allocate_kv_cache`、`get_active_activation_memory`、`num_workers` 均由此中间层实现，子类只需实现 `step()`。

### 4.3 OneRecPrefillOnlyEnginePipeline

**适用模型**: onerec（传统模式）

**step 流程**:

```
┌──────────────────────────────────────────┐
│ 1. prepare_inputs (worker[0])            │
│ 2. get_model_output → prefill            │  ← first token
│ 3. process_sample_output                 │
│ 4. for i in [0, kRecDecodeSteps):        │
│    ├─ prepare_inputs (worker[0])         │
│    ├─ get_model_output → decode          │  ← second/third token
│    └─ process_sample_output              │
│ 5. batch.finish()                        │
└──────────────────────────────────────────┘
```

**get_model_output**: 所有 Worker 异步执行 `step_async`，取 rank 0 的输出，将 embeddings / next_tokens / logprobs / top_tokens / top_logprobs 从 GPU 搬到 CPU，然后 synchronize。

**KV Cache**: 当 `use_legacy_onerec_prefill_only_contract()` 为 true 时，预分配 `kMinimalOneRecMetadataKVBlocks = 2` 个 metadata KV blocks。

### 4.4 OneRecXAttentionEnginePipeline

**适用模型**: onerec（xattention 模式）

**step 流程**:

```
┌──────────────────────────────────────────┐
│ 1. prepare_inputs (worker[0])            │
│ 2. get_model_output (单次推理)            │  ← first token
│    - 多轮 decode 在 device 内部完成       │
│ 3. 处理输出:                              │
│    - 有 beam_sequence_group →             │
│      process_beam_sequence_group         │
│    - 否则 → process_sample_output         │
│ 4. batch.finish()                        │
└──────────────────────────────────────────┘
```

**与 PrefillOnly 的区别**:
- 只做一次前向推理，多轮 decode 在 device/worker 内部完成
- 支持 beam search 输出 (`beam_sequence_group`)
- 内置 debug trace 环境变量: `XLLM_DEBUG_ONEREC_ENGINE_TRACE`、`XLLM_DEBUG_ONEREC_XATTN_STAGE_TIMING`
- 始终预分配 `kMinimalOneRecMetadataKVBlocks` 个 KV blocks

### 4.5 RecMultiRoundEnginePipeline

**适用模型**: qwen2/qwen3（多轮模式，`max_decode_rounds > 0`）

**step 流程**:

```
┌──────────────────────────────────────────┐
│ 1. prepare_inputs (worker[0])            │
│ 2. get_model_output (单次推理)            │  ← first token
│    - 多轮 decode 在 worker/device 内完成  │
│ 3. process_beam_sequence_group           │
│    - 处理多轮 beam search 结果            │
│ 4. batch.finish()                        │
└──────────────────────────────────────────┘
```

**Worker 初始化**: 与 OneRecLocal 类似使用本地 Worker，但单卡场景下通过 `create_process_group()` 创建带 host:port 的 rank_size=1 通信组。

**输出处理**: D2H 搬运 `beam_sequence_group` 和 `beam_search_output.out_logprobs`。

## 5. 工厂选择逻辑

`RecEngine::create_pipeline()` 根据 `RecPipelineType` 创建对应 Pipeline：

```
model_type 判断 (rec_model_utils.h):
│
├─ "onerec"
│   └─ is_onerec_xattention_mode()?
│       ├─ Yes → OneRecXAttentionEnginePipeline
│       └─ No  → OneRecPrefillOnlyEnginePipeline
│
├─ "qwen2" / "qwen3" / "qwen3_moe"
│   └─ is_rec_multi_round_mode()?
│       ├─ Yes → RecMultiRoundEnginePipeline
│       └─ No  → LlmRecEnginePipeline
│
└─ 其他 → kNone (不支持)
```

模式判断条件：
- `is_rec_multi_round_mode()`: `RecConfig::max_decode_rounds() > 0`
- `is_onerec_xattention_mode()`: `RecConfig::max_decode_rounds() > 0`
- `use_legacy_onerec_prefill_only_contract()`: `enable_rec_prefill_only && !is_onerec_xattention_mode()`

## 6. RecEngine 与 Pipeline 的协作

`RecEngine` 负责 Pipeline 无关的公共逻辑，Pipeline 负责具体的执行策略：

```
RecEngine::init()
│
├─ init_model()
│   ├─ ModelLoader::create() → tokenizer, model_args, quant_args
│   ├─ get_rec_model_kind() → get_rec_pipeline_type()
│   ├─ create_pipeline() → pipeline_
│   ├─ pipeline_->setup_workers()
│   ├─ pipeline_->process_group_test()
│   ├─ 创建 ThreadPool (16 线程)
│   ├─ 计算 KV cache 配置 (n_local_kv_heads, head_dim, dtype)
│   └─ pipeline_->init_model_workers()
│
├─ estimate_kv_cache_capacity()
│   ├─ pipeline_->minimal_kv_cache_blocks() > 0?
│   │   └─ 使用固定 block 数 (OneRec metadata)
│   └─ pipeline_->estimate_min_available_memory()
│       └─ 按 max_memory_utilization / max_cache_size 裁剪
│
└─ allocate_kv_cache()
    ├─ 初始化 BlockManagerPool
    └─ pipeline_->allocate_kv_cache()

RecEngine::step(batches)
└─ pipeline_->step(batches)
```

## 7. 关键设计特点

1. **策略模式**: `RecEngine` 通过 `pipeline_` 指针多态调用，不感知具体推理细节，新增模型类型只需添加 Pipeline 子类

2. **异步并行**: 所有 Worker 操作通过 `folly::SemiFuture` 异步并发执行，使用 `folly::collectAll` 同步收集结果

3. **DP + TP 混合并行**: `LlmRecEnginePipeline` 支持 `dp_size × tp_size` 的二维并行，同一 DP rank 的 TP workers 共享输入、协同推理

4. **动态步数控制**: LlmRec 根据 batch 中各请求的 `max_tokens` 动态决定 decode 循环次数，而非固定步数

5. **KV Cache 统一管理**: 容量估算取所有 Worker 的最小可用显存，分配由 `RecEngine` 统一协调 `BlockManagerPool` + Pipeline

6. **Device 内循环优化**: XAttention 和 MultiRound 两种 Pipeline 将多轮 decode 下沉到 device/worker 内部，减少 engine 层的 step 开销

7. **Beam Search 支持**: LlmRec 和 XAttention/MultiRound Pipeline 均支持 beam search 输出处理，包括 `src_blocks_ → blocks_` 的 KV Cache 转移

---

## 8. step 输入组装与输出处理详解

### 8.1 核心数据结构

#### 8.1.1 ForwardInput (Worker 输入)

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

#### 8.1.2 ForwardOutput / RawForwardOutput (Worker 输出)

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

### 8.2 输入组装流程

#### 8.2.1 LlmRecEnginePipeline::prepare_inputs

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

**Step 1: refresh_forward_type**

重置 `batch_forward_type_` 为 EMPTY，遍历所有 sequence 重新推导：

```
sequence.stage() → PREFILL / CHUNKED_PREFILL / DECODE
多条 sequence 合并 → PREFILL / DECODE / CHUNKED_PREFILL / MIXED
```

**Step 2: BatchInputBuilder::process_sequences**

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

**Step 3: state_to_forward_input**

将 Builder 累积的 vector 转换为 Tensor：
```cpp
token_ids     = torch::tensor(flatten_tokens_vec, kInt)
positions     = torch::tensor(flatten_positions_vec, kInt)
block_tables  = create_2d_tensor(block_tables_vec, kInt)
kv_seq_lens   = torch::tensor(seq_lens, kInt)
...
sampling_params.init(sampling_params_vec, selected_token_idxes, ...)
```

**Step 4: DP 元数据同步**

```cpp
// 每个 dp_rank 的 ForwardInput 需要知道全局信息:
dp_global_token_nums[dp_rank] = token_ids.numel()  // 各 rank token 数
dp_is_decode[dp_rank] = (forward_type == DECODE && q_max_seq_len == 1)

// 空 batch 继承第一个非空 batch 的 forward_type
// 确保所有 DP rank 的 batch_forward_type 一致
```

#### 8.2.2 OneRecLocalEnginePipeline (OneRecPrefillOnly / XAttention)

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

#### 8.2.3 RecMultiRoundEnginePipeline

与 OneRecLocal 类似，通过 `workers_[0]->prepare_inputs(batches[0])` 调用，使用 `RecMultiRoundBatchInputBuilder`。多轮 decode 逻辑在 worker/device 内部完成，engine 层只准备一次输入。

### 8.3 Worker 执行与数据传输

#### 8.3.1 LlmRec: 远程 Worker (step_remote_async)

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

#### 8.3.2 OneRec / MultiRound: 本地 Worker (step_async)

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

### 8.4 输出处理流程

#### 8.4.1 LlmRecEnginePipeline::step 输出处理

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

**process_sample_output (RawForwardOutput 版本)**:
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

**process_beam_search_output**:
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

#### 8.4.2 OneRecPrefillOnlyEnginePipeline::step 输出处理

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

#### 8.4.3 OneRecXAttentionEnginePipeline::step 输出处理

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

**process_beam_sequence_group**:
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

#### 8.4.4 RecMultiRoundEnginePipeline::step 输出处理

与 XAttention 类似：
```
output = get_model_output(forward_inputs)
batches[0].process_beam_sequence_group(output)
batches[0].finish()
```

D2H 搬运 `beam_sequence_group` 和 `beam_search_output.out_logprobs`。

### 8.5 完整数据流图

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

### 8.6 各 Pipeline step 对比

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
