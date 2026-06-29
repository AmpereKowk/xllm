# RecEnginePipeline 各实例对比分析

## 1. 类继承体系

```
RecEnginePipeline (抽象基类)
│
├── LlmRecEnginePipeline (final)
│       qwen2/qwen3 系列，通过 DistManager 远程 Worker 执行
│
├── OneRecLocalEnginePipeline (中间层，共享本地 Worker 逻辑)
│   ├── OneRecPrefillOnlyEnginePipeline (final)
│   │       传统 OneRec prefill + decode 模式
│   │
│   └── OneRecXAttentionEnginePipeline (final)
│           OneRec xattention 模式，多轮 decode 在 device 内完成
│
└── RecMultiRoundEnginePipeline (final)
        LLM 多轮推理，多轮 decode 在 worker/device 内完成
```

## 2. 抽象接口一览

| 接口方法 | 签名 | 说明 |
|---------|------|------|
| `setup_workers()` | `void` | 初始化 Worker 连接 |
| `process_group_test()` | `void` | 测试分布式通信组连通性 |
| `init_model_workers(path)` | `bool` | 加载模型权重到各 Worker |
| `estimate_min_available_memory()` | `int64_t` | 估算最小可用 KV Cache 显存 |
| `allocate_kv_cache(shape)` | `bool` | 在各 Worker 上分配 KV Cache |
| `minimal_kv_cache_blocks()` | `int64_t` | 最小 KV blocks 预分配数（默认 0） |
| `step(batches)` | `ForwardOutput` | 执行一次推理前向 |
| `get_active_activation_memory()` | `vector<int64_t>` | 获取各 Worker 激活内存 |
| `num_workers()` | `size_t` | 返回 Worker 数量 |

## 3. 各实例逐接口对比

### 3.1 setup_workers()

| 实例 | 实现 |
|------|------|
| **LlmRecEnginePipeline** | 创建 `DistManager`，通过远程连接获取 `worker_clients_`，计算 `dp_size_` 和 `dp_local_tp_size_` |
| **OneRecLocalEnginePipeline** | 空操作（本地 Worker 在 `init_model_workers` 中创建） |
| **RecMultiRoundEnginePipeline** | 空操作（同上） |

**源码位置**: `rec_engine.cpp:252-260` (LlmRec), `rec_engine.cpp:552-554` (OneRecLocal), `rec_engine.cpp:991-993` (MultiRound)

---

### 3.2 process_group_test()

| 实例 | 实现 |
|------|------|
| **LlmRecEnginePipeline** | 非 NPU 时，对所有远程 `worker_clients_` 异步调用 `process_group_test_async()`，`folly::collectAll` 等待 |
| **OneRecLocalEnginePipeline** | 对本地 `workers_` 异步调用 `process_group_test_async()`，`folly::collectAll` 等待 |
| **RecMultiRoundEnginePipeline** | 与 OneRecLocal 相同逻辑 |

**差异**: LlmRec 通过远程 `WorkerClient` 测试通信组；OneRec/MultiRound 通过本地 `Worker` 测试。

**源码位置**: `rec_engine.cpp:262-277` (LlmRec), `rec_engine.cpp:556-569` (OneRecLocal), `rec_engine.cpp:995-1008` (MultiRound)

---

### 3.3 init_model_workers(path)

| 实例 | Worker 类型 | ProcessGroup 创建方式 | 初始化流程 |
|------|------------|---------------------|-----------|
| **LlmRecEnginePipeline** | 远程（DistManager 管理） | 无需创建 PG | 对所有 `worker_clients_` 异步调用 `init_model_async()` |
| **OneRecLocalEnginePipeline** | 本地 `Worker` | 单卡：轻量 `ProcessGroup(rank, world_size, device)`，无通信后端；多卡 NPU：`create_npu_process_groups()`（HCCL）；多卡非 NPU：`create_local_process_groups()` | 创建 PG → 创建 Worker → 异步 `init_model_async()` |
| **RecMultiRoundEnginePipeline** | 本地 `Worker` | 单卡 NPU：`create_process_group(rank=0, world_size=1, rank_size=1, port, host)` 带真实通信后端；多卡 NPU：`create_npu_process_groups()`；非 NPU：`create_local_process_groups()` | 创建 PG → 创建 Worker → 异步 `init_model_async()` |

**关键差异**:
- **LlmRec** 不需要创建 ProcessGroup，Worker 由 DistManager 远程管理
- **OneRecLocal** 单卡时创建轻量 PG（无通信后端），因为只需要 rank/world_size 元数据
- **RecMultiRound** 单卡 NPU 时创建**带真实通信后端**的 PG（`rank_size=1`），因为模型层构造时需要有效的 `tp_group_`

**源码位置**: `rec_engine.cpp:279-296` (LlmRec), `rec_engine.cpp:571-623` (OneRecLocal), `rec_engine.cpp:1010-1070` (MultiRound)

---

### 3.4 estimate_min_available_memory()

| 实例 | 实现 |
|------|------|
| **LlmRecEnginePipeline** | 对所有远程 `worker_clients_` 调用 `estimate_kv_cache_capacity_async()`，取所有 Worker 的最小可用内存，应用 `max_memory_utilization` 和 `max_cache_size` 裁剪 |
| **OneRecLocalEnginePipeline** | 对所有本地 `workers_` 调用 `estimate_kv_cache_capacity_async()`，逻辑与 LlmRec 完全相同 |
| **RecMultiRoundEnginePipeline** | 与 OneRecLocal 完全相同 |

**差异**: 仅调用的目标对象不同（远程 `WorkerClient` vs 本地 `Worker`），计算逻辑完全一致。

**源码位置**: `rec_engine.cpp:298-333` (LlmRec), `rec_engine.cpp:625-660` (OneRecLocal), `rec_engine.cpp:1072-1108` (MultiRound)

---

### 3.5 allocate_kv_cache(shape)

| 实例 | 实现 |
|------|------|
| **LlmRecEnginePipeline** | 对所有远程 `worker_clients_` 异步调用 `allocate_kv_cache_async(shape)`，`folly::collectAll` 等待 |
| **OneRecLocalEnginePipeline** | 对所有本地 `workers_` 异步调用 `allocate_kv_cache_async(shape)` |
| **RecMultiRoundEnginePipeline** | 与 OneRecLocal 相同 |

**源码位置**: `rec_engine.cpp:335-349` (LlmRec), `rec_engine.cpp:662-676` (OneRecLocal), `rec_engine.cpp:1110-1124` (MultiRound)

---

### 3.6 minimal_kv_cache_blocks()

| 实例 | 返回值 | 说明 |
|------|--------|------|
| **LlmRecEnginePipeline** | `0`（使用基类默认值） | 不需要预分配 metadata KV blocks |
| **OneRecPrefillOnlyEnginePipeline** | `kMinimalOneRecMetadataKVBlocks (2)` 或 `0` | 当 `use_legacy_onerec_prefill_only_contract()` 为 true 时返回 2 |
| **OneRecXAttentionEnginePipeline** | `kMinimalOneRecMetadataKVBlocks (2)` | 始终返回 2 |
| **RecMultiRoundEnginePipeline** | `0`（使用基类默认值） | 不需要预分配 |

**作用**: 当返回值 > 0 时，`RecEngine::estimate_kv_cache_capacity()` 会跳过动态估算，直接使用固定 block 数。这是 OneRec 模型的特殊需求——它只需要少量 metadata KV blocks 而非完整的 KV Cache。

**源码位置**: `rec_engine.cpp:690-695` (PrefillOnly), `rec_engine.cpp:836-839` (XAttention)

---

### 3.7 step(batches) -- 核心差异

这是各 Pipeline 差异最大的接口。

#### LlmRecEnginePipeline::step()

```
输入: vector<Batch>& batches (size = dp_size_)
│
├─ 循环 max_steps 步 (动态，由 batch 中 max_tokens 决定):
│   ├─ prepare_inputs(batches)
│   │   ├─ 按 dp_rank 拆分 batch
│   │   ├─ 每个 dp_rank 独立 prepare_forward_input(args, threadpool)
│   │   └─ 同步 dp_global_token_nums / dp_is_decode / batch_forward_type
│   │
│   ├─ 对所有 worker_clients_ 异步发送:
│   │   └─ worker_clients_[rank]->step_remote_async(forward_inputs[dp_rank])
│   │
│   ├─ folly::collectAll 收集所有 worker 结果 (RawForwardOutput)
│   │
│   └─ 处理输出 (每个 dp_rank 独立):
│       ├─ src_seq_idxes 为空 → process_sample_output(raw_output)
│       └─ src_seq_idxes 非空 → process_beam_search_output(raw_output)
│           └─ seq->kv_state().process_beam_search()  // KV block 转移
│       └─ refresh_sequences_from_groups()
│
└─ batch.finish()
返回: {} (空 ForwardOutput)
```

**特点**:
- 多步循环由 **Engine 层驱动**
- 通过远程 Worker 执行（SHM/brpc）
- 支持 DP + TP 混合并行
- 动态步数控制（`get_max_steps_from_batch`）
- 接收 `RawForwardOutput`（序列化后的传输格式）
- Beam search 输出处理在 Engine 层

**源码位置**: `rec_engine.cpp:404-500`

---

#### OneRecPrefillOnlyEnginePipeline::step()

```
输入: vector<Batch>& batches (size = 1)
│
├─ workers_[0]->prepare_inputs(batches[0])
│
├─ get_model_output(forward_inputs)  ← prefill (first token)
│   ├─ 所有 workers_ 异步 step_async
│   ├─ 取 results.front() (rank 0)
│   └─ D2H: embeddings/next_tokens/logprobs/top_tokens/top_logprobs → CPU
│
├─ process_sample_output(prefill_output.sample_output)
│
├─ 循环 kRecDecodeSteps 次:
│   ├─ workers_[0]->prepare_inputs(batches[0])
│   ├─ get_model_output(forward_inputs)  ← decode
│   └─ process_sample_output(decode_output.sample_output,
│          force_requested_beam_result_size = (i+1 == kRecDecodeSteps))
│
└─ batch.finish()
返回: decode_output (最后一次 decode 的 ForwardOutput)
```

**特点**:
- 1 次 prefill + N 次 decode，**多步循环由 Engine 层驱动**
- 本地 Worker 执行
- 无 beam search
- 使用 `process_sample_output` 处理输出
- 返回最后一次 decode 的 `ForwardOutput`

**源码位置**: `rec_engine.cpp:697-756`

---

#### OneRecXAttentionEnginePipeline::step()

```
输入: vector<Batch>& batches (size = 1)
│
├─ workers_[0]->prepare_inputs(batches[0])
│
├─ get_model_output(forward_inputs)  ← 单次推理
│   ├─ 所有 workers_ 异步 step_async
│   ├─ 取 results.front() (rank 0)
│   ├─ 有 beam 输出时:
│   │   └─ D2H: beam_sequence_group / beam_search_output.out_logprobs → CPU
│   └─ 无 beam 输出时:
│       └─ D2H: embeddings/next_tokens/logprobs/top_tokens/top_logprobs → CPU
│
├─ 处理输出:
│   ├─ beam_sequence_group 非空 → process_beam_sequence_group(output)
│   └─ 否则 → process_sample_output(output.sample_output)
│
└─ batch.finish()
返回: output
```

**特点**:
- **单次推理**，多轮 decode 在 device/worker 内部完成
- 本地 Worker 执行
- 支持 beam search 输出（`beam_sequence_group`）
- 内置 debug trace 环境变量
- 始终预分配 `kMinimalOneRecMetadataKVBlocks` 个 KV blocks

**源码位置**: `rec_engine.cpp:841-871`

---

#### RecMultiRoundEnginePipeline::step()

```
输入: vector<Batch>& batches (size = 1)
│
├─ workers_[0]->prepare_inputs(batches[0])
│
├─ get_model_output(forward_inputs)  ← 单次推理
│   ├─ 所有 workers_ 异步 step_async
│   ├─ 取 results.front() (rank 0)
│   └─ D2H: beam_sequence_group / beam_search_output.out_logprobs → CPU
│
├─ process_beam_sequence_group(output)  ← 始终使用 beam 处理
│
└─ batch.finish()
返回: output
```

**特点**:
- **单次推理**，多轮 decode 在 worker/device 内部完成
- 本地 Worker 执行
- **始终**使用 `process_beam_sequence_group`（无条件）
- 与 XAttention 类似但更简单（无条件分支，无 trace）

**源码位置**: `rec_engine.cpp:1130-1159`

---

### 3.8 get_model_output() -- D2H 传输差异

各本地 Pipeline 的 `get_model_output()` 实现差异：

| 实例 | D2H 传输内容 | 条件 |
|------|-------------|------|
| **OneRecPrefillOnly** | `embeddings`, `next_tokens`, `logprobs`, `top_tokens`, `top_logprobs` | 始终传输 |
| **OneRecXAttention** | 有 beam: `beam_sequence_group`, `out_logprobs`；无 beam: `embeddings`, `next_tokens`, `logprobs`, `top_tokens`, `top_logprobs` | 按 `has_beam_output` 分支 |
| **RecMultiRound** | `beam_sequence_group`, `out_logprobs` | 始终传输 |

**源码位置**: `rec_engine.cpp:758-809` (PrefillOnly), `rec_engine.cpp:873-981` (XAttention), `rec_engine.cpp:1161-1193` (MultiRound)

---

### 3.9 num_workers()

| 实例 | 实现 |
|------|------|
| **LlmRecEnginePipeline** | `dp_size_ > 1` 时返回 `dp_local_tp_size_`；否则返回 `worker_clients_.size()` |
| **OneRecLocalEnginePipeline** | 返回 `workers_.size()` |
| **RecMultiRoundEnginePipeline** | 返回 `workers_.size()` |

**源码位置**: `rec_engine.cpp:351-356` (LlmRec), `rec_engine.cpp:678-680` (OneRecLocal), `rec_engine.cpp:1126-1128` (MultiRound)

---

## 4. 总对比表

| 维度 | LlmRec | OneRecPrefillOnly | OneRecXAttention | RecMultiRound |
|------|--------|-------------------|------------------|---------------|
| **适用模型** | qwen2/qwen3 | onerec (传统) | onerec (xattention) | qwen2/qwen3 (多轮) |
| **Worker 类型** | 远程 (DistManager) | 本地 | 本地 | 本地 |
| **传输方式** | SHM / brpc | 进程内 | 进程内 | 进程内 |
| **返回类型** | RawForwardOutput | ForwardOutput | ForwardOutput | ForwardOutput |
| **输入准备** | prepare_forward_input (DP 拆分) | prepare_inputs (worker) | prepare_inputs (worker) | prepare_inputs (worker) |
| **推理步数** | 动态 (max_tokens) | 1 prefill + N decode | 1 次 | 1 次 |
| **多轮 decode 位置** | Engine 层循环 | Engine 层循环 | Device 内部 | Worker/Device 内部 |
| **采样输出处理** | process_sample_output | process_sample_output | process_sample_output (无 beam 时) | - |
| **Beam 输出处理** | process_beam_search_output | - | process_beam_sequence_group | process_beam_sequence_group |
| **DP 支持** | 是 | 否 | 否 | 否 |
| **KV blocks 预分配** | 0 | 0 或 2 | 2 | 0 |
| **单卡 PG 创建** | 不需要 | 轻量 PG (无后端) | 轻量 PG (无后端) | 真实 PG (rank_size=1) |
| **D2H 传输** | 远程已完成 | 全部字段 | 按 beam 分支 | beam 字段 |
| **debug trace** | 无 | 无 | XLLM_DEBUG_ONEREC_ENGINE_TRACE | 无 |

## 5. 工厂选择逻辑

```cpp
// rec_engine.cpp:1215-1232
RecEngine::create_pipeline(RecPipelineType type, RecEngine& engine) {
    switch (type) {
        case kLlmRecDefault:              → LlmRecEnginePipeline
        case kLlmRecMultiRoundPipeline:   → RecMultiRoundEnginePipeline
        case kOneRecDefault:              → OneRecPrefillOnlyEnginePipeline
        case kOneRecXAttentionPipeline:   → OneRecXAttentionEnginePipeline
    }
}
```

`RecPipelineType` 由 `get_rec_pipeline_type(RecModelKind)` 决定：

```cpp
// rec_model_utils.h:76-91
RecModelKind::kLlmRec:
    max_decode_rounds > 0 → kLlmRecMultiRoundPipeline
    否则                  → kLlmRecDefault

RecModelKind::kOneRec:
    max_decode_rounds > 0 → kOneRecXAttentionPipeline
    否则                  → kOneRecDefault
```

## 6. 对 GE 执行器接入的启示

如果要新增一个 GE Pipeline，它最接近 **RecMultiRoundEnginePipeline** 的模式：

| 维度 | RecMultiRound | GE Pipeline (预期) |
|------|--------------|-------------------|
| 推理次数 | 1 次（多轮在 device 内） | 1 次（多轮 + beam search 在图内） |
| 输出格式 | `beam_sequence_group` | `beam_sequence_group`（需对齐） |
| 输出处理 | `process_beam_sequence_group` | 可复用 |
| Worker 类型 | 本地 | 本地 |
| 差异点 | Worker 内部有多轮循环 | Worker 内部只有 forward，beam 在图内 |

如果 GE 图的输出能对齐到 `ForwardOutput.beam_sequence_group` 格式（`[batch, result_width, total_rounds]` + `out_logprobs`），EnginePipeline 层可以**直接复用 RecMultiRoundEnginePipeline**，只需新增 WorkerPipeline 子类。
