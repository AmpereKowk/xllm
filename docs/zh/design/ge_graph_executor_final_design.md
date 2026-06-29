# GeGraphExecutor 最终设计方案

## 1. 架构概述

### 1.1 设计目标

为 xLLM 框架新增 `GeGraphExecutorImpl`，使用华为 Ascend GE V2 接口执行 epair 模型，实现：
- **继承设计**：EpModel 继承 CausalLM，符合模型抽象
- **简洁架构**：Model 与 Graph 直接绑定，无需单例管理
- **可扩展性**：支持多种模型类型（LLM/VLM/Rec）
- **性能优化**：避免 H2D 拷贝，复用 Worker Stream

### 1.2 核心架构

```
执行链路: Worker -> Engine -> GraphExecutor -> EpModel.forward() -> ModelLoader.RunGraphAsyncWithStream

┌─────────────────────────────────────────────────────────────┐
│                 ExecutorImpl 接口（固定）                     │
│  run(tokens, positions, kv_caches, params)                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               GeGraphExecutorImpl                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  run(): 实现 ExecutorImpl 接口                       │   │
│  │    └─> model_->forward(tokens, positions, ...)      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      EpModel : CausalLM                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  成员变量: EpairModelLoader loader_                  │   │
│  │                                                      │   │
│  │  load_model(loader):                                 │   │
│  │    ├─> GEInitializeV2()（进程级别，首次调用时初始化） │   │
│  │    ├─> 加载 epair 模型                               │   │
│  │    └─> CompileAndLoad()                              │   │
│  │                                                      │   │
│  │  forward(tokens, positions, kv_caches, params):      │   │
│  │    ├─> 从 params.input_tensor_map 构造图输入         │   │
│  │    ├─> RunGraphAsyncWithStream()                     │   │
│  │    └─> 返回 ModelOutput                              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**使用流程**：
```
1. 创建 EpModel（继承 CausalLM）
2. 调用 load_model() 加载 epair 模型（内部完成 GE 初始化）
3. 将 EpModel 传入 GeGraphExecutorImpl
4. Executor 调用 model_->forward() 执行推理
```

**设计要点**：
- EpModel 自己管理 EpairModelLoader，Model 与 Graph 绑定
- GE 初始化在 load_model() 中完成（首次调用时初始化，使用静态变量保证进程级别）
- 不需要单例管理图，每个 EpModel 管理自己的图资源

## 2. 核心设计决策

### 2.1 EpModel 管理 epair 模型

**设计目标**：
- 参考 `RecCausalLM` 的设计，让 `EpModel` 继承 `CausalLM`
- 使用 GE 图执行方式，而非传统的 PyTorch 模型执行
- `EpairModelLoader` 作为 `EpModel` 的成员变量管理 epair 模型
- 在 `load_model()` 中完成模型加载和编译
- 通过 `forward()` 函数执行模型推理

**架构设计**：
```cpp
// EpCausalLM 基类
class EpCausalLM : public CausalLM {
public:
    ~EpCausalLM() override = default;
};

// EpModel 实现
class EpModel : public EpCausalLM {
public:
    EpModel(const torch::TensorOptions& options);
    
    // 实现 CausalLM 接口
    ModelOutput forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& params) override;
    
    void load_model(std::unique_ptr<ModelLoader> loader) override;
    torch::Device device() const override;
    const torch::TensorOptions& options() const override;
    
private:
    td::EpairModelLoader loader_;  // 管理 epair 模型加载
    torch::TensorOptions options_;
    uint64_t device_id_;
};
```

**优点**：
- ✅ 继承设计：EpModel 继承 CausalLM，可无缝接入 Executor
- ✅ 封装性好：模型加载、编译、执行统一管理
- ✅ 职责清晰：EpModel 负责模型执行，Executor 负责调度
- ✅ 易于扩展：可支持多种模型类型（参考 RecCausalLM）

### 2.2 ModelInputParams 扩展设计

**问题**：
- 不同模型的 Graph 输入节点命名和顺序可能不同
- 需要一种灵活的方式将输入 Tensor 映射到 Graph 输入节点

**解决方案**：在 `ModelInputParams` 中新增 `input_tensor_map`
- 使用 `std::unordered_map<std::string, gert::Tensor>` 表示图输入名字与输入 Tensor 的映射关系
- 调用方负责填充这个映射表
- `EpModel::forward()` 直接使用这个映射表构造图输入

**实现**：
```cpp
struct ModelInputParams {
    // 原有字段...
    torch::Tensor tokens;
    torch::Tensor positions;
    // ...
    
    // 新增：图输入名字与输入 Tensor 的映射关系
    // key: Graph 输入节点名称（如 "input_ids", "position_ids", "past_key_values[0].key"）
    // value: 对应的 gert::Tensor
    std::unordered_map<std::string, gert::Tensor> input_tensor_map;
};
```

**调用方职责**：
```cpp
// Worker/Engine 层负责填充 input_tensor_map
ModelInputParams params;
params.tokens = tokens;
params.positions = positions;

// 根据模型类型填充图输入映射
params.input_tensor_map["input_ids"] = ConvertToGeTensor(tokens);
params.input_tensor_map["position_ids"] = ConvertToGeTensor(positions);
for (int i = 0; i < num_layers; ++i) {
    params.input_tensor_map[FormatKey("past_key_values[%d].key", i)] = ConvertToGeTensor(k_caches[i]);
    params.input_tensor_map[FormatKey("past_key_values[%d].value", i)] = ConvertToGeTensor(v_caches[i]);
}

// 调用 EpModel
output = ep_model->forward(params);
```

**优点**：
- ✅ 解耦：Executor 不关心具体的图输入节点命名
- ✅ 灵活：支持任意 Graph 输入结构
- ✅ 可扩展：新模型只需调整填充逻辑

### 2.3 KVCache 原地更新机制

**核心理解**：
- EpModel **不需要处理 KVCache 更新逻辑**
- 只需要把 KVCache tensor 作为**输入**传给 Graph（通过 `input_tensor_map`）
- Graph 内部算子（如 `reshape_paged_cache`）会**自动修改**这些 tensor 的内容
- 类似于传引用机制：`gert::Tensor` 只引用 device memory，Graph 内部直接修改

**实现机制**：
```cpp
// 调用方（Worker/Engine）负责填充 input_tensor_map
void PrepareKVCacheInputs(ModelInputParams& params, std::vector<KVCache>& kv_caches) {
    for (size_t layer = 0; layer < kv_caches.size(); ++layer) {
        torch::Tensor k_cache = kv_caches[layer].get_k_cache();
        torch::Tensor v_cache = kv_caches[layer].get_v_cache();
        
        // TorchToDeviceTensor：只引用 device memory，不拷贝
        gert::Tensor ge_k_cache, ge_v_cache;
        ConvertTorchToGeTensor(k_cache, ge_k_cache);
        ConvertTorchToGeTensor(v_cache, ge_v_cache);
        
        // 填充到 input_tensor_map
        params.input_tensor_map[FormatKey("past_key_values[%zu].key", layer)] = ge_k_cache;
        params.input_tensor_map[FormatKey("past_key_values[%zu].value", layer)] = ge_v_cache;
    }
}

// EpModel 执行时直接使用
ModelOutput EpModel::forward(const ModelInputParams& params) {
    std::vector<gert::Tensor> graph_inputs;
    for (const auto& [name, tensor] : params.input_tensor_map) {
        graph_inputs.push_back(tensor);
    }
    // ... 执行 Graph
}
```

**关键点**：
- **调用方职责**：填充 `input_tensor_map`，将 KVCache tensor 转换为 gert::Tensor
- **EpModel 职责**：只传入 tensor，不处理更新逻辑
- **Graph 职责**：内部算子自动修改 tensor 内容
- **内存管理**：KVCache tensor 由调用方管理，EpModel 不持有所有权
- **性能优化**：无内存拷贝，直接引用 device memory

### 2.4 多卡场景架构

**设计决策**：
- **GEInitializeV2**：进程级别，在 EpModel::load_model() 中首次调用时初始化（使用静态变量）
- **EpModel**：每个 device 独立的 EpModel 实例，由调用方创建
- **EpairModelLoader**：通过 `session_device_id` 指定真正的 deviceId

**实现**：
```cpp
void EpModel::load_model(std::unique_ptr<ModelLoader> loader) {
    // 1. GE 初始化（进程级别，只初始化一次）
    static bool ge_initialized = false;
    static std::mutex ge_mutex;
    
    if (!ge_initialized) {
        std::lock_guard<std::mutex> lock(ge_mutex);
        if (!ge_initialized) {
            if (ge::GEInitializeV2() != ge::SUCCESS) {
                LOG(ERROR) << "Failed to initialize GE";
                return;
            }
            ge_initialized = true;
        }
    }
    
    // 2. 加载 epair 文件
    // 3. CompileAndLoad(device_id_, nullptr)
}
```

**架构图**：
```
进程级别（静态变量）
  │
  └─> GEInitializeV2()  // 首次调用时初始化

每个 Worker/Engine
  │
  ├─> 创建 EpModel(device_id=0)
  │    └─> load_model() -> loader_.CompileAndLoad(session_device_id=0)
  │
  ├─> 创建 EpModel(device_id=1)
  │    └─> load_model() -> loader_.CompileAndLoad(session_device_id=1)
  │
  └─> 创建 EpModel(device_id=N)
       └─> load_model() -> loader_.CompileAndLoad(session_device_id=N)

执行链路
Worker -> Engine -> GraphExecutor
                    └─> EpModel.forward()
                         └─> loader_.RunGraphAsyncWithStream()
```

### 2.5 Stream 管理

**设计决策**：
- **复用 Worker Stream**：通过 `c10_npu::getCurrentNPUStream(device_id)` 获取
- **Init 阶段**：传 `nullptr`（编译阶段与运行时流无关）
- **forward 阶段**：传入真正的 stream（执行阶段）

**流程**：
```cpp
// Worker 执行前设置 stream
c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();

// EpModel::Init() - 编译阶段
Status EpModel::Init(const std::string& epair_path, uint64_t device_id) {
    device_id_ = device_id;
    // 加载 epair 文件
    loader_.Load(epair_path);
    // 编译（stream 传 nullptr）
    loader_.CompileAndLoad(device_id, nullptr);
    return Status::OK();
}

// EpModel::forward() - 执行阶段
ModelOutput EpModel::forward(const ModelInputParams& params) {
    // 获取当前 thread 的 stream
    aclrtStream stream = c10_npu::getCurrentNPUStream(device_id_).stream();
    
    // 执行 Graph
    loader_.RunGraphAsyncWithStream(stream, inputs, outputs);
}
```

### 2.6 Tensor 内存管理

**输入 Tensor**：
- 由调用方（Worker/Engine）管理
- 填充到 `input_tensor_map` 中
- 直接使用 torch tensor 的 device memory（`data_ptr()`）
- 无 H2D 拷贝
- torch tensor 自动管理内存，EpModel 不持有所有权

**输出 Tensor**：
- 预分配 device memory（`aclrtMalloc`）
- Graph 执行后写入
- D2H 拷贝后手动释放（`aclrtFree`）

**实现要点**：
```cpp
// 调用方填充 input_tensor_map
void PrepareInputs(ModelInputParams& params) {
    void* dev_ptr = torch_tensor.data_ptr();  // 已经在 device
    gert::Tensor ge_tensor;
    ge_tensor.SetData(gert::TensorData(dev_ptr, nullptr, bytes, gert::kOnDeviceHbm));
    params.input_tensor_map["input_ids"] = ge_tensor;
    // 注意：不要 aclrtFree(dev_ptr)，torch tensor 会自动管理
}

// EpModel 执行
ModelOutput EpModel::forward(const ModelInputParams& params) {
    // 输入：直接使用 params.input_tensor_map 中的 tensor
    
    // 输出：预分配 device memory
    aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    ge_tensor.SetData(gert::TensorData(dev, nullptr, bytes, gert::kOnDeviceHbm));
    
    // 执行后需要手动释放输出内存
    aclrtFree(dev);
}
```

## 3. 类设计

### 3.1 EpModel（继承 CausalLM）

**设计思路**：
- 参考 `RecCausalLM` 的设计，让 `EpModel` 继承 `CausalLM`
- 使用 GE 图执行方式，而非传统的 PyTorch 模型执行
- `EpairModelLoader` 作为成员变量管理 epair 模型

**职责**：
- 继承 `CausalLM` 接口，作为模型类使用
- 管理 epair 模型的生命周期
- 实现 `forward()` 接口，通过 GE 图执行推理
- 实现 `load_model()` 接口，加载和编译 epair 模型

**接口**：
```cpp
// EpCausalLM 基类（类似 RecCausalLM）
class EpCausalLM : public CausalLM {
public:
    ~EpCausalLM() override = default;
};

// EpModel 实现
class EpModel : public EpCausalLM {
public:
    EpModel(const torch::TensorOptions& options);
    ~EpModel() override;
    
    // 实现 CausalLM 接口
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
    
    // 是否已初始化
    bool IsInitialized() const { return initialized_; }
    
private:
    // 从 input_tensor_map 构造图输入
    std::vector<gert::Tensor> BuildGraphInputs(const ModelInputParams& params);
    
    // 准备输出 tensors（预分配）
    bool PrepareDeviceOutputs(std::vector<gert::Tensor>& device_outputs,
                              std::vector<DevMem>& output_mems);
    
    // 清理 tensors
    void CleanupTensors(std::vector<gert::Tensor>& tensors,
                        std::vector<DevMem>& mems);
    
    // 辅助方法：torch tensor -> gert::Tensor（引用 device memory）
    static bool ConvertTorchToGeTensor(const torch::Tensor& torch_tensor,
                                       gert::Tensor& ge_tensor);
    
    // 辅助方法：gert::Tensor -> torch tensor（拷贝到 host）
    static bool ConvertGeToTorchTensor(gert::Tensor& ge_tensor,
                                       torch::Tensor& torch_tensor);
    
private:
    td::EpairModelLoader loader_;  // epair 模型加载器
    torch::TensorOptions options_;
    uint64_t device_id_ = 0;
    bool initialized_ = false;
    
    // 模型信息（从 epair 中获取）
    std::vector<std::string> input_names_;   // 图输入节点名称列表
    std::vector<std::string> output_names_;  // 图输出节点名称列表
    size_t output_count_ = 1;                 // 输出 tensor 数量
};
```

**核心实现**：
```cpp
// 构造函数
EpModel::EpModel(const torch::TensorOptions& options)
    : options_(options) {
    device_id_ = options.device().index();
}

// load_model()（实现 CausalLM 接口）
void EpModel::load_model(std::unique_ptr<ModelLoader> loader) {
    // 1. GE 初始化（进程级别，只初始化一次）
    static bool ge_initialized = false;
    static std::mutex ge_mutex;
    
    if (!ge_initialized) {
        std::lock_guard<std::mutex> lock(ge_mutex);
        if (!ge_initialized) {
            if (ge::GEInitializeV2() != ge::SUCCESS) {
                LOG(ERROR) << "Failed to initialize GE";
                return;
            }
            ge_initialized = true;
        }
    }
    
    // 2. 从 loader 中获取 epair 路径
    std::string epair_path = loader->GetEpairPath();
    
    // 3. 加载 epair 文件
    if (loader_.Load(epair_path) != td::SUCCESS) {
        LOG(ERROR) << "Failed to load epair: " << epair_path;
        return;
    }
    
    // 4. 编译和加载（编译阶段 stream 传 nullptr）
    if (loader_.CompileAndLoad(device_id_, nullptr) != td::SUCCESS) {
        LOG(ERROR) << "Failed to compile and load model";
        return;
    }
    
    // 5. 获取图输入/输出节点信息
    input_names_ = loader_.GetInputNames();
    output_names_ = loader_.GetOutputNames();
    output_count_ = output_names_.size();
    
    initialized_ = true;
}

// forward()（实现 CausalLM 接口）
ModelOutput EpModel::forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& params) {
    if (!initialized_) {
        LOG(ERROR) << "EpModel not initialized";
        return ModelOutput();
    }
    
    // 1. 从 input_tensor_map 构造图输入
    std::vector<gert::Tensor> graph_inputs = BuildGraphInputs(params);
    if (graph_inputs.empty()) {
        LOG(ERROR) << "Failed to build graph inputs";
        return ModelOutput();
    }
    
    // 2. 获取 stream
    aclrtStream stream = c10_npu::getCurrentNPUStream(device_id_).stream();
    
    // 3. 准备输出 tensors（预分配）
    std::vector<gert::Tensor> device_outputs;
    std::vector<DevMem> output_mems;
    if (!PrepareDeviceOutputs(device_outputs, output_mems)) {
        LOG(ERROR) << "Failed to prepare device outputs";
        return ModelOutput();
    }
    
    // 4. 执行 Graph
    if (loader_.RunGraphAsyncWithStream(stream, graph_inputs, device_outputs) 
        != td::SUCCESS) {
        LOG(ERROR) << "RunGraphAsyncWithStream failed";
        CleanupTensors(device_outputs, output_mems);
        return ModelOutput();
    }
    
    // 5. 同步
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        LOG(ERROR) << "aclrtSynchronizeStream failed";
        CleanupTensors(device_outputs, output_mems);
        return ModelOutput();
    }
    
    // 6. 转换输出
    ModelOutput result;
    if (!device_outputs.empty()) {
        ConvertGeToTorchTensor(device_outputs[0], result.logits);
    }
    
    // 7. 清理输出 tensors
    CleanupTensors(device_outputs, output_mems);
    
    return result;
}

// BuildGraphInputs()
std::vector<gert::Tensor> EpModel::BuildGraphInputs(const ModelInputParams& params) {
    std::vector<gert::Tensor> graph_inputs;
    
    // 按照 input_names_ 的顺序构造图输入
    for (const auto& name : input_names_) {
        auto it = params.input_tensor_map.find(name);
        if (it == params.input_tensor_map.end()) {
            LOG(ERROR) << "Missing input tensor: " << name;
            return {};
        }
        graph_inputs.push_back(it->second);
    }
    
    return graph_inputs;
}
```

### 3.2 ModelInputParams 扩展

**职责**：
- 存储模型输入参数
- 新增 `input_tensor_map` 字段，表示图输入名字与输入 Tensor 的映射关系

**接口**：
```cpp
struct ModelInputParams {
    // 原有字段
    torch::Tensor tokens;
    torch::Tensor positions;
    // ... 其他字段
    
    // 新增：图输入名字与输入 Tensor 的映射关系
    // key: Graph 输入节点名称
    // value: 对应的 gert::Tensor
    std::unordered_map<std::string, gert::Tensor> input_tensor_map;
};
```

### 3.3 GeGraphExecutorImpl

**设计思路**：
- EpModel 继承 CausalLM，可以作为模型类传入 Executor
- Executor 持有 CausalLM 指针（实际是 EpModel）
- 直接调用 `model_->forward()` 执行推理

**职责**：
- 实现 `ExecutorImpl::run()` 接口
- 持有 `CausalLM`（实际是 `EpModel`）实例
- 调用 `model_->forward()` 执行推理

**接口**：
```cpp
class GeGraphExecutorImpl : public ExecutorImpl {
public:
    GeGraphExecutorImpl(CausalLM* model,
                       const ModelArgs& args,
                       const torch::Device& device,
                       const runtime::Options& options);
    
    ~GeGraphExecutorImpl() override;
    
    ForwardInput prepare_inputs(Batch& batch) override;
    
    ModelOutput run(const torch::Tensor& tokens,
                   const torch::Tensor& positions,
                   std::vector<KVCache>& kv_caches,
                   const ModelInputParams& params) override;
    
private:
    CausalLM* model_;  // 持有 CausalLM 指针（实际是 EpModel）
    std::string model_key_;
    uint64_t device_id_;
    bool initialized_;
    
    ModelArgs args_;
    torch::Device device_;
    runtime::Options options_;
};

REGISTER_EXECUTOR("ge", GeGraphExecutorImpl);
```

**核心实现**：
```cpp
// 构造函数
GeGraphExecutorImpl::GeGraphExecutorImpl(CausalLM* model,
                                         const ModelArgs& args,
                                         const torch::Device& device,
                                         const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options), initialized_(false) {
    
    // 1. model 必须是 EpModel 类型
    if (model == nullptr) {
        LOG(ERROR) << "GeGraphExecutorImpl requires model to be EpModel";
        return;
    }
    
    // 2. 验证 model 是否是 EpModel
    EpModel* ep_model = dynamic_cast<EpModel*>(model);
    if (ep_model == nullptr) {
        LOG(ERROR) << "GeGraphExecutorImpl requires model to be EpModel type";
        return;
    }
    
    // 3. 检查 EpModel 是否已初始化
    if (!ep_model->IsInitialized()) {
        LOG(ERROR) << "EpModel not initialized";
        return;
    }
    
    // 4. device_id 来自 torch::Device
    device_id_ = device.index();
    
    initialized_ = true;
}

// run()（委托给 CausalLM）
ModelOutput GeGraphExecutorImpl::run(const torch::Tensor& tokens,
                                     const torch::Tensor& positions,
                                     std::vector<KVCache>& kv_caches,
                                     const ModelInputParams& params) {
    if (!initialized_) {
        LOG(ERROR) << "GeGraphExecutorImpl not initialized";
        return ModelOutput();
    }
    
    // 委托给 CausalLM 执行
    return model_->forward(tokens, positions, kv_caches, params);
}
```

**使用流程**：
```cpp
// 1. 创建 EpModel
auto options = torch::TensorOptions().device(torch::kNPU, device_id);
std::unique_ptr<CausalLM> model = std::make_unique<EpModel>(options);

// 2. 加载模型
std::unique_ptr<ModelLoader> loader = CreateModelLoader(epair_path, config);
model->load_model(std::move(loader));

// 3. 创建 Executor
auto executor = std::make_unique<GeGraphExecutorImpl>(
    model.get(), args, device, options);

// 4. 执行推理
ModelOutput output = executor->run(tokens, positions, kv_caches, params);
```

## 4. 文件结构

```
xllm/core/framework/model/
├── ep_causal_lm.h                   # EpCausalLM 基类定义
├── ep_model.h                       # EpModel 类定义
├── ep_model.cpp                     # EpModel 实现
└── causal_lm.h                      # CausalLM 基类（已存在）

xllm/core/runtime/
├── ge_graph_executor_impl.h        # GeGraphExecutorImpl 定义
├── ge_graph_executor_impl.cpp      # GeGraphExecutorImpl 实现
└── executor_impl_factory.h         # 添加 REGISTER_EXECUTOR("ge", GeGraphExecutorImpl)
```

## 5. 依赖项

### 5.1 外部库
- `ge` (Graph Engine V2): 华为 Ascend GE V2 接口
- `acl` (Ascend Computing Language): Ascend 运行时库
- `gert::Tensor`: GE Runtime Tensor 接口
- `td::EpairModelLoader`: epair 文件加载器（来自 `torch_delegate` 库）
- `torch_npu`: PyTorch NPU 后端

### 5.2 头文件
```cpp
#include <ge/ge_api_v2.h>           // GE V2 接口
#include <acl/acl.h>                // ACL 接口
#include <torch_npu/csrc/core/npu/NPUStream.h>  // NPU Stream
#include "torch_delegate/epair_model_loader.h"  // epair 加载器
```

### 5.3 CMake 链接
```cmake
target_link_libraries(xllm
    ge                          # GE V2 库
    acl                         # ACL 库
    torch_delegate              # epair 加载器库
    torch_npu                   # PyTorch NPU 后端
)
```

## 6. 实现计划

### 6.1 Phase 1：基础实现
1. 实现 `EpCausalLM` 基类和 `EpModel` 类
   - 继承 CausalLM 接口
   - load_model() 方法（包含 GE 初始化、加载和编译 epair 模型）
   - forward() 方法（执行推理）
   - BuildGraphInputs() 方法（从 input_tensor_map 构造图输入）
   - Tensor 内存管理

2. 扩展 `ModelInputParams` 结构体
   - 新增 `input_tensor_map` 字段
   - 定义图输入名字与输入 Tensor 的映射关系

3. 实现 `GeGraphExecutorImpl`
   - 构造函数（接收 CausalLM 指针）
   - run() 方法（委托给 CausalLM）

4. 单元测试
   - EpModel 测试（GE 初始化、load_model、forward）
   - GeGraphExecutorImpl 测试

### 6.2 Phase 2：调用方实现
1. 在 Worker/Engine 层实现输入准备逻辑
   - 将 tokens、positions 转换为 gert::Tensor
   - 将 KVCache 转换为 gert::Tensor
   - 填充 `input_tensor_map`

2. 确认 epair 文件的 Graph 输入节点命名规范
   - 确定 KV Cache tensor 的命名规则
   - 支持量化场景的 scale tensor

### 6.3 Phase 3：扩展支持
1. 支持多种模型类型（VLM、Rec 等）
2. 集成测试
3. 性能优化

## 7. 待确认事项

### 7.1 Graph 输入节点命名规范（高优先级）

**问题**：epair 文件的 Graph 输入节点命名和顺序是什么？

**可能的结构**：
```
方案 A（标准命名）：
  input_ids
  position_ids
  past_key_values[0].key
  past_key_values[0].value
  past_key_values[1].key
  past_key_values[1].value
  ...
  past_key_values[N-1].key
  past_key_values[N-1].value

方案 B（合并命名）：
  input_ids
  position_ids
  past_key_values[0]  // [layers, batch, heads, cache_len, head_dim]
```

**影响**：
- 调用方填充 `input_tensor_map` 的键名
- `EpModel::BuildGraphInputs()` 的查找逻辑

**解决方案**：
- 提供 epair 文件的节点列表文档
- 或者在运行时从 epair 查询节点信息（`loader_.GetInputNames()`）

### 7.2 Graph 输出 Shape 推断

**问题**：输出 tensor 的 shape 如何确定？

**可能方案**：
- 固定 shape（根据模型配置）
- 运行时查询 epair 文件
- 动态 shape（第一次推理后记录）

### 7.3 错误处理策略

**决策**：宽松策略（LOG ERROR + 返回空 ModelOutput）

**理由**：
- GeGraphExecutor 用于在线服务
- 错误不中断推理，适合容错场景
- 调用方检查返回结果是否为空

## 8. 测试计划

### 8.1 单元测试
- `EpModel` 测试：GE 初始化、load_model、forward、BuildGraphInputs
- `GeGraphExecutorImpl` 测试：构造函数、run()
- `ModelInputParams` 测试：input_tensor_map 填充

### 8.2 集成测试
- 端到端推理测试
- 多卡场景测试
- 多线程并发测试

### 8.3 性能测试
- H2D 拷贝优化验证
- Stream 复用验证
- 多卡负载均衡测试

## 9. 总结

**核心优势**：
- ✅ **继承设计**：EpModel 继承 CausalLM，符合模型抽象，可无缝接入 Executor
- ✅ **简洁架构**：Model 与 Graph 直接绑定，无需单例管理图
- ✅ **职责清晰**：EpModel 管理自己的 epair 模型，Worker/Engine 负责输入准备
- ✅ **解耦设计**：通过 `input_tensor_map` 解耦图输入结构
- ✅ **性能优化**：避免 H2D 拷贝，复用 Worker Stream
- ✅ **易于扩展**：支持多种模型类型（参考 RecCausalLM 设计）

**执行链路**：
```
Worker -> Engine -> GraphExecutor -> EpModel.forward() -> ModelLoader.RunGraphAsyncWithStream
```

**实现优先级**：
1. EpCausalLM 基类 + EpModel（继承 CausalLM，包含 GE 初始化）
2. ModelInputParams 扩展（输入映射）
3. GeGraphExecutorImpl（Executor 实现）
4. 单元测试 + 集成测试

**下一步**：
- 确认 epair 文件的节点命名规范
- 开始实现 EpModel（继承 CausalLM）

## 10. Pipeline 设计方案

### 10.1 设计动机

#### 10.1.1 现有 Pipeline 的问题

当前 `RecEnginePipeline` 和 `RecWorkPipeline` 的各实例（LlmRec、OneRec、OneRecXAttention、RecMultiRound）承担了大量本应属于 Model 层的计算逻辑：

| 职责 | 所在层 | 具体操作 |
|------|--------|---------|
| 多轮 decode 循环 | EnginePipeline / WorkPipeline | `for i in [0, kRecDecodeSteps)` 循环驱动多步推理 |
| Beam Search | WorkPipeline | `beam_searcher_->forward()`、`process_beam_search_output()` |
| Sampling | WorkPipeline | `sampler_->forward()`、`rec_sampler_->forward()` |
| 约束解码 | WorkPipeline | `prepare_filter_mask_async()`、`RecSampler` with `filter_mask` |
| KV Cache 轮次管理 | WorkPipeline | 每轮修改 `input_params`、`token_ids`、`positions`、`attn_metadata` |
| 输出后处理 | EnginePipeline | `process_sample_output()`、`process_beam_search_output()`、`process_beam_sequence_group()` |

这些操作在单算子执行模式下是合理的——Host 需要逐步驱动每个计算步骤。但在 **torch_delegate 图模式**下，GE 图已经将整个模型的计算（包括多轮 decode、beam search、sampling）封装为一次 `RunGraphAsyncWithStream()` 调用。Pipeline 层再继续承担这些职责会导致：

1. **职责重叠**：图内部已完成 beam search + sampling，Pipeline 再做一遍是冗余
2. **接口不匹配**：图的输入/输出是 Tensor，不需要 `SamplingParameters`、`filter_mask` 等 Host 侧结构
3. **维护成本**：每新增一种图模式都需要适配复杂的 Pipeline 逻辑

#### 10.1.2 设计目标

为 torch_delegate 图模式新增专用的 `GeGraphEnginePipeline` 和 `GeGraphWorkerPipeline`，实现：

- **职责单一**：Pipeline 只负责 Batch ↔ Tensor 的转换，不承载模型计算逻辑
- **单次推理**：一次 `step()` 调用 = 一次 `executor->forward()` 调用，所有多轮逻辑在图内完成
- **最小依赖**：不需要 Sampler、BeamSearcher、filter_mask 等组件
- **输出对齐**：输出格式对齐现有 `ForwardOutput.beam_sequence_group`，复用 `Batch::process_beam_sequence_group()`

### 10.2 核心架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RecEngine                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  GeGraphEnginePipeline : RecEnginePipeline                    │  │
│  │                                                                │  │
│  │  step(batches):                                               │  │
│  │    1. workers_[0]->prepare_inputs(batches[0])                 │  │
│  │    2. get_model_output(forward_inputs)                        │  │
│  │       └─ 所有 workers 异步 step_async                         │  │
│  │       └─ 取 rank 0 输出                                       │  │
│  │       └─ D2H: beam_sequence_group / out_logprobs → CPU        │  │
│  │    3. batches[0].process_beam_sequence_group(output)          │  │
│  │    4. batches[0].finish()                                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RecWorkerImpl                                   │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  GeGraphWorkerPipeline : RecWorkPipeline                      │  │
│  │                                                                │  │
│  │  prepare_inputs(batch):                                       │  │
│  │    └─ batch.prepare_forward_input() → ForwardInput            │  │
│  │                                                                │  │
│  │  prepare_work_before_execute(inputs, processed_inputs):       │  │
│  │    ├─ H2D 传输 (复用 Base 逻辑)                               │  │
│  │    ├─ KV block swap                                           │  │
│  │    ├─ Schema 驱动填充 input_tensor_map (标准输入)             │  │
│  │    └─ Model override 填充 input_tensor_map (Custom 输入)      │  │
│  │                                                                │  │
│  │  step(input):                                                 │  │
│  │    ├─ executor->forward(tokens, positions, kv_caches, params) │  │
│  │    │   └─ GeGraphExecutorImpl::run()                          │  │
│  │    │       └─ EpModel::forward()                              │  │
│  │    │           └─ RunGraphAsyncWithStream()                   │  │
│  │    └─ 构造 ForwardOutput (beam_sequence_group 格式)           │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

**执行链路**：
```
RecEngine::step()
  └─> GeGraphEnginePipeline::step()
        ├─> prepare_inputs()
        ├─> GeGraphWorkerPipeline::step()
        │     ├─> prepare_work_before_execute()
        │     ├─> GeGraphExecutorImpl::run()
        │     │     └─> EpModel::forward()
        │     │           └─> RunGraphAsyncWithStream()
        │     └─> 构造 ForwardOutput
        ├─> process_beam_sequence_group()
        └─> batch.finish()
```

### 10.3 GeGraphMasterPipeline 设计

#### 10.3.1 四层 Pipeline 全景

Rec 推理链路共有四层 Pipeline 抽象，GE 图模式需要在每一层都提供对应实现：

```
Layer 1: RecMasterPipeline    → 请求构造 (API 输入 → Request)
Layer 2: RecEnginePipeline    → 引擎编排 (Batch → ForwardInput → ForwardOutput)
Layer 3: RecWorkPipeline      → Worker 计算 (ForwardInput → executor→forward → ForwardOutput)
Layer 4: SchedulerPipeline     → 调度批处理 (Sequence → Batch)
```

各层现有实例与 GE 图模式的映射关系：

```
RecPipelineType              → MasterPipeline                    → EnginePipeline                     → WorkPipeline                      → SchedulerPipeline
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
kLlmRecDefault               → LlmRecMasterPipeline              → LlmRecEnginePipeline               → LlmRecWorkPipeline                → LlmRecSchedulerPipeline
kLlmRecWithMmData            → LlmRecWithMmDataMasterPipeline    → (复用 LlmRec)                      → LlmRecWithMmDataWorkPipeline      → (复用 LlmRec)
kLlmRecMultiRoundPipeline    → LlmRecMasterPipeline (复用)       → RecMultiRoundEnginePipeline        → LlmRecMultiRoundPipeline          → RecMultiRoundSchedulerPipeline
kOneRecDefault               → OneRecPrefillOnlyMasterPipeline   → OneRecPrefillOnlyEnginePipeline    → OneRecWorkPipeline                → OneRecSchedulerPipeline
kOneRecXAttentionPipeline    → OneRecXAttentionMasterPipeline    → OneRecXAttentionEnginePipeline     → OneRecXAttentionWorkPipeline      → OneRecXAttentionSchedulerPipeline
kGeGraphPipeline (新增)      → GeGraphMasterPipeline (新增)      → GeGraphEnginePipeline (新增)       → GeGraphWorkerPipeline (新增)      → GeGraphSchedulerPipeline (新增)
```

#### 10.3.2 RecMasterPipeline 的职责

`RecMasterPipeline` 是请求构造层的策略抽象，负责将外部 API 输入（prompt、token_ids、input_tensors、MMData）转换为内部 `Request` 对象。

**核心方法**：
```cpp
class RecMasterPipeline {
public:
    // prompt 输入方式
    virtual std::shared_ptr<Request> generate_request(
        std::string prompt,
        std::optional<std::vector<int>> prompt_tokens,
        std::optional<std::vector<proto::InferInputTensor>> input_tensors,
        const RequestParams& sp,
        OutputCallback callback);

    // raw 输入方式（token_ids + MMData）
    virtual std::shared_ptr<Request> generate_request(
        const std::vector<int>& prompt_tokens,
        std::optional<MMData> mm_data,
        const RequestParams& sp,
        OutputCallback callback);
};
```

**与 Engine/Worker Pipeline 的本质区别**：
- Master Pipeline 不参与模型执行，只负责请求构造
- 不涉及 `step()`、`prepare_inputs()`、`prepare_work_before_execute()` 等执行方法
- 输出是 `std::shared_ptr<Request>`，而非 `ForwardOutput`

#### 10.3.3 GE 图模式下的核心差异

GE 图内部自行完成 **sampling + beam search + 多轮 decode + 停止判断**，这导致 Master 层的 Request 构造与现有 Pipeline 存在以下关键差异：

| 维度 | 现有 Pipeline | GE 图模式 | 影响 |
|------|-------------|----------|------|
| **StoppingChecker** | `build_stop_checker=true`（LlmRec/XAttention）或 `false`（PrefillOnly），Host 逐步检查 | **必须 `false`**——图内部控制停止，Host 无法逐步检查 | `build_request_common()` 参数 |
| **停止条件传递** | 由 Host 侧 `StoppingChecker::check()` 每步执行 | 需要把 `max_tokens`、`eos_token_id`、`stop_token_ids` **作为图输入 Tensor** 传入 GE 图 | Request 需要携带这些元数据到图 |
| **Sampling 参数** | 聚合成 per-token 张量（`temperatures`、`top_p`、`top_k` 等 batch 张量） | 图内处理采样，需要的是**标量配置**而非 per-token 张量 | `SamplingParameters::init()` 的聚合逻辑对 GE 图是冗余的 |
| **beam_width / num_return_sequences** | Host 侧驱动 beam 扩展、`SequencesGroup::process_beam_search()` | 图内完成 beam search，Host **不应** 做 beam 扩展 | `Sequence` 生命周期管理 |
| **Sequence 生命周期** | `append_token()` → `finished()` → `process_beam_search()` 循环 | 图一次性返回完整结果，Sequence **不走逐步循环** | Scheduler/Engine 层的 Sequence 管理 |
| **max_tokens** | 用于 `StoppingChecker` + `seq_capacity` 计算 | 同时作为图的 **最大 decode 步数** 输入 | 需要额外传递到图 |

#### 10.3.4 StoppingChecker 的悖论

```
现有模式:
  Request → Sequence → 每步 append_token → StoppingChecker::check() → 决定是否终止

GE 图模式:
  Request → Sequence → 一次性 forward → 图内部循环 decode + 停止 → 返回完整结果
                                            ↑
                                   图怎么知道什么时候停？
                                   → 需要 max_tokens / eos_token_id / stop_token_ids 作为图输入
```

`StoppingChecker` 在 GE 图模式下有双重角色：
1. **不构建** Host 侧的 `StoppingChecker` 对象（`build_stop_checker=false`）
2. **但要保留** 停止条件的原始数据（`max_tokens`、`eos_token_id`、`stop_token_ids`），让它们能流到图输入

**当前问题**：`build_request_common()` 中，`max_tokens`、`eos_token_id`、`stop_token_ids` 在构建完 `StoppingChecker` 后就丢弃了。GE 图模式需要在 `RequestState` 中保留它们。

#### 10.3.5 数据流差异

```
现有模式:
  RequestParams
    ├─ max_tokens ──→ StoppingChecker (Host 逐步检查)
    ├─ beam_width ──→ RequestSamplingParam ──→ SamplingParameters (per-token 张量)
    ├─ temperature ─→ RequestSamplingParam ──→ SamplingParameters (per-token 张量)
    └─ stop_token_ids → StoppingChecker (Host 逐步检查)

GE 图模式:
  RequestParams
    ├─ max_tokens ──→ RequestState 保留 ──→ input_tensor_map["max_decode_steps"]
    ├─ beam_width ──→ RequestState 保留 ──→ input_tensor_map["beam_width"]
    ├─ temperature ─→ RequestState 保留 ──→ input_tensor_map["temperature"]
    ├─ eos_token_id → RequestState 保留 ──→ input_tensor_map["eos_token_id"]
    └─ stop_token_ids → RequestState 保留 ──→ input_tensor_map["stop_token_ids"]
```

#### 10.3.6 GeGraphMasterPipeline 类定义

```cpp
class GeGraphMasterPipeline final : public RecMasterPipeline {
public:
    explicit GeGraphMasterPipeline(RecMaster& master)
        : RecMasterPipeline(master) {}
    ~GeGraphMasterPipeline() override = default;

    std::shared_ptr<Request> generate_request(
        std::string prompt,
        std::optional<std::vector<int>> prompt_tokens,
        std::optional<std::vector<proto::InferInputTensor>> input_tensors,
        const RequestParams& sp,
        OutputCallback callback) override;
};
```

#### 10.3.7 generate_request() 实现

```cpp
std::shared_ptr<Request> GeGraphMasterPipeline::generate_request(
    std::string prompt,
    std::optional<std::vector<int>> prompt_tokens,
    std::optional<std::vector<proto::InferInputTensor>> input_tensors,
    const RequestParams& sp,
    OutputCallback callback) {

    // 1. tokenize (复用 LlmRec 逻辑)
    std::vector<int> local_prompt_tokens;
    if (prompt_tokens.has_value()) {
        local_prompt_tokens = prompt_tokens.value();
    } else {
        local_prompt_tokens = master_.tokenizer_->encode(prompt);
    }

    if (local_prompt_tokens.empty()) {
        LOG(ERROR) << "Empty prompt tokens";
        return nullptr;
    }

    // 2. 构建 Request (关键差异: build_stop_checker=false)
    return master_.build_request_common(
        prompt,
        local_prompt_tokens,
        /*mm_data=*/std::nullopt,
        sp,
        callback,
        /*build_stop_checker=*/false);  // ← 关键：不构建 Host 侧 StoppingChecker
}
```

**设计要点**：
- `build_stop_checker=false`：GE 图内部控制停止，Host 不需要逐步检查
- 复用 `build_request_common()` 的 `RequestState` 构建逻辑
- 停止条件和采样配置的原始数据需要保留在 `RequestState` 中（见 10.3.8）

#### 10.3.8 RequestState 扩展

GE 图模式需要在 `RequestState` 中保留停止条件和采样配置的原始数据，以便后续流转到图输入：

```cpp
struct RequestState {
    // ... 已有字段 ...

    // 新增：GE 图模式需要的原始配置数据
    // 这些数据当前仅在构建 StoppingChecker 时使用，之后被丢弃
    // GE 图模式需要将它们保留并传递到图输入
    uint32_t max_tokens = 0;                    // 最大生成 token 数 → 图输入 max_decode_steps
    int32_t eos_token_id = -1;                  // EOS token ID → 图输入 eos_token_id
    std::unordered_set<int32_t> stop_token_ids; // 停止 token 集合 → 图输入 stop_token_ids
    std::vector<std::vector<int32_t>> stop_sequences; // 停止序列 → 图输入 stop_sequences

    // 采样配置标量（当前仅在 SamplingParameters::init() 中聚合为张量）
    // GE 图模式需要标量形式作为图输入
    float temperature = 0.0;                    // → 图输入 temperature
    float top_p = 1.0;                          // → 图输入 top_p
    int64_t top_k = -1;                         // → 图输入 top_k
    float repetition_penalty = 1.0;             // → 图输入 repetition_penalty
};
```

**`build_request_common()` 修改**：

```cpp
// 在 build_request_common() 中，构建 RequestState 时保留原始数据
RequestState state;
// ... 已有字段赋值 ...

// 新增：保留 GE 图模式需要的原始数据
state.max_tokens = max_tokens;
state.eos_token_id = eos_token_id;
state.stop_token_ids = stop_tokens;
state.stop_sequences = stop_sequences;
state.temperature = sp.temperature;
state.top_p = sp.top_p;
state.top_k = sp.top_k;
state.repetition_penalty = sp.repetition_penalty;
```

#### 10.3.9 GeGraphSchedulerPipeline

GE 图模式还需要对应的 SchedulerPipeline，负责 Batch 构造和 KV Cache 分配：

```cpp
class GeGraphSchedulerPipeline final : public SchedulerPipeline {
public:
    explicit GeGraphSchedulerPipeline() = default;

    std::vector<Batch> create_batches(FixedStepsScheduler& scheduler,
                                       BatchFactory* batch_factory) override;

    bool requires_kv_cache() const override { return true; }

    bool allocate_kv_cache(KVCacheManager* kv_cache_manager,
                            Sequence* sequence) override;
};
```

**与现有 SchedulerPipeline 的差异**：
- `requires_kv_cache()` 返回 `true`（GE 图需要 KV Cache）
- `create_batches()` 逻辑与 `LlmRecSchedulerPipeline` 类似
- `allocate_kv_cache()` 需要为 GE 图的特殊 KV Cache 布局分配（如果有的话）

#### 10.3.10 工厂映射扩展

**RecMaster::create_pipeline**：
```cpp
std::unique_ptr<RecMasterPipeline> RecMaster::create_pipeline(
    RecPipelineType type, RecMaster& master) {
    switch (type) {
        case RecPipelineType::kLlmRecDefault:
        case RecPipelineType::kLlmRecMultiRoundPipeline:
            return std::make_unique<LlmRecMasterPipeline>(master);
        case RecPipelineType::kLlmRecWithMmData:
            return std::make_unique<LlmRecWithMmDataMasterPipeline>(master);
        case RecPipelineType::kOneRecDefault:
            return std::make_unique<OneRecPrefillOnlyMasterPipeline>(master);
        case RecPipelineType::kOneRecXAttentionPipeline:
            return std::make_unique<OneRecXAttentionMasterPipeline>(master);
        case RecPipelineType::kGeGraphPipeline:          // 新增
            return std::make_unique<GeGraphMasterPipeline>(master);
        default:
            LOG(FATAL) << "Unknown pipeline type";
            return nullptr;
    }
}
```

**FixedStepsScheduler::create_pipeline**：
```cpp
std::unique_ptr<SchedulerPipeline> FixedStepsScheduler::create_pipeline(
    RecPipelineType type) {
    switch (type) {
        case RecPipelineType::kLlmRecDefault:
            return std::make_unique<LlmRecSchedulerPipeline>();
        case RecPipelineType::kOneRecDefault:
            return std::make_unique<OneRecSchedulerPipeline>();
        case RecPipelineType::kOneRecXAttentionPipeline:
            return std::make_unique<OneRecXAttentionSchedulerPipeline>();
        case RecPipelineType::kLlmRecMultiRoundPipeline:
            return std::make_unique<RecMultiRoundSchedulerPipeline>();
        case RecPipelineType::kGeGraphPipeline:          // 新增
            return std::make_unique<GeGraphSchedulerPipeline>();
        default:
            LOG(FATAL) << "Unknown pipeline type";
            return nullptr;
    }
}
```

#### 10.3.11 对 Sequence 生命周期的影响

GE 图模式下，Sequence 的生命周期与现有模式有本质区别：

**现有模式**：
```
Sequence 创建
  → Scheduler 加入 Batch
  → Engine step() → Worker forward() → 返回 1 个 token
  → Sequence::append_token()
  → StoppingChecker::check() → 未完成则继续
  → 循环直到完成
```

**GE 图模式**：
```
Sequence 创建
  → Scheduler 加入 Batch
  → Engine step() → Worker forward() → 图内完成全部 decode
  → 返回完整结果 (beam_sequence_group)
  → Batch::process_beam_sequence_group() 一次性写入所有 token
  → Sequence 直接标记为 FINISHED
```

**关键影响**：
- Sequence 不经历逐步的 `append_token()` → `finished()` 循环
- `StoppingChecker` 不参与 Host 侧检查
- `SequencesGroup::process_beam_search()` 不执行（图内已完成）
- Scheduler 不需要在多个 step 之间维护 Sequence 状态

### 10.4 GeGraphEnginePipeline 设计

#### 10.4.1 职责

`GeGraphEnginePipeline` 是 `RecEnginePipeline` 的子类，负责 Engine 层的推理编排。其职责仅限于：

1. 初始化本地 Worker（复用 `OneRecLocalEnginePipeline` 的 Worker 管理逻辑）
2. 准备 Batch 输入
3. 触发 Worker 执行推理
4. 处理输出并回写 Batch

**不负责的职责**（与现有 Pipeline 的关键区别）：
- 不驱动多轮 decode 循环（图内完成）
- 不执行 beam search 后处理（图内完成，输出已是最终结果）
- 不执行 sampling（图内完成）

#### 10.4.2 类定义

```cpp
class GeGraphEnginePipeline final : public RecEnginePipeline {
public:
    explicit GeGraphEnginePipeline(RecEngine& engine);
    ~GeGraphEnginePipeline() override = default;

    void setup_workers() override;
    void process_group_test() override;
    bool init_model_workers(const std::string& model_path) override;
    int64_t estimate_min_available_memory() override;
    bool allocate_kv_cache(const KVCacheShape& kv_cache_shape) override;
    int64_t minimal_kv_cache_blocks() const override { return 0; }
    
    ForwardOutput step(std::vector<Batch>& batches) override;
    
    std::vector<int64_t> get_active_activation_memory() const override;
    size_t num_workers() const override;

private:
    ForwardInput prepare_inputs(std::vector<Batch>& batches);
    ForwardOutput get_model_output(const ForwardInput& forward_inputs);

private:
    RecEngine& engine_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::unique_ptr<ProcessGroup> process_group_;
};
```

#### 10.4.3 step() 流程

```
GeGraphEnginePipeline::step(batches)
│
├─ 1. prepare_inputs(batches)
│     └─ workers_[0]->prepare_inputs(batches[0])
│          └─ WorkerImpl::prepare_inputs(batch)
│               └─ model_executor_->prepare_inputs(batch)
│                    └─ batch.prepare_forward_input(args)
│
├─ 2. get_model_output(forward_inputs)
│     ├─ 所有 workers_ 异步 step_async(model_inputs)
│     ├─ folly::collectAll(futures).get()
│     ├─ 取 results.front() (rank 0 的输出)
│     └─ D2H: beam_sequence_group → CPU
│             beam_search_output.out_logprobs → CPU
│             Device::synchronize_default_stream()
│
├─ 3. batches[0].process_beam_sequence_group(output)
│
└─ 4. batches[0].finish()

返回: output
```

**与 RecMultiRoundEnginePipeline 的对比**：

| 步骤 | RecMultiRound | GeGraph |
|------|--------------|---------|
| 输入准备 | `workers_[0]->prepare_inputs()` | 相同 |
| 推理调用 | `get_model_output()` → Worker 内部多轮 | `get_model_output()` → Worker 内部单次 forward |
| 输出处理 | `process_beam_sequence_group()` | 相同 |
| finish | `batch.finish()` | 相同 |
| Worker 内部 | 多轮 decode 循环 + sampling + beam search | 单次 `executor->forward()`，图内完成一切 |

#### 10.4.4 Worker 管理

Worker 初始化逻辑复用 `OneRecLocalEnginePipeline` / `RecMultiRoundEnginePipeline` 的本地 Worker 模式：

```cpp
void GeGraphEnginePipeline::setup_workers() {
    // 空操作（本地 Worker 在 init_model_workers 中创建）
}

bool GeGraphEnginePipeline::init_model_workers(const std::string& model_path) {
    // 1. 创建 ProcessGroup
    //    单卡 NPU: create_process_group(rank=0, world_size=1, rank_size=1)
    //    多卡 NPU: create_npu_process_groups()
    //    非 NPU:   create_local_process_groups()
    
    // 2. 创建 Worker (WorkerType::REC)
    for (int rank = 0; rank < world_size; ++rank) {
        workers_.push_back(std::make_unique<Worker>(...));
    }
    
    // 3. 异步初始化模型
    std::vector<folly::SemiFuture<bool>> futures;
    for (auto& worker : workers_) {
        futures.push_back(worker->init_model_async(model_path));
    }
    auto results = folly::collectAll(futures).get();
    return std::all_of(results.begin(), results.end(), 
                       [](auto& r) { return r.value(); });
}
```

### 10.5 GeGraphWorkerPipeline 设计

#### 10.5.1 职责

`GeGraphWorkerPipeline` 是 `RecWorkPipeline` 的子类，负责 Worker 层的推理执行。其职责仅限于：

1. 将 `ForwardInput` 转换为 GE 图所需的 `input_tensor_map`
2. 调用 `executor->forward()` 执行一次推理
3. 将 `ModelOutput` 转换为 `ForwardOutput`（对齐 `beam_sequence_group` 格式）

**不负责的职责**（与现有 WorkPipeline 的关键区别）：
- 不需要 Sampler（图内完成采样）
- 不需要 BeamSearcher（图内完成 beam search）
- 不需要 filter_mask / RecSampler（图内完成约束解码）
- 不需要多轮循环（图内完成多轮 decode）
- 不需要每轮修改 `input_params` / `token_ids` / `positions`

#### 10.5.2 类定义

```cpp
class GeGraphWorkerPipeline final : public RecWorkPipeline {
public:
    explicit GeGraphWorkerPipeline(RecPipelineRuntime& runtime);
    ~GeGraphWorkerPipeline() override = default;

    ForwardInput prepare_inputs(Batch& batch) override;
    
    void prepare_work_before_execute(const ForwardInput& inputs,
                                      ForwardInput& processed_inputs) override;
    
    std::optional<ForwardOutput> step(const ForwardInput& input) override;
};
```

#### 10.5.3 prepare_inputs()

复用基类 `RecWorkPipeline::prepare_inputs()` 逻辑，将 Batch 转换为 `ForwardInput`：

```cpp
ForwardInput GeGraphWorkerPipeline::prepare_inputs(Batch& batch) {
    return RecWorkPipeline::prepare_inputs(batch);
}
```

#### 10.5.4 prepare_work_before_execute()

在 Base 逻辑（H2D + KV block swap）基础上，通过 **Schema 驱动** 填充 `input_tensor_map`（详见 10.12 节）：

```cpp
void GeGraphWorkerPipeline::prepare_work_before_execute(
    const ForwardInput& inputs, ForwardInput& processed_inputs) {
    
    // 1. 复用 Base 逻辑：H2D 传输 + KV block swap
    RecWorkPipeline::prepare_work_before_execute(inputs, processed_inputs);
    
    // 2. 从 Model 获取 Schema（通过 Executor 暴露）
    auto* ge_executor = dynamic_cast<GeGraphExecutorImpl*>(
        runtime().executor->impl());
    const auto& schema = ge_executor->GetGraphInputSchema();
    
    // 3. 通用 Builder 按 Schema 填充 input_tensor_map
    GeGraphInputBuilder::BuildInputTensorMap(
        processed_inputs.input_params,
        schema,
        processed_inputs.token_ids,
        processed_inputs.positions,
        runtime().worker.kv_caches_);
    
    // 4. 模型特有输入补充（kCustom 类输入，由 Model 子类 override）
    auto* ep_model = dynamic_cast<EpModel*>(ge_executor->model());
    if (ep_model) {
        ep_model->PrepareCustomInputs(processed_inputs.input_params, inputs);
    }
}
```

**设计要点**：
- Pipeline 不硬编码任何模型特有的输入名字
- Model 通过 `GetGraphInputSchema()` 声明"我要什么"
- 通用 `GeGraphInputBuilder` 按 Schema 从标准运行时数据中取数据
- 仅 `kCustom` 类输入需要 Model 子类 override `PrepareCustomInputs()`

#### 10.5.5 step()

单次 forward，构造 `ForwardOutput`：

```cpp
std::optional<ForwardOutput> GeGraphWorkerPipeline::step(const ForwardInput& input) {
    auto& mutable_input = const_cast<ForwardInput&>(input);
    
    // 1. 单次 forward 调用（图内完成所有计算）
    auto model_output = runtime().executor->forward(
        mutable_input.token_ids,
        mutable_input.positions,
        runtime().worker.kv_caches_,
        mutable_input.input_params);
    
    // 2. 构造 ForwardOutput
    ForwardOutput output;
    
    // 从 ModelOutput 中提取 beam_sequence_group 格式的输出
    // GE 图的输出已经是 beam search 的最终结果
    if (model_output.beam_sequence_group.defined()) {
        output.beam_sequence_group = model_output.beam_sequence_group;
    }
    if (model_output.beam_search_output.out_logprobs.defined()) {
        output.beam_search_output = model_output.beam_search_output;
    }
    
    // 3. 设置 sampling 标志（从输入参数中透传）
    output.do_sample = mutable_input.sampling_params.do_sample;
    output.logprobs = mutable_input.sampling_params.logprobs;
    output.max_top_logprobs = mutable_input.sampling_params.max_top_logprobs;
    
    return output;
}
```

#### 10.5.6 ForwardInput 字段消费矩阵

| 字段 | GeGraph 使用方式 | 对比 Base RecWorkPipeline |
|------|-----------------|--------------------------|
| `token_ids` | READ → Schema `kTokenIds` → `input_tensor_map` | 相同（READ → executor） |
| `positions` | READ → Schema `kPositions` → `input_tensor_map` | 相同（READ → executor） |
| `input_params` | READ → Schema 驱动填充 `input_tensor_map` + executor | 相同（READ → executor） |
| `input_params.multimodal` | READ（仅 kCustom 类，由 Model override） | **忽略** |
| `sampling_params` | READ（仅透传标志字段到 output） | READ（用于 sampler） |
| `sampling_params.selected_token_idxes` | **忽略**（图内处理） | READ（用于 logits/sampler） |
| `sampling_params.use_beam_search` | **忽略**（图内处理） | READ（用于 beam_search kernel） |
| `sampling_params.acc_logprob` | **忽略**（图内处理） | READ（用于 beam_searcher） |
| `decoder_sampling_params` | **忽略** | 忽略 |
| `step_decode` / `step_meta()` | **忽略**（多轮在图内） | 忽略 |
| `transfer_kv_infos` | **忽略** | READ |
| `onerec_params` / `llmrec_params` | **忽略** | 各 Pipeline 按需 READ |

#### 10.5.7 与现有 WorkPipeline 的对比

| 维度 | Base RecWorkPipeline | OneRecXAttention | LlmRecMultiRound | **GeGraph** |
|------|---------------------|------------------|------------------|-------------|
| executor->forward() 次数 | 1 | 每轮 1~2，N 轮 | 每轮 1，N 轮 | **1** |
| Sampler | `sampler_` | `rec_sampler_` + filter_mask | `rec_sampler_` | **无** |
| Beam Search | `beam_searcher_` | 图内 | 图内 | **图内** |
| 多轮循环 | 无 | Worker 内 N 轮 | Worker 内 N 轮 | **无（图内）** |
| input_params 修改 | 无 | 拷贝+改 flag | 每轮原地修改 | **无** |
| 输出格式 | `sample_output` | `beam_sequence_group` | `beam_sequence_group` | **`beam_sequence_group`** |

### 10.6 数据流图

```
┌─────────────────────────────────────────────────────────────────────┐
│                GeGraphEnginePipeline::step()                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────── 输入组装 ──────────────┐                           │
│  │                                       │                           │
│  │  Batch (sequences)                    │                           │
│  │    │                                  │                           │
│  │    ├─ workers_[0]->prepare_inputs()   │                           │
│  │    │   └─ batch.prepare_forward_input │                           │
│  │    │       ├─ tokens, positions       │                           │
│  │    │       ├─ kv_cache_info           │                           │
│  │    │       └─ sampling_params         │                           │
│  │    │                                  │                           │
│  │    └─ → ForwardInput                 │                           │
│  │                                       │                           │
│  └───────────────────────────────────────┘                           │
│                       │                                               │
│                       ▼                                               │
│  ┌────────────── Worker 执行 ───────────┐                           │
│  │                                       │                           │
│  │  GeGraphWorkerPipeline::step()        │                           │
│  │    │                                  │                           │
│  │    ├─ prepare_work_before_execute()   │                           │
│  │    │   ├─ H2D 传输                    │                           │
│  │    │   ├─ KV block swap              │                           │
│  │    │   ├─ Schema 驱动填充标准输入     │                           │
│  │    │   └─ Model override 填充 Custom  │                           │
│  │    │                                  │                           │
│  │    ├─ executor->forward()             │                           │
│  │    │   └─ GeGraphExecutorImpl::run()  │                           │
│  │    │       └─ EpModel::forward()      │                           │
│  │    │           └─ RunGraphAsyncWithStream()                       │
│  │    │               │                  │                           │
│  │    │               │  ┌──────────────────────────────────┐       │
│  │    │               │  │  GE Graph (epair)                │       │
│  │    │               │  │  ┌────────────────────────┐      │       │
│  │    │               │  │  │ Model Forward          │      │       │
│  │    │               │  │  │ (Transformer Layers)   │      │       │
│  │    │               │  │  └───────────┬────────────┘      │       │
│  │    │               │  │              ▼                    │       │
│  │    │               │  │  ┌────────────────────────┐      │       │
│  │    │               │  │  │ Sampling / TopK / TopP │      │       │
│  │    │               │  │  └───────────┬────────────┘      │       │
│  │    │               │  │              ▼                    │       │
│  │    │               │  │  ┌────────────────────────┐      │       │
│  │    │               │  │  │ Beam Search            │      │       │
│  │    │               │  │  └───────────┬────────────┘      │       │
│  │    │               │  │              ▼                    │       │
│  │    │               │  │  ┌────────────────────────┐      │       │
│  │    │               │  │  │ Multi-round Decode     │      │       │
│  │    │               │  │  │ (loop in graph)        │      │       │
│  │    │               │  │  └───────────┬────────────┘      │       │
│  │    │               │  └──────────────┼───────────────────┘       │
│  │    │               │                  │                           │
│  │    │               ▼                  ▼                           │
│  │    │           ModelOutput                                        │
│  │    │           (beam_sequence_group + out_logprobs)               │
│  │    │                                  │                           │
│  │    └─ → ForwardOutput                │                           │
│  │                                       │                           │
│  └───────────────────────────────────────┘                           │
│                       │                                               │
│                       ▼                                               │
│  ┌────────────── 输出处理 ──────────────┐                           │
│  │                                       │                           │
│  │  D2H: beam_sequence_group → CPU       │                           │
│  │       out_logprobs → CPU              │                           │
│  │                                       │                           │
│  │  process_beam_sequence_group(output)  │                           │
│  │    ├─ 解析 [groups, beam_width, rounds]                           │
│  │    │   token 矩阵                      │                           │
│  │    └─ seq->set_beam_result()          │                           │
│  │                                       │                           │
│  │  batch.finish()                       │                           │
│  │                                       │                           │
│  └───────────────────────────────────────┘                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.7 RecPipelineType 扩展

#### 10.7.1 新增枚举值

```cpp
enum class RecPipelineType : uint8_t {
    kLlmRecDefault = 0,
    kLlmRecWithMmData = 1,
    kOneRecDefault = 2,
    kLlmRecMultiRoundPipeline = 3,
    kOneRecXAttentionPipeline = 4,
    kGeGraphPipeline = 5,  // 新增：torch_delegate GE 图模式
};
```

#### 10.7.2 工厂方法扩展

**RecEngine::create_pipeline**：
```cpp
std::unique_ptr<RecEnginePipeline> RecEngine::create_pipeline(
    RecPipelineType type, RecEngine& engine) {
    switch (type) {
        case RecPipelineType::kLlmRecDefault:
            return std::make_unique<LlmRecEnginePipeline>(engine);
        case RecPipelineType::kLlmRecMultiRoundPipeline:
            return std::make_unique<RecMultiRoundEnginePipeline>(engine);
        case RecPipelineType::kOneRecDefault:
            return std::make_unique<OneRecPrefillOnlyEnginePipeline>(engine);
        case RecPipelineType::kOneRecXAttentionPipeline:
            return std::make_unique<OneRecXAttentionEnginePipeline>(engine);
        case RecPipelineType::kGeGraphPipeline:          // 新增
            return std::make_unique<GeGraphEnginePipeline>(engine);
        default:
            LOG(FATAL) << "Unknown pipeline type: " << static_cast<int>(type);
            return nullptr;
    }
}
```

**RecWorkerImpl::create_pipeline**：
```cpp
std::unique_ptr<RecWorkPipeline> RecWorkerImpl::create_pipeline(
    RecPipelineType type, RecPipelineRuntime& runtime) {
    switch (type) {
        case RecPipelineType::kLlmRecDefault:
            return std::make_unique<LlmRecWorkPipeline>(runtime);
        case RecPipelineType::kOneRecDefault:
            return std::make_unique<OneRecWorkPipeline>(runtime);
        case RecPipelineType::kLlmRecMultiRoundPipeline:
            return std::make_unique<LlmRecMultiRoundPipeline>(runtime);
        case RecPipelineType::kOneRecXAttentionPipeline:
            return std::make_unique<OneRecXAttentionWorkPipeline>(runtime);
        case RecPipelineType::kGeGraphPipeline:          // 新增
            return std::make_unique<GeGraphWorkerPipeline>(runtime);
        default:
            LOG(FATAL) << "Unknown pipeline type: " << static_cast<int>(type);
            return nullptr;
    }
}
```

#### 10.7.3 Pipeline 类型选择逻辑

```cpp
RecPipelineType get_rec_pipeline_type(RecModelKind kind, 
                                       const ModelArgs& args,
                                       bool use_ge_graph) {
    if (use_ge_graph) {
        return RecPipelineType::kGeGraphPipeline;
    }
    
    // 原有逻辑...
    switch (kind) {
        case RecModelKind::kLlmRec:
            return RecConfig::max_decode_rounds() > 0 
                ? RecPipelineType::kLlmRecMultiRoundPipeline
                : RecPipelineType::kLlmRecDefault;
        case RecModelKind::kOneRec:
            return RecConfig::max_decode_rounds() > 0
                ? RecPipelineType::kOneRecXAttentionPipeline
                : RecPipelineType::kOneRecDefault;
        // ...
    }
}
```

### 10.8 ModelOutput 扩展

#### 10.8.1 问题

当前 `ModelOutput` 主要承载 `logits`（hidden states），由 WorkPipeline 中的 Sampler 和 BeamSearcher 处理后生成 `ForwardOutput`。但在 GE 图模式下，图输出已经是最终的 beam search 结果，`ModelOutput` 需要能够承载这些结果。

#### 10.8.2 扩展方案

在 `ModelOutput` 中新增 GE 图输出字段：

```cpp
struct ModelOutput {
    // 原有字段
    torch::Tensor logits;              // hidden states / logits
    torch::Tensor embedding;           // embedding output
    
    // 新增：GE 图模式的直接输出
    torch::Tensor beam_sequence_group; // [groups, beam_width, rounds] token 矩阵
    BeamSearchOutput beam_search_output; // beam search 元数据 (out_logprobs 等)
};
```

#### 10.8.3 EpModel::forward() 输出处理

```cpp
ModelOutput EpModel::forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& params) {
    // ... 执行 Graph ...
    
    ModelOutput result;
    
    // 根据图输出节点名称解析结果
    // 假设 output_names_ = ["beam_sequence_group", "out_logprobs"]
    for (size_t i = 0; i < device_outputs.size(); ++i) {
        const auto& name = output_names_[i];
        torch::Tensor torch_output;
        ConvertGeToTorchTensor(device_outputs[i], torch_output);
        
        if (name == "beam_sequence_group") {
            result.beam_sequence_group = torch_output;
        } else if (name == "out_logprobs") {
            result.beam_search_output.out_logprobs = torch_output;
        } else if (name == "logits" || name == "hidden_states") {
            result.logits = torch_output;
        }
    }
    
    return result;
}
```

### 10.9 与现有方案的对比总结

#### 10.9.1 四层 Pipeline 职责对比

**Master 层（请求构造）**：

| 职责 | LlmRec | OneRecPrefill | OneRecXAttention | **GeGraph** |
|------|--------|--------------|------------------|-------------|
| 输入方式 | prompt + tokenizer | prompt + input_tensors | prompt + input_tensors | **prompt + tokenizer** |
| build_stop_checker | true | false | true | **false** |
| MMData 处理 | 无 | sparse_embedding 等 | sparse_embedding 等 | **无** |
| 停止条件保留 | 仅 StoppingChecker | 不保留 | 仅 StoppingChecker | **RequestState 保留原始数据** |
| 采样配置保留 | 仅 SamplingParam | 仅 SamplingParam | 仅 SamplingParam | **RequestState 保留标量** |

**Engine 层（推理编排）**：

| 职责 | LlmRec | OneRecPrefill | OneRecXAttention | MultiRound | **GeGraph** |
|------|--------|--------------|------------------|------------|-------------|
| Worker 初始化 | DistManager 远程 | 本地 PG + Worker | 本地 PG + Worker | 本地 PG + Worker | **本地 PG + Worker** |
| 输入准备 | DP 拆分 + prepare | prepare_inputs | prepare_inputs | prepare_inputs | **prepare_inputs** |
| forward 次数 | 动态 N 次 | 1+N 次 | N 轮 | N 轮 | **1 次** |
| 输出处理 | process_sample/beam | process_sample | process_beam_group | process_beam group | **process_beam_sequence_group** |

**Worker 层（设备计算）**：

| 职责 | LlmRec | OneRecPrefill | OneRecXAttention | MultiRound | **GeGraph** |
|------|--------|--------------|------------------|------------|-------------|
| Sampling | Worker 层 | Worker 层 | Worker 层 | Worker 层 | **图内** |
| Beam Search | Worker 层 | 无 | Worker 层 | Worker 层 | **图内** |
| 多轮 decode | 无 | Engine 层循环 | Worker 内循环 | Worker 内循环 | **图内** |
| Sampler 组件 | 需要 | 需要 | 需要 | 需要 | **不需要** |
| BeamSearcher 组件 | 需要 | 不需要 | 不需要 | 不需要 | **不需要** |

**Scheduler 层（调度批处理）**：

| 职责 | LlmRec | OneRec | OneRecXAttention | MultiRound | **GeGraph** |
|------|--------|--------|------------------|------------|-------------|
| requires_kv_cache | true | false | true | false | **true** |
| Sequence 生命周期 | 逐步循环 | 逐步循环 | 一次性完成 | 一次性完成 | **一次性完成** |

#### 10.9.2 代码复杂度对比

| 指标 | RecMultiRound WorkPipeline | **GeGraph WorkPipeline** |
|------|---------------------------|--------------------------|
| step() 行数 | ~140 行 | **~20 行** |
| prepare_work_before_execute() 行数 | ~130 行 | **~30 行** |
| 依赖组件 | Executor, RecSampler, KVCache manager | **Executor** |
| 需要理解的领域知识 | 多轮 decode、beam search、sampling、KV cache 轮次管理 | **Batch → Tensor 转换** |

### 10.10 文件结构

```
xllm/core/distributed_runtime/
├── rec_master.h                    # RecMasterPipeline 基类（已有）
├── rec_master.cpp                  # 新增 GeGraphMasterPipeline 实现
├── rec_engine.h                    # RecEnginePipeline 基类（已有）
├── rec_engine.cpp                  # 新增 GeGraphEnginePipeline 实现
└── ge_graph_engine_pipeline.h      # GeGraphEnginePipeline 类定义

xllm/core/runtime/
├── rec_worker_impl.h               # RecWorkPipeline 基类（已有）
├── rec_worker_impl.cpp             # 新增 GeGraphWorkerPipeline 实现
├── ge_graph_worker_pipeline.h      # GeGraphWorkerPipeline 类定义
├── executor_impl.h                 # ExecutorImpl 基类（已有）
├── ge_graph_executor_impl.h        # GeGraphExecutorImpl（已有，扩展 GetGraphInputSchema）
└── ge_graph_executor_impl.cpp      # GeGraphExecutorImpl 实现（已有）

xllm/core/scheduler/
├── fixed_steps_scheduler.h         # SchedulerPipeline 基类（已有）
└── fixed_steps_scheduler.cpp       # 新增 GeGraphSchedulerPipeline 实现

xllm/core/framework/model/
├── ep_model.h                      # EpModel（已有，扩展 GetGraphInputSchema/PrepareCustomInputs）
├── ep_model.cpp                    # EpModel 实现（已有）
├── graph_input_schema.h            # GraphInputSchema/GraphInputSpec/GraphInputSource 定义
└── graph_input_builder.h           # GeGraphInputBuilder 通用填充器

xllm/core/framework/request/
└── request_state.h                 # RequestState（扩展 GE 图原始配置字段）

xllm/core/util/
└── rec_model_utils.h               # RecPipelineType 枚举（扩展 kGeGraphPipeline）
```

### 10.11 实现计划

#### Phase 1：Schema 基础设施
1. 定义 `GraphInputSource` 枚举、`GraphInputSpec`、`GraphInputSchema` 类型
2. 实现 `EpModel::GetGraphInputSchema()`（从 `input_names_` 自动推导）
3. 实现 `ParseKVCacheName()` 支持多种 KVCache 命名模式
4. 实现 `GeGraphInputBuilder::BuildInputTensorMap()` 通用填充器
5. `EpModel` 基类新增 `PrepareCustomInputs()` 虚方法（默认空实现）

#### Phase 2：RequestState 扩展 + Master/Scheduler Pipeline
1. `RequestState` 新增 GE 图原始配置字段（`max_tokens`、`eos_token_id`、`stop_token_ids`、采样标量）
2. `build_request_common()` 修改：保留原始配置数据到 `RequestState`
3. 实现 `GeGraphMasterPipeline`
   - `generate_request()`：tokenize + `build_stop_checker=false`
4. 实现 `GeGraphSchedulerPipeline`
   - `create_batches()`：Batch 构造
   - `requires_kv_cache()`：返回 `true`
5. 工厂方法扩展：`RecMaster::create_pipeline()`、`FixedStepsScheduler::create_pipeline()`

#### Phase 3：基础 Pipeline 实现
1. 新增 `RecPipelineType::kGeGraphPipeline` 枚举值
2. 实现 `GeGraphWorkerPipeline`
   - `prepare_inputs()`：复用基类
   - `prepare_work_before_execute()`：H2D + KV block swap + Schema 驱动填充
   - `step()`：单次 `executor->forward()` + 构造 `ForwardOutput`
3. 实现 `GeGraphEnginePipeline`
   - Worker 管理（复用本地 Worker 模式）
   - `step()`：prepare → get_model_output → process_beam_sequence_group → finish
4. `GeGraphExecutorImpl` 扩展 `GetGraphInputSchema()` 和 `model()` 方法

#### Phase 4：ModelOutput 扩展
1. `ModelOutput` 新增 `beam_sequence_group` 和 `beam_search_output` 字段
2. `EpModel::forward()` 根据图输出节点名称解析结果

#### Phase 5：模型子类扩展（按需）
1. VLM 模型：`VlmEpModel::PrepareCustomInputs()` 补充多模态输入
2. Rec 模型：`RecEpModel::PrepareCustomInputs()` 补充 encoder 输入

#### Phase 6：集成测试
1. 端到端推理测试（Request → Batch → GE 图 → beam_sequence_group → Batch → Response）
2. 与 RecMultiRoundEnginePipeline 的输出一致性对比
3. 多卡场景测试
4. Schema 自动推导正确性测试（覆盖 LLM / VLM / Rec 三种模型类型）
5. Sequence 生命周期验证（确认不走逐步 append_token 循环）

### 10.12 模型输入抽象设计（Schema 驱动）

#### 10.12.1 问题

不同模型的 GE 图输入节点差异很大：

```
LLM:  input_ids, position_ids, past_key_values[0].key, past_key_values[0].value, ...
VLM:  input_ids, position_ids, pixel_values, image_grid_thw, past_key_values[0].key, ...
Rec:  tokens, positions, encoder_tokens, encoder_positions, ...
```

当前 `input_tensor_map`（`model_input_params.h:1082`）已定义但从未被填充。核心矛盾：

| 角色 | 知道什么 | 不知道什么 |
|------|---------|-----------|
| Pipeline | 运行时数据（tokens, positions, kv_caches, attention metadata） | 图要什么输入、叫什么名字 |
| Model (EpModel) | 图要什么（`input_names_` 来自 epair） | 运行时数据在哪、怎么取 |

如果在 Pipeline 中硬编码每个模型的输入名字映射，每新增一种模型都要改 Pipeline，违反开闭原则。

#### 10.12.2 方案选型

| 方案 | 思路 | 优点 | 缺点 |
|------|------|------|------|
| **A: Pipeline 硬编码** | Pipeline 直接写死 `input_tensor_map["input_ids"] = tokens` 等映射 | 简单直接 | 每新增模型改 Pipeline |
| **B: Model 侧填充** | Model 自己从 `ForwardInput` 中取数据填充 `input_tensor_map` | Pipeline 通用 | Model 承担数据搬运职责，职责不清 |
| **C: Schema 驱动** | Model 声明输入 Schema，通用 Builder 按 Schema 填充 | Pipeline 通用 + Model 职责清晰 + 可自动推导 | 需要定义 Schema 数据结构 |

**选择方案 C**。

#### 10.12.3 核心设计

```
┌──────────────────────────────────────────────────────┐
│  EpModel                                              │
│  提供: GetGraphInputSchema()                          │
│  返回: 每个输入节点的 name + source + 变换规则         │
│  来源: 从 epair 的 input_names_ 自动推导              │
└──────────────────────┬───────────────────────────────┘
                       │ Schema
                       ▼
┌──────────────────────────────────────────────────────┐
│  GeGraphInputBuilder (通用，在 Pipeline 中调用)       │
│  按 Schema 从 ForwardInput 中取标准数据               │
│  填充 input_tensor_map                                │
└──────────────────────┬───────────────────────────────┘
                       │ kCustom 类输入
                       ▼
┌──────────────────────────────────────────────────────┐
│  EpModel 子类 override: PrepareCustomInputs()         │
│  仅处理 Schema 无法自动推导的模型特有输入              │
│  (如 VLM 的 pixel_values, Rec 的 encoder_tokens)     │
└──────────────────────────────────────────────────────┘
```

#### 10.12.4 GraphInputSchema 数据结构

```cpp
enum class GraphInputSource : uint8_t {
    kTokenIds,
    kPositions,
    kKVCacheKey,
    kKVCacheValue,
    kAttentionMask,
    kKVSeqLens,
    kQSeqLens,
    kBlockTables,
    kNewCacheSlots,
    kCustom,
};

struct GraphInputSpec {
    std::string name;
    GraphInputSource source;
    int32_t layer_index = -1;  // 仅 KVCache 类输入使用
};

using GraphInputSchema = std::vector<GraphInputSpec>;
```

#### 10.12.5 Schema 自动推导

EpModel 在 `load_model()` 后从 `input_names_` 自动推导 Schema，无需每个模型手动配置：

```cpp
GraphInputSchema EpModel::GetGraphInputSchema() const {
    GraphInputSchema schema;
    for (const auto& name : input_names_) {
        GraphInputSpec spec;
        spec.name = name;
        
        if (name == "input_ids" || name == "tokens") {
            spec.source = GraphInputSource::kTokenIds;
        } else if (name == "position_ids" || name == "positions") {
            spec.source = GraphInputSource::kPositions;
        } else if (auto match = ParseKVCacheName(name)) {
            // 匹配 "past_key_values[N].key" / "past_key_values[N].value"
            // 或 "k_cache[N]" / "v_cache[N]" 等常见命名模式
            spec.source = match->is_key 
                ? GraphInputSource::kKVCacheKey 
                : GraphInputSource::kKVCacheValue;
            spec.layer_index = match->layer;
        } else if (name == "attention_mask") {
            spec.source = GraphInputSource::kAttentionMask;
        } else if (name == "kv_seq_lens") {
            spec.source = GraphInputSource::kKVSeqLens;
        } else if (name == "q_seq_lens") {
            spec.source = GraphInputSource::kQSeqLens;
        } else if (name == "block_tables") {
            spec.source = GraphInputSource::kBlockTables;
        } else if (name == "new_cache_slots") {
            spec.source = GraphInputSource::kNewCacheSlots;
        } else {
            spec.source = GraphInputSource::kCustom;
        }
        
        schema.push_back(spec);
    }
    return schema;
}
```

**KVCache 名字解析**（`ParseKVCacheName`）支持多种常见命名模式：

```cpp
struct KVCacheMatch {
    bool is_key;
    int32_t layer;
};

std::optional<KVCacheMatch> ParseKVCacheName(const std::string& name) {
    // 模式 1: "past_key_values[N].key" / "past_key_values[N].value"
    // 模式 2: "k_cache[N]" / "v_cache[N]"
    // 模式 3: "key_cache[N]" / "value_cache[N]"
    // 使用正则匹配，返回 layer_index 和 is_key
    // ...
}
```

#### 10.12.6 GeGraphInputBuilder

通用 Builder，按 Schema 从标准运行时数据中取数据填充 `input_tensor_map`：

```cpp
class GeGraphInputBuilder {
public:
    static void BuildInputTensorMap(
        ModelInputParams& params,
        const GraphInputSchema& schema,
        const torch::Tensor& tokens,
        const torch::Tensor& positions,
        std::vector<KVCache>& kv_caches) {
        
        for (const auto& spec : schema) {
            torch::Tensor tensor;
            
            switch (spec.source) {
                case GraphInputSource::kTokenIds:
                    tensor = tokens;
                    break;
                case GraphInputSource::kPositions:
                    tensor = positions;
                    break;
                case GraphInputSource::kKVCacheKey:
                    tensor = kv_caches[spec.layer_index].get_k_cache();
                    break;
                case GraphInputSource::kKVCacheValue:
                    tensor = kv_caches[spec.layer_index].get_v_cache();
                    break;
                case GraphInputSource::kKVSeqLens:
                    tensor = params.attention.device.kv_seq_lens;
                    break;
                case GraphInputSource::kQSeqLens:
                    tensor = params.attention.device.q_seq_lens;
                    break;
                case GraphInputSource::kBlockTables:
                    tensor = params.attention.device.block_tables;
                    break;
                case GraphInputSource::kNewCacheSlots:
                    tensor = params.attention.device.new_cache_slots;
                    break;
                case GraphInputSource::kAttentionMask:
                    tensor = params.graph.attn_mask;
                    break;
                case GraphInputSource::kCustom:
                    continue;  // kCustom 由 Model 子类 PrepareCustomInputs() 处理
            }
            
            if (tensor.defined()) {
                params.input_tensor_map[spec.name] = tensor;
            }
        }
    }
};
```

#### 10.12.7 Model 子类 Custom 输入扩展

对于 Schema 无法自动推导的模型特有输入，通过 Model 子类 override 处理：

```cpp
// EpModel 基类：默认空实现
class EpModel : public EpCausalLM {
public:
    // 虚方法，子类可 override
    virtual void PrepareCustomInputs(ModelInputParams& params,
                                      const ForwardInput& input) const {
        // 默认无 custom 输入
    }
};

// VLM 模型子类：补充多模态输入
class VlmEpModel : public EpModel {
public:
    void PrepareCustomInputs(ModelInputParams& params,
                              const ForwardInput& input) const override {
        if (input.input_params.multimodal.mm_data.has_value()) {
            params.input_tensor_map["pixel_values"] = 
                input.input_params.multimodal.mm_data.pixel_values;
            params.input_tensor_map["image_grid_thw"] = 
                input.input_params.multimodal.mm_data.image_grid_thw;
        }
    }
};
```

#### 10.12.8 GeGraphExecutorImpl 暴露 Schema

`GeGraphExecutorImpl` 需要暴露 Schema 和 Model 指针供 Pipeline 使用：

```cpp
class GeGraphExecutorImpl : public ExecutorImpl {
public:
    // 已有方法...
    
    // 新增：获取 Graph 输入 Schema
    const GraphInputSchema& GetGraphInputSchema() const {
        return schema_;
    }
    
    // 新增：获取 Model 指针（供 PrepareCustomInputs 调用）
    CausalLM* model() const { return model_; }

private:
    GraphInputSchema schema_;  // 在构造时从 EpModel 获取并缓存
};

// 构造函数中初始化 Schema
GeGraphExecutorImpl::GeGraphExecutorImpl(CausalLM* model, ...) {
    // ... 已有逻辑 ...
    
    EpModel* ep_model = dynamic_cast<EpModel*>(model_);
    schema_ = ep_model->GetGraphInputSchema();
}
```

#### 10.12.9 完整调用流程

```
RecWorkerImpl::prepare_inputs(batch)
  └─ batch.prepare_forward_input()              ← 模型无关，已有
       → ForwardInput (tokens, positions, attention, sampling, ...)

GeGraphWorkerPipeline::prepare_work_before_execute()
  ├─ Base: H2D + KV block swap                 ← 模型无关，已有
  ├─ GetGraphInputSchema() from Executor        ← Model 声明"我要什么"
  ├─ GeGraphInputBuilder::Build(schema)         ← 通用，按 Schema 填充标准输入
  └─ EpModel::PrepareCustomInputs()             ← 仅 kCustom 输入需 override

GeGraphExecutorImpl::run()
  └─ EpModel::forward()
       └─ BuildGraphInputs(input_names_, input_tensor_map)  ← 按名字查找
```

#### 10.12.10 不同模型类型的 Schema 示例

**LLM 模型**（如 Qwen2）：
```
input_names_ = ["input_ids", "position_ids", 
                "past_key_values[0].key", "past_key_values[0].value",
                "past_key_values[1].key", "past_key_values[1].value", ...]

推导 Schema:
  {name: "input_ids",              source: kTokenIds}
  {name: "position_ids",           source: kPositions}
  {name: "past_key_values[0].key", source: kKVCacheKey,   layer_index: 0}
  {name: "past_key_values[0].value", source: kKVCacheValue, layer_index: 0}
  ...

kCustom 数量: 0 → 无需 override PrepareCustomInputs()
```

**VLM 模型**（如 Qwen2-VL）：
```
input_names_ = ["input_ids", "position_ids", "pixel_values", "image_grid_thw",
                "past_key_values[0].key", "past_key_values[0].value", ...]

推导 Schema:
  {name: "input_ids",        source: kTokenIds}
  {name: "position_ids",     source: kPositions}
  {name: "pixel_values",     source: kCustom}      ← 需要 override
  {name: "image_grid_thw",   source: kCustom}      ← 需要 override
  {name: "past_key_values[0].key", source: kKVCacheKey, layer_index: 0}
  ...

kCustom 数量: 2 → VlmEpModel override PrepareCustomInputs()
```

**Rec 模型**：
```
input_names_ = ["tokens", "positions", "encoder_tokens", "encoder_positions",
                "k_cache[0]", "v_cache[0]", ...]

推导 Schema:
  {name: "tokens",             source: kTokenIds}
  {name: "positions",          source: kPositions}
  {name: "encoder_tokens",     source: kCustom}    ← 需要 override
  {name: "encoder_positions",  source: kCustom}    ← 需要 override
  {name: "k_cache[0]",         source: kKVCacheKey,   layer_index: 0}
  {name: "v_cache[0]",         source: kKVCacheValue, layer_index: 0}
  ...

kCustom 数量: 2 → RecEpModel override PrepareCustomInputs()
```

#### 10.12.11 方案对比总结

| 维度 | A: Pipeline 硬编码 | B: Model 填充 | **C: Schema 驱动** |
|------|-------------------|--------------|-------------------|
| 新增模型改动 | 改 Pipeline | 改 Model | **只改 Schema（可自动推导）** |
| Pipeline 通用性 | 差（每模型一份） | 好 | **好** |
| Model 职责 | 干净 | 重（承担数据搬运） | **适中（只声明 + 少量 Custom）** |
| 自动推导能力 | 无 | 无 | **有（从 input_names_ 推导）** |
| 扩展性 | 差 | 好 | **好（kCustom + override）** |
| 代码量 | Pipeline 膨胀 | Model 膨胀 | **Builder 固定，Schema 自动生成** |

### 10.13 待确认事项

#### 10.13.1 GE 图输出格式（高优先级）

**问题**：epair 图的输出 Tensor 格式是什么？

**需要确认**：
- 输出是否包含 `beam_sequence_group`（`[groups, beam_width, rounds]` 的 token 矩阵）？
- 输出是否包含 `out_logprobs`（每个 beam 的累积 logprob）？
- 如果图输出不包含 beam search 结果，而是输出 logits，则需要在 Pipeline 中补充 Sampler 和 BeamSearcher（退化为类似 Base RecWorkPipeline 的模式）

**影响**：
- 决定 `GeGraphWorkerPipeline::step()` 的输出构造逻辑
- 决定 `ModelOutput` 的扩展方式

#### 10.13.2 输出 D2H 策略

**问题**：GE 图输出 Tensor 在 device 上，如何高效搬到 host？

**方案**：
- 使用异步 D2H 拷贝 + `Device::synchronize_default_stream()`
- 参考 `OneRecXAttentionEnginePipeline::get_model_output()` 的 D2H 逻辑

#### 10.13.3 非 beam search 场景

**问题**：如果某些请求不使用 beam search（纯 sampling），GE 图是否支持？

**可能方案**：
- 方案 A：GE 图始终执行 beam search，非 beam 请求 beam_width=1
- 方案 B：GE 图输出 logits，Pipeline 根据请求类型选择 sampling 或 beam search 后处理
- 方案 C：准备两种 epair 图（beam / non-beam），根据请求类型选择