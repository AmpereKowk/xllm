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