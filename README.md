## EquiQuant

EquiQuant 是一个在 Ascend NPU 上**自动完成模型权重量化与精度调优**的工具链封装，串联了：

- **ModelSlim**：负责权重量化与敏感层分析；
- **vLLM & vLLM-Ascend**：负责以 OpenAI 风格 API 暴露量化模型；
- **AISBench**：负责在本地对 API 做数据集精度 / 性能评测。

整体流程为：

1. **环境检查**：确认 `msmodelslim`、`vllm(-ascend)`、`ais_bench` 等可用；
2. **初始量化**：用默认 ModelSlim YAML 量化一版权重；
3. **拉起 vLLM-Ascend 服务**（非阻塞）；
4. **启动 AISBench 评测**（阻塞），统计各数据集精度；
5. **精度判定与回退**：若所有数据集都在目标精度的 \(\pm 1\%\)（或自定义区间）内，则保存当前权重；否则根据敏感层信息扩大“回退层”（保留为浮点权重、不再量化），重新生成 ModelSlim 配置，回到 Step 2 继续迭代。

---

## 使用前准备

### 依赖安装

请确保已正确安装以下组件，并在当前环境中可用：

- **ModelSlim / msmodelslim**（含 CLI）；
- **vLLM 与 vLLM-Ascend**（含对应 Ascend 驱动 / 固件）；
- **AISBench**（带 OpenAI 风格 LLM API 评测能力）；
- 以及 `requirements.txt` 中列出的 Python 依赖。

你可以先在 shell 中分别运行以下命令确认安装情况（命令名称需与本地环境实际保持一致）：

- `msmodelslim --help`
- `python -m vllm ...` 或 `vllm --help`
- `ais_bench --help`

如果上述命令不可用，请参考各自官方文档完成安装与环境配置。

---

## 配置文件说明

EquiQuant 的**唯一用户配置文件**为 `config/config.yaml`。首次使用时需要：

1. 将模板文件复制 / 重命名为真实配置文件：

   ```bash
   cd /path/to/EquiQuant
   cp config/config_template.yaml config/config.yaml
   ```

2. 打开 `config/config.yaml`，根据实际环境与模型**逐项修改**。

下面对主要字段进行分组说明（以 `config/config_template.yaml` 为例）。

### 1. 模型与工作区相关

- **`base_model_path`**  
  原始模型权重路径（通常是 HuggingFace / 本地权重目录）。  
  例如：`"/mnt/nfs/weight/Qwen3-32B"`。

- **`workspace_base_dir`**  
  EquiQuant 的工作区根目录，保存量化配置、临时权重、日志等。

- **`workspace_current_run_dir`**  
  当前运行使用的子目录名，例如 `"current_run"`，实际路径为  
  `workspace_base_dir/workspace_current_run_dir`。

- **`workspace_history_dir`**  
  历史运行记录存放目录名，例如 `"history"`。

- **`workspace_best_weights_dir`**  
  满足精度要求的“最佳权重”输出目录名，例如 `"best_weights"`。

- **`workspace_quant_config_name`**  
  每轮生成的 ModelSlim 配置 YAML 文件名，例如 `"generated_modelslim_config.yaml"`。

- **`workspace_quant_weights_dir`**  
  量化后权重的存放目录名，例如 `"quantized_weights"`。

> **建议**：在修改 `workspace_*` 相关字段前，先确认本地磁盘布局与读写权限。

### 2. 核心调优 / 回退策略

- **`strategy_initial_fallback_layers`**  
  初始就被视为“敏感层”、直接保留为浮点权重的层列表。  
  模板中给出了一个针对 Qwen3-32B 的示例，从 `lm_head` 到  
  `model.layers.63.mlp.down_proj` 的一系列投影层。  
  无 AQT 模式下现在只使用这份固定列表，不再自动追加回退层；AQT 模式下回退层完全由 AQT 产出。

### 3. 评估 / 精度相关

- **`evaluation_tolerance_ratio`**  
  默认的**相对容忍区间**，例如 `1.0` 表示允许在目标精度的 \(\pm 1\%\) 区间内判定为通过。

- **`disable_qwen_thinking`**  
  若为 `true`，则在请求参数中自动注入 `chat_template_kwargs={"enable_thinking": False}`，  
  用于关闭 Qwen 类模型的思维链 / 思考输出，以便与基线对齐。

- **`evaluation_datasets`**  
  需要评测的数据集列表，每个键是一个数据集 ID，例如：

  - **`ceval`**  
    - `config_name`: AISBench 中对应的配置名，例如 `"ceval_gen_5_shot_str"`；  
    - `mode`: AISBench 执行模式，如 `"all"`；  
    - `target_accuracy`: 该数据集的目标精度（基线精度）。

  - **`aime2024`**（示例）  
    - `config_name`: `"aime2024_gen_0_shot_chat_prompt"`；  
    - `mode`: `"all"`；  
    - `target_accuracy`: `0.900`；  
    - `tolerance_ratio`: `0.067`（若设置则覆盖全局 `evaluation_tolerance_ratio`）；  
    - `metric_keys`: 精度从 AISBench 日志中提取的字段名列表，如 `"aime2024_accuracy"`。

若某个数据集未显式设置 `tolerance_ratio`，则使用全局 `evaluation_tolerance_ratio`。

### 4. AISBench 相关配置

- **`aisbench_timeout`**  
  单次 AISBench 评测的超时时间（秒），例如 `7200`。

- **`aisbench_request_rate`**  
  请求速率（QPS），例如 `1`。

- **`aisbench_retry`**  
  AISBench 失败后的重试次数。

- **`aisbench_batch_size`**  
  Batch size，需与数据集配置兼容。

- **`aisbench_max_out_len`**  
  最大生成长度（token 数），若只关心选择题，可设为较小值。

- **`aisbench_pred_postprocessor`**  
  预测结果后处理函数名，例如 `"extract_non_reasoning_content"`，  
  用于从回答中剥离推理过程，仅保留最终答案。

- **`aisbench_generation_kwargs`**  
  传给服务端的生成参数，常见字段包括：
  - `temperature`
  - `top_k`
  - `top_p`
  - `seed`
  - `repetition_penalty`

- **`aisbench_log_dir`**  
  AISBench 日志输出目录，例如 `"workspace/aisbench_logs"`。

- **`aisbench_default_metric_keys`**  
  当数据集未配置专门的 `metric_keys` 时，默认尝试从日志中解析这些字段：  
  `"final_accuracy"`, `"accuracy"`, `"score"` 等。

- **`aisbench_use_chat_template`**  
  - `true`: 使用 Chat 模板（`VLLMCustomAPIChat`, `vllm_api_general_chat*`）；  
  - `false`: 使用普通模板（`VLLMCustomAPI`, `vllm_api_general*`）。

- **`aisbench_cleanup_model_config`**  
  评测结束后是否清理 AISBench 临时模型配置。

- **`aisbench_extra_args`**  
  额外传给 AISBench CLI 的参数（若不需要可设为 `null`）。

### 5. vLLM-Ascend 服务相关

- **`vllm_port`**  
  vLLM-Ascend 服务监听的端口，例如 `1234`。

- **`vllm_env_vars`**  
  启动 vLLM-Ascend 时设置的环境变量，例如：
  - `HCCL_BUFFSIZE`
  - `VLLM_VERSION`
  - `ASCEND_RT_VISIBLE_DEVICES`
  - 以及可选的 `HCCL_OP_EXPANSION_MODE` 等。

- **`vllm_args`**  
  启动 vLLM-Ascend 的额外命令行参数，例如：
  - `trust-remote-code`
  - `tensor-parallel-size`
  - `data-parallel-size`
  - `quantization: "ascend"`
  - `enable-prefix-caching`
  - `max-model-len`
  - `max-num-batched-tokens`
  - `gpu-memory-utilization`（在 Ascend 环境中同样用于控制显存 / HBM 利用率）

- **`vllm_args.additional_config`**  
  更底层的后端配置，例如：
  - `ascend_scheduler_config.enabled`：是否启用 Ascend 调度器；
  - `enable_weight_nz_layout`：是否打开非零权重布局优化。

> **注意**：vLLM 的入口命令、host、健康检查等通用参数已经在代码里封装，通常无需在配置文件中重复设置。

### 6. 量化器（ModelSlim）相关

- **`quantization_visible_devices`**  
  参与量化的 NPU 设备 ID 列表，例如 `"0,1,2,3"`。

- **`quantization_model_type`**  
  模型类型标识，需要与 ModelSlim 支持列表匹配，例如 `"Qwen3-32B"`。

- **`quantization_device`**  
  量化运行设备类型，在 Ascend 上通常为 `"npu"`。

- **`quantization_trust_remote_code`**  
  是否允许加载第三方 `trust_remote_code` 模型。

- **`quantization_template_config`**  
  ModelSlim 的 YAML 模板片段，会被 EquiQuant 在运行时补充 / 修改后写入到真正的量化配置文件中（`workspace_quant_config_name`）。  
  其中常见字段包括：

  - `apiversion`: 例如 `modelslim_v0`；
  - `metadata`: 包含 `config_id`、`score`、`verified_model_types`、`label` 等；
  - `spec.calib_cfg`: 例如 `w_bit`、`a_bit`、`mm_tensor`、`pdmix` 等量化超参；
  - `spec.calib_dataset`: 标定数据集文件名，如 `"mix_calib.jsonl"`；
  - `spec.calib_save_params.part_file_size`: 每个分片最大大小。

你可以根据目标模型与量化策略修改这些字段，例如调整比特宽度、切换稀疏模式、修改校准数据集等。

---

## 运行方式

在完成配置文件编辑后，直接运行：

```bash
cd /path/to/EquiQuant
python main.py
```

程序会按以下步骤自动执行：

1. **Step 0 – 环境检测**  
   在 `modules/env_checker.py` 中检查当前环境是否安装并可调用 ModelSlim、vLLM(-Ascend)、AISBench 等组件。

2. **Step 1 – ModelSlim 初始量化**  
   在 `modules/quantizer.py` 中，根据 `quantization_*` 与 `quantization_template_config` 生成 ModelSlim YAML，并调用量化工具，对 `base_model_path` 对应模型进行权重量化，输出到 `workspace_quant_weights_dir`。

3. **Step 2 – 启动 vLLM-Ascend 服务**（非阻塞）  
   在 `modules/server.py` 中，使用最新的量化权重与 `vllm_*` 配置，拉起 vLLM-Ascend 服务，暴露 OpenAI 风格的 HTTP API（端口为 `vllm_port`）。

4. **Step 3 – AISBench 评测**（阻塞）  
   在 `modules/bencher.py` 中，将 `evaluation_datasets` 中的各个数据集转换为 AISBench 所需的 Python 配置脚本，随后调用 AISBench CLI，对每个数据集执行评测并记录日志到 `aisbench_log_dir`。

5. **Step 4 – 精度判定与回退迭代**  
   - 将 AISBench 解析得到的精度结果与 `evaluation_datasets.*.target_accuracy` 比对；  
   - 若所有数据集都在 `target_accuracy ± tolerance`（全局 / 局部）区间内：  
     - 当前量化权重与配置信息会被保存到 `workspace_best_weights_dir`；  
     - 程序结束。  
   - 若至少一个数据集不满足目标精度：  
     - 基于敏感层分析扩展 `strategy_initial_fallback_layers`（回退更多层到浮点）；  
     - 重新生成 ModelSlim YAML 并回到 Step 1，继续量化与评测，直至满足精度或达到你在代码中设定的终止条件（如最大迭代次数）。

整个过程用户只需**准备好配置文件并保证依赖可用**，其余步骤均由 EquiQuant 自动执行。

---

## 常见使用建议

- **关于初始回退层**  
  模板给出的 `strategy_initial_fallback_layers` 较为保守，适合作为安全起点。  
  若你希望更激进地压缩，可减少初始回退层；若精度难以收敛，可手工添加更多关键层。

- **关于目标精度与容忍度**  
  - 建议先通过全精度模型或已有基线记录，确定 `target_accuracy`；  
  - 对于高噪声或难数据集，可以单独调大 `tolerance_ratio`。

- **关于多 NPU 环境**  
  - `quantization_visible_devices` 与 `vllm_env_vars.ASCEND_RT_VISIBLE_DEVICES` 需要与你实际的 NPU 拓扑匹配；  
  - 多卡并行时注意 HCCL 环境与调度配置是否正确。

---

## 目录结构概览

- `main.py`：EquiQuant 程序入口，串联整个工作流；
- `config/config.yaml`：用户唯一需要编辑的配置文件（由 `config_template.yaml` 拷贝而来）；
- `core/engine.py`：核心调度与状态管理；
- `core/strategy.py`：回退策略与敏感层相关逻辑；
- `modules/quantizer.py`：ModelSlim 量化调用实现；
- `modules/server.py`：vLLM-Ascend 服务启动与管理；
- `modules/bencher.py`：AISBench 评测配置生成与执行；
- `modules/env_checker.py`：环境与依赖检查；
- `utils/*`: 日志记录、配置读取、Shell 调用等工具函数。
