# EquiQuant

自动化完成 ModelSlim 量化 → vLLM-Ascend 推理服务 → AISBench 精度验证 的一体化流程。

## 运行

```shell
python main.py
```

程序会依次执行：

1. **环境检测**：确认 `msmodelslim`、`vllm(-ascend)`、`ais_bench` 等依赖可用。
2. **ModelSlim 量化**：根据 `config/` 中的模板生成 YAML，并输出量化权重。
3. **vLLM-Ascend 服务**：用最新权重拉起 OpenAI 风格 API。
4. **AISBench 精度评测**：自动生成 `vllm_api` 模型配置文件，按数据集执行 `ais_bench --models ... --datasets ... --mode ...`，解析日志精度。
5. **策略回退**：若任一数据集命中 `target_accuracy ± tolerance` 范围之外，则扩展回退层并重新迭代；若全部满足，则将当前权重与配置快照保存到 `workspace/best_weights`.

## 关键配置

`config/config.yaml` 采用扁平化键名，约定俗成地通过前缀区分不同模块，常见字段包括：

- `base_model_path`：原始模型路径。
- `workspace_*`：运行中产生的量化配置、权重、日志目录等。
- `strategy_layer_pattern` / `strategy_initial_fallback_layers`：回退策略。
- `evaluation_tolerance_ratio` 与 `evaluation_datasets`：分别定义全局容忍度以及每个数据集的 `config_name / target_accuracy / tolerance_ratio / metric_keys`。
- `aisbench_*`：AISBench 运行参数（请求速率、生成参数、日志目录等）。`disable_qwen_thinking` 会自动向 `generation_kwargs` 注入 `chat_template_kwargs={"enable_thinking": False}`。
- `vllm_port`、`vllm_env_vars`、`vllm_args`：控制 vLLM-Ascend 端口及附加参数（`entrypoint/host/health check/startup timeout/served-model-name` 等已内置，用户无需配置）。
- `quantization_*`：ModelSlim 运行所需的设备、模板配置等。

完整示例见 `config/config_template.yaml`。*** End Patch
