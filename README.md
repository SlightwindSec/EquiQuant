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

- `evaluation.tolerance_ratio`：全局默认容忍度（例如 `0.01` 即 ±1%）。
- `evaluation.aisbench`：定义 `ais_bench` 二进制、模型配置模板、请求速率、生成参数、日志目录等。
- `evaluation.datasets`：为每个评测数据集提供
  - `config_name`：`ais_bench --datasets` 对应的 python 配置名。
  - `target_accuracy` 和可选的 `tolerance_ratio`（覆盖全局值）。
  - `metric_keys` / `result_regex`（可选）用于解析日志得分。

更多字段示例见 `config/config.yaml`。
