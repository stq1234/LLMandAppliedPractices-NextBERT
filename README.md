# NextBERT 设计与验证

本目录保存中文英文 8K 双向编码器的模型选择、可复现的 CPU 程序验证和后续证伪方案。默认路线从公开的 mmBERT-base 检查点继续适配；另附约 300M 参数的从头训练蓝图作为预算对照。当前没有下载检查点、预训练语料或评测数据，也没有进行语义基准测试。

本次实验的代码及相关说明可在该仓库中获取。仓库地址：[LLMandAppliedPractices-NextBERT](https://github.com/stq1234/LLMandAppliedPractices-NextBERT)。

## 从仓库根目录运行

使用 Python 3.12。复现环境固定见 `requirements.txt`；在仓库根目录安装依赖并运行：

```powershell
python -m pip install -r ".\作业3.2_NextBERT设计验证\requirements.txt"
python ".\作业3.2_NextBERT设计验证\src\run_verification.py"
```

程序将 CPU 线程数限制为 2，不调用 CUDA API，不下载模型权重。它通过官方 `ModernBertForMaskedLM` 模块在 meta device 上统计 mmBERT 配置参数，在小型模型上执行 MLM、连续片段 MLM、分类、抽取、对比检索和重排的前向、反向与参数更新检查，并尝试一次 8K CPU 前向。可用 `--skip-8k` 跳过较占内存的 8K 检查。

输出写入 `results/verification.json` 和 `results/parameter_counts.csv`。输入为合成随机 token ID，因此损失值只证明接口可运行，不能解释成语言任务效果。8K 时间和进程内存只描述本机的小型 CPU 检查，不代表 300M 部署速度。

## 文件

- `configs/mmbert_adaptation.json`：固定公开检查点 revision 与配置。
- `configs/scratch_blueprint.json`：48K 共享词表的参数预算替代方案；未训练。
- `configs/experiment_plan.json`：拟议数据快照、100M-token pilot 预算、三组固定对照及预注册门槛；未执行。
- `data/manifest.schema.json`：未来中英文预训练语料清单的校验结构。本次没有填充语料清单。
- `src/nextbert_lab.py`：参数核算和 CPU 验证实现。
- `src/run_verification.py`：仓库根目录入口。
- `设计与验证报告.md`：架构选择、实测结果和预注册式消融计划。
