# NextBERT 双尺度编码器：设计与复现

本仓库实现从 mmBERT-base warm-start 的四种注意力配置、MLM 适配、分类、抽取式 QA、双编码检索和交叉编码重排，以及评测与长上下文诊断。当前方案是每个完整适配分支 67,108,864 输入 token 的候选配方。GPU 7 的真实批次短预跑已完成，用于核对速度与显存；模型质量由正式训练后的固定任务评测判断。正式训练矩阵尚未启动，等待用户预算决定。

代码及复现材料：[https://github.com/stq1234/LLMandAppliedPractices-NextBERT](https://github.com/stq1234/LLMandAppliedPractices-NextBERT)。

## 模型与注意力变体

权重和 tokenizer 来自固定 revision 的 [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base/tree/8c1f42296af5de623b0ad6d3a5ed6a5a1838d82c)。底座有 22 层、隐藏维度 768、12 个注意力头、GeGLU 单分支宽度 1152、256,000 词表和 8,192 位置上限；MLM 参数量约 307.8M。实现保留嵌入、QKV 与输出投影形状、RoPE、Pre-LayerNorm、GeGLU、输入与 MLM 嵌入权重绑定以及原始 token 位置。基座 warm-start 后只调整注意力连接方式和任务头。

| 变体 | 全局注意力层 | 双尺度层 | MLM 目标 |
|---|---|---|---|
| V0 | 0、3、6、9、12、15、18、21 | 无 | 全程 token MLM |
| V1 | 0、6、12、18、21 | 无 | 全程 token MLM |
| V2 | 0、6、12、18、21 | 膨胀率 2：2、8、14、20；膨胀率 4：4、10、16 | 全程 token MLM |
| V3 | 0、6、12、18、21 | 与 V2 相同 | 与 V2 共享前 384 步；最后 128 步在各长度阶段内均衡交替 token/span MLM |

其余层使用半径 64 的双向局部注意力。双尺度层的前六个头保留近邻路径，后六个头通过每头一个可学习 sigmoid gate 混合近邻与膨胀路径；七层共 42 个 gate。膨胀路径按原始 RoPE 位置计算，再按余数类重排键和值以执行局部核，不丢弃或重编号 token。局部 SDPA 按查询块计算，不构造完整的长度平方局部掩码。gate 初始 logit 为 0，前 64 次更新以线性 ramp 启用新增路径。

主要接口位于 `src/modeling_nextbert.py`：`NextBert(model_dir, variant, attention_backend="auto")`。模型提供 `forward_mlm`、`encode`、`forward_classification`、`forward_qa`、`encode_retrieval` 和 `forward_reranking`。模型和 tokenizer 从本地目录加载；脚本不会自行下载权重。

## MLM 适配预算

每个长度阶段每次更新处理 131,072 个输入 token。表中有效序列数等于物理 batch 乘梯度累积次数；Wiki 窗口为单篇文章内的满长连续片段，不跨文档拼接，不把 padding 计入 token 预算。

| 最大长度 | 物理 batch | 累积更新 | 有效序列/更新 | 更新数 | 阶段 token 数 |
|---:|---:|---:|---:|---:|---:|
| 512 | 64 | 4 | 256 | 192 | 25,165,824 |
| 1,024 | 32 | 4 | 128 | 192 | 25,165,824 |
| 4,096 | 8 | 4 | 32 | 64 | 8,388,608 |
| 8,192 | 4 | 4 | 16 | 64 | 8,388,608 |
| **每分支** |  |  |  | **512** | **67,108,864** |

V0 和 V1 各执行 512 步。V2/V3 共用 512 与 1,024 长度阶段的 384 步及其优化器状态，再各自完成 4,096 与 8,192 阶段的 128 步。因此四个分支合计执行 1,664 个独立优化更新、处理 218,103,808 个输入 token。V3 的最终 128 步在最后两个阶段各进行 32 个 token MLM 和 32 个 span MLM 更新。MLM mask 比率为 30%；span 目标保持与 token MLM 相同的监督 token 数，平均 span 长度约 3，最长 10。

训练使用全参数 AdamW、BF16 autocast、梯度检查点和仅对监督位置计算 MLM 输出。主干学习率为 1e-5，gate 学习率为 1e-4，weight decay 0.01，梯度裁剪 1，warmup 5%，随后线性衰减。完整数值配置见 `configs/experiment_plan.json`。

## 下游任务与比较范围

所有下游阶段从同变体的 MLM checkpoint 初始化主干和 gate，任务头用指定 seed 新建。主干继续训练；QA 的 span loss 和解码只使用上下文 token。

| 任务 | 训练量 | 有效 batch | 更新预算/模型阶段 |
|---|---|---:|---:|
| XNLI 分类 | 中、英文各 10,000 条，训练 2 epoch | 64 句对 | 625 |
| SQuAD/CMRC QA 常规窗口 | 每语言 5,000 个训练问题，训练 2 epoch | 32 features | 625 |
| QA 长证据增强 | 每语言 500 条，仅由训练集构造 | 32 features | 63 |
| MLDR 检索 | 每语言 2,000 个训练查询，训练 2 epoch | 64 queries（gradient-cache）；物理 document batch 4、query microbatch 32 | 125 |
| MLDR 重排 | 共 4,000 个查询、一正二负，训练 2 epoch | 8 个 query groups（物理 4、累积 2） | 1,000 |

QA 共 688 步；长增强从训练文章构造，不能使用评测文章。检索用 InfoNCE，包含显式负例和逻辑 batch 内负例；重排使用一正二负组内交叉熵。检索与重排的学习率、窗口和负例细节见实验计划。

各模型的每项任务仅训练一次：V0/V3 的四任务共 8 个阶段，V1/V2 的 QA 与检索共 4 个阶段，合计 12 个下游阶段（2 个分类、4 个 QA、4 个检索和 2 个重排）。程序使用固定随机状态保证可复现，不进行不同随机种子的重复训练。

## 数据与本地准备

数据 profile `full_67m` 对应新的 67M-token 候选配方；旧的 `full` 是既有小规模计时和基线准备数据，不代表新方案的数据量。准备工作在可联网的本地机器进行，远程训练环境保持离线。profile 完成后，典型准备与检查命令如下：

```powershell
python -m pip install -r requirements.txt
python src/prepare_assets.py
python src/prepare_data.py --profile full_67m
python src/prepare_data.py --profile full_67m --merge-manifest
python tools/validate_prepared.py data/prepared/full_67m
```

模型资产固定为 `artifacts/mmbert-base/`。Wiki 文章在切窗前分配 train/dev/test，训练数组 `wiki_{length}.npy` 只包含 train 行，并配有文章、语言、split 和窗口 sidecar。分类及 QA 使用中英文 XNLI、SQuAD 1.1 和 CMRC2018；QA 保留 overflow windows、`context_mask`、offset、原始文本及答案。评估按问题聚合全部窗口并使用 SQuAD/CMRC 官方 EM/F1。检索与重排来自 MLDR；研究语料池为每语言 5,000 篇，检索评估标记为缩减语料实验，重排所有模型共用固定 BM25 top-20 候选。

固定数据 revision、字段、split、许可状态和重建方式记录在 `data/README.md`、`src/prepare_data.py` 与生成的 manifests。准备数据是可复现的所选子集，不声称下载了各上游数据集的完整快照。体积较大的 prepared 文件需单独归档或重建；`tools/archive_prepared.py` 只归档小于 96 MiB 的文件。

远程 Linux 环境使用项目配置的 `a6000`、conda 环境 `llm`，并设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1` 和本地 `NLTK_DATA`。传输前将代码、模型和数据分别打包；通过共享 `remote-upload` 工具校验 SHA-256。离线安装和数据字段说明见 [`data/README.md`](data/README.md) 与项目 AGENTS.md。

## 真实批次校准

短预跑核对实际模型和实际数据 batch 的吞吐、显存和恢复开销。默认每个训练案例执行 2 次 warm-up 和 3 次计时更新，结果逐项写入 JSON/JSONL。新配方的 MLM、下游任务和优化后的 span 校准分别保存在 `results/pilot_standard67m_mlm7`、`results/pilot_standard67m_tasks7` 和 `results/pilot_standard67m_span_optimized7`，综合投影保存在 `results/calibration_standard67m_summary.json`。这些测量用于计算运行预算，不作为模型质量结果。

矩阵保留最终 MLM checkpoint 的 optimizer、scheduler 和 RNG 状态，以支持继续适配；下游最终 checkpoint 保存纯权重。双路调度只在依赖阶段成功结束后清理已消费的 MLM 中间 checkpoint；V2 的共享前缀会等 V2 与 V3 两路续训均完成后再清理。最终 8192-token MLM checkpoint 与全部下游 checkpoint 会保留。

完整校准入口：

```powershell
python tools/launch_pilot.py --gpu 7 --reserve-gib 6 `
  --record results/gpu7-preflight-67m.json -- `
  --model-dir V0=artifacts/mmbert-base `
  --model-dir V1=artifacts/mmbert-base `
  --model-dir V2=artifacts/mmbert-base `
  --model-dir V3=artifacts/mmbert-base `
  --mlm-data-dir data/prepared/full_67m `
  --classification-data data/prepared/full_67m/classification_train.jsonl `
  --qa-data data/prepared/full_67m/qa_train.jsonl `
  --retrieval-data data/prepared/full_67m/retrieval_train.jsonl `
  --evaluation-corpus data/prepared/full_67m/retrieval_corpus.jsonl `
  --evaluation-retrieval-queries data/prepared/full_67m/retrieval_queries_test.jsonl `
  --evaluation-reranking-data data/prepared/full_67m/reranking_test.jsonl `
  --output-dir results/pilot_67m
```

完整训练入口为 `tools/run_all_training.py`。默认只打印计划且不等待 GPU；加 `--execute` 后先等待 GPU 7 满足资源条件，再自动按固定的 5 波双路计划完成全部 14 个 MLM 阶段和 12 个下游阶段并保存 checkpoint。每波两条 lane 同时运行，lane 内顺序执行，波间设置 barrier；如需串行，可传入 `--max-workers 1`。评估不包含在训练调度中，训练后继续使用下文现有评估 CLI。

带 `--execute` 时，程序每 30 秒检查一次 GPU 计算进程和显存：计算进程数须为 0，或仅有一个 owner 为 `shaotianqi` 的进程；同时默认双路要求至少空闲 30 GiB，串行模式要求 18 GiB。条件未满足时继续等待，不设置等待截止时间或预留开始时段；一旦满足便立即自动进入训练。检查只使用 `nvidia-smi` 和进程 owner 查询，不会预先导入 PyTorch 或初始化 CUDA。训练开始后，各 worker 按正常 PyTorch 显存分配方式运行，不设置每进程显存额度，也不增加阶段间显存门槛；训练参数和 batch 保持既定配置。48 GiB A6000 若有 15 GiB 常驻占用，理论空闲约 33 GiB，仍须通过执行时的实际检查。启动命令：

```powershell
python tools/run_all_training.py --gpu 7 --output-dir results/formal_67m --execute
```

脚本会拒绝含有 checkpoint 的既有输出目录；阶段失败时会中止剩余任务并保留日志与未消费的 checkpoint。双路模式只会终止本调度器启动的另一 lane worker。每个 job 的独立日志位于输出目录下 `logs/<run timestamp>/`，总控同时逐项打印开始/结束状态。成功完成时会在依赖波次结束后清理已消费的 MLM 中间 checkpoint，保留四个最终 MLM checkpoint 和 12 个下游 checkpoint；评测命令单独执行。

## 六项门控微调补训

`tools/retrain_gated_downstream.py` 是独立的六任务入口，复用各任务训练命令，不调用或修改原五波调度器。本轮固定只使用 GPU 5 和 GPU 7：GPU 5 同时运行 V2 检索、V3 检索、V3 QA、V3 分类；GPU 7 同时运行 V2 QA、V3 重排。启动前按既有预跑峰值与每进程约 1 GiB CUDA 开销检查各卡空闲显存，不设置每进程显存上限。

所有任务从 `results/formal_67m/mlm/` 中相应版本的最终预训练 checkpoint 初始化，保留原数据、batch、步数和学习率。下游微调显式启用结构门控，checkpoint metadata 保存其运行状态。新 checkpoint 和训练记录写入 `results/formal_67m_gatefix/`；入口汇总各 GPU 的实际任务数、更新步数、进程计时与显存峰值。该卡号安排仅用于本轮补训，不改变原训练入口默认 GPU 7。

```bash
python tools/retrain_gated_downstream.py --execute
```

## 评估与长程诊断

各任务评估通过本地 checkpoint 执行；`--model-dir` 指向原始 mmBERT 目录，提供配置和 tokenizer。例：

```powershell
python src/evaluate_nextbert.py --task classification --variant V0 `
  --model-dir artifacts/mmbert-base `
  --checkpoint results/formal_67m/downstream/V0/classification/17/V0_classification_checkpoint.pt `
  --data data/prepared/full_67m/classification_test.jsonl `
  --output results/evaluation/V0-classification.json

python src/evaluate_nextbert.py --task qa --variant V0 `
  --model-dir artifacts/mmbert-base `
  --checkpoint results/formal_67m/downstream/V0/qa/17/V0_qa_checkpoint.pt `
  --data data/prepared/full_67m/qa_test.jsonl `
  --output results/evaluation/V0-qa.json `
  --predictions-output results/evaluation/V0-qa-predictions.jsonl

python src/evaluate_nextbert.py --task retrieval --variant V0 `
  --model-dir artifacts/mmbert-base `
  --checkpoint results/formal_67m/downstream/V0/retrieval/17/V0_retrieval_checkpoint.pt `
  --data data/prepared/full_67m/retrieval_queries_test.jsonl `
  --corpus data/prepared/full_67m/retrieval_corpus.jsonl `
  --output results/evaluation/V0-retrieval.json
```

长程诊断使用 held-out QA 问题和 held-out Wikipedia 背景，所得长输入是受控合成文档，不是自然长文 QA 质量估计。每语言按 50 个 source groups 构造 1K/4K/8K 输入、首/中/尾证据位置；比较 intact、evidence deletion、counterfactual answer replacement 和 irrelevant-background replacement，并以首 512 token、固定 512 滑窗和完整输入为三种视图。按 source group 聚合并保存逐组预测；counterfactual 找不到 token 长度和类型匹配答案时记录跳过，不伪造样本。

```powershell
python src/diagnostics_nextbert.py --mode long --profile full_67m `
  --model-dir artifacts/mmbert-base --variant V0 `
  --qa-data data/prepared/full_67m/qa_test.jsonl `
  --wiki-background-dir artifacts/raw/full_67m/wikipedia `
  --max-source-groups 50 --eval-batch-size 4 `
  --output results/diagnostics/V0-long.json `
  --predictions-output results/diagnostics/V0-long-predictions.jsonl
```

冻结探针加载原始底座或本地适配 checkpoint，冻结 encoder 与 attention gates，只训练新 seed 初始化的任务头；QA 训练只用长度不超过 512 且 `window_type != "long"` 的 regular 行，dev 评测保留普通 overflow windows。显式指定 100 步：

```powershell
python src/diagnostics_nextbert.py --mode probe --profile full_67m `
  --model-dir artifacts/mmbert-base --variant V3 `
  --checkpoint results/formal_67m/mlm/V3/8192/V3_mlm_checkpoint.pt `
  --probe-tasks classification,qa `
  --classification-train data/prepared/full_67m/classification_train.jsonl `
  --classification-dev data/prepared/full_67m/classification_dev.jsonl `
  --qa-train data/prepared/full_67m/qa_train.jsonl `
  --qa-dev data/prepared/full_67m/qa_dev.jsonl `
  --steps 100 --seed 17 --output results/diagnostics/V3-retention.json
```

指标和输出字段以 `src/evaluate_nextbert.py`、`src/qa_metrics.py`、`src/diagnostics_nextbert.py` 及 JSON 结果中的实际 schema 为准。正式模型质量仅在正式模型与冻结数据完成后评估；短校准结果不能替代质量结果。

## 检查与参考资料

下游测评的机器可读汇总位于 `results/downstream_evaluation/summary.json`，语言分项与双语宏平均数据位于同目录 `metrics.csv`。`bm25_baseline.json` 使用已冻结的 BM25 top-20 候选原顺序计算，无模型推理；可通过 `python tools/evaluate_saved_bm25_order.py` 重算。

`nextbert-formal-results.tar.gz` 归档正式训练、六项补训与下游测评的日志、状态、逐题预测和文档向量，不包含模型 checkpoint。归档保留测评时的源码快照与 QA 汇总更正前的原始 JSON。QA 的逐题 EM/F1 未改变；`tools/correct_qa_language_metrics.py` 仅在 CPU 上按语言重新求数值平均，修正将非零 F1 当作布尔值的问题。当前 `evaluate_nextbert.py` 已直接采用数值均值。

下游测评入口为 `python tools/run_all_evaluation.py --phase tasks --gpu 7 --execute`，V0/V1 使用原任务 checkpoint，V2/V3 使用门控修复后的任务 checkpoint。已有结果目录会拒绝覆盖；复现实验应指定新的 `--output-dir`。`tools/summarize_downstream_evaluation.py` 核对任务完整性、数据来源、样本数及 checkpoint 家族，并从保存的逐题 QA 分数核验汇总值。

本地测试：

```powershell
python -m pytest -q
```

`设计与验证报告.md`、`src/run_verification.py`、`src/nextbert_lab.py` 与早期 `results/verification.json` 仅记录随机初始化缩比模块的程序验证，不是当前 mmBERT warm-start 模型的训练结果或下游分数。

- Warner et al., [ModernBERT](https://arxiv.org/abs/2412.13663)。现代编码器的长上下文训练阶段采用近似恒定的 batch token 数；本项目 131,072 token/update 是固定计算对照选择，不是文献推荐值或普适收敛门槛。
- Marone et al., [mmBERT: A Modern Multilingual Encoder](https://arxiv.org/abs/2509.06888)；固定权重、模型卡与配置：[jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base/tree/8c1f42296af5de623b0ad6d3a5ed6a5a1838d82c)。
- Ding et al., [LongNet](https://arxiv.org/abs/2307.02486)，膨胀注意力的相关工作。
- Devlin et al., [BERT](https://arxiv.org/abs/1810.04805)；Joshi et al., [SpanBERT](https://arxiv.org/abs/1907.10529)。
- 数据 revision、来源卡与许可状态见 `src/prepare_assets.py`、`src/prepare_data.py` 及其 manifests；包括 [XNLI](https://huggingface.co/datasets/facebook/xnli)、[SQuAD](https://huggingface.co/datasets/rajpurkar/squad)、[CMRC2018](https://huggingface.co/datasets/hfl/cmrc2018) 和 [MLDR](https://huggingface.co/datasets/Shitao/MLDR)。
