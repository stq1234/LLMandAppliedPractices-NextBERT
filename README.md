# NextBERT 双尺度编码器：实现与复现

本仓库实现 mmBERT-base 适配、四个注意力变体、MLM 与四类下游任务的训练入口，以及评测、运行校准和长上下文诊断。当前 704 步、每分支约 1153 万输入 token 的矩阵属于小规模架构验证；其训练量和有效 batch 尚未建立充分适配或收敛的证据，正式训练规模等待选择。矩阵入口默认仅打印命令，加入 `--execute` 才会启动。校准日志用于运行成本估计，不能代替下游任务质量结果。

代码与说明：[https://github.com/stq1234/LLMandAppliedPractices-NextBERT](https://github.com/stq1234/LLMandAppliedPractices-NextBERT)。

## 模型与训练实现

基础模型和 tokenizer 来自固定版本的 [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base/tree/8c1f42296af5de623b0ad6d3a5ed6a5a1838d82c)。实现保留 ModernBERT/mmBERT 的嵌入、QKV 与输出投影、RoPE、Pre-LayerNorm、GeGLU 和 MLM 输出权重绑定；仅注意力层连接方式按变体替换。局部注意力按 query 分块执行，避免构造长度平方的局部掩码或分数矩阵。`attention_backend=auto` 在 FlashAttention 2 可用时采用该后端，否则采用 SDPA；`sdpa` 和 `flash_attention_2` 也可显式选择。

| 变体 | 全局注意力层 | 双尺度门控层 | 适配目标 |
|---|---|---|---|
| V0 | 0、3、6、9、12、15、18、21 | 无 | token MLM |
| V1 | 0、6、12、18、21 | 无 | token MLM |
| V2 | 0、6、12、18、21 | 2、8、14、20 使用膨胀率 2；4、10、16 使用膨胀率 4 | token MLM |
| V3 | 与 V2 相同 | 与 V2 相同 | 先完成共享 token MLM 前缀，再在最后两个长度阶段平衡交替 token/span MLM |

未列为全局或双尺度的层使用半径 64 的双向局部注意力。V2/V3 的双尺度层中，前六个头保留近邻路径，后六个头以可学习 gate 融合近邻和膨胀路径。各 gate 初值为 0 logit；结构适配的前 64 次更新采用线性 ramp。校准工具可将 ramp 固定为 1，以测量稳定双路执行开销。

模型实现位于 `src/modeling_nextbert.py`，主接口为 `NextBert(model_dir, variant, attention_backend="auto")`。统一编码器提供 MLM、三分类、跨度 QA、双编码检索和组内重排接口。训练加载本地模型目录并打开 gradient checkpointing，不会解析远程模型名或自行下载权重。

### MLM 阶段与恢复

MLM 预处理数组须为 `int32 [N, length]`，训练数组只包含 `train` 行；旁边的 JSONL split sidecar 如存在会被核对。训练每个更新都使用固定 seed 的 epoch 排列和游标，完整遍历一个排列后才重新洗牌。mask ratio 为 30%，80/10/10 替换规则仅作用于普通内容 token，并排除全部 special IDs。Span MLM 按相同目标 token 数采样连续片段，平均长度约 3、最大长度 10。

`--steps` 是**本模型分支已完成更新数的绝对目标**，不是本次命令额外执行的步数。所有长度阶段的 LR scheduler 总长固定为 704，恢复时用 `--resume-from` 载入模型、AdamW、scheduler 和 RNG 状态。计划长度阶段如下：

| 长度 | 绝对更新区间 | 物理 batch | 累积步数 | 阶段完成目标 |
|---:|---:|---:|---:|---:|
| 512 | 0–255 | 16 | 2 | `--steps 256` |
| 1024 | 256–511 | 8 | 2 | `--steps 512` |
| 4096 | 512–639 | 4 | 1 | `--steps 640` |
| 8192 | 640–703 | 2 | 1 | `--steps 704` |

V0/V1/V2 全阶段使用 token MLM。V3 与 V2 共用到绝对更新 512 的 512/1024 前缀，包括优化器、scheduler 和随机状态；V2 从该 checkpoint 继续 token MLM，V3 从同一 checkpoint 用 `--shared-prefix-resume` 分支，并在 4096/8192 阶段使用 `--mode balanced`。balanced 按绝对更新交替 token/span，不改变每步 mask 目标数。其余阶段边界、路径和恢复参数由 `tools/run_training_matrix.py` 生成。

阶段中的断点默认保存完整 optimizer、scheduler 和 RNG 状态。矩阵只在最终 8192 MLM 阶段和每个下游任务的最终 checkpoint 使用 `--final-weights-only`：文件标记 `checkpoint_kind="weights_only"`，可用于评估或通过 `--init-from` 初始化后续任务，**不能用于 optimizer resume**。运行到目标步数前若使用 `--stop-after-steps`，仍保存完整可恢复 checkpoint；恢复后到达目标才写入纯权重文件。矩阵在下一阶段成功读取旧 checkpoint 后清理已消费的中间状态，并保留 V2 的共享前缀直到 V3 已读取。

示例：

```bash
python tools/run_training_matrix.py \
  --model-dir artifacts/mmbert-base \
  --data-dir data/prepared/full \
  --output-dir results/formal \
  --gpu 7
```

默认命令只打印将要执行的各阶段命令，不训练。确认数据和输出目录后，加 `--execute` 才开始矩阵。该入口显式离线运行，不自动下载模型或数据；输出目录中已有 checkpoint 时会拒绝覆盖，应选择新目录或逐阶段恢复。

### 下游训练矩阵

所有任务从对应 MLM checkpoint 初始化主干和 gate，任务头按指定 seed 新建。`--init-from` 不载入来源 checkpoint 中未训练的随机任务头；同任务断点续训才使用 `--resume-from`。所有任务执行全参数 AdamW 微调。

| 任务 | 模型 | 下游 seed | 训练步数 | 训练细节 |
|---|---|---|---:|---|
| 分类 | V0、V3 | 17、29 | 250 | 中英 XNLI，内容 token masked-mean，三分类交叉熵；物理 batch 16，累积 2 |
| QA | V0、V3 | 17、29；V1、V2 为 17 | 由数据行数计算 | 英文 SQuAD 与中文 CMRC2018； regular 与 4K augmentation 分阶段训练，context-only span loss |
| 检索 | V0、V3 | 17；V1、V2 为 17 | 125 | 中英 MLDR，逻辑 batch 固定为 32 queries；InfoNCE 负例含跨行 negatives，gradient-cache 前向与 RNG 重放反向 |
| 重排 | V0、V3 | 17 | 500 | 每组一个正例和两个负例；组内交叉熵，物理 4 组、累积 2 |

seed 17 矩阵包含 V0/V3 四项任务及 V1/V2 的 QA、检索；seed 29 额外训练 V0/V3 的分类和 QA。MLDR 训练段按确定性哈希分配短、中、长文档窗口并保留自然短文长度；BM25 只用于评测阶段的固定 top-20 候选。正式矩阵入口对检索任务显式使用 `--physical-document-batch 4`。单独调用训练入口时默认依逻辑 batch 的最大文档长度自动选 2（最大长度至少 4096）或 4，也可用同名参数覆写。

单阶段训练 CLI 总是要求显式给出 `--task`、`--variant`、本地 `--model-dir`、`--data` 和绝对 `--steps`。直接恢复某段时需要保持相同 task/seed/scheduler 配置；只有 V2→V3 的共享 MLM 前缀使用 `--shared-prefix-resume`。完整参数以以下只读帮助为准：

```bash
python src/train_nextbert.py --help
python tools/run_training_matrix.py --help
```

例如，直接训练一项分类任务：

```bash
python src/train_nextbert.py \
  --task classification --variant V0 \
  --model-dir artifacts/mmbert-base \
  --data data/prepared/full/classification_train.jsonl \
  --steps 250 --seed 17 \
  --init-from results/formal/mlm/V0/8192/V0_mlm_checkpoint.pt \
  --final-weights-only \
  --output-dir results/formal/downstream/V0/classification/17
```

## 数据、模型权重与离线部署

所有 Hub 获取和 tokenizer 处理只在可联网的本地机器执行。远程训练服务器离线运行：不要在那里调用 `src/prepare_assets.py` 或 `src/prepare_data.py`。

本地 Windows PowerShell 从仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
python src/prepare_assets.py
python src/prepare_data.py --profile full
python src/prepare_data.py --profile full --merge-manifest
python tools/validate_prepared.py data/prepared/full
python tools/archive_prepared.py
```

`prepare_assets.py` 获取固定 mmBERT 权重、tokenizer、配置和来源卡，并在 `artifacts/assets_manifest.json` 记录资产散列。`prepare_data.py` 获取冻结版本的公开数据并写入 `data/prepared/full/` 和 `artifacts/raw/full/`；默认 `pilot` profile 写入 `data/prepared/`。需要联网的命令仅在本地运行。

仓库归档只包含体积小于 96 MiB 的 prepared 文件。`tools/archive_prepared.py` 按 pilot/full profile 与任务生成 `data/archives/*.tar.gz`，并写入 `data/archive_manifest.json`。解压会按仓库相对路径恢复已归档文件：

```powershell
Get-ChildItem data/archives/*.tar.gz | ForEach-Object {
    tar -xzf $_.FullName -C .
}
```

这些归档不包含超过体积上限的大文件，因此不能单独恢复完整 full profile。尤其 full MLDR corpus、retrieval train 和 reranking 数据会被排除；manifest 记录其路径、SHA-256 与重建命令。若本地完整原始资产可用，在仓库根目录重建 full profile：

```powershell
python src/prepare_data.py --profile full
python src/prepare_data.py --profile full --merge-manifest
python tools/validate_prepared.py data/prepared/full
```

完整远程运行还需单独传输排除在 Git 归档外的大型数据与模型资产；本地生成的数据和资产应保留。不要在离线远程服务器上重建或下载数据。

远程目录及环境由项目配置指定：SSH alias `a6000`，工作目录 `~/LLM/nextbert`，conda 环境 `llm`。`requirements-linux.txt` 记录远程 Linux/Python 依赖版本；离线服务器不使用 PyPI 或 Hugging Face 下载。准备新的远程环境时，需先将匹配版本的完整 wheelhouse 与 `artifacts/nltk_data/tokenizers/punkt_tab` 从本地传入。已配置服务器可直接激活环境：

```bash
ssh a6000
cd ~/LLM/nextbert
conda activate llm
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NLTK_DATA="$PWD/artifacts/nltk_data"
```

文件上传使用共享校验工具，它逐文件传输并校验 SHA-256。先在仓库根目录构造代码/配置归档；大型权重与被 Git 归档排除的数据须通过单独资产包传输，`data/archives/*.tar.gz` 仅包含 manifest 中列出的中小型 prepared 文件：

```powershell
python tools/pack_offline.py --output artifacts/bundles/nextbert-code.tar `
    configs src tests tools environment requirements-linux.txt
python tools/pack_offline.py --output artifacts/bundles/nextbert-model.tar `
    artifacts/assets_manifest.json artifacts/mmbert-base artifacts/nltk_data
ssh a6000 "mkdir -p ~/LLM/nextbert/artifacts/bundles ~/LLM/nextbert/data/archives"

python "$env:USERPROFILE\.codex\tools\remote-upload\remote_upload.py" `
    --host a6000 --source artifacts/bundles/nextbert-code.tar `
    --target "~/LLM/nextbert/artifacts/bundles/nextbert-code.tar"
python "$env:USERPROFILE\.codex\tools\remote-upload\remote_upload.py" `
    --host a6000 --source artifacts/bundles/nextbert-code.tar.json `
    --target "~/LLM/nextbert/artifacts/bundles/nextbert-code.tar.json"
python "$env:USERPROFILE\.codex\tools\remote-upload\remote_upload.py" `
    --host a6000 --source artifacts/bundles/nextbert-model.tar `
    --target "~/LLM/nextbert/artifacts/bundles/nextbert-model.tar"
python "$env:USERPROFILE\.codex\tools\remote-upload\remote_upload.py" `
    --host a6000 --source artifacts/bundles/nextbert-model.tar.json `
    --target "~/LLM/nextbert/artifacts/bundles/nextbert-model.tar.json"
Get-ChildItem data/archives/*.tar.gz | ForEach-Object {
    python "$env:USERPROFILE\.codex\tools\remote-upload\remote_upload.py" `
      --host a6000 --source $_.FullName `
      --target "~/LLM/nextbert/data/archives/$($_.Name)"
}
```

`remote-upload` 只替换目标文件，不执行归档。目标父目录须已存在。将数据归档及大型资产包逐个传入服务器后，在远程仓库根目录安全安装代码、模型和已归档数据：

```bash
python tools/install_bundle.py artifacts/bundles/nextbert-code.tar \
  --expected-bundle artifacts/bundles/<previous-deployed-code.tar>
python tools/install_bundle.py artifacts/bundles/nextbert-model.tar
for archive in data/archives/*.tar.gz; do
  python tools/install_bundle.py "$archive"
done
```

只有在另行传入 large-file 资产包、确认目录齐全后，才运行 `python tools/validate_prepared.py data/prepared/full`。归档中缺少的大文件应来自本地已准备资产；远程服务器保持离线。

更新已有代码时，`--expected-bundle` 应指向远程已部署的上一归档；安装器只接受归档内容与当前文件相同或与该上一版本散列相符的覆盖。首次写入不存在的目标时可省略此参数。`artifacts/mmbert-base` 是大文件资产，不纳入 Git 版本控制；应通过本地获取、散列清单和离线资产包交付。

数据处理遵循“论文源 revision、读取范围与 split manifest”为准。当前冻结来源如下：

| 用途 | 数据源与 revision | 许可/来源说明 |
|---|---|---|
| 双语 MLM | [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia/tree/b04c8d1ceb2f5cd4588862100d08de323dccfbaa) | Hub 卡记载 CC-BY-SA-3.0 与 GFDL |
| 双语分类 | [XNLI](https://huggingface.co/datasets/facebook/xnli/tree/b8dd5d7af51114dbda02c0e3f6133f332186418e) | 固定 Hub 卡未声明许可；不自行补充许可 |
| 英文 QA | [SQuAD](https://huggingface.co/datasets/rajpurkar/squad/tree/7b6d24c440a36b6815f21b70d25016731768db1f) | Hub 卡记载 CC-BY-SA-4.0 |
| 中文 QA | [CMRC2018](https://huggingface.co/datasets/hfl/cmrc2018/tree/137f2c45a24275fb68f6961c4d357f46288886aa) | Hub 卡记载 CC-BY-SA-4.0 |
| 双语检索与重排 | [MLDR](https://huggingface.co/datasets/Shitao/MLDR/tree/d67138e705d963e346253a80e59676ddb418810a) | Hub 卡记载 MIT |

Wiki 训练数组 `wiki_{length}.npy` 是不跨文章、无 padding 的满长连续窗口；train/dev/test 以文章分组且由 sidecar 标记。分类 JSONL 含 `input_ids`、`label`、语言和来源 ID。QA JSONL 含全部 overflow windows、`question_id`、`context_mask`、offset、原始文本和答案；dev/test 评测先按问题聚合窗口，不以 gold 标签挑窗口。检索训练 JSONL 含 `query_ids`、`positive_ids`、单条 `negative_ids`，另保留两条带 `input_ids` 的 `negative_candidates` 用于一正二负重排组。检索评测 query 使用 `input_ids` 和 `relevant_doc_ids`；语料行使用 `id`、`language`、`input_ids`。详细字段、split 和许可记载见 [`data/README.md`](data/README.md) 与各任务 manifest。

## 运行校准、训练与评估

### 真实批次校准

`src/run_pilot.py` 只运行 CLI 要求的校准步骤，不启动正式矩阵。默认每个请求阶段有两次 warm-up 和三次 measured update，并将阶段结果逐条写入 JSONL。使用正式数据检查训练 batch、保存恢复、显存和评测/建库计时：

```bash
python tools/launch_pilot.py --gpu 7 --reserve-gib 6 \
  --record results/gpu7-preflight.json -- \
  --model-dir V0=artifacts/mmbert-base \
  --model-dir V1=artifacts/mmbert-base \
  --model-dir V2=artifacts/mmbert-base \
  --model-dir V3=artifacts/mmbert-base \
  --mlm-data-dir data/prepared/full \
  --classification-data data/prepared/full/classification_train.jsonl \
  --qa-data data/prepared/full/qa_train.jsonl \
  --retrieval-data data/prepared/full/retrieval_train.jsonl \
  --evaluation-corpus data/prepared/full/retrieval_corpus.jsonl \
  --evaluation-retrieval-queries data/prepared/full/retrieval_queries_test.jsonl \
  --evaluation-reranking-data data/prepared/full/reranking_test.jsonl \
  --output-dir results/pilot
```

对检索物理 document batch 做单独校准时，保持逻辑 query batch 不变，只传入 `--retrieval-physical-document-batch N`。例如要比较固定 8 个文档的微批次：

```bash
python tools/launch_pilot.py --gpu 7 --reserve-gib 6 \
  --record results/retrieval-doc8-preflight.json -- \
  --tasks retrieval --variants V0 \
  --model-dir V0=artifacts/mmbert-base \
  --retrieval-data data/prepared/full/retrieval_train.jsonl \
  --retrieval-physical-document-batch 8 \
  --skip-save-resume --output-dir results/pilot/retrieval-doc8
```

校准报告和逐步日志保存在指定的 `results/` 目录。计时结果用于执行计划；它们不包含正式任务的模型质量结论。正式矩阵入口 `tools/run_training_matrix.py --execute` 为检索选用 document batch 4，单阶段训练入口仍支持自动 2/4 与手动覆写。

如先前训练校准已完成而只需测建库、检索和重排评测耗时，可跳过所有训练案例：

```bash
python tools/launch_pilot.py --gpu 7 --reserve-gib 6 \
  --record results/evaluation-only-preflight.json -- \
  --evaluation-only --variants V0 V1 V2 V3 \
  --model-dir V0=artifacts/mmbert-base \
  --model-dir V1=artifacts/mmbert-base \
  --model-dir V2=artifacts/mmbert-base \
  --model-dir V3=artifacts/mmbert-base \
  --evaluation-corpus data/prepared/full/retrieval_corpus.jsonl \
  --evaluation-retrieval-queries data/prepared/full/retrieval_queries_test.jsonl \
  --evaluation-reranking-data data/prepared/full/reranking_test.jsonl \
  --output-dir results/pilot/evaluation-only
```

该模式的报告标记 `evaluation_only=true`、training coverage 为空，只记录指定的评测探针；它不会运行 MLM 或下游训练校准。

### 任务评估

训练结束后，评测都显式指定本地 checkpoint；`--model-dir` 提供原始 checkpoint 的配置、tokenizer 及基座结构。评估输出 JSON；QA 可另存原文预测 JSONL。

```bash
# 分类：accuracy 与 macro-F1
python src/evaluate_nextbert.py --task classification --variant V0 \
  --model-dir artifacts/mmbert-base \
  --checkpoint results/formal/downstream/V0/classification/17/V0_classification_checkpoint.pt \
  --data data/prepared/full/classification_test.jsonl \
  --output results/evaluation/V0-classification.json

# QA：按 question_id 聚合 overflow windows，使用官方 SQuAD/CMRC EM/F1
python src/evaluate_nextbert.py --task qa --variant V0 \
  --model-dir artifacts/mmbert-base \
  --checkpoint results/formal/downstream/V0/qa/17/V0_qa_checkpoint.pt \
  --data data/prepared/full/qa_test.jsonl \
  --output results/evaluation/V0-qa.json \
  --predictions-output results/evaluation/V0-qa-predictions.jsonl

# 双编码检索：共用固定评测语料和 query qrels
python src/evaluate_nextbert.py --task retrieval --variant V0 \
  --model-dir artifacts/mmbert-base \
  --checkpoint results/formal/downstream/V0/retrieval/17/V0_retrieval_checkpoint.pt \
  --data data/prepared/full/retrieval_queries_test.jsonl \
  --corpus data/prepared/full/retrieval_corpus.jsonl \
  --output results/evaluation/V0-retrieval.json

# 交叉编码重排：固定 BM25 top-20 候选
python src/evaluate_nextbert.py --task reranking --variant V0 \
  --model-dir artifacts/mmbert-base \
  --checkpoint results/formal/downstream/V0/reranking/17/V0_reranking_checkpoint.pt \
  --data data/prepared/full/reranking_test.jsonl \
  --corpus data/prepared/full/retrieval_corpus.jsonl --require-20-candidates \
  --output results/evaluation/V0-reranking.json
```

QA 指标实现位于 `src/qa_metrics.py`，英文按 SQuAD 规范化，中文使用 [CMRC2018 官方 v6 scorer](https://raw.githubusercontent.com/ymcui/cmrc2018/master/baseline/cmrc2018_evaluate.py) 的 punctuation、NLTK 分词和最长连续公共子串口径；离线 NLTK 数据随模型资产包提供。检索输出 nDCG@10、Recall@10/100；重排输出 nDCG@10、MRR@10 和候选 Recall@20。

以固定 mask 在 held-out Wiki 上测 MLM 保留能力：

```bash
python tools/evaluate_mlm.py --variant V0 --model-dir artifacts/mmbert-base \
  --checkpoint results/formal/mlm/V0/8192/V0_mlm_checkpoint.pt \
  --data data/prepared/full/wiki_512_test.npy \
  --output results/evaluation/V0-mlm-512-test.json --device cuda
```

性能探针可测输入前向吞吐、延迟分位数及显存：

```bash
python src/evaluate_nextbert.py --task performance --variant V0 \
  --model-dir artifacts/mmbert-base --data data/prepared/full/wiki_8192.npy \
  --length 8192 --warmup-steps 2 --measured-steps 10 \
  --output results/evaluation/V0-performance-8192.json --device cuda
```

### 长上下文诊断与冻结头保留探针

长文诊断从 QA held-out 问题和 held-out Wikipedia 背景构造控制输入；它不是自然长文 QA 测试。完整模式按 source group 聚合，比较证据位置、事实干预及截断/滑窗/完整输入视图：

```bash
python src/diagnostics_nextbert.py --mode long --profile full \
  --model-dir artifacts/mmbert-base --variant V0 \
  --tokenizer-path artifacts/mmbert-base \
  --qa-data data/prepared/full/qa_test.jsonl \
  --wiki-background-dir artifacts/raw/full/wikipedia \
  --max-source-groups 100 --output results/diagnostics/V0-long.json \
  --predictions-output results/diagnostics/V0-long-predictions.jsonl
```

冻结头探针冻结 encoder 与 attention gates，只训练新建分类/QA 头；checkpoint 可省略以测预训练参考，也可传入适配或下游 checkpoint：

```bash
python src/diagnostics_nextbert.py --mode probe --profile full \
  --model-dir artifacts/mmbert-base --variant V3 \
  --checkpoint results/formal/mlm/V3/8192/V3_mlm_checkpoint.pt \
  --tokenizer-path artifacts/mmbert-base --probe-tasks classification,qa \
  --classification-train data/prepared/full/classification_train.jsonl \
  --classification-dev data/prepared/full/classification_dev.jsonl \
  --qa-train data/prepared/full/qa_train.jsonl \
  --qa-dev data/prepared/full/qa_dev.jsonl --steps 100 --seed 17 \
  --output results/diagnostics/V3-retention.json
```

## 检查与历史记录

本地 CPU 回归检查：

```bash
python -m pytest -q
```

`tools/validate_prepared.py data/prepared/full` 检查 prepared arrays、QA answer/context offsets、文章级 split 隔离、retrieval negatives、BM25 top-20 和 manifest 散列。

`设计与验证报告.md`、`src/run_verification.py`、`src/nextbert_lab.py` 与 `results/verification.json` 保留较早阶段的架构参数计数和**随机初始化缩比 ModernBERT 模块程序验证**。该记录用于追溯当时的结构与接口检查，不是当前 mmBERT 权重适配后的训练结果、下游质量分数或性能对照。当前模型质量以实际 checkpoint 在对应 held-out 数据上运行上述评测 CLI 所产生的结果为准。校准记录位于 `results/pilot_*/`，也只解释为真实批次的预跑/计时日志。

## 参考来源

- Marone et al. [mmBERT: A Modern Multilingual Encoder](https://arxiv.org/abs/2509.06888)；固定模型卡与配置：[jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base/tree/8c1f42296af5de623b0ad6d3a5ed6a5a1838d82c)。
- Warner et al. [ModernBERT](https://arxiv.org/abs/2412.13663)；Ding et al. [LongNet](https://arxiv.org/abs/2307.02486) 的膨胀注意力背景。
- Devlin et al. [BERT](https://arxiv.org/abs/1810.04805)；Joshi et al. [SpanBERT](https://arxiv.org/abs/1907.10529)。
- XNLI、[SQuAD](https://huggingface.co/datasets/rajpurkar/squad/tree/7b6d24c440a36b6815f21b70d25016731768db1f)、[CMRC2018](https://huggingface.co/datasets/hfl/cmrc2018/tree/137f2c45a24275fb68f6961c4d357f46288886aa) 与 [MLDR](https://huggingface.co/datasets/Shitao/MLDR/tree/d67138e705d963e346253a80e59676ddb418810a) 的 revision 与许可状态见上表及 prepared manifest。
