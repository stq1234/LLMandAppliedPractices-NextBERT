# NextBERT 架构预算与程序验证

本目录记录面向中英文长文本理解的 8K 双向编码器设计、约 300M 参数预算及本地程序验证。架构设计以 mmBERT-base 公开配置为适配参照；本地结构计数按该配置构造，功能链路采用随机初始化缩比模型验证。

本次实验的代码及相关说明可在该仓库中获取：[LLMandAppliedPractices-NextBERT](https://github.com/stq1234/LLMandAppliedPractices-NextBERT)。

mmBERT-base 固定配置 revision：8c1f42296af5de623b0ad6d3a5ed6a5a1838d82c。模型与 tokenizer 元数据记录在 configs/mmbert_adaptation.json。

## 复现程序检查与图表

程序使用 Python 3.12、PyTorch 2.12.0、Transformers 4.51.3、psutil 5.9.0 和 Matplotlib 3.9.2。以下命令从仓库根目录的 Windows PowerShell 执行：

1. 安装依赖：<code>python -m pip install -r ".\作业3.2_NextBERT设计验证\requirements.txt"</code>
2. 运行 CPU 程序检查：<code>python ".\作业3.2_NextBERT设计验证\src\run_verification.py"</code>
3. 如需略过 8K CPU 探针：<code>python ".\作业3.2_NextBERT设计验证\src\run_verification.py" --skip-8k</code>
4. 根据保存的参数统计和架构配置生成报告图表：<code>python ".\作业3.2_NextBERT设计验证\src\plot_report_figures.py"</code>

程序将完整结果写入 results/verification.json，并将参数计数写入 results/parameter_counts.csv。绘图脚本根据这两项记录生成参数分配图和 22 层注意力布局图。报告中的实验环境和数值以本地验证记录为准。

## 验证记录

mmBERT 参数计数来自 Transformers 模型类按固定配置在 meta device 上构造后的唯一参数统计；输入与输出绑定词嵌入仅计一次。从头训练蓝图根据层参数公式估算。

主要功能检查使用 4 层、隐藏维度 64、4 个注意力头和 512 词表的随机初始化模型，参数量为 177,280；输入为 batch 3、长度 18 的合成序列，含 BOS、SEP 与右侧 padding。CPU 线程数为 2，优化器为 AdamW，学习率 0.001。数据批次固定 seed 1701，token MLM、span MLM 和特殊 token 探针分别使用 seed 101、202、303 与 404。

掩码候选位置由 attention mask 与特殊 token 排除规则定义；候选内容位置约 30% 被选为 MLM 目标。动态 token MLM 无放回选取离散位置，span MLM 采样长度 1–4 的连续片段并填足目标数。已选目标按 80% MASK、10% 随机词表替换、10% 保留原值扰动，其他标签设为 ignore index。预测目标位置排除 special IDs {0,1,2,3,4,106,107} 与 padding；随机替换值由合成词表的 ID 区间 [10,512) 抽取。

下游头输出为分类 [3,3]、抽取 [3,18,2]、检索相似度矩阵 [3,3] 和重排分数 [3]。四项损失均为有限值，组合更新覆盖共享编码器与分类、抽取、投影及重排头。检索头采用两次独立 dropout 视图、共享投影、L2 归一化及温度 0.07 的批内对比交叉熵；重排头将合成文本对拼接编码并使用二元交叉熵。程序还核对了 eval 重复输出、padding masked mean pooling 不变性、有效 span loss 对 padding logits 的屏蔽，以及七类 special ID 不作为 MLM 预测目标。

8K CPU 探针使用 3 层、隐藏维度 16、1 个注意力头、15,616 参数的 SDPA 模型，输入长度 8,192，完成一次前向耗时 0.688 秒。验证进程 peak working set 记录为 1,984.4 MiB，测量范围包含解释器和初始化等整个进程开销。

## 目录中的设计记录

- configs/mmbert_adaptation.json：模型架构参数及特殊 token 元数据。
- configs/scratch_blueprint.json：从头训练候选的结构参数与公式预算。
- configs/experiment_plan.json：后续训练和下游评测的拟议对照方案；其中计划不作为本报告实测结果。
- data/manifest.schema.json：冻结预训练语料清单时使用的校验结构。
- src/nextbert_lab.py 与 src/run_verification.py：参数统计、功能检查和仓库根目录运行入口。
- src/plot_report_figures.py：读取本地计数生成报告图表。
- 设计与验证报告.md：研究设计、参数分配、程序验证结果及方法边界。
