"""Generate report figures from the saved parameter and architecture evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EVIDENCE_PATH = RESULTS / "verification.json"
PARAMETER_PATH = RESULTS / "parameter_counts.csv"

COLORS = {
    "embedding": "#365F91",
    "blocks": "#55A6A2",
    "head": "#D49A48",
    "global": "#D9852B",
    "local": "#347DA1",
    "ink": "#243447",
    "grid": "#D9E1E8",
    "input": "#EEF3F7",
}


def load_evidence() -> tuple[dict, dict[str, dict[str, int]]]:
    with EVIDENCE_PATH.open("r", encoding="utf-8") as file:
        evidence = json.load(file)

    counts: dict[str, dict[str, int]] = {}
    with PARAMETER_PATH.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            counts.setdefault(row["model"], {})[row["component"]] = int(row["parameters"])

    actual = counts["mmBERT-base pinned config"]
    scratch = counts["48K scratch alternative"]
    selected = evidence["selected_pretrained_route"]
    alternative = evidence["scratch_alternative"]
    actual_total = sum(value for key, value in actual.items() if key != "total unique trainable parameters")
    scratch_total = sum(value for key, value in scratch.items() if key != "total estimated parameters")

    if actual_total != selected["total_parameters"]:
        raise ValueError("mmBERT component counts do not match verification.json")
    if actual["total unique trainable parameters"] != selected["total_parameters"]:
        raise ValueError("mmBERT total row does not match verification.json")
    if scratch_total != alternative["estimated_total_parameters"]:
        raise ValueError("scratch component counts do not match verification.json")
    if scratch["total estimated parameters"] != alternative["estimated_total_parameters"]:
        raise ValueError("scratch total row does not match verification.json")
    return evidence, counts


def parameter_budget_figure(counts: dict[str, dict[str, int]]) -> None:
    mmbert = counts["mmBERT-base pinned config"]
    scratch = counts["48K scratch alternative"]
    rows = [
        (
            "mmBERT-base（配置构造）",
            mmbert["total unique trainable parameters"],
            mmbert["token embedding"],
            mmbert["transformer blocks"],
            mmbert["embedding norms and MLM head"],
        ),
        (
            "从头训练蓝图（公式估算）",
            scratch["total estimated parameters"],
            scratch["token embedding"],
            scratch["transformer blocks"],
            scratch["MLM head and norms"],
        ),
    ]

    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelcolor": COLORS["ink"],
            "axes.edgecolor": COLORS["grid"],
            "text.color": COLORS["ink"],
            "axes.unicode_minus": False,
        }
    )
    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    y_positions = [1.1, 0.0]
    components = [
        ("绑定词嵌入", 2, COLORS["embedding"]),
        ("Transformer blocks", 3, COLORS["blocks"]),
        ("MLM head 与归一化", 4, COLORS["head"]),
    ]

    for y, row in zip(y_positions, rows):
        label, total, embedding, blocks, head = row
        values = {2: embedding, 3: blocks, 4: head}
        left = 0.0
        for display_name, key, color in components:
            percent = 100 * values[key] / total
            ax.barh(
                y,
                percent,
                left=left,
                height=0.48,
                color=color,
                edgecolor="white",
                linewidth=0.8,
            )
            if percent >= 8:
                ax.text(
                    left + percent / 2,
                    y,
                    f"{percent:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10,
                    fontweight="bold",
                )
            left += percent

        head_percent = 100 * head / total
        ax.text(102, y + 0.12, f"总量 {total / 1e6:.2f} M", ha="left", va="center", fontsize=10)
        ax.text(
            102,
            y - 0.16,
            f"head + norm {head / 1e6:.2f} M（{head_percent:.1f}%）",
            ha="left",
            va="center",
            fontsize=9,
            color="#596A79",
        )

    ax.set_yticks(y_positions, [row[0] for row in rows])
    ax.set_xlim(0, 138)
    ax.set_ylim(-0.62, 1.72)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("总参数占比（%）")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=COLORS["grid"], linewidth=0.8)
    ax.yaxis.grid(False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)

    legend_items = [
        Patch(facecolor=COLORS["embedding"], label="绑定词嵌入"),
        Patch(facecolor=COLORS["blocks"], label="Transformer blocks"),
        Patch(facecolor=COLORS["head"], label="MLM head 与归一化"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.16),
        ncol=3,
        frameon=False,
        columnspacing=1.8,
    )
    fig.suptitle("约 300M 参数预算的结构分配", fontsize=16, fontweight="bold", y=0.98)
    fig.text(
        0.08,
        0.015,
        "mmBERT 为固定配置的唯一参数计数；从头训练蓝图为结构公式估算。比例按各自总量计算。",
        fontsize=8.5,
        color="#596A79",
    )
    fig.tight_layout(rect=(0.04, 0.07, 0.98, 0.88))
    fig.savefig(RESULTS / "fig_parameter_budget.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def attention_schedule_figure(evidence: dict) -> None:
    route = evidence["selected_pretrained_route"]
    config = route["config"]
    layout = route["attention_layout"]
    layers = config["num_hidden_layers"]
    every = config["global_attn_every_n_layers"]
    global_layers = list(range(0, layers, every))
    local_layers = [index for index in range(layers) if index not in global_layers]
    if global_layers != layout["global_layer_indices_zero_based"]:
        raise ValueError("derived global layer schedule does not match verification.json")
    if local_layers != layout["local_layer_indices_zero_based"]:
        raise ValueError("derived local layer schedule does not match verification.json")
    if len(global_layers) != layout["global_layer_count"] or len(local_layers) != layout["local_layer_count"]:
        raise ValueError("derived attention layer counts do not match verification.json")

    fig, ax = plt.subplots(figsize=(13.8, 4.55))
    ax.set_xlim(0, 22.7)
    ax.set_ylim(0.15, 3.65)
    ax.axis("off")
    fig.suptitle("mmBERT-base 的局部与全局注意力调度", fontsize=16, fontweight="bold", y=0.97)

    start_x = 0.2
    step = 1.02
    block_width = 0.8
    input_box = FancyBboxPatch(
        (start_x, 2.91),
        22.22,
        0.48,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=COLORS["input"],
        edgecolor=COLORS["grid"],
        linewidth=1.0,
    )
    ax.add_patch(input_box)
    ax.text(
        11.31,
        3.15,
        f"输入序列上限 {config['max_position_embeddings']:,} tokens   ·   {config['vocab_size'] // 1000}K 绑定词嵌入",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="medium",
    )

    first_center = start_x + block_width / 2
    ax.add_patch(
        FancyArrowPatch(
            (first_center, 2.89),
            (first_center, 2.53),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color="#768797",
        )
    )

    for index in range(layers):
        x = start_x + index * step
        is_global = index in global_layers
        color = COLORS["global"] if is_global else COLORS["local"]
        block = FancyBboxPatch(
            (x, 1.88),
            block_width,
            0.58,
            boxstyle="round,pad=0.025,rounding_size=0.07",
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.add_patch(block)
        ax.text(x + block_width / 2, 2.17, str(index), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        if index < layers - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + block_width + 0.015, 2.17),
                    (x + step - 0.02, 2.17),
                    arrowstyle="-|>",
                    mutation_scale=6,
                    linewidth=0.7,
                    color="#8998A6",
                )
            )

    ax.text(
        start_x,
        1.46,
        f"全局层（{len(global_layers)} 层）：{', '.join(map(str, global_layers))}；每 {every} 层一次，覆盖完整序列",
        ha="left",
        va="center",
        fontsize=9.5,
        color=COLORS["ink"],
    )
    ax.text(
        start_x,
        1.04,
        f"局部层（{len(local_layers)} 层）：窗口 {config['local_attention']} tokens；其余层按局部窗口交互",
        ha="left",
        va="center",
        fontsize=9.5,
        color=COLORS["ink"],
    )
    ax.text(
        start_x,
        0.61,
        f"RoPE theta：局部与全局均为 {config['local_rope_theta']:,}   ·   层索引从 0 开始",
        ha="left",
        va="center",
        fontsize=9.5,
        color="#596A79",
    )
    ax.text(
        start_x,
        0.27,
        "示意图依据公开模型配置与本地注意力模块布局生成。",
        ha="left",
        va="center",
        fontsize=8.3,
        color="#596A79",
    )
    fig.tight_layout(rect=(0.025, 0.04, 0.985, 0.91))
    fig.savefig(RESULTS / "fig_attention_schedule.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    evidence, counts = load_evidence()
    RESULTS.mkdir(parents=True, exist_ok=True)
    parameter_budget_figure(counts)
    attention_schedule_figure(evidence)
    print("Generated results/fig_parameter_budget.png")
    print("Generated results/fig_attention_schedule.png")


if __name__ == "__main__":
    main()
