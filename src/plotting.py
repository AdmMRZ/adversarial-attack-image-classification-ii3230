from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_metric_plot(metrics_csv: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(metrics_csv)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epsilon"], df["accuracy_after_attack"], marker="o", label="Accuracy after attack")
    ax.plot(df["epsilon"], df["attack_success_rate"], marker="o", label="Attack success rate")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Score")
    ax.set_title("Pengaruh FGSM terhadap Akurasi Model")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_adversarial_examples(example_records: list[dict], output_path: str | Path) -> None:
    if not example_records:
        return

    max_items = min(len(example_records), 12)
    records = example_records[:max_items]
    fig, axes = plt.subplots(max_items, 2, figsize=(5, 2 * max_items))

    if max_items == 1:
        axes = [axes]

    for row_idx, record in enumerate(records):
        original = record["original"].squeeze().numpy()
        perturbed = record["perturbed"].squeeze().numpy()

        axes[row_idx][0].imshow(original, cmap="gray")
        axes[row_idx][0].set_title(
            f"Original | true={record['true_label']} pred={record['before_pred']}"
        )
        axes[row_idx][0].axis("off")

        axes[row_idx][1].imshow(perturbed, cmap="gray")
        axes[row_idx][1].set_title(
            f"FGSM eps={record['epsilon']} | pred={record['after_pred']}"
        )
        axes[row_idx][1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
