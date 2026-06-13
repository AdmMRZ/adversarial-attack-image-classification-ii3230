from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import optim

from data import get_mnist_loaders
from fgsm import evaluate_clean, evaluate_fgsm, train_one_epoch
from model import SimpleCNN
from plotting import save_adversarial_examples, save_metric_plot
from utils import ensure_dir, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eksperimen FGSM pada model CNN untuk klasifikasi gambar MNIST."
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--result-dir", type=str, default="results")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--max-test-samples", type=int, default=2000)
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
        help="Daftar epsilon untuk FGSM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    result_dir = ensure_dir(args.result_dir)
    model_dir = ensure_dir(args.model_dir)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = get_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, device, train_loader, optimizer, epoch)
        train_losses.append(train_loss)
        print(f"Epoch {epoch} train loss: {train_loss:.4f}")

    model_path = model_dir / "mnist_cnn.pt"
    torch.save(model.state_dict(), model_path)

    clean_accuracy = evaluate_clean(model, device, test_loader)
    print(f"Clean accuracy: {clean_accuracy:.4f}")

    attack_metrics, example_records = evaluate_fgsm(
        model=model,
        device=device,
        test_loader=test_loader,
        epsilons=args.epsilons,
    )

    rows = []
    for item in attack_metrics:
        rows.append(
            {
                "epsilon": item.epsilon,
                "total_samples": item.total_samples,
                "correct_after_attack": item.correct_after_attack,
                "accuracy_after_attack": item.accuracy_after_attack,
                "attack_success_rate": item.attack_success_rate,
            }
        )

    df = pd.DataFrame(rows)
    metrics_csv = result_dir / "metrics.csv"
    metrics_json = result_dir / "metrics.json"
    df.to_csv(metrics_csv, index=False)

    save_json(
        {
            "clean_accuracy": clean_accuracy,
            "train_losses": train_losses,
            "epsilons": args.epsilons,
            "metrics": rows,
        },
        metrics_json,
    )

    save_adversarial_examples(example_records, result_dir / "fgsm_examples.png")
    save_metric_plot(metrics_csv, result_dir / "fgsm_metrics_plot.png")

    print("\nSelesai. Output utama:")
    print(f"- {metrics_csv}")
    print(f"- {metrics_json}")
    print(f"- {result_dir / 'fgsm_examples.png'}")
    print(f"- {result_dir / 'fgsm_metrics_plot.png'}")
    print(f"- {model_path}")


if __name__ == "__main__":
    main()
