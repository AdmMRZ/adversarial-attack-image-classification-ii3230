from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class AttackMetric:
    epsilon: float
    total_samples: int
    correct_after_attack: int
    accuracy_after_attack: float
    attack_success_rate: float


def fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> torch.Tensor:
    """Membuat adversarial example dengan Fast Gradient Sign Method.

    Formula umum:
        x_adv = x + epsilon * sign(gradient_x(loss))

    Hasil akhir di-clamp ke rentang [0, 1] karena nilai piksel MNIST berada pada
    rentang tersebut.
    """
    perturbed_image = image + epsilon * data_grad.sign()
    return torch.clamp(perturbed_image, 0, 1)


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0

    progress = tqdm(train_loader, desc=f"Train epoch {epoch}", leave=False)
    for data, target in progress:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(train_loader.dataset)


def evaluate_clean(model: nn.Module, device: torch.device, test_loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return correct / total


def evaluate_fgsm(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    epsilons: list[float],
    max_examples_per_epsilon: int = 5,
) -> tuple[list[AttackMetric], list[dict]]:
    """Evaluasi model terhadap beberapa nilai epsilon FGSM.

    attack_success_rate dihitung sebagai proporsi sampel yang awalnya benar,
    tetapi menjadi salah setelah diberi perturbasi.
    """
    model.eval()
    metrics: list[AttackMetric] = []
    example_records: list[dict] = []

    for epsilon in epsilons:
        correct_after_attack = 0
        total_samples = 0
        initially_correct = 0
        successful_attacks = 0
        examples_for_this_epsilon = 0

        progress = tqdm(test_loader, desc=f"FGSM epsilon={epsilon}", leave=False)
        for data, target in progress:
            data, target = data.to(device), target.to(device)
            data.requires_grad = True

            output = model(data)
            initial_pred = output.argmax(dim=1)
            loss = F.cross_entropy(output, target)

            model.zero_grad()
            loss.backward()
            data_grad = data.grad.detach()

            perturbed_data = fgsm_attack(data, epsilon, data_grad)
            output_after_attack = model(perturbed_data)
            final_pred = output_after_attack.argmax(dim=1)

            correct_after_attack += (final_pred == target).sum().item()
            total_samples += target.size(0)

            initially_correct_mask = initial_pred == target
            attack_success_mask = initially_correct_mask & (final_pred != target)
            initially_correct += initially_correct_mask.sum().item()
            successful_attacks += attack_success_mask.sum().item()

            # Simpan contoh gambar yang berhasil menipu model.
            for i in range(data.size(0)):
                if examples_for_this_epsilon >= max_examples_per_epsilon:
                    break
                if attack_success_mask[i].item():
                    example_records.append(
                        {
                            "epsilon": float(epsilon),
                            "original": data[i].detach().cpu(),
                            "perturbed": perturbed_data[i].detach().cpu(),
                            "true_label": int(target[i].item()),
                            "before_pred": int(initial_pred[i].item()),
                            "after_pred": int(final_pred[i].item()),
                        }
                    )
                    examples_for_this_epsilon += 1

        accuracy_after_attack = correct_after_attack / total_samples
        attack_success_rate = successful_attacks / initially_correct if initially_correct else 0.0

        metrics.append(
            AttackMetric(
                epsilon=float(epsilon),
                total_samples=total_samples,
                correct_after_attack=correct_after_attack,
                accuracy_after_attack=accuracy_after_attack,
                attack_success_rate=attack_success_rate,
            )
        )

    return metrics, example_records
