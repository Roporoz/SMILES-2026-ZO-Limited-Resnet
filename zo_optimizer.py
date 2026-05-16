from __future__ import annotations

from typing import Callable
import os
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA


class ZeroOrderOptimizer:


    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.02,
        eps: float = 0.5,
        perturbation_mode: str = "gaussian",
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.perturbation_mode = perturbation_mode

        self.layer_names = ["fc.weight_pca_hard_low_rank"]

        self.hard_class_ids = [
            55, 93, 72, 32, 35, 80, 50, 3, 74, 14,
            7, 45, 26, 11, 59, 65, 67, 77, 84, 47,
            44, 4, 27, 64, 98, 79, 51, 63, 66, 96,
        ]

        pca_extra_class_ids = [
            73, 52, 46, 36, 18, 24, 13, 81, 90, 91, 95
        ]
        self.pca_class_ids = sorted(set(self.hard_class_ids + pca_extra_class_ids))

        self.rank = 32
        self.relative_correction_scale = 0.10

        self.samples_per_class = 40
        self.seed = 42
        self.data_root = os.environ.get("DATA_DIR", "./data")

        # Adam-like ZO
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.adam_eps = 1e-8
        self.t = 0
        self.step_count = 0

        self.max_grad_norm = 10.0
        self.max_update_norm = 0.08
        self.accept_only_if_improves = True

        self._init_state()

    def _init_state(self) -> None:
        fc = self.model.fc
        if not isinstance(fc, nn.Linear):
            raise TypeError("Expected model.fc to be nn.Linear")

        device = fc.weight.device
        dtype = fc.weight.dtype

        out_features, in_features = fc.weight.shape
        if in_features != 512:
            raise ValueError(f"Expected 512 input features, got {in_features}")
        if self.rank > in_features:
            raise ValueError(f"rank={self.rank} cannot exceed {in_features}")

        self.base_weight = fc.weight.detach().clone()
        self.base_bias = fc.bias.detach().clone() if fc.bias is not None else None

        self.hard_ids = torch.tensor(
            self.hard_class_ids,
            device=device,
            dtype=torch.long,
        )

        hard_base = self.base_weight.index_select(0, self.hard_ids)
        self.base_norm = hard_base.norm(dim=1).mean().clamp_min(1e-12)
        self.correction_scale = self.relative_correction_scale * self.base_norm

        self.B = self._build_pca_basis(
            rank=self.rank,
            device=device,
            dtype=dtype,
        )

        self.A = torch.zeros(
            len(self.hard_class_ids),
            self.rank,
            device=device,
            dtype=dtype,
        )

        self.m = torch.zeros_like(self.A)
        self.v = torch.zeros_like(self.A)

        self._apply_current_weight()

    def _build_pca_basis(
        self,
        rank: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        transform = weights.transforms()

        train_dataset = torchvision.datasets.CIFAR100(
            root=self.data_root,
            train=True,
            download=True,
            transform=transform,
        )

        targets = np.array(train_dataset.targets)
        rng = np.random.default_rng(self.seed)

        selected_indices = []
        selected_labels = []

        # Тот же balanced subset 40/class, что и в head_init.py
        for class_id in range(100):
            class_indices = np.where(targets == class_id)[0]
            chosen = rng.choice(
                class_indices,
                size=self.samples_per_class,
                replace=False,
            )
            selected_indices.extend(chosen.tolist())
            selected_labels.extend([class_id] * self.samples_per_class)

        selected_indices = np.array(selected_indices)
        selected_labels = np.array(selected_labels)

        # Для PCA оставляем только hard classes + confusers
        mask = np.isin(selected_labels, self.pca_class_ids)
        pca_indices = selected_indices[mask]

        subset = Subset(train_dataset, pca_indices.tolist())

        loader = DataLoader(
            subset,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        feature_model = models.resnet18(weights=weights)
        feature_model.fc = nn.Identity()
        feature_model.eval()
        feature_model.to(device)

        X_list = []

        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                features = feature_model(images).cpu().numpy()
                X_list.append(features)

        X = np.concatenate(X_list, axis=0)

        pca = PCA(
            n_components=rank,
            whiten=False,
            random_state=self.seed,
        )
        pca.fit(X)

        B_np = pca.components_.astype(np.float32)

        del feature_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return torch.tensor(B_np, device=device, dtype=dtype).contiguous()

    def _current_weight(self) -> torch.Tensor:
        current = self.base_weight.clone()

        correction = self.correction_scale * (self.A @ self.B)
        hard_base_rows = current.index_select(0, self.hard_ids)
        hard_new_rows = hard_base_rows + correction

        current.index_copy_(0, self.hard_ids, hard_new_rows)
        return current

    def _apply_current_weight(self) -> None:
        with torch.no_grad():
            self.model.fc.weight.copy_(self._current_weight())

            if self.base_bias is not None and self.model.fc.bias is not None:
                self.model.fc.bias.copy_(self.base_bias)

    def _loss_value(self, loss_fn: Callable[[], float]) -> float:
        with torch.no_grad():
            loss = loss_fn()

        if isinstance(loss, torch.Tensor):
            return float(loss.detach().cpu().item())
        return float(loss)

    def _sample_direction(self) -> torch.Tensor:
        if self.perturbation_mode == "uniform":
            direction = torch.rand_like(self.A) * 2.0 - 1.0
        else:
            direction = torch.randn_like(self.A)

        norm = direction.norm()
        if norm > 0:
            direction = direction / norm

        return direction

    def _estimate_grad(self, loss_fn: Callable[[], float]) -> torch.Tensor:
        direction = self._sample_direction()

        with torch.no_grad():
            self.A.add_(self.eps * direction)
            self._apply_current_weight()
            f_plus = self._loss_value(loss_fn)

            self.A.sub_(2.0 * self.eps * direction)
            self._apply_current_weight()
            f_minus = self._loss_value(loss_fn)

            self.A.add_(self.eps * direction)
            self._apply_current_weight()

        dim = self.A.numel()
        grad = dim * ((f_plus - f_minus) / (2.0 * self.eps)) * direction

        grad_norm = grad.norm()
        if grad_norm > self.max_grad_norm:
            grad = grad * (self.max_grad_norm / (grad_norm + 1e-12))

        return grad

    def _adam_update(self, grad: torch.Tensor) -> torch.Tensor:
        self.t += 1

        self.m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
        self.v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)

        update = self.lr * m_hat / (v_hat.sqrt() + self.adam_eps)

        update_norm = update.norm()
        if update_norm > self.max_update_norm:
            update = update * (self.max_update_norm / (update_norm + 1e-12))

        return update

    def step(self, loss_fn: Callable[[], float]) -> float:
        self.step_count += 1

        self._apply_current_weight()
        loss_before = self._loss_value(loss_fn)

        old_A = self.A.detach().clone()
        old_m = self.m.detach().clone()
        old_v = self.v.detach().clone()
        old_t = self.t

        grad = self._estimate_grad(loss_fn)
        update = self._adam_update(grad)

        with torch.no_grad():
            self.A.sub_(update)
            self._apply_current_weight()

        loss_after = self._loss_value(loss_fn)

        if self.accept_only_if_improves and loss_after > loss_before:
            with torch.no_grad():
                self.A.copy_(old_A)
                self.m.copy_(old_m)
                self.v.copy_(old_v)
                self.t = old_t
                self._apply_current_weight()
            accepted = False
        else:
            accepted = True

        if self.step_count % 8 == 0:
            with torch.no_grad():
                current_w = self._current_weight()
                delta = current_w - self.base_weight

                rel_delta = delta.norm() / (self.base_weight.norm() + 1e-12)

                hard_delta = delta.index_select(0, self.hard_ids)
                hard_base = self.base_weight.index_select(0, self.hard_ids)

                hard_row_rel_delta = hard_delta.norm(dim=1).mean() / (
                    hard_base.norm(dim=1).mean() + 1e-12
                )

                max_hard_row_rel_delta = (
                    hard_delta.norm(dim=1) /
                    (hard_base.norm(dim=1) + 1e-12)
                ).max()

                print(
                    f"[HardPCALowRankZO] step={self.step_count:03d} "
                    f"loss_before={loss_before:.4f} "
                    f"loss_after={loss_after:.4f} "
                    f"accepted={accepted} "
                    f"rel_delta={rel_delta.item():.4f} "
                    f"hard_row_rel_delta={hard_row_rel_delta.item():.4f} "
                    f"max_hard_row_rel_delta={max_hard_row_rel_delta.item():.4f} "
                    f"A_norm={self.A.norm().item():.4f}"
                )
        return float(loss_before)
