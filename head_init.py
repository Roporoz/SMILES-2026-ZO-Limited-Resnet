import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.utils.data import DataLoader, Subset
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def init_last_layer(layer: nn.Linear) -> None:
    """
    Blend40 head:
      - 40 images per class
      - LDA + scaled multinomial Logistic Regression
      - W = alpha_lda * W_LDA + (1 - alpha_lda) * W_LogReg
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples_per_class = 80
    batch_size = 256
    seed = 42

    lda_shrinkage = 0.3
    logreg_C = 0.020
    alpha_lda = 0.10
    alpha_logreg = 1.0 - alpha_lda

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    train_dataset = torchvision.datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    targets = np.array(train_dataset.targets)
    rng = np.random.default_rng(seed)

    selected_indices = []
    for class_id in range(100):
        class_indices = np.where(targets == class_id)[0]
        chosen = rng.choice(
            class_indices,
            size=samples_per_class,
            replace=False,
        )
        selected_indices.extend(chosen.tolist())

    subset = Subset(train_dataset, selected_indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    feature_model = models.resnet18(weights=weights)
    feature_model.fc = nn.Identity()
    feature_model.eval()
    feature_model.to(device)

    X_list = []
    y_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = feature_model(images).cpu().numpy()
            X_list.append(features)
            y_list.append(labels.numpy())

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # ----- LDA -----
    lda = LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage=lda_shrinkage,
    )
    lda.fit(X, y)

    lda_weight = torch.tensor(lda.coef_, dtype=torch.float32)
    lda_bias = torch.tensor(lda.intercept_, dtype=torch.float32)

    # ----- Logistic Regression -----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logreg = LogisticRegression(
        C=logreg_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        n_jobs=-1,
    )
    logreg.fit(X_scaled, y)

    coef_scaled = logreg.coef_
    intercept_scaled = logreg.intercept_

    logreg_coef = coef_scaled / scaler.scale_[None, :]
    logreg_intercept = intercept_scaled - np.sum(
        coef_scaled * scaler.mean_[None, :] / scaler.scale_[None, :],
        axis=1,
    )

    logreg_weight = torch.tensor(logreg_coef, dtype=torch.float32)
    logreg_bias = torch.tensor(logreg_intercept, dtype=torch.float32)

    # ----- Blend -----
    final_weight = alpha_lda * lda_weight + alpha_logreg * logreg_weight
    final_bias = alpha_lda * lda_bias + alpha_logreg * logreg_bias

    with torch.no_grad():
        layer.weight.copy_(final_weight.to(layer.weight.device))
        layer.bias.copy_(final_bias.to(layer.bias.device))

    del feature_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
