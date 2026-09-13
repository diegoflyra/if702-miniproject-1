"""Auditoria do carregamento em tensor/GPU e do particionamento holdout/K-fold.

1) Sem augmentation, os batches são idênticos a ToTensor() + Normalize(0.5, 0.5) do torchvision.
2) Com augmentation, cada imagem é exatamente um RandomCrop(32, padding=4) (+ flip opcional) da original.
3) Holdout e K-fold são estratificados, disjuntos, cobrem o treino e são determinísticos por seed.
Uso: python tests/check_data_loader.py   (usa ./data; baixa o CIFAR-10 se necessário)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from data_loader import TensorBatchLoader, get_fold_loaders, load_cifar10, make_splits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = os.environ.get("DATA_DIR", "./data")
data = load_cifar10(data_dir, device)
checks = 0


def check(condition, message):
    global checks
    assert condition, message
    checks += 1


check(data["train_x"].shape == (50000, 3, 32, 32) and data["train_x"].dtype == torch.uint8, "train_x")
check(data["test_x"].shape == (10000, 3, 32, 32) and len(data["test_y"]) == 10000, "test_x")
check(data["train_x"].device.type == device.type, "dados no device do modelo")

# 1) Equivalência com o pipeline torchvision original
reference = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=T.Compose([
    T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
idx = np.arange(64)
x, y = next(iter(TensorBatchLoader(data["train_x"], data["train_y"], idx, batch_size=64)))
ref_x = torch.stack([reference[i][0] for i in idx]).to(device)
ref_y = torch.tensor([reference[i][1] for i in idx], device=device)
check(torch.allclose(x, ref_x, atol=1e-6) and torch.equal(y, ref_y), "normalização idêntica ao torchvision")
check(x.min() >= -1 and x.max() <= 1, "intervalo [-1, 1]")

# 2) Augmentation: cada saída é um crop (com padding 4 de zeros) da imagem original, possivelmente espelhado
loader = TensorBatchLoader(data["train_x"], data["train_y"], np.arange(16), batch_size=16, augment=True, seed=1)
aug, _ = next(iter(loader))
original = data["train_x"][:16].float().div(255).sub(0.5).div(0.5)
padded = torch.nn.functional.pad(original.add(1), (4, 4, 4, 4)).sub(1)  # pixel 0 → -1 após normalizar
n_flipped = 0
for i in range(16):
    candidates = [padded[i, :, r:r + 32, c:c + 32] for r in range(9) for c in range(9)]
    plain = any(torch.allclose(aug[i], cand, atol=1e-6) for cand in candidates)
    flipped = any(torch.allclose(aug[i], cand.flip(2), atol=1e-6) for cand in candidates)
    check(plain or flipped, f"imagem {i} não corresponde a um crop/flip válido")
    n_flipped += int(flipped and not plain)
check(aug.shape == (16, 3, 32, 32), "shape após augmentation")
aug2, _ = next(iter(TensorBatchLoader(data["train_x"], data["train_y"], np.arange(16), batch_size=16, augment=True, seed=1)))
check(torch.equal(aug, aug2), "augmentation reproduzível pela seed")

# Ordem: shuffle muda a ordem, mas cobre todos os índices uma única vez
loader = TensorBatchLoader(data["train_x"], data["train_y"], np.arange(1000), batch_size=128, shuffle=True, seed=3)
seen = torch.cat([yb for _, yb in loader])
check(len(loader) == 8 and len(seen) == 1000, "número de batches/amostras")
check(torch.equal(seen.sort().values, data["train_y"][:1000].sort().values), "shuffle cobre todas as amostras")

# 3) Particionamento
labels = data["train_y"].cpu().numpy()
holdout = make_splits(data["train_y"], k_folds=1, val_split=0.1, seed=42)
check(len(holdout) == 1, "holdout = 1 split")
tr, va = holdout[0]
check(len(va) == 5000 and len(tr) == 45000 and not set(tr) & set(va), "holdout 45k/5k disjunto")
check(np.all(np.bincount(labels[va], minlength=10) == 500), "holdout estratificado (500 por classe)")

folds = make_splits(data["train_y"], k_folds=5, seed=42)
check(len(folds) == 5, "5 folds")
all_val = np.concatenate([va for _, va in folds])
check(len(all_val) == 50000 and len(np.unique(all_val)) == 50000, "cada amostra é validação em exatamente um fold")
for k, (tr, va) in enumerate(folds, 1):
    check(len(tr) == 40000 and len(va) == 10000 and not set(tr) & set(va), f"fold {k} 40k/10k disjunto")
    check(np.all(np.bincount(labels[va], minlength=10) == 1000), f"fold {k} estratificado")
again = make_splits(data["train_y"], k_folds=5, seed=42)
check(all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again)), "folds determinísticos por seed")

sub = make_splits(data["train_y"], k_folds=3, subset_size=300, seed=42)
check(sum(len(va) for _, va in sub) == 300, "subset_size com K-fold")
loaders = get_fold_loaders(data, *sub[0], batch_size=32, eval_train=True, subset_size=300)
check(loaders[2].num_samples == 300 and loaders[3] is not None, "loaders do fold (teste reduzido e train_eval)")

print(f"Auditoria do data loader ({device}): {checks} verificações OK ({n_flipped}/16 imagens espelhadas)")
