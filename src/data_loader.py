"""Download, preparação e particionamento (holdout ou K-fold) do CIFAR-10.

O dataset inteiro é mantido como tensor uint8 no mesmo device do modelo
(~180 MB na GPU). Normalização e data augmentation são feitas por batch,
vetorizadas no próprio device, eliminando o gargalo de CPU do DataLoader
tradicional (PIL → ToTensor → Normalize imagem a imagem).

A transformação é equivalente à dos notebooks originais:
ToTensor() + Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), e o augmentation
equivale a RandomCrop(32, padding=4) + RandomHorizontalFlip().
"""

import numpy as np
import torch
import torchvision
from sklearn.model_selection import StratifiedKFold, train_test_split

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
NUM_CLASSES = len(CIFAR10_CLASSES)
INPUT_SHAPE = (3, 32, 32)

_MEAN = 0.5
_STD = 0.5
_CROP_PADDING = 4


def load_cifar10(data_dir="./data", device="cpu"):
    """Baixa (se necessário) e carrega o CIFAR-10 como tensores no device.

    Returns:
        dict com train_x/test_x (uint8, N×3×32×32) e train_y/test_y (int64).
    """
    train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)

    def to_tensors(ds):
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()  # N×H×W×C → N×C×H×W
        y = torch.as_tensor(ds.targets, dtype=torch.long)
        return x.to(device), y.to(device)

    train_x, train_y = to_tensors(train)
    test_x, test_y = to_tensors(test)
    return {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y}


def make_splits(train_y, k_folds=1, val_split=0.1, subset_size=None, seed=42):
    """Particiona o conjunto de treino em pares (índices_treino, índices_validação).

    - k_folds == 1: holdout estratificado com `val_split` para validação;
    - k_folds > 1: StratifiedKFold; cada fold usa 1/k dos dados como validação.

    Com a mesma seed, todos os experimentos usam exatamente as mesmas partições,
    o que torna as comparações entre configurações pareadas.
    `subset_size` limita o treino a N amostras aleatórias (testes rápidos).
    """
    labels = train_y.cpu().numpy()
    indices = np.arange(len(labels))
    if subset_size is not None:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.permutation(indices)[:subset_size])

    if k_folds == 1:
        train_idx, val_idx = train_test_split(
            indices, test_size=val_split, stratify=labels[indices], random_state=seed
        )
        return [(np.sort(train_idx), np.sort(val_idx))]

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    return [(indices[tr], indices[va]) for tr, va in skf.split(indices, labels[indices])]


class TensorBatchLoader:
    """Iterador de batches sobre tensores já no device (substitui o DataLoader)."""

    def __init__(self, images, labels, indices, batch_size, shuffle=False, augment=False, seed=0):
        device = images.device
        self.images = images
        self.labels = labels
        self.indices = torch.as_tensor(np.asarray(indices), dtype=torch.long, device=device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.num_samples = len(self.indices)
        # Geradores próprios: a ordem dos batches e o augmentation são reproduzíveis por seed.
        self._cpu_gen = torch.Generator().manual_seed(seed)
        self._dev_gen = torch.Generator(device=device).manual_seed(seed)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def _random_crop_flip(self, x):
        n, _, h, w = x.shape
        p = _CROP_PADDING
        padded = torch.nn.functional.pad(x, (p, p, p, p))  # zeros = pixel preto, como no RandomCrop
        dev = x.device
        top = torch.randint(0, 2 * p + 1, (n, 1), device=dev, generator=self._dev_gen)
        left = torch.randint(0, 2 * p + 1, (n, 1), device=dev, generator=self._dev_gen)
        rows = (top + torch.arange(h, device=dev))[:, :, None]  # n×h×1
        cols = (left + torch.arange(w, device=dev))[:, None, :]  # n×1×w
        batch = torch.arange(n, device=dev)[:, None, None]
        x = padded.permute(0, 2, 3, 1)[batch, rows, cols].permute(0, 3, 1, 2)
        flip = torch.rand(n, device=dev, generator=self._dev_gen) < 0.5
        return torch.where(flip[:, None, None, None], x.flip(3), x)

    def __iter__(self):
        order = self.indices
        if self.shuffle:
            perm = torch.randperm(self.num_samples, generator=self._cpu_gen).to(order.device)
            order = order[perm]
        for start in range(0, self.num_samples, self.batch_size):
            idx = order[start:start + self.batch_size]
            x = self.images[idx].float().div_(255.0)
            if self.augment:
                x = self._random_crop_flip(x)
            x = x.sub_(_MEAN).div_(_STD)
            yield x, self.labels[idx]


def get_fold_loaders(data, train_idx, val_idx, batch_size=64, augment=False, eval_train=False,
                     subset_size=None, seed=42):
    """Cria os loaders de um fold.

    Returns:
        (train_loader, val_loader, test_loader, train_eval_loader | None)
        train_eval_loader percorre o mesmo treino sem augmentation e sem shuffle,
        para medir as métricas de treino em modo eval.
    """
    train_x, train_y = data["train_x"], data["train_y"]
    test_n = len(data["test_y"]) if subset_size is None else min(subset_size, len(data["test_y"]))

    train_loader = TensorBatchLoader(train_x, train_y, train_idx, batch_size, shuffle=True, augment=augment, seed=seed)
    val_loader = TensorBatchLoader(train_x, train_y, val_idx, batch_size)
    test_loader = TensorBatchLoader(data["test_x"], data["test_y"], np.arange(test_n), batch_size)
    train_eval_loader = TensorBatchLoader(train_x, train_y, train_idx, batch_size) if eval_train else None
    return train_loader, val_loader, test_loader, train_eval_loader
