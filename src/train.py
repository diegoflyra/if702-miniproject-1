"""Loop de treinamento com persistência local (CSV/JSON/.pth) e logging no Weights & Biases.

A ordem é sempre: grava em disco primeiro, depois envia ao wandb. Falhas do wandb
nunca interrompem o treino nem a gravação local.
"""

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn import metrics

from experiment_logger import BEST_WEIGHTS_FILE, CONFUSION_FILE, now_iso


class SoftmaxMSELoss(nn.Module):
    """MSE entre softmax(logits) e o rótulo one-hot (formulação clássica de MSE para classificação)."""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.mse = nn.MSELoss()

    def forward(self, logits, targets):
        one_hot = nn.functional.one_hot(targets, self.num_classes).to(logits.dtype)
        return self.mse(torch.softmax(logits, dim=1), one_hot)


def get_loss_fn(name, num_classes):
    name = name.lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "mse":
        return SoftmaxMSELoss(num_classes)
    raise ValueError(f"Função de erro '{name}' não suportada. Use 'cross_entropy' ou 'mse'.")


def get_optimizer(name, params, lr, weight_decay=0.0, momentum=0.9):
    """weight_decay é a penalidade L2 aplicada pelo próprio otimizador."""
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Otimizador '{name}' não suportado.")


def compute_metrics(targets, predictions, class_names):
    """Métricas globais e por classe (precision, recall, f1) via scikit-learn."""
    labels = list(range(len(class_names)))
    scores = {
        "accuracy": metrics.accuracy_score(targets, predictions),
        "balanced_accuracy": metrics.balanced_accuracy_score(targets, predictions),
        "precision_macro": metrics.precision_score(targets, predictions, labels=labels, average="macro", zero_division=0),
        "recall_macro": metrics.recall_score(targets, predictions, labels=labels, average="macro", zero_division=0),
        "f1_macro": metrics.f1_score(targets, predictions, labels=labels, average="macro", zero_division=0),
        "precision_weighted": metrics.precision_score(targets, predictions, labels=labels, average="weighted", zero_division=0),
        "recall_weighted": metrics.recall_score(targets, predictions, labels=labels, average="weighted", zero_division=0),
        "f1_weighted": metrics.f1_score(targets, predictions, labels=labels, average="weighted", zero_division=0),
    }
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        targets, predictions, labels=labels, average=None, zero_division=0
    )
    for i, name in enumerate(class_names):
        scores[f"precision_per_class/{name}"] = precision[i]
        scores[f"recall_per_class/{name}"] = recall[i]
        scores[f"f1_per_class/{name}"] = f1[i]
    return scores


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Retorna (loss média, targets, predições) acumulados durante a época."""
    model.train()
    total_loss, total = 0.0, 0
    all_preds, all_targets = [], []
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        all_preds.append(outputs.detach().argmax(dim=1).cpu().numpy())
        all_targets.append(labels.cpu().numpy())
    return total_loss / total, np.concatenate(all_targets), np.concatenate(all_preds)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Retorna (loss média, targets, predições)."""
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_targets = [], []
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        total += labels.size(0)
        all_preds.append(outputs.argmax(dim=1).cpu().numpy())
        all_targets.append(labels.cpu().numpy())
    return total_loss / total, np.concatenate(all_targets), np.concatenate(all_preds)


def _prefixed(prefix, scores):
    return {f"{prefix}/{k}": v for k, v in scores.items()}


def safe_wandb(fn, *args, **kwargs):
    """Executa uma chamada do wandb sem deixar que uma falha derrube o experimento."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] aviso: {type(exc).__name__}: {exc} (resultados locais não foram afetados)", flush=True)
        return None


def fit(model, loaders, config, device, class_names, logger, fold_summary):
    """Treina UM fold, persiste cada época em disco e avalia no conjunto de teste.

    `loaders` = (train, val, test, train_eval | None).
    `config` precisa conter: epochs, lr, optimizer, weight_decay, momentum, patience, loss_fn.
    Early stopping monitora val/loss e os melhores pesos (menor val/loss) são
    restaurados antes da avaliação no teste.

    Métricas de treino: por padrão são acumuladas durante a época (modo train,
    com dropout ativo e pesos mudando a cada batch). Se o loader train_eval for
    fornecido, são recalculadas ao fim da época em modo eval, o que torna o gap
    treino-validação diretamente comparável entre redes com e sem dropout.
    """
    train_loader, val_loader, test_loader, train_eval_loader = loaders
    fold = fold_summary["fold"]
    fold_tag = f"[fold {fold}/{logger.k_folds}] " if logger.k_folds > 1 else ""
    best_weights_path = logger.fold_path(BEST_WEIGHTS_FILE, fold)

    criterion = get_loss_fn(config.loss_fn, len(class_names))
    optimizer = get_optimizer(
        config.optimizer, model.parameters(), config.lr,
        weight_decay=config.weight_decay, momentum=config.momentum,
    )
    monitor = "val/loss"
    logger.summary["metrica_monitorada"] = monitor

    best_loss = float("inf")
    best_state = None
    best_row = None
    best_epoch = 0
    patience_counter = 0
    row = None

    for epoch in range(1, config.epochs + 1):
        start = time.time()
        train_loss, train_targets, train_preds = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if train_eval_loader is not None:
            train_loss, train_targets, train_preds = evaluate(model, train_eval_loader, criterion, device)
        val_loss, val_targets, val_preds = evaluate(model, val_loader, criterion, device)

        row = {
            "fold": fold,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_metrics_mode": "eval" if train_eval_loader is not None else "running",
            "train/loss": train_loss,
            **_prefixed("train", compute_metrics(train_targets, train_preds, class_names)),
            "val/loss": val_loss,
            **_prefixed("val", compute_metrics(val_targets, val_preds, class_names)),
        }
        # Gap de generalização: positivo e crescente indica overfitting.
        row["gap/loss"] = val_loss - train_loss
        row["gap/accuracy"] = row["train/accuracy"] - row["val/accuracy"]

        diverged = not (math.isfinite(train_loss) and math.isfinite(val_loss))
        is_best = not diverged and val_loss < best_loss
        if is_best:
            best_loss, best_epoch, patience_counter = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            logger.save_weights(model, best_weights_path)
        else:
            patience_counter += 1

        row["is_best_epoch"] = int(is_best)
        row["epoch_time_s"] = time.time() - start
        if is_best:
            best_row = row

        # 1) disco  2) wandb
        logger.log_epoch(row)
        fold_summary.update({"epocas_concluidas": epoch, "melhor_epoca": best_epoch,
                             "melhor_valor_monitorado": best_loss if best_state is not None else None})
        logger.save_results()
        safe_wandb(wandb.log, {k: v for k, v in row.items() if not isinstance(v, str)}, step=epoch)

        print(f"{fold_tag}Epoch {epoch}/{config.epochs} | train_loss {train_loss:.4f} | "
              f"train_acc {row['train/accuracy']:.4f} | val_loss {val_loss:.4f} | val_acc {row['val/accuracy']:.4f} | "
              f"{row['epoch_time_s']:.1f}s" + (" *" if is_best else ""), flush=True)

        if diverged:
            # Loss NaN/inf não se recupera; a época fica registrada no CSV e o motivo em resultados.json.
            fold_summary["divergiu_na_epoca"] = epoch
            logger.save_results()
            print(f"{fold_tag}Treino divergiu na época {epoch} (loss não finita). Interrompendo.", flush=True)
            break
        if config.patience > 0 and patience_counter >= config.patience:
            fold_summary["early_stopping"] = True
            print(f"{fold_tag}Early stopping na época {epoch} (melhor época: {best_epoch}).", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        # Nenhuma época com loss finita: ainda assim melhor_modelo.pth precisa existir.
        logger.save_weights(model, best_weights_path)
        best_row = row
        fold_summary["aviso"] = "Nenhuma época com val/loss finita; melhor_modelo.pth contém os pesos finais."

    test_loss, test_targets, test_preds = evaluate(model, test_loader, criterion, device)
    test_scores = compute_metrics(test_targets, test_preds, class_names)
    test_log = {"test/loss": test_loss, **_prefixed("test", test_scores)}

    logger.save_confusion_matrix(
        metrics.confusion_matrix(test_targets, test_preds, labels=list(range(len(class_names)))),
        class_names, logger.fold_path(CONFUSION_FILE, fold),
    )
    fold_summary.update({
        "melhor_val": {k: v for k, v in best_row.items() if k.startswith(("train/", "val/", "gap/"))},
        "teste": test_log,
        "avaliado_em": now_iso(),
    })
    logger.save_results()

    safe_wandb(lambda: wandb.log({
        **test_log,
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=test_targets.tolist(), preds=test_preds.tolist(), class_names=list(class_names)
        ),
    }))
    if wandb.run is not None:
        safe_wandb(wandb.run.summary.update, {**test_log, "best_epoch": best_epoch})

    print(f"\n{fold_tag}Teste (pesos da melhor época {best_epoch}):")
    for key in ("loss", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"):
        value = test_loss if key == "loss" else test_scores[key]
        print(f"  {key:<18} {value:.4f}")

    return model, test_log
