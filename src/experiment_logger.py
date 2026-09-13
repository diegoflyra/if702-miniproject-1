"""Persistência local dos experimentos em ./outputs/{exp_name}/.

Independente do Weights & Biases: todo experimento gera, fisicamente em disco,

    parametros.json                 hiperparâmetros exatos da rodada (gravado antes do treino)
    historico_treino.csv            uma linha por (fold, época): loss, acurácia, precision/recall/f1
                                    (globais e por classe) de treino e validação, gap treino-validação
    historico_treino_agregado.csv   média e desvio padrão entre folds, por época
    resultados.json                 status, resultado de cada fold, média ± desvio da validação
                                    (melhor época de cada fold) e do teste
    melhor_modelo.pth               pesos com a menor val/loss (do melhor fold, quando K-fold)
    modelo_final.pth                pesos ao término (sempre gravado, inclusive em falhas)
    matriz_confusao_teste.csv       matriz de confusão no teste (do melhor fold)
    folds/fold_k/                   melhor_modelo.pth e matriz de cada fold (apenas quando k_folds > 1)

JSONs e .pth são gravados de forma atômica (arquivo temporário + os.replace) e
cada linha do CSV é sincronizada em disco (fsync) assim que a época termina,
de modo que uma queda no meio do treino preserva todas as épocas concluídas.
"""

import csv
import json
import math
import os
import shutil
from datetime import datetime

import numpy as np
import torch

PARAMS_FILE = "parametros.json"
HISTORY_FILE = "historico_treino.csv"
AGG_HISTORY_FILE = "historico_treino_agregado.csv"
RESULTS_FILE = "resultados.json"
BEST_WEIGHTS_FILE = "melhor_modelo.pth"
FINAL_WEIGHTS_FILE = "modelo_final.pth"
CONFUSION_FILE = "matriz_confusao_teste.csv"


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def to_builtin(value):
    """Converte tipos numpy/torch para tipos nativos serializáveis (NaN/inf viram None)."""
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, torch.Tensor):
        return to_builtin(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value is not None


def mean_std(dicts):
    """Média e desvio padrão (populacional) chave a chave, ignorando valores não numéricos/None."""
    keys = [k for k in dicts[0] if all(_is_number(d.get(k)) for d in dicts)] if dicts else []
    mean = {k: float(np.mean([d[k] for d in dicts])) for k in keys}
    std = {k: float(np.std([d[k] for d in dicts])) for k in keys}
    return mean, std


def _parse_csv_value(value):
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() and "." not in value and "e" not in value.lower() else number


class ExperimentLogger:
    def __init__(self, output_dir, exp_name, k_folds=1, overwrite=False, resume=False):
        self.dir = os.path.join(output_dir, exp_name)
        self.k_folds = k_folds
        self.history_path = os.path.join(self.dir, HISTORY_FILE)
        self._csv_fields = None
        self._rows = []  # cópia em memória do histórico, para o CSV agregado
        self.current_fold = None
        self.resumed = False

        exists = os.path.isdir(self.dir) and bool(os.listdir(self.dir))
        if exists and overwrite:
            shutil.rmtree(self.dir)
        elif exists and resume and os.path.isfile(self.path(RESULTS_FILE)):
            self._load_existing()
            return
        elif exists:
            raise FileExistsError(
                f"'{self.dir}' já existe e não está vazio. Use outro --exp_name, --resume para "
                f"completar folds pendentes ou --overwrite para apagar os resultados anteriores."
            )
        os.makedirs(self.dir, exist_ok=True)
        # Preenchido ao longo da execução e gravado em resultados.json.
        self.summary = {"status": "em_execucao", "iniciado_em": now_iso(), "k_folds": k_folds, "folds": []}

    def _load_existing(self):
        """Retoma um experimento: mantém folds concluídos e descarta os incompletos (CSV e arquivos)."""
        with open(self.path(RESULTS_FILE), encoding="utf-8") as f:
            self.summary = json.load(f)
        if self.summary.get("k_folds") != self.k_folds:
            raise ValueError(f"--k_folds={self.k_folds} difere do experimento existente "
                             f"(k_folds={self.summary.get('k_folds')}).")

        done = [f for f in self.summary.get("folds", []) if f.get("status") == "concluido"]
        dropped = [f["fold"] for f in self.summary.get("folds", []) if f.get("status") != "concluido"]
        done_ids = {f["fold"] for f in done}
        self.summary["folds"] = done
        self.summary["status"] = "em_execucao"
        self.summary.setdefault("retomado_em", []).append(now_iso())

        if os.path.isfile(self.history_path):
            with open(self.history_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self._csv_fields = list(reader.fieldnames or [])
                rows = [{k: _parse_csv_value(v) for k, v in r.items()} for r in reader]
            self._rows = [r for r in rows if r.get("fold") in done_ids]
            if len(self._rows) != len(rows):
                self._rewrite_history()
        for fold in dropped:
            if self.k_folds > 1:
                shutil.rmtree(os.path.join(self.dir, "folds", f"fold_{fold}"), ignore_errors=True)
        self.resumed = True
        if dropped:
            print(f"[resume] folds incompletos descartados e que serão refeitos se solicitados: {dropped}", flush=True)
        self.save_results()

    def _rewrite_history(self):
        tmp_path = self.history_path + ".tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields, restval="", extrasaction="ignore")
            writer.writeheader()
            for row in self._rows:
                writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.history_path)

    def completed_folds(self):
        return sorted(f["fold"] for f in self.summary["folds"] if f.get("status") == "concluido")

    # ------------------------------------------------------------------ caminhos
    def path(self, filename):
        return os.path.join(self.dir, filename)

    def fold_path(self, filename, fold):
        """Com K-fold, arquivos por fold ficam em folds/fold_k/; em holdout, na raiz."""
        if self.k_folds == 1:
            return self.path(filename)
        directory = os.path.join(self.dir, "folds", f"fold_{fold}")
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, filename)

    # ------------------------------------------------------------------ escrita
    def write_json(self, filename, data):
        final_path = self.path(filename)
        tmp_path = final_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(to_builtin(data), f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
        return final_path

    def save_params(self, params):
        return self.write_json(PARAMS_FILE, params)

    def save_results(self):
        return self.write_json(RESULTS_FILE, self.summary)

    def log_epoch(self, row):
        """Anexa uma linha (fold, época) ao CSV. As colunas são fixadas pela primeira linha."""
        row = to_builtin(row)
        if self._csv_fields is None:
            self._csv_fields = list(row.keys())
        with open(self.history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields, restval="", extrasaction="ignore")
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        self._rows.append(row)

    def save_weights(self, model, path):
        tmp_path = path + ".tmp"
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
        return path

    def save_confusion_matrix(self, matrix, class_names, path):
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["real\\predito", *class_names])
            for name, row in zip(class_names, matrix):
                writer.writerow([name, *[int(v) for v in row]])
        os.replace(tmp_path, path)

    def _copy(self, src, dst):
        if os.path.abspath(src) != os.path.abspath(dst) and os.path.isfile(src):
            tmp_path = dst + ".tmp"
            shutil.copyfile(src, tmp_path)
            os.replace(tmp_path, dst)

    # ------------------------------------------------------------------ folds
    def start_fold(self, fold, n_train, n_val):
        self.summary["folds"] = [f for f in self.summary["folds"] if f["fold"] != fold]
        self.current_fold = {
            "fold": fold, "status": "em_execucao", "n_train": n_train, "n_val": n_val,
            "epocas_concluidas": 0, "early_stopping": False,
        }
        self.summary["folds"].append(self.current_fold)
        self.save_results()
        return self.current_fold

    def finish_fold(self, fold_summary):
        """Atualiza agregados, promove o melhor fold para a raiz e grava tudo em disco."""
        fold_summary["status"] = "concluido"
        self.summary["folds"].sort(key=lambda f: f["fold"])
        done = [f for f in self.summary["folds"] if f["status"] == "concluido"]
        self.summary["folds_concluidos"] = len(done)
        self.summary["folds_concluidos_ids"] = [f["fold"] for f in done]

        val_mean, val_std = mean_std([f["melhor_val"] for f in done])
        test_mean, test_std = mean_std([f["teste"] for f in done])
        self.summary.update({
            "validacao_media": val_mean, "validacao_std": val_std,
            "teste": test_mean, "teste_std": test_std,
            "melhor_epoca_media": float(np.mean([f["melhor_epoca"] for f in done])),
        })

        candidates = [f for f in done if _is_number(f.get("melhor_valor_monitorado"))] or done
        best = min(candidates, key=lambda f: f.get("melhor_valor_monitorado") or math.inf)
        if self.summary.get("melhor_fold") != best["fold"]:
            self._copy(self.fold_path(BEST_WEIGHTS_FILE, best["fold"]), self.path(BEST_WEIGHTS_FILE))
            self._copy(self.fold_path(CONFUSION_FILE, best["fold"]), self.path(CONFUSION_FILE))
        self.summary.update({
            "melhor_fold": best["fold"], "melhor_epoca": best["melhor_epoca"],
            "melhor_valor_monitorado": best.get("melhor_valor_monitorado"),
        })

        self.save_aggregated_history()
        self.save_results()

    def save_aggregated_history(self):
        """Média e desvio padrão por época entre os folds (folds com early stopping contam até parar)."""
        if not self._rows:
            return
        numeric = [k for k in self._csv_fields if k not in ("fold", "epoch") and _is_number(self._rows[0].get(k))]
        epochs = sorted({r["epoch"] for r in self._rows if r.get("epoch") is not None})
        fields = ["epoch", "n_folds"] + [f"{k}_{s}" for k in numeric for s in ("mean", "std")]
        tmp_path = self.path(AGG_HISTORY_FILE) + ".tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for epoch in epochs:
                rows = [r for r in self._rows if r["epoch"] == epoch]
                line = [epoch, len(rows)]
                for k in numeric:
                    values = [r[k] for r in rows if _is_number(r.get(k))]
                    line += [float(np.mean(values)), float(np.std(values))] if values else ["", ""]
                writer.writerow(line)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.path(AGG_HISTORY_FILE))
