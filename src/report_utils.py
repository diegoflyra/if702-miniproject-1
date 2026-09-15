"""Análise dos experimentos a partir EXCLUSIVAMENTE dos arquivos locais em outputs/.

Uso nos notebooks:
    import sys; sys.path.insert(0, "src")
    import report_utils as rep
    rep.grid_ranking("mlp_b1_topologia")                       # ranking do bloco pela validação (sem teste)
    rep.heatmap("mlp_b1_topologia", row="mlp_layers", col="mlp_neurons")
    rep.show_champion("mlp_b1_topologia"); rep.plot_finalists("mlp_b1_topologia")
    rep.final_report(["mlp_b0_referencia", "mlp_b1_topologia"])  # teste revelado só para os campeões
    rep.plot_curves("mlp_b1_topologia__*")                     # treino (tracejado) e validação (sólido)

Com K-fold, as métricas são média ± desvio padrão entre folds (validação na melhor
época de cada fold; teste com os pesos da melhor época de cada fold) e as curvas
mostram a média por época com uma faixa de ±1 desvio padrão.

Tabelas (.csv) e figuras (.png) também são gravadas em {output_dir}/_relatorio/
para uso direto no PPT.
"""

import glob
import json
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

from data_loader import CIFAR10_CLASSES

# Parâmetros operacionais que não descrevem o experimento (ficam fora da tabela de ablação).
_NON_HPARAMS = {
    "exp_name", "output_dir", "overwrite", "project", "entity", "tags", "notes", "wandb_mode", "device",
    "data_dir", "iniciado_em", "arquitetura", "gpu", "python", "torch", "torchvision",
    "scikit_learn", "wandb", "git_commit", "comando", "validacao", "n_train_por_fold", "n_val_por_fold", "n_test",
    "subset_size",
}
_SUMMARY_METRICS = ("train/loss", "train/accuracy", "val/loss", "val/accuracy", "val/f1_macro", "gap/loss", "gap/accuracy")
_TEST_METRICS = ("test/loss", "test/accuracy", "test/balanced_accuracy", "test/precision_macro",
                 "test/recall_macro", "test/f1_macro")


def output_dir():
    return os.environ.get("EXP_OUTPUT_DIR", "./outputs")


def report_dir():
    path = os.path.join(output_dir(), "_relatorio")
    os.makedirs(path, exist_ok=True)
    return path


def _patterns(pattern):
    return [pattern] if isinstance(pattern, str) else list(pattern)


def _slug(pattern):
    parts = [p.replace("*", "").replace("?", "").strip("_") for p in _patterns(pattern)]
    return "+".join(p for p in parts if p) or "todos"


def experiment_dirs(pattern="*"):
    """Pastas de experimento que casam com o(s) padrão(ões), ex.: 'cnn_f3_*' ou ['cnn_f2_3blocos', 'cnn_f3_*'].

    Mantém a ordem dos padrões (útil para colocar a referência primeiro) e ignora
    pastas internas como _relatorio.
    """
    dirs = []
    for pat in _patterns(pattern):
        for d in sorted(glob.glob(os.path.join(output_dir(), pat))):
            name = os.path.basename(d)
            if d not in dirs and not name.startswith("_") and os.path.isfile(os.path.join(d, "parametros.json")):
                dirs.append(d)
    return dirs


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_history(pattern="*", aggregated=True):
    """{exp_name: DataFrame}. aggregated=True lê historico_treino_agregado.csv (média/desvio por época);
    aggregated=False lê historico_treino.csv (uma linha por fold e época)."""
    filename = "historico_treino_agregado.csv" if aggregated else "historico_treino.csv"
    histories = {}
    for d in experiment_dirs(pattern):
        path = os.path.join(d, filename)
        if os.path.isfile(path):
            histories[os.path.basename(d)] = pd.read_csv(path)
    return histories


def load_summary(pattern="*"):
    """Uma linha por experimento: hiperparâmetros, validação (melhor época) e teste, em média ± desvio entre folds."""
    rows = []
    for d in experiment_dirs(pattern):
        name = os.path.basename(d)
        params = _read_json(os.path.join(d, "parametros.json"))
        results_path = os.path.join(d, "resultados.json")
        results = _read_json(results_path) if os.path.isfile(results_path) else {}

        row = {"exp_name": name, "status": results.get("status", "sem_resultados")}
        row.update({k: v for k, v in params.items() if k not in _NON_HPARAMS})
        row["folds_concluidos"] = results.get("folds_concluidos", 0)
        row["melhor_epoca_media"] = results.get("melhor_epoca_media")

        val_mean, val_std = results.get("validacao_media", {}), results.get("validacao_std", {})
        for key in _SUMMARY_METRICS:
            row[f"melhor_{key}"] = val_mean.get(key)
            row[f"melhor_{key}_std"] = val_std.get(key)
        test_mean, test_std = results.get("teste", {}), results.get("teste_std", {})
        for key in _TEST_METRICS:
            row[key] = test_mean.get(key)
            row[f"{key}_std"] = test_std.get(key)

        agg_path = os.path.join(d, "historico_treino_agregado.csv")
        if os.path.isfile(agg_path):
            agg = pd.read_csv(agg_path)
            if "gap/accuracy_mean" in agg:
                row["max_gap/accuracy"] = agg["gap/accuracy_mean"].max()
            if "epoch_time_s_mean" in agg:
                row["tempo_medio_epoca_s"] = agg["epoch_time_s_mean"].mean()
        hist_path = os.path.join(d, "historico_treino.csv")
        if os.path.isfile(hist_path):
            row["tempo_total_treino_min"] = pd.read_csv(hist_path, usecols=["epoch_time_s"])["epoch_time_s"].sum() / 60
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_table(pattern="*", save=True, include_test=False):
    """Resumo mostrando apenas os hiperparâmetros que VARIAM entre os experimentos do padrão.

    Com K-fold, diferenças menores que o desvio padrão (colunas *_std) não devem ser
    tratadas como melhora real. As métricas de teste ficam ocultas por padrão: a escolha
    entre configurações deve ser feita pela validação (ver final_report).
    """
    df = load_summary(pattern)
    if df.empty:
        print(f"Nenhum experimento encontrado para '{pattern}' em {output_dir()}")
        return df
    metric_prefixes = ("melhor_", "max_gap", "test/", "tempo_", "folds_")
    metric_cols = [c for c in df.columns if c.startswith(metric_prefixes)]
    hparam_cols = [c for c in df.columns if c not in metric_cols and c not in ("exp_name", "status")]
    if not include_test:
        metric_cols = [c for c in metric_cols if not c.startswith("test/")]
    varying = [c for c in hparam_cols if df[c].astype(str).nunique() > 1]
    table = df[["exp_name", "status", *varying, *metric_cols]]
    if save:
        suffix = "_com_teste" if include_test else ""
        path = os.path.join(report_dir(), f"ablacao_{_slug(pattern)}{suffix}.csv")
        table.to_csv(path, index=False)
        print(f"Tabela salva em {path}")
    return table


def plot_curves(pattern="*", metrics=("loss", "accuracy"), save=True):
    """Curvas por época (média entre folds): treino tracejado, validação sólida com faixa ±1 desvio padrão."""
    histories = load_history(pattern, aggregated=True)
    if not histories:
        print(f"Nenhum histórico encontrado para '{pattern}' em {output_dir()}")
        return None
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 4.5), squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, hist) in enumerate(histories.items()):
        color = colors[i % len(colors)]
        for ax, metric in zip(axes[0], metrics):
            train_col, val_col = f"train/{metric}_mean", f"val/{metric}_mean"
            if train_col in hist:
                ax.plot(hist["epoch"], hist[train_col], "--", color=color, alpha=0.6)
            if val_col in hist:
                ax.plot(hist["epoch"], hist[val_col], "-", color=color, label=name)
                std = hist[f"val/{metric}_std"].fillna(0)
                ax.fill_between(hist["epoch"], hist[val_col] - std, hist[val_col] + std, color=color, alpha=0.15)
    for ax, metric in zip(axes[0], metrics):
        ax.set_title(f"{metric} (tracejado = treino, sólido = validação ± 1 desvio)")
        ax.set_xlabel("época")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    if save:
        path = os.path.join(report_dir(), f"curvas_{_slug(pattern)}.png")
        fig.savefig(path, dpi=150)
        print(f"Figura salva em {path}")
    plt.show()
    return fig


def per_class_table(pattern="*", metric="recall", split="test"):
    """Métrica por classe (linhas = classes, colunas = experimentos), em média entre folds.

    split='test' usa o teste; split='val'/'train' usa a melhor época de cada fold.
    """
    data = {}
    for d in experiment_dirs(pattern):
        results_path = os.path.join(d, "resultados.json")
        if not os.path.isfile(results_path):
            continue
        results = _read_json(results_path)
        source = results.get("teste", {}) if split == "test" else results.get("validacao_media", {})
        values = {cls: source.get(f"{split}/{metric}_per_class/{cls}") for cls in CIFAR10_CLASSES}
        if any(v is not None for v in values.values()):
            data[os.path.basename(d)] = values
    return pd.DataFrame(data)


def plot_per_class(pattern="*", metric="recall", split="test", save=True):
    table = per_class_table(pattern, metric, split)
    if table.empty:
        print(f"Nenhum resultado por classe para '{pattern}'")
        return table
    ax = table.plot.bar(figsize=(12, 4.5), width=0.8)
    ax.set_title(f"{metric} por classe ({split})")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save:
        slug = _slug(pattern)
        table.to_csv(os.path.join(report_dir(), f"{metric}_por_classe_{split}_{slug}.csv"))
        path = os.path.join(report_dir(), f"{metric}_por_classe_{split}_{slug}.png")
        plt.savefig(path, dpi=150)
        print(f"Figura salva em {path}")
    plt.show()
    return table


def _pct(value, std=None):
    text = "—" if value is None else f"{value * 100:.1f}%".replace(".", ",")
    return text if std is None else f"{text} ± {std * 100:.1f}".replace(".", ",")


def class_metrics_table(exp_name, split="test", save=True):
    """Métricas gerais e por classe de UM experimento, em média ± desvio entre folds.

    Linhas: as 10 classes e 'geral'. Colunas: acuracia, precision, recall, f1 (+ *_std).
    Na linha 'geral', precision/recall/f1 são médias macro. Por classe, a acurácia é a fração
    das imagens daquela classe classificadas corretamente, ou seja, igual ao recall.
    split='test' usa o teste (modelo de cada fold); 'val'/'train' usam a melhor época de cada fold.
    """
    results = _read_json(os.path.join(output_dir(), exp_name, "resultados.json"))
    mean = results.get("teste", {}) if split == "test" else results.get("validacao_media", {})
    std = results.get("teste_std", {}) if split == "test" else results.get("validacao_std", {})
    rows = {}
    for cls in CIFAR10_CLASSES:
        row = {}
        for metric in ("precision", "recall", "f1"):
            row[metric] = mean.get(f"{split}/{metric}_per_class/{cls}")
            row[f"{metric}_std"] = std.get(f"{split}/{metric}_per_class/{cls}")
        row["acuracia"], row["acuracia_std"] = row["recall"], row["recall_std"]
        rows[cls] = row
    rows["geral"] = {"acuracia": mean.get(f"{split}/accuracy"), "acuracia_std": std.get(f"{split}/accuracy")}
    for metric in ("precision", "recall", "f1"):
        rows["geral"][metric] = mean.get(f"{split}/{metric}_macro")
        rows["geral"][f"{metric}_std"] = std.get(f"{split}/{metric}_macro")
    cols = [c for m in ("acuracia", "precision", "recall", "f1") for c in (m, f"{m}_std")]
    table = pd.DataFrame(rows).T[cols].astype(float)
    table.index.name = "classe"
    if save:
        path = os.path.join(report_dir(), f"metricas_por_classe_{split}_{exp_name}.csv")
        table.to_csv(path)
        print(f"Tabela salva em {path}")
    return table


def plot_class_metrics(exp_name, split="test", save=True):
    """Figura para slide: acurácia, precision, recall e F1 gerais (macro) no topo e
    precision, recall e F1 de cada classe abaixo, em % (média ± desvio entre folds).
    Retorna a tabela formatada em % para exibição no notebook."""
    table = class_metrics_table(exp_name, split, save=save)
    results = _read_json(os.path.join(output_dir(), exp_name, "resultados.json"))
    split_name = {"test": "teste", "val": "validação", "train": "treino"}[split]
    folds = results.get("folds_concluidos")

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    ax.set_xlim(0, 7.2)
    ax.set_ylim(15.2, 0)
    ax.axis("off")
    ax.text(0, 0.35, exp_name, fontsize=11, fontweight="bold", va="center")
    ax.text(0, 0.95, f"{split_name.capitalize()} · média ± desvio padrão entre {folds} folds", fontsize=8.5, color="#555555", va="center")

    # geral: quatro caixas
    geral = table.loc["geral"]
    for k, (metric, label) in enumerate([("acuracia", "Acurácia"), ("precision", "Precision (macro)"),
                                         ("recall", "Recall (macro)"), ("f1", "F1 (macro)")]):
        x = k * 1.8
        ax.add_patch(plt.Rectangle((x, 1.5), 1.7, 1.9, facecolor="#eef3fa", edgecolor="#c9d6ea", linewidth=1))
        ax.text(x + 0.12, 1.95, label, fontsize=8, color="#555555", va="center")
        ax.text(x + 0.12, 2.65, _pct(geral[metric]), fontsize=15, fontweight="bold", va="center")
        ax.text(x + 0.12, 3.12, f"± {geral[f'{metric}_std'] * 100:.1f} p.p.".replace(".", ",", 1), fontsize=8, color="#555555", va="center")

    # por classe
    top, row_h, label_w, col_w = 4.3, 0.95, 1.8, 1.8
    for j, label in enumerate(["Precision", "Recall", "F1"]):
        ax.text(label_w + j * col_w + col_w / 2, top, label, fontsize=9, fontweight="bold", ha="center", va="center")
    ax.text(0, top, "Classe", fontsize=9, fontweight="bold", va="center")
    cmap = plt.cm.Blues
    for i, cls in enumerate(CIFAR10_CLASSES):
        y = top + 0.55 + i * row_h
        ax.text(0, y + row_h / 2, cls, fontsize=9, va="center")
        for j, metric in enumerate(("precision", "recall", "f1")):
            v, s = table.loc[cls, metric], table.loc[cls, f"{metric}_std"]
            t = min(1, max(0, (v - 0.3) / 0.7))
            ax.add_patch(plt.Rectangle((label_w + j * col_w + 0.04, y + 0.05), col_w - 0.08, row_h - 0.1,
                                       facecolor=cmap(0.08 + 0.8 * t), linewidth=0))
            ax.text(label_w + j * col_w + col_w / 2, y + row_h / 2, _pct(v, s), fontsize=8.5, ha="center", va="center",
                    color="white" if t > 0.6 else "black", fontweight="bold" if metric == "f1" else "normal")
    ax.text(0, top + 0.55 + 10 * row_h + 0.25,
            "Por classe, a acurácia é a fração das imagens da classe classificadas corretamente (= recall).",
            fontsize=7.5, color="#555555", va="center")
    fig.tight_layout()
    if save:
        path = os.path.join(report_dir(), f"metricas_por_classe_{split}_{exp_name}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Figura salva em {path}")
    plt.show()

    shown = pd.DataFrame({m: [_pct(table.loc[i, m], table.loc[i, f"{m}_std"]) for i in table.index]
                          for m in ("acuracia", "precision", "recall", "f1")}, index=table.index)
    return shown


def champion_class_metrics(block, split="test", save=True):
    """plot_class_metrics para o campeão do bloco (escolhido pela validação)."""
    name = champion_name(block)
    if name is None:
        print(f"Campeão do bloco {block} ainda não definido.")
        return pd.DataFrame()
    print(f"Campeão de {block}: {name}")
    return plot_class_metrics(name, split, save)


# ============================================================================ grid search em blocos
def _grid_dir(block):
    return os.path.join(output_dir(), "_grids", block)


def grid_ranking(block, stage="triagem", save=True):
    """Ranking do bloco pela validação (sem métricas de teste).

    stage='triagem': todas as configurações, média nos folds de triagem;
    stage='final': finalistas, média nos K folds.
    """
    path = os.path.join(_grid_dir(block), f"ranking_{stage}.csv")
    if not os.path.isfile(path):
        print(f"Ranking '{stage}' do bloco {block} ainda não existe ({path}).")
        return pd.DataFrame()
    df = pd.read_csv(path).drop(columns=["hyperparams"], errors="ignore")
    df.columns = [c.replace("eixo:", "") for c in df.columns]
    if save:
        df.to_csv(os.path.join(report_dir(), f"ranking_{stage}_{block}.csv"), index=False)
    return df


def discarded_configs(block):
    """Combinações descartadas antes do treino (dimensões inválidas, parâmetros demais) e o motivo."""
    path = os.path.join(_grid_dir(block), "configs.csv")
    if not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.replace("eixo:", "") for c in df.columns]
    return df[~df["valida"]].reset_index(drop=True)


def show_champion(block):
    path = os.path.join(_grid_dir(block), "campeao.json")
    if not os.path.isfile(path):
        print(f"Campeão do bloco {block} ainda não definido.")
        return None
    champion = _read_json(path)
    k = champion["hyperparams"].get("k_folds")
    print(f"Campeão de {block}: {champion['exp_name']}")
    print(f"  critério: {champion['criterio']}")
    print(f"  eixos: {champion['axes']}")
    print(f"  val/accuracy: {champion['val_mean']['val/accuracy']:.4f} ± {champion['val_std']['val/accuracy']:.4f} "
          f"({k} folds) | gap/accuracy: {champion['val_mean'].get('gap/accuracy', float('nan')):.4f} | "
          f"parâmetros: {champion['num_parameters']:,}")
    return champion


def heatmap(block, row, col, value="val/accuracy_mean", facet=None, stage="triagem", save=True):
    """Heatmap da métrica de validação do grid. `col` pode ser uma lista de eixos (concatenados);
    `facet` gera um heatmap por valor de outro eixo. Células vazias = combinação descartada."""
    df = grid_ranking(block, stage, save=False)
    if df.empty:
        return None
    cols = [col] if isinstance(col, str) else list(col)
    # Ordena pelos valores originais (numéricos quando possível), não pelo texto: 128, 256, ..., 2048.
    df = df.sort_values(cols + [row], kind="stable").copy()
    df["_col"] = df[cols].astype(str).agg(" | ".join, axis=1)
    col_order = list(dict.fromkeys(df["_col"]))
    facets = [None] if facet is None else sorted(df[facet].astype(str).unique())
    vmin, vmax = df[value].min(), df[value].max()  # mesma escala de cores em todos os painéis
    fig, axes = plt.subplots(1, len(facets), figsize=(max(5, 1.3 * df["_col"].nunique()) * len(facets), 4.2),
                             squeeze=False)
    for ax, fval in zip(axes[0], facets):
        sub = df if fval is None else df[df[facet].astype(str) == fval]
        pivot = sub.pivot_table(index=row, columns="_col", values=value, aggfunc="first")
        pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns]).sort_index()
        im = ax.imshow(pivot.values.astype(float), cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_xlabel(" | ".join(cols))
        ax.set_ylabel(row)
        ax.set_title(f"{value}" + (f" — {facet}={fval}" if fval is not None else ""))
        mid = (vmin + vmax) / 2
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not pd.isna(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                            color="white" if v < mid else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if save:
        metric = value.replace("/", "-")
        path = os.path.join(report_dir(), f"heatmap_{block}_{row}_x_{'-'.join(cols)}__{metric}.png")
        fig.savefig(path, dpi=150)
        print(f"Figura salva em {path}")
    plt.show()
    return fig


def plot_grid_bars(block, stage="triagem", save=True):
    """Todas as configurações do bloco ordenadas pela validação, com barras de erro (±1 desvio entre folds)."""
    df = grid_ranking(block, stage, save=False)
    if df.empty:
        return None
    df = df.sort_values("val/accuracy_mean")
    labels = df["exp_name"].str.replace(f"{block}__", "", regex=False)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(df))))
    ax.barh(labels, df["val/accuracy_mean"], xerr=df["val/accuracy_std"], color="#4c72b0", alpha=0.85)
    ax.set_xlabel("val/accuracy (média ± desvio entre folds)")
    lo = df["val/accuracy_mean"].min()
    ax.set_xlim(max(0, lo - 0.05), min(1, df["val/accuracy_mean"].max() + 0.03))
    ax.set_title(f"{block} — {stage}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if save:
        path = os.path.join(report_dir(), f"barras_{stage}_{block}.png")
        fig.savefig(path, dpi=150)
        print(f"Figura salva em {path}")
    plt.show()
    return fig


def plot_finalists(block, metrics=("loss", "accuracy")):
    """Curvas de treino/validação (média entre os K folds) das finalistas do bloco."""
    df = grid_ranking(block, "final", save=False)
    if df.empty:
        return None
    return plot_curves(list(df["exp_name"]), metrics=metrics)


def final_report(blocks, save=True):
    """Revela o TESTE apenas para o(s) campeão(ões) de cada bloco, escolhidos pela validação.

    Blocos de referência (sem grid, ex.: *_b0_referencia) mostram todas as suas configurações.
    """
    rows = []
    for block in blocks:
        spec_path = os.path.join(_grid_dir(block), "spec.json")
        ranking = grid_ranking(block, "final", save=False)
        if ranking.empty:
            continue
        is_reference = os.path.isfile(spec_path) and not _read_json(spec_path).get("grid") \
            and not _read_json(spec_path).get("grid_from_ranking")
        names = list(ranking["exp_name"]) if is_reference else list(ranking["exp_name"][:1])
        for name in names:
            results = _read_json(os.path.join(output_dir(), name, "resultados.json"))
            val_m, val_s = results.get("validacao_media", {}), results.get("validacao_std", {})
            test_m, test_s = results.get("teste", {}), results.get("teste_std", {})
            params = _read_json(os.path.join(output_dir(), name, "parametros.json"))
            rows.append({
                "bloco": block, "papel": "referência" if is_reference else "campeão", "exp_name": name,
                "folds": results.get("folds_concluidos"), "num_parameters": params.get("num_parameters"),
                "val/accuracy": val_m.get("val/accuracy"), "val/accuracy_std": val_s.get("val/accuracy"),
                "test/accuracy": test_m.get("test/accuracy"), "test/accuracy_std": test_s.get("test/accuracy"),
                "test/f1_macro": test_m.get("test/f1_macro"), "test/f1_macro_std": test_s.get("test/f1_macro"),
                "test/precision_macro": test_m.get("test/precision_macro"),
                "test/recall_macro": test_m.get("test/recall_macro"),
                "gap/accuracy": val_m.get("gap/accuracy"),
            })
    df = pd.DataFrame(rows)
    if save and not df.empty:
        path = os.path.join(report_dir(), f"relatorio_final_{_slug(blocks)}.csv")
        df.to_csv(path, index=False)
        print(f"Tabela salva em {path}")
    return df


# ============================================================================ comparação pareada por fold
def champion_name(block):
    """exp_name do campeão de um bloco (None se ainda não definido)."""
    path = os.path.join(_grid_dir(block), "campeao.json")
    return _read_json(path)["exp_name"] if os.path.isfile(path) else None


def base_config_name(block):
    """exp_name da configuração do bloco idêntica à receita herdada (referência pareada), ou None."""
    path = os.path.join(_grid_dir(block), "spec.json")
    return _read_json(path).get("config_base") if os.path.isfile(path) else None


def _fold_metric(exp_name, metric):
    results = _read_json(os.path.join(output_dir(), exp_name, "resultados.json"))
    section = "teste" if metric.startswith("test/") else "melhor_val"
    return {f["fold"]: f.get(section, {}).get(metric) for f in results.get("folds", [])
            if f.get("status") == "concluido" and f.get(section, {}).get(metric) is not None}


def paired_comparison(reference, others, metric="val/accuracy", save=True):
    """Compara experimentos com uma referência FOLD A FOLD (mesmas partições, mesma seed).

    Como cada fold tem a mesma divisão treino/validação em todos os experimentos, a diferença
    por fold elimina a variação causada pela partição e revela efeitos pequenos mas consistentes
    ("venceu em 5 de 5 folds") que a sobreposição de média ± desvio esconde.
    `others` aceita nomes de experimento ou padrões (ex.: "cnn_b4_augmentation__*").
    Use métricas de validação para decidir; métricas de teste apenas para relatar.
    """
    ref = _fold_metric(reference, metric)
    names = [os.path.basename(d) for d in experiment_dirs(others) if os.path.basename(d) != reference]
    rows = []
    for name in names:
        values = _fold_metric(name, metric)
        common = sorted(set(ref) & set(values))
        if not common:
            continue
        deltas = [values[k] - ref[k] for k in common]
        n = len(deltas)
        mean = sum(deltas) / n
        std = (sum((d - mean) ** 2 for d in deltas) / (n - 1)) ** 0.5 if n > 1 else float("nan")
        rows.append({"exp_name": name, "referencia": reference, "metrica": metric, "folds_comuns": n,
                     "delta_medio": mean, "delta_std": std, "vitorias": sum(d > 0 for d in deltas),
                     "derrotas": sum(d < 0 for d in deltas),
                     **{f"delta_fold_{k}": d for k, d in zip(common, deltas)}})
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"Nenhum experimento com folds em comum com {reference}.")
        return df
    df = df.sort_values("delta_medio", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.55 * len(df) + 1)))
    fold_cols = [c for c in df.columns if c.startswith("delta_fold_")]
    for i, row in df.iterrows():
        values = [row[c] for c in fold_cols if pd.notna(row[c])]
        ax.scatter(values, [i] * len(values), color="#4c72b0", alpha=0.6, s=28, zorder=3)
        ax.scatter([row["delta_medio"]], [i], color="black", marker="|", s=400, zorder=4)
    ax.axvline(0, color="#c44e52", lw=1)
    ax.set_yticks(range(len(df)), [f"{n}  ({v}/{v + d})" for n, v, d in zip(df["exp_name"], df["vitorias"], df["derrotas"])],
                  fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"Δ {metric} por fold (pontos = folds, traço = média)")
    ax.set_title(f"Referência: {reference}", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if save:
        slug = metric.replace("/", "-")
        df.to_csv(os.path.join(report_dir(), f"pareado_{reference}__{slug}.csv"), index=False)
        path = os.path.join(report_dir(), f"pareado_{reference}__{slug}.png")
        fig.savefig(path, dpi=150)
        print(f"Tabela e figura salvas em {report_dir()}/pareado_{reference}__{slug}.*")
    plt.show()
    return df
