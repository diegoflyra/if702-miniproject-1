"""Extrai dos arquivos de resultados todos os números usados no caderno e no .docx.

Uso (na raiz do repositório):
    python tools/relatorio/extrair_dados.py

Lê outputs_mlp/, outputs_cnn/, outputs_final/ (complementares) e outputs_augmentation/ (bônus)
e grava relatorio/dados/doc_data.json e relatorio/dados/curves.json. As métricas por classe dos campeões,
as matrizes de confusão (somadas nos 5 folds) e as conclusões por hiperparâmetro vêm de conclusoes.py.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

import conclusoes

M, C, F, A = "outputs_mlp", "outputs_cnn", "outputs_final", "outputs_augmentation"
OUT = os.path.join("relatorio", "dados")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def rank(root, block, stage):
    path = f"{root}/_grids/{block}/ranking_{stage}.csv"
    if not os.path.isfile(path):
        return []
    df = pd.read_csv(path)
    df.columns = [c.replace("eixo:", "") for c in df.columns]
    rows = df[[c for c in df.columns if c != "hyperparams"]].replace({np.nan: None}).to_dict("records")
    for r in rows:
        r["label"] = r["exp_name"].split("__", 1)[1]
    return rows


def res(root, exp):
    return json.load(open(f"{root}/{exp}/resultados.json"))


def summary(root, exp):
    r = res(root, exp)
    p = json.load(open(f"{root}/{exp}/parametros.json"))
    return {"exp": exp, "label": exp.split("__", 1)[1], "val": r["validacao_media"].get("val/accuracy"),
            "val_std": r["validacao_std"].get("val/accuracy"), "test": r["teste"].get("test/accuracy"),
            "test_std": r["teste_std"].get("test/accuracy"), "f1": r["teste"].get("test/f1_macro"),
            "train": r["validacao_media"].get("train/accuracy"), "gap": r["validacao_media"].get("gap/accuracy"),
            "params": p.get("num_parameters"), "folds": r.get("folds_concluidos"),
            "best_epoch": float(np.mean([f["melhor_epoca"] for f in r["folds"]])),
            "best_epochs": [f["melhor_epoca"] for f in r["folds"]], "epochs_limit": p.get("epochs"),
            "augment": p.get("augment"), "conv_blocks": p.get("conv_blocks"),
            "dropout": p.get("cnn_dropout") if p.get("model") == "cnn" else p.get("dropout")}


def recall(root, exp):
    t = res(root, exp)["teste"]
    return [t.get(f"test/recall_per_class/{c}") for c in CLASSES]


def confusions(root, exp, n=5):
    cm = pd.read_csv(f"{root}/{exp}/matriz_confusao_teste.csv", index_col=0)
    off = cm.values.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((int(off[i, j]), CLASSES[i], CLASSES[j]) for i in range(10) for j in range(10)), reverse=True)[:n]
    return [{"n": a, "real": b, "pred": c} for a, b, c in pairs]


def paired(root, ref, others, key="val/accuracy", section="melhor_val"):
    base = {f["fold"]: f[section][key] for f in res(root, ref)["folds"]}
    out = []
    for other in others:
        values = {f["fold"]: f[section][key] for f in res(root, other)["folds"]}
        common = sorted(set(base) & set(values))
        deltas = [values[k] - base[k] for k in common]
        out.append({"exp": other, "delta": float(np.mean(deltas)), "wins": int(sum(d > 0 for d in deltas)), "n": len(deltas)})
    return out


def grid_time(root, prefix):
    return float(sum(pd.read_csv(f"{d}/historico_treino.csv", usecols=["epoch_time_s"]).epoch_time_s.sum()
                     for d in glob.glob(f"{root}/{prefix}__*")) / 60)


def exps(root, prefix):
    return sorted(os.path.basename(d) for d in glob.glob(f"{root}/{prefix}__*"))


def curve(root, exp, folds):
    """Média entre folds por época, só nas épocas em que todos os folds ainda treinavam."""
    h = pd.read_csv(f"{root}/{exp}/historico_treino.csv",
                    usecols=["fold", "epoch", "train/accuracy", "val/accuracy", "train/loss", "val/loss"])
    h = h[h.fold.isin(folds)]
    last = h.groupby("fold").epoch.max()
    upto = int(last.min())
    g = h[h.epoch <= upto].groupby("epoch").agg(["mean", "std"])
    points = [[int(ep), *(round(float(row[(k, s)]), 5) for k in ["train/accuracy", "val/accuracy", "train/loss", "val/loss"]
                          for s in ["mean", "std"])] for ep, row in g.iterrows()]
    return {"exp": exp, "folds": folds, "upto": upto, "stops": [int(x) for x in last.values], "points": points}


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {"classes": CLASSES}
    data["mlp"] = {
        "b0": rank(M, "mlp_b0_referencia", "final"),
        "b1_triagem": rank(M, "mlp_b1_topologia", "triagem"), "b1_final": rank(M, "mlp_b1_topologia", "final"),
        "b2_triagem": rank(M, "mlp_b2_otimizacao", "triagem"), "b2_final": rank(M, "mlp_b2_otimizacao", "final"),
        "chk_final": rank(M, "mlp_b2_checagem", "final"),
        "b3_triagem": rank(M, "mlp_b3_regularizacao", "triagem"), "b3_final": rank(M, "mlp_b3_regularizacao", "final"),
        "extraA": rank(F, "mlp_b3b_dropout_longo", "final"),
        "stages": [summary(M, e) for e in ["mlp_b0_referencia__linear", "mlp_b0_referencia__original_64-128-64",
                                          "mlp_b1_topologia__L3_N256", "mlp_b2_otimizacao__optadam_lr0.0003",
                                          "mlp_b2_checagem__rank2_L4_N256", "mlp_b3_regularizacao__do0.2_losscross_entropy"]],
        "extraA_summary": [summary(F, e) for e in exps(F, "mlp_b3b_dropout_longo")],
        "paired_A": paired(F, "mlp_b3b_dropout_longo__do0.2_losscross_entropy",
                           [e for e in exps(F, "mlp_b3b_dropout_longo") if not e.endswith("do0.2_losscross_entropy")]),
        "recall_original": recall(M, "mlp_b0_referencia__original_64-128-64"),
        "recall_final": recall(M, "mlp_b3_regularizacao__do0.2_losscross_entropy"),
        "confusions_final": confusions(M, "mlp_b3_regularizacao__do0.2_losscross_entropy"),
    }
    data["cnn"] = {
        "b0": rank(C, "cnn_b0_referencia", "final"),
        "b1_triagem": rank(C, "cnn_b1_topologia", "triagem"), "b1_final": rank(C, "cnn_b1_topologia", "final"),
        "b2_triagem": rank(C, "cnn_b2_otimizacao", "triagem"), "b2_final": rank(C, "cnn_b2_otimizacao", "final"),
        "chk_final": rank(C, "cnn_b2_checagem", "final"),
        "b3_triagem": rank(C, "cnn_b3_regularizacao", "triagem"), "b3_final": rank(C, "cnn_b3_regularizacao", "final"),
        "extraB": rank(F, "cnn_b3b_pooling", "final"),
        "stages": [summary(C, e) for e in ["cnn_b0_referencia__lenet_original", "cnn_b1_topologia__B4_k3_padsame_maxpool",
                                          "cnn_b2_otimizacao__optadam_lr0.001", "cnn_b3_regularizacao__pool2_do0.5_bn1"]]
                  + [summary(F, "cnn_b3b_pooling__B3_pool2")],
        "extraB_summary": [summary(F, e) for e in exps(F, "cnn_b3b_pooling")],
        "paired_B": paired(F, "cnn_b3b_pooling__B4_pool2", [e for e in exps(F, "cnn_b3b_pooling") if not e.endswith("B4_pool2")]),
        "recall_lenet": recall(C, "cnn_b0_referencia__lenet_original"),
        "recall_final": recall(F, "cnn_b3b_pooling__B3_pool2"),
        "confusions_final": confusions(F, "cnn_b3b_pooling__B3_pool2"),
    }

    # ------------------------------------------------------------ bônus: augmentation
    aug = {}
    for net, ref, champ, extra_paired in (
            ("cnn", "cnn_b4_augmentation__aug0_B3_do0.5", "cnn_b4_augmentation__aug1_B4_do0.5",
             ("cnn_b4_augmentation__aug1_B3_do0.5", "cnn_b4_augmentation__aug1_B4_do0.5")),
            ("mlp", "mlp_b4_augmentation__aug0_do0.2", "mlp_b4_augmentation__aug1_do0.2", None)):
        names = exps(A, f"{net}_b4_augmentation")
        aug[net] = {"rows": [summary(A, e) for e in names], "ref": ref, "champ": champ,
                    "paired_val": paired(A, ref, [e for e in names if e != ref]),
                    "paired_test": paired(A, ref, [champ], "test/accuracy", "teste")[0],
                    "recall_ref": recall(A, ref), "recall_champ": recall(A, champ),
                    "confusions_champ": confusions(A, champ),
                    "tempo_gpu_min": grid_time(A, f"{net}_b4_augmentation")}
        if extra_paired:
            aug[net]["depth_with_aug"] = paired(A, extra_paired[0], [extra_paired[1]])[0]
    data["aug"] = aug

    # ------------------------------------------------------------ campeões por classe e hiperparâmetros
    champs = [conclusoes.champion(*c) for c in conclusoes.CHAMPIONS]
    data["champions"] = champs
    data["class_conclusions"] = conclusoes.class_conclusions(champs)
    data["hyper"] = conclusoes.hyperparameters(data)
    json.dump(data, open(os.path.join(OUT, "doc_data.json"), "w"), ensure_ascii=False, default=float, allow_nan=False)

    # ------------------------------------------------------------ curvas por época
    F13, F15 = [1, 2, 3], [1, 2, 3, 4, 5]
    spec = {
        "mlp_b1": (M, F13, [("mlp_b1_topologia__L1_N256", "1 × 256"), ("mlp_b1_topologia__L3_N256", "3 × 256 (campeão)"),
                            ("mlp_b1_topologia__L5_N2048", "5 × 2048 (maior)")], []),
        "mlp_b2": (M, F13, [("mlp_b2_otimizacao__optadam_lr0.0003", "Adam 3e-4 (campeão)"), ("mlp_b2_otimizacao__optsgd_lr0.01", "SGD 1e-2"),
                            ("mlp_b2_otimizacao__optsgd_lr0.0001", "SGD 1e-4")], [("mlp_b2_otimizacao__optadam_lr0.03", "Adam 3e-2 (colapso)")]),
        "mlp_b3": (M, F13, [("mlp_b3_regularizacao__do0_losscross_entropy", "sem dropout"),
                            ("mlp_b3_regularizacao__do0.2_losscross_entropy", "dropout 0,2 (campeão)"),
                            ("mlp_b3_regularizacao__do0.5_losscross_entropy", "dropout 0,5")], []),
        "cnn_b1": (C, F13, [("cnn_b1_topologia__B4_k3_padsame_maxpool", "4 blocos · maxpool (campeão)"),
                            ("cnn_b1_topologia__B4_k3_padsame_stride2", "4 blocos · stride 2"),
                            ("cnn_b1_topologia__B2_k3_padsame_maxpool", "2 blocos · maxpool")], []),
        "cnn_b3": (C, F13, [("cnn_b3_regularizacao__pool2_do0.5_bn0", "sem BatchNorm"),
                            ("cnn_b3_regularizacao__pool2_do0.5_bn1", "com BatchNorm (campeão)")], []),
        "cnn_b": (F, F15, [("cnn_b3b_pooling__B4_pool2", "4 blocos · pool 2"), ("cnn_b3b_pooling__B3_pool2", "3 blocos · pool 2 (novo campeão)"),
                           ("cnn_b3b_pooling__B2_pool2", "2 blocos · pool 2")], []),
        "cnn_aug": (A, F13, [("cnn_b4_augmentation__aug0_B3_do0.5", "sem augmentation · 3 blocos"),
                             ("cnn_b4_augmentation__aug1_B3_do0.5", "com augmentation · 3 blocos"),
                             ("cnn_b4_augmentation__aug1_B4_do0.5", "com augmentation · 4 blocos (campeão)")], []),
        "mlp_aug": (A, F15, [("mlp_b4_augmentation__aug0_do0.2", "sem augmentation · dropout 0,2"),
                             ("mlp_b4_augmentation__aug1_do0", "com augmentation · sem dropout"),
                             ("mlp_b4_augmentation__aug1_do0.2", "com augmentation · dropout 0,2 (campeão)")], []),
    }
    curves = {key: {"folds": folds,
                    "series": [{**curve(root, e, folds), "label": label} for e, label in series],
                    "context": [{**curve(root, e, folds), "label": label} for e, label in context]}
              for key, (root, folds, series, context) in spec.items()}
    json.dump(curves, open(os.path.join(OUT, "curves.json"), "w"), ensure_ascii=False, allow_nan=False)
    print("dados:", os.path.join(OUT, "doc_data.json"), "| curvas:", ", ".join(curves))


if __name__ == "__main__":
    main()
