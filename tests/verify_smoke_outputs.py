"""Verificações pós-execução do test_local.sh (arquivos, retomada, grids, campeões e report_utils)."""

import csv
import json
import os
import sys
from argparse import Namespace

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch

import report_utils as rep
from data_loader import CIFAR10_CLASSES
from models import build_model

OUT = os.environ["EXP_OUTPUT_DIR"]
LOGS = os.path.join(OUT, "_logs_smoke")
REQUIRED = ["parametros.json", "historico_treino.csv", "historico_treino_agregado.csv", "resultados.json",
            "melhor_modelo.pth", "modelo_final.pth", "matriz_confusao_teste.csv"]


def load(exp):
    d = os.path.join(OUT, exp)
    params = json.load(open(os.path.join(d, "parametros.json")))
    results = json.load(open(os.path.join(d, "resultados.json")))
    rows = list(csv.DictReader(open(os.path.join(d, "historico_treino.csv"))))
    return d, params, results, rows


def check_experiment(exp, k, folds, epochs_per_fold, status):
    d, params, results, rows = load(exp)
    missing = [f for f in REQUIRED if not os.path.isfile(os.path.join(d, f))]
    assert not missing, f"{exp}: arquivos ausentes {missing}"
    assert params["exp_name"] == exp and params["k_folds"] == k
    assert results["status"] == status, f"{exp}: status {results['status']} != {status}"
    assert results["folds_concluidos_ids"] == folds, f"{exp}: folds {results['folds_concluidos_ids']} != {folds}"
    foreign = ("kernel_size", "conv_blocks", "cnn_dropout") if params["model"] == "mlp" else ("mlp_layers", "dropout", "batch_norm")
    assert all(params[f] is None for f in foreign), f"{exp}: parâmetros do outro modelo deveriam ser null"
    assert not any(k in params for k in ("folds", "resume", "overwrite")), f"{exp}: argumentos operacionais em parametros.json"

    cols = rows[0].keys()
    for col in ("fold", "epoch", "lr", "gap/loss", "gap/accuracy", "is_best_epoch", "epoch_time_s", "train_metrics_mode"):
        assert col in cols, f"{exp}: coluna {col} ausente"
    for split in ("train", "val"):
        for metric in ("loss", "accuracy", "precision_macro", "recall_macro", "f1_macro"):
            assert f"{split}/{metric}" in cols, f"{exp}: coluna {split}/{metric} ausente"
        for metric in ("precision", "recall", "f1"):
            for cls in CIFAR10_CLASSES:
                assert f"{split}/{metric}_per_class/{cls}" in cols, f"{exp}: {split}/{metric}_per_class/{cls} ausente"
    assert len(rows) == len(folds) * epochs_per_fold, f"{exp}: {len(rows)} linhas, esperado {len(folds) * epochs_per_fold}"
    assert sorted({int(r["fold"]) for r in rows}) == folds, f"{exp}: folds no CSV"
    assert all(v != "" for r in rows for v in r.values()), f"{exp}: valores vazios no CSV"

    agg = list(csv.DictReader(open(os.path.join(d, "historico_treino_agregado.csv"))))
    assert len(agg) == epochs_per_fold and all(int(a["n_folds"]) == len(folds) for a in agg), f"{exp}: agregado incorreto"
    mean_acc = sum(float(r["val/accuracy"]) for r in rows if r["epoch"] == "1") / len(folds)
    assert abs(float(agg[0]["val/accuracy_mean"]) - mean_acc) < 1e-9, f"{exp}: média agregada incorreta"

    fold_accs = [f["teste"]["test/accuracy"] for f in results["folds"]]
    assert abs(results["teste"]["test/accuracy"] - sum(fold_accs) / len(folds)) < 1e-9, f"{exp}: média do teste"
    best = min(results["folds"], key=lambda f: f["melhor_valor_monitorado"])
    assert results["melhor_fold"] == best["fold"], f"{exp}: melhor fold incorreto"

    model = build_model(Namespace(**params))
    for weights in ("melhor_modelo.pth", "modelo_final.pth"):
        model.load_state_dict(torch.load(os.path.join(d, weights), map_location="cpu"))
    if k > 1:
        root_best = torch.load(os.path.join(d, "melhor_modelo.pth"), map_location="cpu")
        for fold in folds:
            for f in ("melhor_modelo.pth", "matriz_confusao_teste.csv"):
                assert os.path.isfile(os.path.join(d, "folds", f"fold_{fold}", f)), f"{exp}: fold_{fold}/{f} ausente"
        fold_best = torch.load(os.path.join(d, "folds", f"fold_{results['melhor_fold']}", "melhor_modelo.pth"), map_location="cpu")
        assert all(torch.equal(root_best[n], fold_best[n]) for n in root_best), f"{exp}: raiz != melhor fold"
    return params, results, rows


# ============================================================================ experimentos isolados
check_experiment("smoke_mlp_holdout", 1, [1], 1, "concluido")
check_experiment("smoke_mlp_kfold", 3, [1, 2, 3], 2, "concluido")
check_experiment("smoke_cnn_kfold", 3, [1, 2, 3], 2, "concluido")
print("  OK experimentos isolados: holdout, K-fold MLP e CNN (arquivos, CSV por fold×época, agregados, pesos)")

d, _, results, rows = load("smoke_divergente")
assert all(os.path.isfile(os.path.join(d, f)) for f in REQUIRED)
assert results["status"] == "concluido" and all(f.get("divergiu_na_epoca") for f in results["folds"])
assert len(rows) == sum(f["divergiu_na_epoca"] for f in results["folds"])
print(f"  OK divergência: folds pararam nas épocas {[f['divergiu_na_epoca'] for f in results['folds']]}, arquivos preservados")

# Retomada: log 1 executou só o fold 2; log 2 descartou o fold 2 interrompido e executou 2 e 3
log1 = open(os.path.join(LOGS, "resume_1.log")).read()
log2 = open(os.path.join(LOGS, "resume_2.log")).read()
assert "folds concluídos: [1] | a executar: [2]" in log1, "retomada 1 deveria executar apenas o fold 2"
assert "a executar: [2, 3]" in log2 and "descartados" in log2, "retomada 2 deveria refazer o fold 2 e executar o 3"
params, results, rows = check_experiment("smoke_resume", 3, [1, 2, 3], 2, "concluido")
assert len(results["retomado_em"]) == 2
print("  OK retomada: fold pendente executado, fold interrompido refeito, hiperparâmetro divergente recusado")


# ============================================================================ grids
def grid_file(block, name):
    return os.path.join(OUT, "_grids", block, name)


def read_csv(path):
    return list(csv.DictReader(open(path)))


def champion(block):
    return json.load(open(grid_file(block, "campeao.json")))


def no_test_leak(block):
    for name in ("ranking_triagem.csv", "ranking_final.csv", "campeao.json", "configs.csv"):
        path = grid_file(block, name)
        if os.path.isfile(path):
            assert "test/" not in open(path).read(), f"{block}/{name} contém métricas de teste"


# Bloco 1 MLP: 4 combinações, 2 descartadas por max_parameters; 2 finalistas completam os 3 folds
configs = read_csv(grid_file("smoke_mlp_b1", "configs.csv"))
assert len(configs) == 4 and sum(c["valida"] == "False" for c in configs) == 2
assert all("max_parameters" in c["motivo"] for c in configs if c["valida"] == "False")
triage = read_csv(grid_file("smoke_mlp_b1", "ranking_triagem.csv"))
final = read_csv(grid_file("smoke_mlp_b1", "ranking_final.csv"))
assert len(triage) == 2 and len(final) == 2 and all(int(r["folds"]) == 3 for r in final)
accs = [float(r["val/accuracy_mean"]) for r in triage]
assert accs == sorted(accs, reverse=True), "ranking de triagem não está ordenado pela validação"
b1 = champion("smoke_mlp_b1")
assert b1["exp_name"] == final[0]["exp_name"] and b1["criterio"].startswith("maior val/accuracy")
for r in final:
    check_experiment(r["exp_name"], 3, [1, 2, 3], 2, "concluido")
no_test_leak("smoke_mlp_b1")
print(f"  OK grid Bloco 1: 2 válidas/2 descartadas, triagem em 2 folds, finalistas em 3, campeão {b1['exp_name']}")

# Bloco 2: herda topologia do campeão do Bloco 1; só a finalista completa os folds
b2_rank = read_csv(grid_file("smoke_mlp_b2", "ranking_triagem.csv"))
assert len(b2_rank) == 2
for r in b2_rank:
    p = json.load(open(os.path.join(OUT, r["exp_name"], "parametros.json")))
    assert p["mlp_layers"] == b1["hyperparams"]["mlp_layers"] and p["mlp_neurons"] == b1["hyperparams"]["mlp_neurons"]
b2_final = read_csv(grid_file("smoke_mlp_b2", "ranking_final.csv"))
assert len(b2_final) == 1
finalist = b2_final[0]["exp_name"]
other = next(r["exp_name"] for r in b2_rank if r["exp_name"] != finalist)
check_experiment(finalist, 3, [1, 2, 3], 2, "concluido")
check_experiment(other, 3, [1, 2], 2, "parcial")
b2 = champion("smoke_mlp_b2")
spec_b2 = json.load(open(grid_file("smoke_mlp_b2", "spec.json")))
assert spec_b2["base_herdada_de"]["exp_name"] == b1["exp_name"]
no_test_leak("smoke_mlp_b2")
print(f"  OK grid Bloco 2: topologia herdada de {b1['exp_name']}, não finalista parou na triagem (parcial)")

# Checagem: 2º do Bloco 1 com otimizador campeão do Bloco 2, em todos os folds
chk = read_csv(grid_file("smoke_mlp_checagem", "ranking_final.csv"))
assert len(chk) == 1
p = json.load(open(os.path.join(OUT, chk[0]["exp_name"], "parametros.json")))
second = json.loads(final[1]["hyperparams"])
assert p["mlp_layers"] == second["mlp_layers"] and p["mlp_neurons"] == second["mlp_neurons"]
assert p["optimizer"] == b2["hyperparams"]["optimizer"] and p["lr"] == b2["hyperparams"]["lr"]
check_experiment(chk[0]["exp_name"], 3, [1, 2, 3], 2, "concluido")
print("  OK checagem: 2º colocado do Bloco 1 treinado com o otimizador campeão do Bloco 2")

# Bloco 3: base = melhor entre campeões do Bloco 2 e checagem; epochs sobrescrito para 3
candidates = [champion("smoke_mlp_b2"), champion("smoke_mlp_checagem")]
best = max(candidates, key=lambda c: c["val_mean"]["val/accuracy"])
spec_b3 = json.load(open(grid_file("smoke_mlp_b3", "spec.json")))
assert spec_b3["base_herdada_de"]["exp_name"] == best["exp_name"]
for r in read_csv(grid_file("smoke_mlp_b3", "ranking_triagem.csv")):
    p = json.load(open(os.path.join(OUT, r["exp_name"], "parametros.json")))
    assert p["epochs"] == 3 and p["dropout"] == 0.3 and p["mlp_layers"] == best["hyperparams"]["mlp_layers"]
b3 = champion("smoke_mlp_b3")
check_experiment(b3["exp_name"], 3, [1, 2, 3], 3, "concluido")
print(f"  OK grid Bloco 3: base herdada de {best['exp_name']} (melhor entre Bloco 2 e checagem)")

# CNN: B6 com maxpool é descartado (mapa 0×0); B6 com stride 2 é válido
cnn_configs = read_csv(grid_file("smoke_cnn_b1", "configs.csv"))
invalid = [c for c in cnn_configs if c["valida"] == "False"]
assert len(cnn_configs) == 4 and len(invalid) == 1 and "B6" in invalid[0]["exp_name"] and "maxpool" in invalid[0]["exp_name"]
assert len(read_csv(grid_file("smoke_cnn_b1", "ranking_triagem.csv"))) == 3
check_experiment(champion("smoke_cnn_b1")["exp_name"], 3, [1, 2, 3], 2, "concluido")
# confirm_also: a referência B2_maxpool completa os 3 folds mesmo sem ser finalista
check_experiment("smoke_cnn_b1__B2_maxpool", 3, [1, 2, 3], 2, "concluido")
assert "smoke_cnn_b1__B2_maxpool" in {r["exp_name"] for r in read_csv(grid_file("smoke_cnn_b1", "ranking_final.csv"))}
assert json.load(open(os.path.join(OUT, "smoke_cnn_b1__B2_maxpool", "parametros.json")))["augment"] is True
print("  OK grid CNN: mapa 0×0 descartado, campeão com 3 folds, referência de confirm_also completa, augmentation herdado da base")

# confirm_also com uma configuração NÃO finalista (finalists=0): só ela completa os folds
check_experiment("smoke_cnn_confirm__B2_stride2", 3, [1, 2, 3], 2, "concluido")
check_experiment("smoke_cnn_confirm__B2_maxpool", 3, [1, 2], 2, "parcial")
assert [r["exp_name"] for r in read_csv(grid_file("smoke_cnn_confirm", "ranking_final.csv"))] == ["smoke_cnn_confirm__B2_stride2"]
assert champion("smoke_cnn_confirm")["exp_name"] == "smoke_cnn_confirm__B2_stride2"
assert json.load(open(grid_file("smoke_cnn_confirm", "spec.json")))["config_base"] == "smoke_cnn_confirm__B2_stride2"
assert rep.base_config_name("smoke_cnn_confirm") == "smoke_cnn_confirm__B2_stride2"
print("  OK confirm_also @base: a configuração igual à receita herdada completou os folds sem ser finalista")

rerun = open(os.path.join(LOGS, "grid_rerun.log")).read()
assert "0 a executar" in rerun and "ok smoke" not in rerun, "reexecução do grid deveria pular tudo"
print("  OK retomada do grid: reexecução não treinou nada")

# ============================================================================ report_utils
ranking = rep.grid_ranking("smoke_mlp_b1")
assert not any(c.startswith("test/") for c in ranking.columns) and "mlp_layers" in ranking.columns
assert len(rep.discarded_configs("smoke_mlp_b1")) == 2
rep.show_champion("smoke_mlp_b1")
rep.heatmap("smoke_mlp_b1", row="mlp_layers", col="mlp_neurons")
rep.heatmap("smoke_cnn_b1", row="conv_blocks", col="reducao")
rep.heatmap("smoke_mlp_b3", row="dropout", col="loss_fn")
rep.plot_grid_bars("smoke_mlp_b2")
rep.plot_finalists("smoke_mlp_b1")
table = rep.ablation_table("smoke_mlp_b1__*")
assert not any(c.startswith("test/") for c in table.columns), "ablation_table não deveria mostrar teste por padrão"
report = rep.final_report(["smoke_mlp_b1", "smoke_mlp_b2", "smoke_mlp_checagem", "smoke_mlp_b3"])
assert list(report["exp_name"]) == [b1["exp_name"], b2["exp_name"], champion("smoke_mlp_checagem")["exp_name"], b3["exp_name"]]
assert report["test/accuracy"].notna().all() and (report["folds"] == 3).all()
per_class = rep.plot_per_class([b1["exp_name"], b3["exp_name"]], metric="recall")
assert list(per_class.index) == list(CIFAR10_CLASSES)
assert rep.champion_name("smoke_mlp_b1") == b1["exp_name"]
paired = rep.paired_comparison(final[1]["exp_name"], "smoke_mlp_b1__*")
assert list(paired["exp_name"]) == [b1["exp_name"]] and paired.loc[0, "folds_comuns"] == 3
assert paired.loc[0, "vitorias"] + paired.loc[0, "derrotas"] <= 3
figs = os.listdir(os.path.join(OUT, "_relatorio"))
for expected in ("heatmap_smoke_mlp_b1_mlp_layers_x_mlp_neurons__val-accuracy_mean.png", "barras_triagem_smoke_mlp_b2.png"):
    assert expected in figs, f"figura ausente: {expected}"
print("  OK report_utils: rankings sem teste, heatmaps, barras, curvas das finalistas e relatório final com teste só dos campeões")
