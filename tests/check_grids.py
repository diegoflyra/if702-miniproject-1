"""Valida todos os grids em grids/*.json sem treinar.

Para cada bloco: expande as combinações, valida cada uma com o argparse e a construção real do
modelo (dimensões de todos os blocos + forward) e confere as contagens esperadas. Blocos que
dependem de campeões ainda inexistentes usam, como campeão fictício, cada configuração válida
do bloco anterior — ou seja, TODA combinação possível de campeão é testada.
Uso: python tests/check_grids.py
"""

import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ["EXP_OUTPUT_DIR"] = os.path.join(ROOT, "outputs_check_grids_inexistente")

import torch

import grid_search as gs
from models import build_model
from run_experiment import parse_args

EXPECTED = {  # bloco: (combinações, válidas) com o campeão fictício padrão
    "mlp_b0_referencia": (2, 2), "mlp_b1_topologia": (25, 25), "mlp_b2_otimizacao": (14, 14),
    "mlp_b3_regularizacao": (8, 8),
    "cnn_b0_referencia": (1, 1), "cnn_b1_topologia": (24, 20), "cnn_b2_otimizacao": (14, 14),
    "mlp_b3b_dropout_longo": (4, 4), "cnn_b3b_pooling": (9, 6),
    "cnn_b4_augmentation": (8, 8), "mlp_b4_augmentation": (4, 4),
}

specs = {gs.load_spec(p)["block"]: p for p in sorted(glob.glob(os.path.join(ROOT, "grids", "*.json")))}
assert specs, "nenhum grid encontrado"


def fake_champion(config, block):
    return {"block": block, "exp_name": config["exp_name"], "hyperparams": config["params"],
            "val_mean": {gs.RANK_METRIC: 0.5}}


def valid_configs(block, champion_params=None):
    """Configurações válidas de um bloco, assumindo um campeão para o(s) bloco(s) de base."""
    spec = gs.load_spec(specs[block])

    def assume(blocks):
        if champion_params is not None:
            return fake_champion({"exp_name": "fake", "params": champion_params}, blocks[0])
        return fake_champion(valid_configs(blocks[0])[0][0], blocks[0])

    base, _ = gs.resolve_base(spec, assume=assume)
    configs = gs.expand(spec, base)
    return [c for c in configs if gs.validate(c, spec["max_parameters"])[0]], configs, spec


def forward_ok(config):
    args = parse_args(["--exp_name", config["exp_name"], *gs.to_argv(config["params"])])
    with torch.no_grad():
        out = build_model(args).eval()(torch.zeros(2, 3, 32, 32))
    return out.shape == (2, 10)


total = 0
for block, (n_all, n_valid) in EXPECTED.items():
    valid, configs, spec = valid_configs(block)
    assert len(configs) == n_all, f"{block}: {len(configs)} combinações, esperado {n_all}"
    assert len(valid) == n_valid, f"{block}: {len(valid)} válidas, esperado {n_valid}"
    assert all(forward_ok(c) for c in valid), f"{block}: forward falhou"
    assert len({c["exp_name"] for c in configs}) == len(configs), f"{block}: exp_name duplicado"
    assert all(c["params"]["model"] == spec["model"] and c["params"]["k_folds"] == spec["k_folds"] for c in configs)
    names = [c["exp_name"] for c in valid]
    print(f"  OK {block:<22} {len(configs):>3} combinações, {len(valid):>3} válidas  ex.: {names[0]}")
    total += len(valid)

# Herança: o Bloco 2 usa a topologia do campeão (fictício) do Bloco 1 e só varia otimizador/lr
b1 = valid_configs("mlp_b1_topologia")[0]
champ = b1[7]["params"]
b2 = valid_configs("mlp_b2_otimizacao", champion_params=champ)[0]
assert all(c["params"]["mlp_layers"] == champ["mlp_layers"] and c["params"]["mlp_neurons"] == champ["mlp_neurons"]
           for c in b2), "Bloco 2 não herdou a topologia do campeão"
assert {(c["params"]["optimizer"], c["params"]["lr"]) for c in b2} == {(o, l) for o in ("sgd", "adam") for l in gs.load_spec(specs["mlp_b2_otimizacao"])["grid"]["lr"]}
print(f"  OK herança MLP: Bloco 2 fixa L={champ['mlp_layers']} N={champ['mlp_neurons']} e varia 14 combinações de otimização")

# Bloco 3 com campeão herdando epochs/patience sobrescritos pela base do bloco
b3 = valid_configs("mlp_b3_regularizacao", champion_params={**b2[0]["params"]})[0]
assert all(c["params"]["epochs"] == 50 and c["params"]["patience"] == 7 for c in b3), "base do Bloco 3 não sobrescreveu"

# CNN: todo campeão possível do Bloco 1 gera Blocos 2 e 3 válidos (Bloco 3 descarta apenas o que excede limites)
cnn_b1 = valid_configs("cnn_b1_topologia")[0]
for champion in cnn_b1:
    b2_valid = valid_configs("cnn_b2_otimizacao", champion_params=champion["params"])[0]
    assert len(b2_valid) == 14, f"Bloco 2 inválido para campeão {champion['exp_name']}"
    b3_valid, b3_all, _ = valid_configs("cnn_b3_regularizacao", champion_params=champion["params"])
    assert len(b3_all) == 18 and len(b3_valid) >= 6, f"Bloco 3 com poucas válidas para {champion['exp_name']}: {len(b3_valid)}"
    assert all(forward_ok(c) for c in b3_valid)
    total += len(b3_valid)
print(f"  OK CNN: para cada um dos {len(cnn_b1)} campeões possíveis do Bloco 1, Blocos 2 (14) e 3 (6-18 válidas) são construíveis")

# Todo campeão possível da MLP gera um Bloco 3 válido
for champion in valid_configs("mlp_b1_topologia")[0]:
    assert len(valid_configs("mlp_b3_regularizacao", champion_params=champion["params"])[0]) == 8

# Checagem: specs apontam para blocos existentes e para chaves de topologia
for block in ("mlp_b2_checagem", "cnn_b2_checagem"):
    spec = gs.load_spec(specs[block])
    ref = spec["grid_from_ranking"]
    assert ref["block"] in specs and ref["ranks"] == [2, 3] and spec["screening_folds"] == [1, 2, 3, 4, 5]
    assert all(b in specs for b in spec["base_from"])
for block in ("mlp_b3_regularizacao", "cnn_b3_regularizacao"):
    assert set(gs.load_spec(specs[block])["base_from"]) == {block.replace("b3_regularizacao", "b2_otimizacao"),
                                                            block.replace("b3_regularizacao", "b2_checagem")}

# @base: com um campeão realista no bloco anterior, a configuração igual à receita herdada existe no grid
def pick(block, label, **overrides):
    config = next(c for c in valid_configs(block)[0] if c["exp_name"].endswith("__" + label))
    return {**config["params"], **overrides}


for block, champion_params, expected in (
        ("mlp_b3b_dropout_longo", pick("mlp_b3_regularizacao", "do0.2_losscross_entropy"), "do0.2_losscross_entropy"),
        ("mlp_b4_augmentation", pick("mlp_b3b_dropout_longo", "do0.2_losscross_entropy", mlp_layers=4), "aug0_do0.2"),
        ("cnn_b3b_pooling", {**pick("cnn_b3_regularizacao", "pool2_do0.5_bn1"), "conv_blocks": 4}, "B4_pool2"),
        ("cnn_b4_augmentation", pick("cnn_b3b_pooling", "B3_pool2", cnn_dropout=0.5), "aug0_B3_do0.5")):
    valid, configs, spec = valid_configs(block, champion_params=champion_params)
    base, _ = gs.resolve_base(spec, assume=lambda blocks: fake_champion({"exp_name": "fake", "params": champion_params}, blocks[0]))
    ref = gs.base_config(spec, base, valid)
    assert ref is not None and ref["exp_name"].endswith("__" + expected), f"{block}: config_base = {ref and ref['exp_name']}"
assert gs.load_spec(specs["cnn_b4_augmentation"])["confirm_also"] == ["@base"]
print("  OK @base: referência pareada resolvida nos blocos complementares e de augmentation")

print(f"\nGrids validados: {len(specs)} blocos, {total} configurações construídas com sucesso")
