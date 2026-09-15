"""Grid search em blocos com triagem, confirmação e seleção automática do campeão.

Uso:
    python src/grid_search.py grids/mlp_b1_topologia.json [--gpus N] [--workers_per_gpu W] [--dry_run]

Fluxo de um bloco (especificado em JSON, ver grids/):
  1. base: hiperparâmetros fixos do bloco + (opcional) o campeão de bloco(s) anterior(es), lido de
     {output_dir}/_grids/{bloco}/campeao.json — nenhum valor é escolhido manualmente;
  2. expansão: produto cartesiano dos eixos do grid (e/ou lista explícita de configurações);
     combinações inválidas (dimensões 0×0, parâmetros demais) são descartadas e registradas;
  3. triagem: cada configuração roda os folds de triagem (ex.: 1-3 de 5), em paralelo entre GPUs;
  4. ranking pela VALIDAÇÃO (média dos folds de triagem) — o teste nunca participa da escolha;
  5. confirmação: as N finalistas (e as referências em "confirm_also"; "@base" = a configuração idêntica
     à receita herdada) completam os folds restantes
     (--resume, sem refazer nada);
  6. campeão: maior média de validação nos K folds entre as finalistas → campeao.json.

Tudo é retomável: experimentos/folds já concluídos em disco são pulados.
Arquivos do bloco em {output_dir}/_grids/{bloco}/: spec.json, configs.csv, ranking_triagem.csv,
ranking_final.csv, campeao.json e logs/.
"""

import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

# Hiperparâmetros herdados do campeão (tudo que define o experimento, exceto identificação e operação).
NON_INHERITED = {"exp_name", "output_dir", "overwrite", "resume", "folds", "project", "entity", "tags", "notes",
                 "wandb_mode", "device", "data_dir", "iniciado_em"}
# Abreviações usadas nos nomes dos experimentos.
ABBREV = {
    "mlp_layers": "L", "mlp_neurons": "N", "activation": "act", "dropout": "do", "batch_norm": "bn",
    "optimizer": "opt", "lr": "lr", "weight_decay": "wd", "momentum": "mom", "loss_fn": "loss",
    "conv_blocks": "B", "filters": "f", "filters_growth": "fg", "convs_per_block": "cpb", "kernel_size": "k",
    "stride": "s", "padding": "pad", "pool_size": "pool", "fc_neurons": "fc", "cnn_dropout": "do",
    "cnn_batch_norm": "bn", "batch_size": "bs", "augment": "aug", "epochs": "ep", "patience": "pat",
}
RANK_METRIC = "val/accuracy"


def output_dir():
    return os.environ.get("EXP_OUTPUT_DIR", "./outputs")


def grid_dir(block):
    return os.path.join(output_dir(), "_grids", block)


def load_spec(path):
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    for key in ("block", "model", "k_folds"):
        if key not in spec:
            raise ValueError(f"{path}: campo obrigatório '{key}' ausente")
    spec.setdefault("base", {})
    spec.setdefault("grid", {})
    spec.setdefault("configs", [])
    spec.setdefault("screening_folds", list(range(1, spec["k_folds"] + 1)))
    spec.setdefault("finalists", 3)
    spec.setdefault("max_parameters", None)
    return spec


def _json_write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _csv_write(path, rows, fields=None):
    fields = fields or (list(rows[0].keys()) if rows else [])
    for row in rows:
        fields += [k for k in row if k not in fields]
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    os.replace(tmp, path)


# ============================================================================ base / campeões
def load_champion(block):
    path = os.path.join(grid_dir(block), "campeao.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def best_champion(blocks):
    """Entre os campeões de vários blocos, o de maior média de validação nos K folds."""
    champions = []
    for block in blocks:
        champion = load_champion(block)
        if champion is None:
            raise FileNotFoundError(
                f"Campeão do bloco '{block}' não encontrado em {grid_dir(block)}/campeao.json. "
                f"Execute esse bloco antes.")
        champions.append(champion)
    return max(champions, key=lambda c: c["val_mean"][RANK_METRIC])


def resolve_base(spec, assume=None):
    """Base do bloco = hiperparâmetros do campeão de base_from (se houver) sobrescritos por spec['base'].

    `assume` (usado em dry-run/testes) fornece um campeão fictício quando o real ainda não existe.
    """
    base, source = {}, None
    if spec.get("base_from"):
        try:
            champion = best_champion(spec["base_from"])
        except FileNotFoundError:
            if assume is None:
                raise
            champion = assume(spec["base_from"])
        base.update(champion["hyperparams"])
        source = {"block": champion["block"], "exp_name": champion["exp_name"],
                  "val_mean": champion["val_mean"].get(RANK_METRIC)}
    base.update(spec["base"])
    base["model"] = spec["model"]
    base["k_folds"] = spec["k_folds"]
    return base, source


# ============================================================================ expansão
def _axis_items(name, values):
    """Um eixo pode ter valores simples ou dicionários que alteram vários parâmetros juntos (com _label)."""
    items = []
    for value in values:
        if isinstance(value, dict):
            params = {k: v for k, v in value.items() if k != "_label"}
            label = value.get("_label") or "_".join(f"{ABBREV.get(k, k)}{_fmt(v)}" for k, v in params.items())
        else:
            params = {name: value}
            label = f"{ABBREV.get(name, name)}{_fmt(value)}"
        items.append((params, label, {name: value.get("_label", label) if isinstance(value, dict) else value}))
    return items


def _fmt(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return "-".join(_fmt(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def expand(spec, base):
    """Lista de configurações {exp_name, params, axes} do bloco."""
    configs = []
    axes = list(spec["grid"].items())
    if axes:
        for combo in itertools.product(*[_axis_items(name, values) for name, values in axes]):
            params, labels, axis_values = dict(base), [], {}
            for p, label, av in combo:
                params.update(p)
                labels.append(label)
                axis_values.update(av)
            configs.append((params, labels, axis_values))

    for extra in spec["configs"]:
        overrides = {k: v for k, v in extra.items() if k != "_label"}
        label = extra.get("_label") or "_".join(f"{ABBREV.get(k, k)}{_fmt(v)}" for k, v in overrides.items())
        configs.append(({**base, **overrides}, [label], {"config": label}))

    if spec.get("grid_from_ranking"):
        ref = spec["grid_from_ranking"]
        ranking = read_ranking(ref["block"], "final") or read_ranking(ref["block"], "triagem")
        if not ranking:
            raise FileNotFoundError(f"Ranking do bloco '{ref['block']}' não encontrado para grid_from_ranking.")
        for row in ranking:
            if int(row["rank"]) in ref["ranks"]:
                hp = json.loads(row["hyperparams"])
                overrides = {k: hp[k] for k in ref["keys"]}
                label = "_".join(f"{ABBREV.get(k, k)}{_fmt(v)}" for k, v in overrides.items())
                configs.append(({**base, **overrides}, [f"rank{row['rank']}", label],
                                {"origem": f"{ref['block']} rank {row['rank']}", **overrides}))

    result, seen = [], set()
    for params, labels, axis_values in configs:
        name = f"{spec['block']}__{'_'.join(labels)}".replace("/", "-")
        if name in seen:
            continue
        seen.add(name)
        result.append({"exp_name": name, "params": params, "axes": axis_values})
    return result


def base_config(spec, base, configs):
    """Configuração do grid idêntica à receita herdada em todos os eixos (ex.: a versão sem augmentation do campeão).

    Serve de referência pareada e de checagem de reprodutibilidade; None se a receita não estiver no grid.
    """
    keys = []
    for name, values in spec["grid"].items():
        dict_keys = {k for v in values if isinstance(v, dict) for k in v if k != "_label"}
        keys += sorted(dict_keys) if dict_keys else [name]
    if not keys:
        return None
    from run_experiment import build_parser
    parser = build_parser()
    expected = {k: base[k] if k in base else parser.get_default(k) for k in keys}  # ausente = padrão do argparse
    return next((c for c in configs if all(c["params"].get(k, parser.get_default(k)) == expected[k] for k in keys)), None)


def to_argv(params):
    argv = []
    for key, value in params.items():
        if key in NON_INHERITED or value is None or value is False:
            continue
        if value is True:
            argv.append(f"--{key}")
        elif isinstance(value, (list, tuple)):
            argv += [f"--{key}", *[str(v) for v in value]]
        else:
            argv += [f"--{key}", str(value)]
    return argv


def validate(config, max_parameters=None):
    """Valida com o argparse e a construção real do modelo. Retorna (ok, motivo, n_parâmetros)."""
    import contextlib
    import io

    from models import build_model, count_parameters
    from run_experiment import parse_args

    argv = ["--exp_name", config["exp_name"], *to_argv(config["params"])]
    try:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            args = parse_args(argv)
    except SystemExit:
        return False, err.getvalue().strip().splitlines()[-1], None
    try:
        n_params = count_parameters(build_model(args))
    except ValueError as exc:
        return False, str(exc), None
    if max_parameters and n_params > max_parameters:
        return False, f"{n_params:,} parâmetros excede max_parameters={max_parameters:,}", n_params
    return True, "", n_params


# ============================================================================ execução
def experiment_status(exp_name):
    path = os.path.join(output_dir(), exp_name, "resultados.json")
    if not os.path.isfile(path):
        return set()
    try:
        with open(path, encoding="utf-8") as f:
            results = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set()
    return {f["fold"] for f in results.get("folds", []) if f.get("status") == "concluido"}


def run_jobs(jobs, log_dir, gpus, workers_per_gpu, stage):
    """Executa run_experiment.py para cada job, distribuindo entre GPUs. Retorna {exp_name: returncode}."""
    os.makedirs(log_dir, exist_ok=True)
    pending = [j for j in jobs if not set(j["folds"]) <= experiment_status(j["exp_name"])]
    skipped = len(jobs) - len(pending)
    print(f"\n[{stage}] {len(jobs)} experimentos | {skipped} já concluídos em disco | {len(pending)} a executar "
          f"| {max(gpus, 1)} GPU(s) × {workers_per_gpu} worker(s)", flush=True)
    if not pending:
        return {}

    slots = [str(g) for g in range(gpus) for _ in range(workers_per_gpu)] or [None] * workers_per_gpu
    free = list(slots)
    lock = threading.Lock()
    done_count = [0]
    results = {}
    started = time.time()

    def run(job):
        with lock:
            slot = free.pop()
        try:
            env = dict(os.environ)
            if slot is not None:
                env["CUDA_VISIBLE_DEVICES"] = slot
            cmd = [sys.executable, os.path.join(SRC_DIR, "run_experiment.py"), "--exp_name", job["exp_name"],
                   *to_argv(job["params"]), "--folds", *map(str, job["folds"]), "--resume"]
            log_path = os.path.join(log_dir, f"{job['exp_name']}.log")
            t0 = time.time()
            code = None
            for attempt in (1, 2):  # uma nova tentativa para falhas transitórias (ex.: CUDA)
                with open(log_path, "a", encoding="utf-8") as log:
                    log.write(f"\n===== {stage} | tentativa {attempt} | {' '.join(cmd)}\n")
                    log.flush()
                    code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
                if code == 0:
                    break
            return job, code, time.time() - t0, slot
        finally:
            with lock:
                free.append(slot)

    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(run, job) for job in pending]
        for future in as_completed(futures):
            job, code, elapsed, slot = future.result()
            done_count[0] += 1
            results[job["exp_name"]] = code
            summary = _fold_summary(job["exp_name"], job["folds"])
            status = "ok" if code == 0 else f"FALHOU (código {code}, ver log)"
            total = (time.time() - started) / 60
            print(f"  [{done_count[0]}/{len(pending)}] {status} {job['exp_name']} (gpu {slot}) "
                  f"{summary} | {elapsed / 60:.1f} min | decorrido {total:.0f} min", flush=True)
    return results


def _fold_summary(exp_name, folds):
    stats = collect(exp_name, folds)
    if stats["n_folds"] == 0:
        return "sem folds concluídos"
    return f"val_acc {stats['val_mean'].get(RANK_METRIC, float('nan')):.4f} ({stats['n_folds']} folds)"


# ============================================================================ ranking
def collect(exp_name, folds):
    """Média/desvio das métricas de validação (melhor época) nos folds indicados que estão concluídos."""
    exp_dir = os.path.join(output_dir(), exp_name)
    out = {"exp_name": exp_name, "n_folds": 0, "val_mean": {}, "val_std": {}, "melhor_epoca_media": None,
           "tempo_min": None, "num_parameters": None, "hyperparams": {}, "divergiu": False}
    try:
        with open(os.path.join(exp_dir, "resultados.json"), encoding="utf-8") as f:
            results = json.load(f)
        with open(os.path.join(exp_dir, "parametros.json"), encoding="utf-8") as f:
            params = json.load(f)
    except (OSError, json.JSONDecodeError):
        return out
    chosen = [f for f in results.get("folds", []) if f.get("status") == "concluido" and f["fold"] in folds]
    out["n_folds"] = len(chosen)
    out["num_parameters"] = params.get("num_parameters")
    out["hyperparams"] = {k: v for k, v in params.items() if k in _cli_dests() and k not in NON_INHERITED}
    out["divergiu"] = any(f.get("divergiu_na_epoca") for f in chosen)
    if not chosen:
        return out
    keys = ("val/accuracy", "val/loss", "val/f1_macro", "train/accuracy", "gap/accuracy")
    for key in keys:
        values = [f.get("melhor_val", {}).get(key) for f in chosen]
        values = [v if isinstance(v, (int, float)) and v is not None and math.isfinite(v) else float("nan") for v in values]
        mean = sum(values) / len(values)
        out["val_mean"][key] = mean
        out["val_std"][key] = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    out["melhor_epoca_media"] = sum(f.get("melhor_epoca") or 0 for f in chosen) / len(chosen)
    hist = os.path.join(exp_dir, "historico_treino.csv")
    if os.path.isfile(hist):
        with open(hist, newline="", encoding="utf-8") as f:
            out["tempo_min"] = sum(float(r["epoch_time_s"]) for r in csv.DictReader(f)
                                   if r.get("epoch_time_s") and int(r["fold"]) in folds) / 60
    return out


_DESTS = None


def _cli_dests():
    global _DESTS
    if _DESTS is None:
        from run_experiment import build_parser
        _DESTS = {a.dest for a in build_parser()._actions if a.dest != "help"}
    return _DESTS


def _score(stats):
    value = stats["val_mean"].get(RANK_METRIC, float("nan"))
    return value if math.isfinite(value) else -1.0


def build_ranking(configs, folds, required_folds):
    """Ordena por val/accuracy média (desc). Configurações sem todos os folds exigidos vão para o fim."""
    rows = []
    for config in configs:
        stats = collect(config["exp_name"], folds)
        complete = stats["n_folds"] == len(required_folds)
        rows.append((complete, _score(stats), config, stats))
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
    ranking = []
    for i, (complete, _, config, stats) in enumerate(rows, start=1):
        row = {"rank": i, "exp_name": config["exp_name"], **{f"eixo:{k}": v for k, v in config["axes"].items()},
               "folds": stats["n_folds"], "completo": complete, "divergiu": stats["divergiu"],
               "val/accuracy_mean": stats["val_mean"].get("val/accuracy"),
               "val/accuracy_std": stats["val_std"].get("val/accuracy"),
               "val/loss_mean": stats["val_mean"].get("val/loss"),
               "val/f1_macro_mean": stats["val_mean"].get("val/f1_macro"),
               "train/accuracy_mean": stats["val_mean"].get("train/accuracy"),
               "gap/accuracy_mean": stats["val_mean"].get("gap/accuracy"),
               "melhor_epoca_media": stats["melhor_epoca_media"], "num_parameters": stats["num_parameters"],
               "tempo_treino_min": stats["tempo_min"], "hyperparams": stats["hyperparams"]}
        ranking.append(row)
    return ranking


def read_ranking(block, stage):
    path = os.path.join(grid_dir(block), f"ranking_{stage}.csv")
    if not os.path.isfile(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ============================================================================ bloco
def run_block(spec_path, gpus=None, workers_per_gpu=1, dry_run=False, assume=None):
    spec = load_spec(spec_path)
    block, k = spec["block"], spec["k_folds"]
    screening = sorted(spec["screening_folds"])
    all_folds = list(range(1, k + 1))
    out = grid_dir(block)

    base, source = resolve_base(spec, assume=assume)
    configs = expand(spec, base)
    table, valid = [], []
    for config in configs:
        ok, reason, n_params = validate(config, spec["max_parameters"])
        table.append({"exp_name": config["exp_name"], **{f"eixo:{a}": v for a, v in config["axes"].items()},
                      "valida": ok, "motivo": reason, "num_parameters": n_params})
        if ok:
            valid.append(config)

    print(f"Bloco {block} ({spec['model']}): {len(configs)} combinações | {len(valid)} válidas | "
          f"{len(configs) - len(valid)} descartadas", flush=True)
    if source:
        print(f"Base herdada do campeão {source['exp_name']} (bloco {source['block']}, "
              f"val_acc {source['val_mean']:.4f})", flush=True)
    for row in table:
        if not row["valida"]:
            print(f"  descartada: {row['exp_name']} — {row['motivo']}", flush=True)
    if dry_run:
        return {"spec": spec, "base": base, "configs": configs, "valid": valid, "table": table}

    os.makedirs(out, exist_ok=True)
    reference = base_config(spec, base, valid)
    _json_write(os.path.join(out, "spec.json"), {**spec, "base_resolvida": base, "base_herdada_de": source,
                                                 "config_base": reference["exp_name"] if reference else None})
    _csv_write(os.path.join(out, "configs.csv"), table)

    if gpus is None:
        import torch
        gpus = torch.cuda.device_count()
    log_dir = os.path.join(out, "logs")

    # 1) Triagem
    run_jobs([{"exp_name": c["exp_name"], "params": c["params"], "folds": screening} for c in valid],
             log_dir, gpus, workers_per_gpu, "triagem")
    triage = build_ranking(valid, screening, screening)
    _csv_write(os.path.join(out, "ranking_triagem.csv"), triage)

    # 2) Confirmação das finalistas nos folds restantes
    finalists_n = spec["finalists"] if len(screening) < k else len(valid)
    finalists = [r for r in triage if r["completo"]][:finalists_n]
    names = {r["exp_name"] for r in finalists}
    # Referências que precisam dos K folds mesmo sem estar entre as finalistas (ex.: a versão sem augmentation).
    labels = [label for label in spec.get("confirm_also", []) if label != "@base"]
    names |= {f"{block}__{label}" for label in labels} & {c["exp_name"] for c in valid}
    if "@base" in spec.get("confirm_also", []) and reference:
        names.add(reference["exp_name"])
    finalist_configs = [c for c in valid if c["exp_name"] in names]
    if len(screening) < k and finalist_configs:
        run_jobs([{"exp_name": c["exp_name"], "params": c["params"], "folds": all_folds} for c in finalist_configs],
                 log_dir, gpus, workers_per_gpu, "confirmação")
    final = build_ranking(finalist_configs, all_folds, all_folds)
    _csv_write(os.path.join(out, "ranking_final.csv"), final)

    # 3) Campeão
    complete = [r for r in final if r["completo"]]
    if not complete:
        print(f"ATENÇÃO: nenhuma finalista completou os {k} folds; campeão não definido.", flush=True)
        return None
    top = complete[0]
    stats = collect(top["exp_name"], all_folds)
    champion = {
        "block": block, "exp_name": top["exp_name"], "criterio": f"maior {RANK_METRIC} média em {k} folds",
        "axes": next(c["axes"] for c in valid if c["exp_name"] == top["exp_name"]),
        "hyperparams": stats["hyperparams"], "val_mean": stats["val_mean"], "val_std": stats["val_std"],
        "num_parameters": stats["num_parameters"], "finalistas": [r["exp_name"] for r in final],
    }
    _json_write(os.path.join(out, "campeao.json"), champion)
    print(f"\nCampeão do bloco {block}: {top['exp_name']} | val_acc "
          f"{stats['val_mean'][RANK_METRIC]:.4f} ± {stats['val_std'][RANK_METRIC]:.4f} ({k} folds)", flush=True)
    return champion


def main():
    p = argparse.ArgumentParser(description="Grid search em blocos (triagem + confirmação + campeão)")
    p.add_argument("spec", help="Arquivo JSON do bloco (ver grids/)")
    p.add_argument("--gpus", type=int, default=None, help="Número de GPUs (padrão: todas as visíveis)")
    p.add_argument("--workers_per_gpu", type=int, default=1, help="Experimentos simultâneos por GPU")
    p.add_argument("--dry_run", action="store_true", help="Apenas expande e valida o grid")
    args = p.parse_args()
    result = run_block(args.spec, gpus=args.gpus, workers_per_gpu=args.workers_per_gpu, dry_run=args.dry_run)
    if not args.dry_run and result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
