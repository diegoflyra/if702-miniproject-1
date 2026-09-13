"""Ponto de entrada dos experimentos.

Exemplos:
    python src/run_experiment.py --model mlp --exp_name mlp_baseline --mlp_layers 3 --mlp_neurons 64 128 64
    python src/run_experiment.py --model cnn --exp_name cnn_lenet --conv_blocks 2 --filters 32 --kernel_size 3 \\
        --padding same --pool_size 2 --fc_neurons 120 84

Todo experimento grava em {output_dir}/{exp_name}/ (padrão ./outputs; ver experiment_logger.py):
parametros.json, historico_treino.csv, melhor_modelo.pth, modelo_final.pth,
resultados.json e matriz_confusao_teste.csv — independentemente do wandb.
Execute `python src/run_experiment.py --help` para a lista completa de hiperparâmetros.
"""

import argparse
import json
import os
import platform
import random
import re
import subprocess
import sys
import traceback

import numpy as np
import sklearn
import torch
import torchvision
import wandb

from data_loader import CIFAR10_CLASSES, INPUT_SHAPE, NUM_CLASSES, get_fold_loaders, load_cifar10, make_splits
from experiment_logger import FINAL_WEIGHTS_FILE, ExperimentLogger, now_iso
from models import build_model, count_parameters
from train import fit, safe_wandb


MLP_ONLY = ("mlp_layers", "mlp_neurons", "dropout", "batch_norm")
CNN_ONLY = ("conv_blocks", "filters", "filters_growth", "convs_per_block", "kernel_size", "stride",
            "padding", "pool_size", "fc_neurons", "cnn_dropout", "cnn_batch_norm")


def padding_type(value):
    if value.lower() in ("same", "valid"):
        return value.lower()
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("use 'same', 'valid' ou um inteiro >= 0")
    if number < 0:
        raise argparse.ArgumentTypeError("padding deve ser >= 0")
    return number


def rate_type(value):
    rate = float(value)
    if not 0.0 <= rate < 1.0:
        raise argparse.ArgumentTypeError("deve estar no intervalo [0.0, 1.0)")
    return rate


def build_parser():
    p = argparse.ArgumentParser(
        description="Experimentos CIFAR-10 (MLP/CNN) com logging local (outputs/) e Weights & Biases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("experimento")
    g.add_argument("--model", choices=["mlp", "cnn"], required=True)
    g.add_argument("--exp_name", required=True, help="Nome do experimento (run no wandb e pasta em --output_dir)")
    g.add_argument("--output_dir", default=os.environ.get("EXP_OUTPUT_DIR", "./outputs"),
                   help="Resultados em {output_dir}/{exp_name}/ (padrão pode vir da variável EXP_OUTPUT_DIR)")
    g.add_argument("--overwrite", action="store_true",
                   help="Apaga {output_dir}/{exp_name}/ se já existir (sem a flag: aborta)")
    g.add_argument("--resume", action="store_true",
                   help="Se {output_dir}/{exp_name}/ existir, mantém os folds concluídos e roda apenas os pendentes "
                        "(exige os mesmos hiperparâmetros)")
    g.add_argument("--project", default="cifar10-rn", help="Projeto no wandb")
    g.add_argument("--entity", default=None, help="Usuário/time no wandb")
    g.add_argument("--tags", nargs="*", default=[])
    g.add_argument("--notes", default=None, help="Hipótese/motivação do experimento")
    g.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default=None,
                   help="Sobrescreve WANDB_MODE (None = online)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")

    g = p.add_argument_group("dados")
    g.add_argument("--data_dir", default="./data")
    g.add_argument("--batch_size", type=int, default=64)
    g.add_argument("--k_folds", type=int, default=1,
                   help="1 = holdout com --val_split; K > 1 = validação cruzada estratificada em K folds "
                        "(treina K modelos; tempo ≈ K × holdout)")
    g.add_argument("--folds", type=int, nargs="+", default=None,
                   help="Subconjunto de folds a executar (1..K), ex.: --folds 1 2 3 para triagem; "
                        "combine com --resume para completar os demais depois")
    g.add_argument("--val_split", type=float, default=0.1,
                   help="Fração do treino usada como validação no holdout (k_folds=1), em (0, 1)")
    g.add_argument("--subset_size", type=int, default=None,
                   help="Limita treino e teste a N amostras (testes rápidos)")
    g.add_argument("--augment", action="store_true", help="RandomCrop + HorizontalFlip no treino")

    g = p.add_argument_group("otimização (MLP e CNN)")
    g.add_argument("--epochs", type=int, default=20)
    g.add_argument("--optimizer", choices=["sgd", "adam", "adamw", "rmsprop"], default="adam")
    g.add_argument("--lr", type=float, default=1e-3, help="Taxa de aprendizagem")
    g.add_argument("--momentum", type=float, default=0.9, help="Momentum do SGD/RMSprop")
    g.add_argument("--weight_decay", type=float, default=0.0, help="Regularização L2 (via otimizador)")
    g.add_argument("--loss_fn", choices=["cross_entropy", "mse"], default="cross_entropy",
                   help="mse = MSE entre softmax(logits) e one-hot")
    g.add_argument("--patience", type=int, default=5,
                   help="Early stopping em val/loss (0 desativa; use 0 para observar overfitting)")
    g.add_argument("--eval_train", action="store_true",
                   help="Recalcula métricas de treino em modo eval ao fim de cada época "
                        "(gap treino-validação justo com dropout/augmentation; custa uma passada extra)")
    g.add_argument("--activation", choices=["relu", "tanh", "leaky_relu", "gelu", "elu", "sigmoid"],
                   default="relu", help="Função de ativação das camadas ocultas")

    g = p.add_argument_group("arquitetura MLP")
    g.add_argument("--mlp_layers", type=int, default=3, help="Número de camadas ocultas (0 = regressão logística)")
    g.add_argument("--mlp_neurons", type=int, nargs="+", default=[128],
                   help="Neurônios por camada oculta: 1 valor (repetido) ou 1 valor por camada")
    g.add_argument("--dropout", type=rate_type, default=0.0, help="Dropout após cada camada oculta da MLP")
    g.add_argument("--batch_norm", action="store_true", help="BatchNorm1d após cada Linear oculta da MLP")

    g = p.add_argument_group("arquitetura CNN")
    g.add_argument("--conv_blocks", type=int, default=2, help="Blocos Conv2d → (BN) → Ativação → MaxPool")
    g.add_argument("--filters", type=int, default=32, help="Filtros do primeiro bloco")
    g.add_argument("--filters_growth", choices=["double", "constant"], default="double",
                   help="double: filters × 2^i no bloco i | constant: mesmo número em todos")
    g.add_argument("--convs_per_block", type=int, default=1, help="Convoluções empilhadas por bloco")
    g.add_argument("--kernel_size", type=int, default=3, help="Kernel k×k de todas as convoluções")
    g.add_argument("--stride", type=int, default=1, help="Stride da (primeira) convolução de cada bloco")
    g.add_argument("--padding", type=padding_type, default="same", help="'same', 'valid' ou inteiro")
    g.add_argument("--pool_size", type=int, default=2, help="MaxPool p×p com stride p ao fim de cada bloco (0/1 desativa)")
    g.add_argument("--fc_neurons", type=int, nargs="+", default=[128],
                   help="Neurônios da(s) camada(s) densa(s) antes da saída de 10 classes")
    g.add_argument("--cnn_dropout", type=rate_type, default=0.0, help="Dropout nas camadas densas da CNN")
    g.add_argument("--cnn_batch_norm", action="store_true", help="BatchNorm2d após cada convolução")

    return p


def parse_args(argv=None):
    p = build_parser()
    args = p.parse_args(argv)

    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", args.exp_name):
        p.error("--exp_name deve conter apenas letras, números, '_', '-' e '.' (vira nome de pasta).")
    if not 0.0 < args.val_split < 1.0:
        p.error("--val_split deve estar em (0, 1): o melhor modelo é escolhido pela validação.")
    if args.k_folds < 1:
        p.error("--k_folds deve ser >= 1.")
    if args.folds is not None:
        if any(f < 1 or f > args.k_folds for f in args.folds) or len(set(args.folds)) != len(args.folds):
            p.error(f"--folds deve conter valores distintos entre 1 e --k_folds ({args.k_folds}).")
        args.folds = sorted(args.folds)
    if args.overwrite and args.resume:
        p.error("--overwrite e --resume são mutuamente exclusivos.")
    if args.k_folds > 1 and args.val_split != p.get_default("val_split"):
        p.error("--val_split não se aplica com --k_folds > 1 (cada fold usa 1/K dos dados como validação).")
    for name in ("epochs", "batch_size"):
        if getattr(args, name) < 1:
            p.error(f"--{name} deve ser >= 1.")
    if args.lr <= 0:
        p.error("--lr deve ser > 0.")
    if args.weight_decay < 0:
        p.error("--weight_decay deve ser >= 0.")

    # Impede que um argumento do outro tipo de rede seja passado e ignorado em
    # silêncio, o que invalidaria a comparação entre experimentos.
    foreign = CNN_ONLY if args.model == "mlp" else MLP_ONLY
    misused = [f"--{name}" for name in foreign if getattr(args, name) != p.get_default(name)]
    if misused:
        hint = " (na CNN use --cnn_dropout / --cnn_batch_norm)" if args.model == "cnn" else ""
        p.error(f"{', '.join(misused)} não se aplica(m) a --model {args.model}{hint}.")
    return args


# Argumentos operacionais: não alteram o experimento e não entram em parametros.json.
OPERATIONAL = ("overwrite", "resume", "folds")
# Argumentos que podem mudar entre execuções de um mesmo experimento retomado.
RESUME_IGNORED = ("output_dir", "project", "entity", "tags", "notes", "wandb_mode", "device", "data_dir")


def experiment_params(args):
    """Hiperparâmetros da rodada; os que não pertencem ao modelo escolhido ficam como null."""
    params = {"exp_name": args.exp_name, **{k: v for k, v in vars(args).items() if k not in OPERATIONAL}}
    for name in (CNN_ONLY if args.model == "mlp" else MLP_ONLY):
        params[name] = None
    if args.k_folds > 1:
        params["val_split"] = None
    return params


def check_resume_compatible(existing, new):
    """Garante que um experimento retomado usa exatamente os mesmos hiperparâmetros."""
    keys = [k for k in new if k not in RESUME_IGNORED and k != "iniciado_em"]
    diff = {k: (existing.get(k), new[k]) for k in keys if existing.get(k) != new[k]}
    if diff:
        details = ", ".join(f"{k}: {a!r} → {b!r}" for k, (a, b) in diff.items())
        raise ValueError(f"--resume com hiperparâmetros diferentes do experimento existente ({details}). "
                         f"Use outro --exp_name ou --overwrite.")


def load_dotenv(path=".env"):
    """Carrega KEY=VALUE de um .env local (ex.: wandb_api_key) sem sobrescrever o ambiente.

    As chaves são convertidas para maiúsculas, pois o wandb lê WANDB_API_KEY.
    """
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.removeprefix("export ").strip().upper()
            os.environ.setdefault(key, value.strip().strip("'\""))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def environment_info(device):
    return {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "scikit_learn": sklearn.__version__,
        "wandb": wandb.__version__,
        "git_commit": git_commit(),
        "comando": " ".join([os.path.basename(sys.executable), *sys.argv]),
    }


def sanity_check_shapes(model, loader, device):
    """Valida as dimensões de entrada/saída com um batch real antes de treinar."""
    images, _ = next(iter(loader))
    assert images.shape[1:] == INPUT_SHAPE, f"Entrada inesperada: {tuple(images.shape)}"
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
    expected = (images.size(0), NUM_CLASSES)
    assert outputs.shape == expected, f"Saída {tuple(outputs.shape)} != esperado {expected}"
    print(f"[shape check] entrada {tuple(images.shape)} -> saída {tuple(outputs.shape)} OK", flush=True)


def init_wandb(args, params, fold):
    """Inicializa o wandb (uma run por fold, agrupadas pelo exp_name); se falhar, segue desativado."""
    name = args.exp_name if args.k_folds == 1 else f"{args.exp_name}-fold{fold}"
    kwargs = dict(
        project=args.project, entity=args.entity, name=name, group=args.exp_name,
        config={**params, "fold": fold}, tags=args.tags, notes=args.notes,
        job_type="train" if args.k_folds == 1 else "cv-fold",
    )
    try:
        return wandb.init(mode=args.wandb_mode, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] falha ao inicializar ({type(exc).__name__}: {exc}). "
              f"Continuando sem wandb — os arquivos locais serão gravados normalmente.", flush=True)
        return wandb.init(mode="disabled", **kwargs)


def main():
    args = parse_args()
    load_dotenv()

    # Criado antes de qualquer outra coisa: se algo falhar daqui em diante,
    # a pasta e os parâmetros da rodada já existem em disco.
    params = experiment_params(args)
    previous_params_path = os.path.join(args.output_dir, args.exp_name, "parametros.json")
    previous = None
    if args.resume and os.path.isfile(previous_params_path):
        # Verificado antes de tocar em qualquer arquivo do experimento existente.
        with open(previous_params_path, encoding="utf-8") as f:
            previous = json.load(f)
        check_resume_compatible(previous, params)

    logger = ExperimentLogger(args.output_dir, args.exp_name, k_folds=args.k_folds,
                              overwrite=args.overwrite, resume=args.resume)
    params["iniciado_em"] = (previous or {}).get("iniciado_em") if logger.resumed else logger.summary["iniciado_em"]
    logger.save_params(params)
    logger.save_results()
    print(f"Resultados em: {os.path.abspath(logger.dir)}", flush=True)

    requested = args.folds or list(range(1, args.k_folds + 1))
    pending = [f for f in requested if f not in logger.completed_folds()]
    if logger.resumed:
        print(f"[resume] folds concluídos: {logger.completed_folds()} | a executar: {pending}", flush=True)
    if not pending:
        logger.summary["status"] = "concluido" if len(logger.completed_folds()) == args.k_folds else "parcial"
        logger.save_results()
        print(f"Nada a executar. Status: {logger.summary['status']} | arquivos em {logger.dir}", flush=True)
        return

    model = None
    run = None
    try:
        set_seed(args.seed)
        device = resolve_device(args.device)

        # Construído antes dos dados: configurações inválidas falham em segundos.
        model = build_model(args, input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
        print(model)
        params.update({
            "num_parameters": count_parameters(model),
            **model.describe(),
            "arquitetura": str(model),
            **environment_info(device),
        })
        logger.save_params(params)
        print(f"Parâmetros treináveis: {params['num_parameters']:,} | device: {device}", flush=True)
        for key, value in model.describe().items():
            print(f"  {key}: {value}", flush=True)

        data = load_cifar10(args.data_dir, device)
        splits = make_splits(data["train_y"], k_folds=args.k_folds, val_split=args.val_split,
                             subset_size=args.subset_size, seed=args.seed)
        params.update({
            "validacao": "holdout estratificado" if args.k_folds == 1 else f"{args.k_folds}-fold estratificado",
            "n_train_por_fold": [len(tr) for tr, _ in splits],
            "n_val_por_fold": [len(va) for _, va in splits],
            "n_test": len(data["test_y"]) if args.subset_size is None else min(args.subset_size, len(data["test_y"])),
        })
        logger.save_params(params)

        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            if fold not in pending:
                continue
            if args.k_folds > 1:
                print(f"\n{'=' * 20} Fold {fold}/{args.k_folds} "
                      f"(treino {len(train_idx)}, validação {len(val_idx)}) {'=' * 20}", flush=True)
            # Mesma seed por fold em todos os experimentos: inicialização e ordem dos batches pareadas.
            fold_seed = args.seed + fold - 1
            set_seed(fold_seed)
            model = build_model(args, input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES).to(device)
            loaders = get_fold_loaders(data, train_idx, val_idx, batch_size=args.batch_size, augment=args.augment,
                                       eval_train=args.eval_train, subset_size=args.subset_size, seed=fold_seed)
            if fold == pending[0]:
                sanity_check_shapes(model, loaders[0], device)

            fold_summary = logger.start_fold(fold, n_train=len(train_idx), n_val=len(val_idx))
            run = init_wandb(args, params, fold)
            safe_wandb(wandb.watch, model, log="gradients", log_freq=100)
            try:
                fit(model, loaders, args, device, CIFAR10_CLASSES, logger, fold_summary)
            finally:
                safe_wandb(run.finish)
                run = None
            logger.finish_fold(fold_summary)

        logger.summary["status"] = "concluido" if len(logger.completed_folds()) == args.k_folds else "parcial"
        if args.k_folds > 1:
            val_m, val_s = logger.summary["validacao_media"], logger.summary["validacao_std"]
            test_m, test_s = logger.summary["teste"], logger.summary["teste_std"]
            print(f"\nfolds {logger.completed_folds()} de {args.k_folds} | val_acc {val_m['val/accuracy']:.4f} ± {val_s['val/accuracy']:.4f} | "
                  f"test_acc {test_m['test/accuracy']:.4f} ± {test_s['test/accuracy']:.4f} | "
                  f"melhor fold: {logger.summary['melhor_fold']}", flush=True)

    except KeyboardInterrupt:
        logger.summary["status"] = "interrompido"
        raise
    except BaseException as exc:
        logger.summary["status"] = "falhou"
        logger.summary["erro"] = f"{type(exc).__name__}: {exc}"
        logger.summary["traceback"] = traceback.format_exc()
        raise
    finally:
        # Executa em 100% dos casos: sucesso, erro ou Ctrl+C.
        if logger.current_fold is not None and logger.current_fold["status"] == "em_execucao":
            logger.current_fold["status"] = logger.summary["status"]
        if model is not None:
            try:
                path = logger.save_weights(model, logger.path(FINAL_WEIGHTS_FILE))
                logger.summary["pesos_finais"] = FINAL_WEIGHTS_FILE
                print(f"Pesos finais salvos em {path}", flush=True)
            except Exception as exc:  # noqa: BLE001
                logger.summary["erro_ao_salvar_pesos"] = f"{type(exc).__name__}: {exc}"
        logger.summary["finalizado_em"] = now_iso()
        logger.save_results()
        if run is not None:
            safe_wandb(run.finish)
        print(f"Status: {logger.summary['status']} | arquivos em {logger.dir}", flush=True)


if __name__ == "__main__":
    main()
