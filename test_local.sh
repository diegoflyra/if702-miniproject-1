#!/usr/bin/env bash
# Smoke test local: dados reduzidos, poucas épocas, wandb offline.
#   1) auditorias sem treino: flags → arquitetura, data loader em GPU/K-fold, todos os grids dos notebooks
#   2) experimentos isolados: holdout, K-fold, divergência e retomada (--folds/--resume)
#   3) cadeia de grid search em miniatura: Bloco 1 → Bloco 2 → checagem → Bloco 3 (MLP) e Bloco 1 (CNN)
#   4) verificação dos arquivos exportados, rankings, campeões, herança entre blocos e report_utils
# Uso: ./test_local.sh   (com o venv ativado, ou define PYTHON=/caminho/python)
set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        PYTHON="$VIRTUAL_ENV/bin/python"
    elif [[ -x venv/bin/python ]]; then
        PYTHON="venv/bin/python"
    else
        PYTHON="python3"
    fi
fi

export WANDB_MODE=offline
export WANDB_SILENT=true
export MPLBACKEND=Agg
export EXP_OUTPUT_DIR=./outputs_smoke_test
LOGS="$EXP_OUTPUT_DIR/_logs_smoke"

COMMON=(--subset_size 300 --batch_size 32 --project cifar10-rn-smoke-test --tags smoke_test)

section() {
    echo
    echo "=================================================================="
    echo ">>> $*"
    echo "=================================================================="
}

run() {
    section "$*"
    "$PYTHON" src/run_experiment.py "$@" "${COMMON[@]}"
}

echo "Python: $PYTHON ($("$PYTHON" --version))"
rm -rf "$EXP_OUTPUT_DIR"  # pasta exclusiva do smoke test (ignorada pelo git)
mkdir -p "$LOGS"

section "Auditoria de hiperparâmetros"
"$PYTHON" tests/check_hyperparams.py

section "Carregamento em GPU e particionamento"
"$PYTHON" tests/check_data_loader.py

section "Grids dos notebooks (expansão e construção de todas as configurações)"
"$PYTHON" tests/check_grids.py

section "Comandos dos notebooks Kaggle"
"$PYTHON" tests/check_notebook_commands.py

# ------------------------------------------------------------------ experimentos isolados
run --model mlp --exp_name smoke_mlp_holdout --epochs 1 --val_split 0.25 --mlp_layers 3 --mlp_neurons 64 128 64
run --model mlp --exp_name smoke_mlp_kfold --k_folds 3 --epochs 2 --patience 0 --mlp_layers 2 --mlp_neurons 256 \
    --activation tanh --dropout 0.3 --batch_norm --weight_decay 1e-4 --loss_fn mse --optimizer sgd --lr 0.01 --eval_train
run --model cnn --exp_name smoke_cnn_kfold --k_folds 3 --epochs 2 --patience 0 --conv_blocks 3 --filters 16 --kernel_size 5 \
    --stride 2 --padding same --pool_size 0 --fc_neurons 64 --cnn_dropout 0.5 --cnn_batch_norm --weight_decay 5e-4 \
    --optimizer sgd --lr 0.01 --augment --eval_train
run --model mlp --exp_name smoke_divergente --k_folds 2 --epochs 3 --patience 0 --mlp_layers 2 --mlp_neurons 64 \
    --optimizer sgd --lr 1e8

# Retomada: fold 1 → (--resume) folds 1-2 → hiperparâmetro diferente é recusado → fold 2 "interrompido" é refeito
RESUME=(--model mlp --exp_name smoke_resume --k_folds 3 --epochs 2 --patience 0 --mlp_layers 1 --mlp_neurons 32)
run "${RESUME[@]}" --folds 1
run "${RESUME[@]}" --folds 1 2 --resume | tee "$LOGS/resume_1.log"
section "Retomada com lr diferente deve falhar"
if "$PYTHON" src/run_experiment.py "${RESUME[@]}" "${COMMON[@]}" --lr 0.5 --folds 1 2 3 --resume > "$LOGS/resume_mismatch.log" 2>&1; then
    echo "ERRO: retomada com hiperparâmetros diferentes deveria falhar"; exit 1
fi
grep -q "hiperparâmetros diferentes" "$LOGS/resume_mismatch.log" && echo "OK: recusada ($(grep -o 'lr: [^)]*' "$LOGS/resume_mismatch.log" | head -1))"
"$PYTHON" - <<'EOF'
import json, os
path = os.path.join(os.environ["EXP_OUTPUT_DIR"], "smoke_resume", "resultados.json")
r = json.load(open(path))
assert r["status"] == "parcial" and [f["fold"] for f in r["folds"]] == [1, 2], "a tentativa recusada alterou o experimento"
r["folds"][1]["status"] = "interrompido"  # simula queda durante o fold 2
json.dump(r, open(path, "w"))
print("fold 2 marcado como interrompido para testar o descarte")
EOF
run "${RESUME[@]}" --folds 1 2 3 --resume | tee "$LOGS/resume_2.log"

# ------------------------------------------------------------------ grid search em miniatura
GRID=(--gpus 1 --workers_per_gpu 2)
for block in smoke_mlp_b1 smoke_mlp_b2 smoke_mlp_checagem smoke_mlp_b3 smoke_cnn_b1; do
    section "Grid $block"
    "$PYTHON" src/grid_search.py "tests/grids_smoke/$block.json" "${GRID[@]}" | tee "$LOGS/grid_$block.log"
done
section "Grid smoke_mlp_b1 novamente (tudo deve ser pulado)"
"$PYTHON" src/grid_search.py tests/grids_smoke/smoke_mlp_b1.json "${GRID[@]}" | tee "$LOGS/grid_rerun.log"

# ------------------------------------------------------------------ verificação
section "Verificando arquivos exportados, grids e report_utils"
"$PYTHON" tests/verify_smoke_outputs.py

echo
echo "Smoke test concluído com sucesso: sem erros de dimensão e com todos os arquivos exportados."
