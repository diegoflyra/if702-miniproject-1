"""Auditoria: cada flag do argparse altera de fato a arquitetura/otimização construída.

Não baixa dados nem treina; roda em segundos (após o import do torch).
Uso: python tests/check_hyperparams.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn

from models import CNN, MLP, build_model
from run_experiment import experiment_params, parse_args
from train import SoftmaxMSELoss, get_loss_fn, get_optimizer

BASE = ["--exp_name", "audit"]
X = torch.randn(4, 3, 32, 32)
checks = 0


def args_for(*flags):
    return parse_args([*BASE, *flags])


def model_for(*flags):
    return build_model(args_for(*flags))


def layers(model, kind):
    return [m for m in model.modules() if isinstance(m, kind)]


def check(condition, message):
    global checks
    assert condition, message
    checks += 1


def expect_error(fn, message):
    global checks
    try:
        fn()
    except (SystemExit, ValueError):
        checks += 1
        return
    raise AssertionError(f"Deveria falhar: {message}")


def silence_stderr(fn):
    def wrapper():
        stderr, sys.stderr = sys.stderr, open(os.devnull, "w")
        try:
            fn()
        finally:
            sys.stderr.close()
            sys.stderr = stderr
    return wrapper


# ----------------------------------------------------------------------------- MLP
m = model_for("--model", "mlp", "--mlp_layers", "4", "--mlp_neurons", "256")
lin = layers(m, nn.Linear)
check(len(lin) == 5, "--mlp_layers 4 deve gerar 4 Linear ocultas + 1 de saída")
check([l.out_features for l in lin] == [256, 256, 256, 256, 10], "--mlp_neurons 256 repetido em todas as camadas")
check(lin[0].in_features == 3072, "entrada da MLP deve ser 3*32*32")
check(m(X).shape == (4, 10), "saída MLP")

m = model_for("--model", "mlp", "--mlp_layers", "3", "--mlp_neurons", "64", "128", "64")
check([l.out_features for l in layers(m, nn.Linear)] == [64, 128, 64, 10], "--mlp_neurons como lista")

m = model_for("--model", "mlp", "--mlp_layers", "0")
check(len(layers(m, nn.Linear)) == 1 and m(X).shape == (4, 10), "--mlp_layers 0 = regressão logística")

expect_error(silence_stderr(lambda: model_for("--model", "mlp", "--mlp_layers", "2", "--mlp_neurons", "1", "2", "3")),
             "--mlp_neurons com tamanho incompatível")

for name, cls in (("relu", nn.ReLU), ("tanh", nn.Tanh), ("leaky_relu", nn.LeakyReLU)):
    m = model_for("--model", "mlp", "--mlp_layers", "2", "--activation", name)
    acts = [mod for mod in m.modules() if isinstance(mod, (nn.ReLU, nn.Tanh, nn.LeakyReLU))]
    check(len(acts) == 2 and all(isinstance(a, cls) for a in acts), f"--activation {name} (MLP)")

m = model_for("--model", "mlp", "--mlp_layers", "3", "--dropout", "0.4")
drops = layers(m, nn.Dropout)
check(len(drops) == 3 and all(d.p == 0.4 for d in drops), "--dropout 0.4 em cada camada oculta")
check(not layers(model_for("--model", "mlp"), nn.Dropout), "sem --dropout não há Dropout")
expect_error(silence_stderr(lambda: args_for("--model", "mlp", "--dropout", "1.5")), "--dropout fora de [0,1)")

m = model_for("--model", "mlp", "--mlp_layers", "3", "--batch_norm")
check(len(layers(m, nn.BatchNorm1d)) == 3, "--batch_norm insere BatchNorm1d por camada oculta")
check(not layers(model_for("--model", "mlp"), nn.BatchNorm1d), "sem --batch_norm não há BatchNorm1d")

# ----------------------------------------------------------------------------- CNN
m = model_for("--model", "cnn", "--conv_blocks", "3", "--filters", "32")
convs = layers(m, nn.Conv2d)
check(len(convs) == 3, "--conv_blocks 3 gera 3 convoluções")
check([c.out_channels for c in convs] == [32, 64, 128], "--filters dobra a cada bloco")
check(len(layers(m, nn.MaxPool2d)) == 3, "um MaxPool por bloco")
check(m.block_output_shapes == [[32, 16, 16], [64, 8, 8], [128, 4, 4]], "shapes por bloco")
check(m(X).shape == (4, 10), "saída CNN")

m = model_for("--model", "cnn", "--conv_blocks", "3", "--filters", "48", "--filters_growth", "constant")
check([c.out_channels for c in layers(m, nn.Conv2d)] == [48, 48, 48], "--filters_growth constant")

m = model_for("--model", "cnn", "--conv_blocks", "2", "--convs_per_block", "2")
check(len(layers(m, nn.Conv2d)) == 4, "--convs_per_block 2")

m = model_for("--model", "cnn", "--conv_blocks", "3", "--kernel_size", "5")
check(all(c.kernel_size == (5, 5) for c in layers(m, nn.Conv2d)), "--kernel_size 5 em TODAS as convoluções")
check(m.block_output_shapes[-1] == [128, 4, 4] and m(X).shape == (4, 10), "kernel 5 + same preserva H×W")

m = model_for("--model", "cnn", "--conv_blocks", "3", "--stride", "2", "--pool_size", "0")
convs = layers(m, nn.Conv2d)
check(all(c.stride == (2, 2) for c in convs), "--stride 2 aplicado")
check(not layers(m, nn.MaxPool2d), "--pool_size 0 remove o pooling")
check(m.block_output_shapes == [[32, 16, 16], [64, 8, 8], [128, 4, 4]], "stride 2 reduz H×W pela metade")
check(m(X).shape == (4, 10), "saída stride 2")

m = model_for("--model", "cnn", "--conv_blocks", "3", "--padding", "valid")
check(all(c.padding == (0, 0) for c in layers(m, nn.Conv2d)), "--padding valid")
check(m.block_output_shapes == [[32, 15, 15], [64, 6, 6], [128, 2, 2]], "shapes com padding valid")

m = model_for("--model", "cnn", "--conv_blocks", "2", "--kernel_size", "5", "--padding", "1")
check(all(c.padding == (1, 1) for c in layers(m, nn.Conv2d)), "--padding inteiro")
check(m.block_output_shapes == [[32, 15, 15], [64, 6, 6]], "shapes com padding 1 e kernel 5")

m = model_for("--model", "cnn", "--conv_blocks", "2", "--pool_size", "3")
pools = layers(m, nn.MaxPool2d)
check(all(p.kernel_size == 3 and p.stride == 3 for p in pools), "--pool_size 3")
check(m.block_output_shapes == [[32, 10, 10], [64, 3, 3]], "shapes com pool 3")

expect_error(lambda: model_for("--model", "cnn", "--conv_blocks", "6", "--pool_size", "2"),
             "6 blocos com pool 2 reduzem 32px a 0")
expect_error(silence_stderr(lambda: args_for("--model", "cnn", "--padding", "meio")), "--padding inválido")

m = model_for("--model", "cnn", "--conv_blocks", "2", "--fc_neurons", "256")
lin = layers(m, nn.Linear)
check([l.out_features for l in lin] == [256, 10], "--fc_neurons 256")
check(lin[0].in_features == 64 * 8 * 8, "flatten dim calculado automaticamente")
m = model_for("--model", "cnn", "--fc_neurons", "120", "84")
check([l.out_features for l in layers(m, nn.Linear)] == [120, 84, 10], "--fc_neurons com 2 camadas")

m = model_for("--model", "cnn", "--fc_neurons", "120", "84", "--cnn_dropout", "0.5")
drops = layers(m, nn.Dropout)
check(len(drops) == 2 and all(d.p == 0.5 for d in drops), "--cnn_dropout nas camadas densas")

m = model_for("--model", "cnn", "--conv_blocks", "3", "--convs_per_block", "2", "--cnn_batch_norm")
check(len(layers(m, nn.BatchNorm2d)) == 6, "--cnn_batch_norm após cada convolução")
check(not layers(model_for("--model", "cnn"), nn.BatchNorm2d), "sem --cnn_batch_norm não há BatchNorm2d")

m = model_for("--model", "cnn", "--activation", "tanh")
check(all(not isinstance(a, nn.ReLU) for a in m.modules()) and layers(m, nn.Tanh), "--activation tanh (CNN)")

# ------------------------------------------------------------- argumentos cruzados
expect_error(silence_stderr(lambda: args_for("--model", "cnn", "--dropout", "0.3")), "--dropout em CNN")
expect_error(silence_stderr(lambda: args_for("--model", "cnn", "--batch_norm")), "--batch_norm em CNN")
expect_error(silence_stderr(lambda: args_for("--model", "mlp", "--kernel_size", "5")), "--kernel_size em MLP")
expect_error(silence_stderr(lambda: args_for("--model", "mlp", "--cnn_dropout", "0.3")), "--cnn_dropout em MLP")
expect_error(silence_stderr(lambda: args_for("--model", "mlp", "--val_split", "0")), "--val_split 0")
expect_error(silence_stderr(lambda: args_for("--model", "mlp", "--k_folds", "0")), "--k_folds 0")
expect_error(silence_stderr(lambda: args_for("--model", "mlp", "--k_folds", "5", "--val_split", "0.2")),
             "--val_split junto com --k_folds > 1")
check(args_for("--model", "cnn", "--k_folds", "5").k_folds == 5, "--k_folds 5")
check(experiment_params(args_for("--model", "mlp", "--k_folds", "5"))["val_split"] is None,
      "parametros.json: val_split fica null com K-fold")

params = experiment_params(args_for("--model", "mlp", "--mlp_layers", "2"))
check(params["kernel_size"] is None and params["mlp_layers"] == 2, "parametros.json: flags da CNN ficam null na MLP")
params = experiment_params(args_for("--model", "cnn", "--kernel_size", "5"))
check(params["mlp_layers"] is None and params["dropout"] is None and params["kernel_size"] == 5,
      "parametros.json: flags da MLP ficam null na CNN")

# ------------------------------------------------------------- otimização e perda
params = list(MLP(num_layers=1, neurons=8).parameters())
opt = get_optimizer("sgd", params, lr=0.05, weight_decay=1e-4, momentum=0.9)
check(isinstance(opt, torch.optim.SGD), "--optimizer sgd")
check(opt.param_groups[0]["lr"] == 0.05 and opt.param_groups[0]["weight_decay"] == 1e-4
      and opt.param_groups[0]["momentum"] == 0.9, "SGD recebe lr/weight_decay/momentum")
opt = get_optimizer("adam", params, lr=3e-4, weight_decay=5e-4)
check(isinstance(opt, torch.optim.Adam), "--optimizer adam")
check(opt.param_groups[0]["lr"] == 3e-4 and opt.param_groups[0]["weight_decay"] == 5e-4, "Adam recebe lr/weight_decay")

logits, targets = torch.randn(4, 10, requires_grad=True), torch.tensor([0, 1, 2, 3])
check(isinstance(get_loss_fn("cross_entropy", 10), nn.CrossEntropyLoss), "--loss_fn cross_entropy")
mse = get_loss_fn("mse", 10)
check(isinstance(mse, SoftmaxMSELoss), "--loss_fn mse")
loss = mse(logits, targets)
loss.backward()
expected = ((torch.softmax(logits, 1) - nn.functional.one_hot(targets, 10).float()) ** 2).mean()
check(torch.allclose(loss, expected) and logits.grad is not None, "MSE = mean((softmax - one_hot)^2) com gradiente")

# Um passo de otimização precisa alterar os pesos (lr efetivamente aplicado)
for name in ("sgd", "adam"):
    model = model_for("--model", "cnn", "--conv_blocks", "1", "--filters", "4", "--fc_neurons", "8")
    before = [p.detach().clone() for p in model.parameters()]
    opt = get_optimizer(name, model.parameters(), lr=0.01)
    nn.CrossEntropyLoss()(model(X), torch.tensor([0, 1, 2, 3])).backward()
    opt.step()
    check(any(not torch.equal(b, p) for b, p in zip(before, model.parameters())), f"{name} atualiza os pesos")

print(f"Auditoria de hiperparâmetros: {checks} verificações OK")
