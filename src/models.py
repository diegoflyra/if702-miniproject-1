"""Arquiteturas MLP e CNN com hiperparâmetros dinâmicos.

Toda a estrutura das redes é derivada dos argumentos do __init__, de modo que
cada flag da linha de comando altera fisicamente as camadas criadas.
"""

import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "sigmoid": nn.Sigmoid,
}


def get_activation(name):
    try:
        return ACTIVATIONS[name.lower()]()
    except KeyError:
        raise ValueError(f"Ativação '{name}' não suportada. Opções: {list(ACTIVATIONS)}")


def resolve_neurons(num_layers, neurons):
    """Normaliza --mlp_layers/--mlp_neurons para uma lista com um valor por camada oculta.

    - neurons inteiro ou lista de 1 valor: repetido em todas as camadas;
    - lista com N valores: N precisa ser igual a num_layers.
    """
    if num_layers < 0:
        raise ValueError("--mlp_layers deve ser >= 0.")
    if isinstance(neurons, int):
        neurons = [neurons]
    neurons = list(neurons)
    if any(n <= 0 for n in neurons):
        raise ValueError("--mlp_neurons deve conter apenas valores positivos.")
    if len(neurons) == 1:
        return neurons * num_layers
    if len(neurons) != num_layers:
        raise ValueError(
            f"--mlp_neurons recebeu {len(neurons)} valores, mas --mlp_layers={num_layers}. "
            f"Passe 1 valor (repetido) ou exatamente {num_layers}."
        )
    return neurons


class MLP(nn.Module):
    """Perceptron multicamadas: [Linear → (BatchNorm1d) → Ativação → (Dropout)] × mlp_layers → Linear.

    Exemplos:
        MLP(num_layers=3, neurons=[64, 128, 64])   # arquitetura do notebook original
        MLP(num_layers=4, neurons=1024, dropout=0.3, batch_norm=True)
        MLP(num_layers=0)                           # regressão logística (sem camadas ocultas)
    """

    def __init__(
        self,
        input_size=32 * 32 * 3,
        num_classes=10,
        num_layers=3,
        neurons=128,
        activation="relu",
        dropout=0.0,
        batch_norm=False,
    ):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout deve estar em [0, 1).")
        self.hidden_sizes = resolve_neurons(num_layers, neurons)

        layers = [nn.Flatten()]
        in_features = input_size
        for units in self.hidden_sizes:
            layers.append(nn.Linear(in_features, units))
            if batch_norm:
                layers.append(nn.BatchNorm1d(units))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = units
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def describe(self):
        return {"mlp_neuronios_por_camada": self.hidden_sizes}


def conv_padding(padding, kernel_size, stride):
    """Converte --padding ('same' | 'valid' | int) para o argumento do nn.Conv2d.

    'same' com stride=1 usa o modo nativo do PyTorch (preserva H×W, inclusive
    kernels pares). Com stride>1 o PyTorch não aceita 'same', então usa-se
    kernel_size // 2, que resulta em ceil(H / stride) para kernels ímpares.
    """
    if isinstance(padding, str):
        padding = padding.lower()
        if padding == "valid":
            return 0
        if padding == "same":
            return "same" if stride == 1 else kernel_size // 2
        raise ValueError(f"padding '{padding}' inválido. Use 'same', 'valid' ou um inteiro >= 0.")
    if padding < 0:
        raise ValueError("padding deve ser >= 0.")
    return padding


class CNN(nn.Module):
    """CNN configurável.

    Bloco i (i = 0 .. conv_blocks-1):
        [Conv2d(k, stride, padding) → (BatchNorm2d) → Ativação] × convs_per_block → MaxPool2d(pool_size)

    - filtros do bloco i: filters * 2**i (filters_growth='double') ou filters ('constant');
    - o stride é aplicado na primeira convolução de cada bloco;
    - pool_size 0 ou 1 desativa o pooling;
    - cabeça: Flatten → [Linear(fc) → Ativação → (Dropout)] × len(fc_neurons) → Linear(10).

    Exemplo (LeNet-5 adaptada do notebook original):
        CNN(conv_blocks=2, filters=32, kernel_size=3, padding='same', pool_size=2, fc_neurons=[120, 84])
    """

    def __init__(
        self,
        input_shape=(3, 32, 32),
        num_classes=10,
        conv_blocks=2,
        filters=32,
        filters_growth="double",
        convs_per_block=1,
        kernel_size=3,
        stride=1,
        padding="same",
        pool_size=2,
        fc_neurons=(128,),
        activation="relu",
        dropout=0.0,
        batch_norm=False,
    ):
        super().__init__()
        if conv_blocks < 1 or convs_per_block < 1:
            raise ValueError("conv_blocks e convs_per_block devem ser >= 1.")
        if filters < 1 or kernel_size < 1 or stride < 1 or pool_size < 0:
            raise ValueError("filters, kernel_size e stride devem ser >= 1; pool_size >= 0.")
        if filters_growth not in ("double", "constant"):
            raise ValueError("filters_growth deve ser 'double' ou 'constant'.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout deve estar em [0, 1).")
        if isinstance(fc_neurons, int):
            fc_neurons = [fc_neurons]

        self.block_filters = [
            filters * (2 ** i) if filters_growth == "double" else filters for i in range(conv_blocks)
        ]
        self.block_output_shapes = []

        blocks = []
        in_channels = input_shape[0]
        probe = torch.zeros(1, *input_shape)
        for i, out_channels in enumerate(self.block_filters):
            layers = []
            for j in range(convs_per_block):
                s = stride if j == 0 else 1
                layers.append(nn.Conv2d(
                    in_channels, out_channels, kernel_size=kernel_size, stride=s,
                    padding=conv_padding(padding, kernel_size, s),
                ))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(get_activation(activation))
                in_channels = out_channels
            if pool_size > 1:
                layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
            block = nn.Sequential(*layers)

            # Propaga um tensor fictício bloco a bloco: valida as dimensões e
            # registra o shape de saída de cada bloco (útil para o relatório).
            try:
                with torch.no_grad():
                    probe = block.eval()(probe)
            except RuntimeError as exc:
                raise ValueError(
                    f"Configuração inválida no bloco {i + 1}: entrada {tuple(probe.shape[1:])} ficou pequena demais "
                    f"para kernel_size={kernel_size}, stride={stride}, padding={padding}, pool_size={pool_size}. "
                    f"Reduza --conv_blocks/--stride/--pool_size"
                    + (" ou use --padding same." if padding != "same" else ".")
                ) from exc
            if probe.shape[-1] < 1 or probe.shape[-2] < 1:
                raise ValueError(f"Bloco {i + 1} reduziu o mapa de ativação para {tuple(probe.shape[1:])}.")
            self.block_output_shapes.append(list(probe.shape[1:]))
            blocks.append(block)
        self.features = nn.Sequential(*blocks)
        self.flatten_dim = probe.numel()

        head = [nn.Flatten()]
        in_features = self.flatten_dim
        for units in fc_neurons:
            head.append(nn.Linear(in_features, units))
            head.append(get_activation(activation))
            if dropout > 0:
                head.append(nn.Dropout(dropout))
            in_features = units
        head.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x):
        return self.classifier(self.features(x))

    def describe(self):
        return {
            "cnn_filtros_por_bloco": self.block_filters,
            "cnn_shape_saida_por_bloco": self.block_output_shapes,
            "cnn_flatten_dim": self.flatten_dim,
        }


def build_model(args, input_shape=(3, 32, 32), num_classes=10):
    """Instancia o modelo a partir dos argumentos da linha de comando."""
    if args.model == "mlp":
        c, h, w = input_shape
        return MLP(
            input_size=c * h * w,
            num_classes=num_classes,
            num_layers=args.mlp_layers,
            neurons=args.mlp_neurons,
            activation=args.activation,
            dropout=args.dropout,
            batch_norm=args.batch_norm,
        )
    if args.model == "cnn":
        return CNN(
            input_shape=input_shape,
            num_classes=num_classes,
            conv_blocks=args.conv_blocks,
            filters=args.filters,
            filters_growth=args.filters_growth,
            convs_per_block=args.convs_per_block,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.padding,
            pool_size=args.pool_size,
            fc_neurons=args.fc_neurons,
            activation=args.activation,
            dropout=args.cnn_dropout,
            batch_norm=args.cnn_batch_norm,
        )
    raise ValueError(f"Modelo '{args.model}' desconhecido.")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
