"""Métricas por classe dos campeões e conclusões por hiperparâmetro, compartilhadas pelo caderno e pelo .docx.

Os textos usam **negrito** e `código`; os números são calculados a partir de doc_data.json e dos
arquivos de cada experimento, para que caderno, .docx e resultados nunca divirjam.
"""

import glob
import json

import numpy as np
import pandas as pd

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
PT = {"airplane": "avião", "automobile": "automóvel", "bird": "pássaro", "cat": "gato", "deer": "cervo",
      "dog": "cachorro", "frog": "sapo", "horse": "cavalo", "ship": "navio", "truck": "caminhão"}
PL = {"airplane": "aviões", "automobile": "automóveis", "bird": "pássaros", "cat": "gatos", "deer": "cervos",
      "dog": "cachorros", "frog": "sapos", "horse": "cavalos", "ship": "navios", "truck": "caminhões"}
SPLITS = {"train": "validacao_media", "val": "validacao_media", "test": "teste"}
STD = {"validacao_media": "validacao_std", "teste": "teste_std"}

# (rede, chave curta, rótulo, subtítulo, pasta, experimento)
CHAMPIONS = [
    ("mlp", "B0 lin", "Linear (B0)", "regressão logística", "outputs_mlp", "mlp_b0_referencia__linear"),
    ("mlp", "B0 orig", "MLP original (B0)", "64-128-64", "outputs_mlp", "mlp_b0_referencia__original_64-128-64"),
    ("mlp", "B1", "Topologia (B1)", "3 × 256", "outputs_mlp", "mlp_b1_topologia__L3_N256"),
    ("mlp", "B2", "Otimização (B2)", "3 × 256 · Adam lr 3e-4", "outputs_mlp", "mlp_b2_otimizacao__optadam_lr0.0003"),
    ("mlp", "CHK", "Checagem", "4 × 256 · Adam lr 3e-4", "outputs_mlp", "mlp_b2_checagem__rank2_L4_N256"),
    ("mlp", "B3", "Regularização (B3) · campeã", "4 × 256 · dropout 0,2", "outputs_mlp", "mlp_b3_regularizacao__do0.2_losscross_entropy"),
    ("mlp", "AUG", "Bônus: augmentation", "4 × 256 · dropout 0,2 · crop + flip", "outputs_augmentation", "mlp_b4_augmentation__aug1_do0.2"),
    ("cnn", "B0", "LeNet original (B0)", "2 blocos · 32/64 filtros", "outputs_cnn", "cnn_b0_referencia__lenet_original"),
    ("cnn", "B1", "Topologia (B1)", "4 blocos · k3 · maxpool", "outputs_cnn", "cnn_b1_topologia__B4_k3_padsame_maxpool"),
    ("cnn", "B2", "Otimização (B2)", "4 blocos · Adam lr 1e-3", "outputs_cnn", "cnn_b2_otimizacao__optadam_lr0.001"),
    ("cnn", "B3", "Regularização (B3)", "4 blocos · BatchNorm · dropout 0,5", "outputs_cnn", "cnn_b3_regularizacao__pool2_do0.5_bn1"),
    ("cnn", "+B", "Pooling (+B) · campeã", "3 blocos · pooling 2×2", "outputs_final", "cnn_b3b_pooling__B3_pool2"),
    ("cnn", "AUG", "Bônus: augmentation", "4 blocos · dropout 0,5 · crop + flip", "outputs_augmentation", "cnn_b4_augmentation__aug1_B4_do0.5"),
]


# ============================================================================ formatação (vírgula decimal)
def pct(x, d=1):
    return f"{x * 100:.{d}f}%".replace(".", ",")


def pp(x, d=1):
    return ("+" if x >= 0 else "−") + f"{abs(x) * 100:.{d}f}".replace(".", ",") + " p.p."


def num(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def mil(n):
    return f"{n / 1e6:.2f} M".replace(".", ",") if n >= 1e6 else f"{n / 1e3:.0f} mil"


def lrfmt(x):
    m, e = f"{x:.0e}".split("e")
    return f"{m}e{int(e)}"


def milhar(n):
    return f"{int(n):,}".replace(",", ".")


# ============================================================================ campeões por classe
def confusion_sum(root, exp):
    """Soma das matrizes de teste dos 5 folds (cada fold avalia seu próprio modelo nas 10.000 imagens)."""
    files = sorted(glob.glob(f"{root}/{exp}/folds/fold_*/matriz_confusao_teste.csv"))
    if files:
        return sum(pd.read_csv(f, index_col=0).values for f in files).astype(int), len(files)
    cm = pd.read_csv(f"{root}/{exp}/matriz_confusao_teste.csv", index_col=0).values
    return cm.astype(int), 1


def champion(net, short, label, sub, root, exp):
    r = json.load(open(f"{root}/{exp}/resultados.json"))
    p = json.load(open(f"{root}/{exp}/parametros.json"))
    classes, overall = {}, {}
    for split, sec in SPLITS.items():
        mean, std = r[sec], r[STD[sec]]
        classes[split] = {m: {"mean": [mean[f"{split}/{m}_per_class/{c}"] for c in CLASSES],
                              "std": [std.get(f"{split}/{m}_per_class/{c}") for c in CLASSES]}
                          for m in ("precision", "recall", "f1")}
        overall[split] = {k: {"mean": mean.get(f"{split}/{k}"), "std": std.get(f"{split}/{k}")}
                          for k in ("accuracy", "loss", "precision_macro", "recall_macro", "f1_macro")}
    cm, nfolds = confusion_sum(root, exp)
    ch = {"net": net, "short": short, "label": label, "sub": sub, "exp": exp, "root": root,
          "params": p.get("num_parameters"), "folds": r.get("folds_concluidos"), "cm_folds": nfolds,
          "best_epoch": float(np.mean([f["melhor_epoca"] for f in r["folds"]])),
          "classes": classes, "overall": overall, "confusion": cm.tolist()}
    ch["reading"] = reading(ch)
    return ch


def reading(ch):
    """Leitura automática de um campeão: extremos, assimetria precision × recall, confusão e overfitting por classe."""
    t = ch["classes"]["test"]
    P, R, F = (np.array(t[m]["mean"]) for m in ("precision", "recall", "f1"))
    Rtr = np.array(ch["classes"]["train"]["recall"]["mean"])
    cm = np.array(ch["confusion"])
    off = cm.copy()
    np.fill_diagonal(off, 0)
    rows = cm.sum(axis=1)
    order = np.argsort(F)
    out = [f"**Melhor classe:** {PT[CLASSES[order[-1]]]} (F1 {pct(F[order[-1]])}). "
           f"**Pior:** {PT[CLASSES[order[0]]]} (F1 {pct(F[order[0]])}), {num((F[order[-1]] - F[order[0]]) * 100)} p.p. abaixo."]
    k = int(np.argmax(np.abs(P - R)))
    c = CLASSES[k]
    if P[k] > R[k]:
        dest = CLASSES[int(off[k].argmax())]
        out.append(f"**{PT[c].capitalize()} é subprevisto:** precision {pct(P[k])} × recall {pct(R[k])}. Quando a rede diz "
                   f"“{PT[c]}”, costuma acertar, mas deixa passar {pct(1 - R[k], 0)} dos {PL[c]} verdadeiros, mais "
                   f"frequentemente classificados como {PT[dest]}.")
    else:
        src = CLASSES[int(off[:, k].argmax())]
        wrong = int(off[:, k].sum())
        out.append(f"**{PT[c].capitalize()} funciona como destino das dúvidas:** recall {pct(R[k])} × precision {pct(P[k])}. "
                   f"Das previsões “{PT[c]}”, {milhar(wrong)} eram outra classe (principalmente {PT[src]}).")
    i, j = np.unravel_index(off.argmax(), off.shape)
    out.append(f"**Maior confusão:** {PT[CLASSES[i]]} → {PT[CLASSES[j]]}, {pct(off[i, j] / rows[i])} dos {PL[CLASSES[i]]} "
               f"({milhar(off[i, j])} de {milhar(rows[i])} previsões somando os folds).")
    g = Rtr - R
    k = int(np.argmax(g))
    out.append(f"**Maior distância treino–teste:** {PT[CLASSES[k]]} (recall {pct(Rtr[k])} no treino × {pct(R[k])} no teste, "
               f"{num(g[k] * 100)} p.p.; média das classes {num(g.mean() * 100)} p.p.).")
    return _tidy(out)


def _tidy(x):
    """Evita "p.p.." quando uma frase termina num valor em pontos percentuais."""
    if isinstance(x, str):
        return x.replace("p.p..", "p.p.")
    if isinstance(x, list):
        return [_tidy(v) for v in x]
    if isinstance(x, dict):
        return {k: _tidy(v) for k, v in x.items()}
    return x


def class_conclusions(champs):
    """Conclusões transversais sobre as classes, com números calculados dos 13 campeões."""
    by = {(c["net"], c["short"]): c for c in champs}
    f1 = lambda ch: np.array(ch["classes"]["test"]["f1"]["mean"])
    worst = {CLASSES[int(np.argmin(f1(c)))] for c in champs}
    tops = [tuple(sorted(CLASSES[i] for i in np.argsort(f1(c))[-2:])) for c in champs]
    cat_i, dog_i, bird_i = CLASSES.index("cat"), CLASSES.index("dog"), CLASSES.index("bird")
    catdog = 0
    for c in champs:
        off = np.array(c["confusion"])
        np.fill_diagonal(off, 0)
        catdog += int(off[cat_i].argmax() == dog_i and off[dog_i].argmax() == cat_i)
    bird_prec = sum(1 for c in champs if c["classes"]["test"]["precision"]["mean"][bird_i] > c["classes"]["test"]["recall"]["mean"][bird_i])
    mlp, cnn, aug = by[("mlp", "B3")], by[("cnn", "+B")], by[("cnn", "AUG")]
    gain = f1(cnn) - f1(mlp)
    gain_aug = f1(aug) - f1(cnn)
    rtr = lambda ch: np.array(ch["classes"]["train"]["recall"]["mean"]) - np.array(ch["classes"]["test"]["recall"]["mean"])
    items = [
        f"**Gato é a pior classe nos {len(champs)} campeões**" + ("" if worst == {"cat"} else f" (exceções: {', '.join(PT[w] for w in worst - {'cat'})})") +
        f", e automóvel e navio são as duas melhores em {sum(1 for t in tops if t == ('automobile', 'ship'))} de {len(champs)}. "
        f"Em {catdog} de {len(champs)} campeões, o erro mais comum do gato é cachorro e o do cachorro é gato. "
        "Gatos e cachorros compartilham pelagem, pose, tamanho e fundo (ambientes internos); em 32×32 pixels, o que os separa "
        "(focinho, orelhas, olhos) ocupa poucos pixels. Veículos têm silhuetas rígidas e fundos característicos (estrada, mar), "
        "com muito menos variação de forma entre exemplos.",
        f"**A CNN ganha mais justamente onde a forma importa.** Do campeão MLP para o campeão CNN, o F1 sobe "
        f"{pp(gain.min(), 0)} a {pp(gain.max(), 0)} por classe; os maiores ganhos são "
        + ", ".join(f"{PT[CLASSES[i]]} ({pp(gain[i], 0)})" for i in np.argsort(gain)[::-1][:3])
        + " e os menores, " + " e ".join(f"{PT[CLASSES[i]]} ({pp(gain[i], 0)})" for i in np.argsort(gain)[:2])
        + ". Avião e navio, que a MLP já reconhecia em boa parte pelo fundo azul (céu e mar), têm menos a ganhar; "
        "animais sobre grama ou floresta dividem o mesmo fundo e só se separam por forma e textura, que a convolução captura e a imagem achatada perde.",
        f"**Precision e recall contam histórias diferentes.** Recall baixo significa que a rede não reconhece a classe; precision "
        f"baixa, que ela usa a classe como resposta para imagens de outras. No campeão CNN, gato tem precision "
        f"{pct(cnn['classes']['test']['precision']['mean'][cat_i])} e recall {pct(cnn['classes']['test']['recall']['mean'][cat_i])}: "
        "o erro é nos dois sentidos, porque gato e cachorro trocam de lugar entre si. Já pássaro tem precision acima do recall "
        f"em {bird_prec} de {len(champs)} campeões: quando a rede não tem certeza sobre um pássaro, prefere outra classe (cervo, avião), "
        "e acerta quando se compromete.",
        f"**Overfitting por classe.** No campeão CNN, a distância entre recall de treino e de teste vai de "
        f"{num(rtr(cnn).min() * 100)} p.p. ({PT[CLASSES[int(np.argmin(rtr(cnn)))]]}) a {num(rtr(cnn).max() * 100)} p.p. "
        f"({PT[CLASSES[int(np.argmax(rtr(cnn)))]]}). As classes com mais variação visual entre exemplos são as que a rede mais "
        f"decora: ela memoriza as imagens de treino sem encontrar um padrão que valha para as novas. Com augmentation, a distância "
        f"média cai de {num(rtr(cnn).mean() * 100)} p.p. para {num(rtr(aug).mean() * 100)} p.p., e o F1 sobe em todas as classes "
        f"({pp(gain_aug.min(), 0)} a {pp(gain_aug.max(), 0)}), mais em " + PT[CLASSES[int(np.argmax(gain_aug))]] + ".",
    ]
    return _tidy(items)


# ============================================================================ hiperparâmetros
def _rows(rows, **eq):
    return [r for r in rows if all((abs(r[k] - v) < 1e-9 if isinstance(v, float) else r[k] == v) for k, v in eq.items())]


def _one(rows, **eq):
    return _rows(rows, **eq)[0]


V = "val/accuracy_mean"


def _marg(rows, key, values, fmt, champ, **eq):
    out = []
    for v in values:
        rs = _rows(rows, **{key: v, **eq})
        out.append({"label": fmt(v), "mean": float(np.mean([r[V] for r in rs])), "best": float(max(r[V] for r in rs)),
                    "ep": float(np.mean([r["melhor_epoca_media"] for r in rs])), "n": len(rs), "champ": v == champ})
    return out


def _pairs(rows, a, b):
    """Diferenças entre pares que só mudam o rótulo a → b (triagem, média dos folds 1–3)."""
    byl = {r["label"]: r for r in rows}
    return [byl[k][V] - byl[k.replace(a, b)][V] for k in byl if a in k and k.replace(a, b) in byl]


def hyperparameters(D):
    M, C, A = D["mlp"], D["cnn"], D["aug"]
    b1m, b2m, b3m, b1c, b2c, b3c = M["b1_triagem"], M["b2_triagem"], M["b3_triagem"], C["b1_triagem"], C["b2_triagem"], C["b3_triagem"]

    lay = _marg(b1m, "mlp_layers", [1, 2, 3, 4, 5], lambda v: f"{v} camada{'s' if v > 1 else ''}", 3)
    neu = _marg(b1m, "mlp_neurons", [128, 256, 512, 1024, 2048], lambda v: f"{v} neurônios", 256)
    wide1 = _one(b1m, mlp_layers=1, mlp_neurons=2048)
    small3 = _one(b1m, mlp_layers=3, mlp_neurons=128)
    big = _one(b1m, mlp_layers=5, mlp_neurons=2048)
    chk = M["chk_final"]
    l4 = next(r for r in chk if r["mlp_layers"] == 4)
    b2champ = M["b2_final"][0]

    best = lambda rows, opt: max(_rows(rows, optimizer=opt), key=lambda r: r[V])
    ma, ms, cs = best(b2m, "adam"), best(b2m, "sgd"), best(b2c, "sgd")
    ca, ca3 = _one(b2c, optimizer="adam", lr=0.001), _one(b2c, optimizer="adam", lr=0.0003)
    lrlab = lambda r: lrfmt(r["lr"])
    opt_m = [{"label": f"{o.upper() if o == 'sgd' else 'Adam'} · lr {lrlab(r)}", "mean": r[V], "best": r[V], "ep": r["melhor_epoca_media"],
              "n": 1, "champ": o == "adam"} for o, r in (("adam", ma), ("sgd", ms))]
    opt_c = [{"label": f"{o.upper() if o == 'sgd' else 'Adam'} · lr {lrlab(r)}", "mean": r[V], "best": r[V], "ep": r["melhor_epoca_media"],
              "n": 1, "champ": o == "adam"} for o, r in (("adam", ca), ("sgd", cs))]

    LR = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
    lr_adam_c = _marg(b2c, "lr", LR, lambda v: f"Adam · lr {lrfmt(v)}", 0.001, optimizer="adam")
    a3c, a2c = _one(b2c, optimizer="adam", lr=0.003), _one(b2c, optimizer="adam", lr=0.01)
    a4m, a3m = _one(b2m, optimizer="adam", lr=0.0001), _one(b2m, optimizer="adam", lr=0.0003)
    s1m, s1c = _one(b2m, optimizer="sgd", lr=0.1), _one(b2c, optimizer="sgd", lr=0.1)
    a4c = _one(b2c, optimizer="adam", lr=0.0001)
    sgdlow = _one(b2m, optimizer="sgd", lr=0.0001)

    do_m = _marg(b3m, "dropout", [0.0, 0.2, 0.3, 0.5], lambda v: f"dropout {v:g}".replace(".", ","), 0.2, loss_fn="cross_entropy")
    ea = {r["label"]: r for r in M["extraA"]}
    pa = {p["exp"].split("__")[1]: p for p in M["paired_A"]}
    loss_m = _marg(b3m, "loss_fn", ["cross_entropy", "mse"], lambda v: "entropia cruzada" if v == "cross_entropy" else "MSE", "cross_entropy", dropout=0.2)

    mp = _pairs(b1c, "maxpool", "stride2")
    red = _marg(b1c, "reducao", ["maxpool", "stride2"], lambda v: "max pooling 2×2" if v == "maxpool" else "convolução stride 2", "maxpool")
    best_stride = max(_rows(b1c, reducao="stride2"), key=lambda r: r[V])
    worst_pool = min(_rows(b1c, reducao="maxpool"), key=lambda r: r[V])
    kp = [d for d in _pairs(_rows(b1c, reducao="maxpool"), "_k3_", "_k5_")]
    ks = [d for d in _pairs(_rows(b1c, reducao="stride2"), "_k3_", "_k5_")]
    ker = _marg(b1c, "kernel_size", [3, 5], lambda v: f"kernel {v}×{v}", 3, reducao="maxpool")
    pdp = _pairs(_rows(b1c, reducao="maxpool"), "padsame", "padvalid")
    pda = _pairs(b1c, "padsame", "padvalid")
    pad = _marg(b1c, "padding", ["same", "valid"], lambda v: f"padding {v}", "same", reducao="maxpool")

    blk = _marg(b1c, "conv_blocks", [2, 3, 4], lambda v: f"{v} blocos", 4, reducao="maxpool", kernel_size=3, padding="same")
    eb = {r["label"]: r for r in C["extraB"]}
    pb = {p["exp"].split("__")[1]: p for p in C["paired_B"]}
    ac = {r["exp"].split("__")[1]: r for r in A["cnn"]["rows"]}
    depth_aug = A["cnn"]["depth_with_aug"]
    blk_bn = [{"label": f"{b} blocos · com BN", "mean": eb[f"B{b}_pool2"][V], "best": eb[f"B{b}_pool2"][V],
               "ep": eb[f"B{b}_pool2"]["melhor_epoca_media"], "n": 1, "champ": b == 3} for b in (2, 3, 4)]

    bn_pairs = [(d, _one(b3c, cnn_dropout=d, cnn_batch_norm=True)[V] - _one(b3c, cnn_dropout=d, cnn_batch_norm=False)[V]) for d in (0.0, 0.3, 0.5)]
    bn = _marg(b3c, "cnn_batch_norm", [False, True], lambda v: "com BatchNorm" if v else "sem BatchNorm", True)
    bn0, bn1 = _rows(b3c, cnn_batch_norm=False), _rows(b3c, cnn_batch_norm=True)
    dc = lambda bnv: _marg(b3c, "cnn_dropout", [0.0, 0.3, 0.5], lambda v: f"dropout {v:g}".replace(".", ",") + (" · BN" if bnv else " · sem BN"), 0.5 if bnv else None, cnn_batch_norm=bnv)
    do_c_bn, do_c_nobn = dc(True), dc(False)
    g_bn = do_c_bn[2]["mean"] - do_c_bn[0]["mean"]
    g_nobn = do_c_nobn[2]["mean"] - do_c_nobn[0]["mean"]
    gap_bn = {d: _one(b3c, cnn_dropout=d, cnn_batch_norm=True)["gap/accuracy_mean"] for d in (0.0, 0.5)}

    def mapa(b, p):
        m = 32
        for _ in range(b):
            m //= p
        return m
    pool = [{"label": f"{b} blocos · pool {p} → {mapa(b, p)}×{mapa(b, p)}", "mean": eb[f"B{b}_pool{p}"][V], "best": eb[f"B{b}_pool{p}"][V],
             "ep": eb[f"B{b}_pool{p}"]["melhor_epoca_media"], "n": 1, "champ": (b, p) == (3, 2)}
            for b, p in ((2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (4, 2))]

    am = {r["exp"].split("__")[1]: r for r in A["mlp"]["rows"]}
    pvm = {p["exp"].split("__")[1]: p for p in A["mlp"]["paired_val"]}
    pvc = {p["exp"].split("__")[1]: p for p in A["cnn"]["paired_val"]}
    aug_rows = [{"label": "CNN sem aug", "mean": ac["aug0_B3_do0.5"]["val"], "best": ac["aug0_B3_do0.5"]["val"], "ep": ac["aug0_B3_do0.5"]["best_epoch"], "n": 1, "champ": False},
                {"label": "CNN com aug", "mean": ac["aug1_B4_do0.5"]["val"], "best": ac["aug1_B4_do0.5"]["val"], "ep": ac["aug1_B4_do0.5"]["best_epoch"], "n": 1, "champ": True},
                {"label": "MLP sem aug", "mean": am["aug0_do0.2"]["val"], "best": am["aug0_do0.2"]["val"], "ep": am["aug0_do0.2"]["best_epoch"], "n": 1, "champ": False},
                {"label": "MLP com aug", "mean": am["aug1_do0.2"]["val"], "best": am["aug1_do0.2"]["val"], "ep": am["aug1_do0.2"]["best_epoch"], "n": 1, "champ": True}]

    dense = lambda flat: flat * 512 + 512
    H = [
        # ---------------------------------------------------------------- capacidade
        {"id": "hp-camadas", "group": "Capacidade", "name": "Camadas ocultas da MLP", "nets": ["MLP"], "values": "1 · 2 · 3 · 4 · 5",
         "choice": "3 no Bloco 1 · 4 após a checagem (empate)", "verdict": "moderado",
         "effect": pp(lay[2]["mean"] - lay[0]["mean"]), "effect_note": "1 → 3 camadas, média das 5 larguras",
         "marg": lay, "marg_note": "média da triagem (folds 1–3) sobre as 5 larguras",
         "evidence": [f"Com 1 camada, a média da triagem é {pct(lay[0]['mean'])}; com 2 a 5, fica entre {pct(min(x['mean'] for x in lay[1:]))} e {pct(max(x['mean'] for x in lay[1:]))}.",
                      f"`1 × 2048` ({mil(wide1['num_parameters'])} parâmetros) fica em {pct(wide1[V])}, abaixo de `3 × 128` ({mil(small3['num_parameters'])}, {pct(small3[V])}).",
                      f"Na checagem com Adam 3e-4, `4 × 256` ({pct(l4[V])}) e `3 × 256` ({pct(b2champ[V])}) empataram: {pp(l4[V] - b2champ[V])}."],
         "why": "Uma camada oculta combina os pixels uma única vez: cada neurônio é um molde da imagem inteira. A segunda camada combina esses moldes, "
                "o que permite reconhecer como a mesma coisa padrões que aparecem com cores ou posições diferentes; daí o salto de 1 para 2 camadas, "
                "que nenhuma largura compensa. Depois disso, a MLP esbarra no que não tem: sem compartilhamento de pesos nem noção de vizinhança, "
                "camadas extras não criam invariância à posição, só mais capacidade de decorar, e o early stopping corta essa capacidade antes que vire ganho."},
        {"id": "hp-neuronios", "group": "Capacidade", "name": "Neurônios por camada da MLP", "nets": ["MLP"], "values": "128 · 256 · 512 · 1024 · 2048",
         "choice": "256", "verdict": "pequeno",
         "effect": pp(neu[1]["mean"] - min(x["mean"] for x in neu)), "effect_note": "256 × pior largura, média das 5 profundidades",
         "marg": neu, "marg_note": "média da triagem (folds 1–3) sobre as 5 profundidades",
         "evidence": [f"256 neurônios tem a maior média ({pct(neu[1]['mean'])}); acima disso, a média cai para {pct(neu[3]['mean'])}–{pct(neu[2]['mean'])}.",
                      f"Com 1024–2048 neurônios, a melhor época chega mais cedo ({num(np.mean([neu[3]['ep'], neu[4]['ep']]))} contra {num(neu[0]['ep'])} com 128).",
                      f"A maior rede, `5 × 2048`, tem {mil(big['num_parameters'])} parâmetros e fica em {pct(big[V])}."],
         "why": "Quase todos os parâmetros da MLP estão na primeira camada (3.072 × N pesos, um por pixel e canal). Aumentar N multiplica pesos que "
                "olham pixels isolados, e com 40.000 imagens a rede consegue usá-los para memorizar cada imagem de treino em vez de aprender o que se "
                "repete entre elas. O sintoma é a validação atingir o pico mais cedo e depois cair enquanto o treino continua subindo (curvas do Bloco 1). "
                "256 neurônios já esgotam o que a entrada achatada permite extrair."},
        {"id": "hp-blocos", "group": "Capacidade", "name": "Blocos convolucionais", "nets": ["CNN"], "values": "2 · 3 · 4",
         "choice": "4 no Bloco 1 · 3 com BatchNorm (+B) · 4 com augmentation", "verdict": "depende do contexto",
         "effect": pp(blk[2]["mean"] - blk[0]["mean"]), "effect_note": "2 → 4 blocos sem regularização (k3 · same · maxpool)",
         "marg": blk + blk_bn, "marg_note": "sem BN: triagem do Bloco 1 · com BN: Complementar B, 5 folds",
         "evidence": [f"Sem regularização, cada bloco a mais ajudou: {' → '.join(pct(x['mean']) for x in blk)}.",
                      f"Com BatchNorm, 3 blocos venceu 4 em {pb['B3_pool2']['wins']} de {pb['B3_pool2']['n']} folds ({pp(pb['B3_pool2']['delta'])}), com melhor época {num(eb['B3_pool2']['melhor_epoca_media'], 0)} contra {num(eb['B4_pool2']['melhor_epoca_media'], 0)}.",
                      f"Com augmentation, 4 blocos voltou a vencer 3: {pp(depth_aug['delta'])}, {depth_aug['wins']} de {depth_aug['n']} folds."],
         "why": "Cada bloco reduz a resolução pela metade e dobra os filtros: o bloco novo enxerga uma região maior da imagem e combina as bordas e texturas "
                "do anterior em partes de objetos. Mais blocos significam uma hierarquia mais rica, mas também mais capacidade para decorar, e o número ótimo "
                "depende de qual dos dois é o gargalo. Sem regularização, o limite era extrair boas features, e o 4º bloco ajudava. Com BatchNorm, as features "
                "melhoraram e a rede de 3 blocos passou a treinar por mais épocas úteis, enquanto a de 4 chegava ao ótimo cedo e passava a decorar. Com augmentation, "
                "que torna a memorização muito mais difícil, a capacidade extra voltou a virar acerto."},
        # ---------------------------------------------------------------- extração espacial
        {"id": "hp-reducao", "group": "Extração espacial", "name": "Redução espacial: max pooling × stride 2", "nets": ["CNN"], "values": "maxpool 2×2 · conv stride 2",
         "choice": "max pooling", "verdict": "decisivo",
         "effect": pp(red[0]["mean"] - red[1]["mean"]), "effect_note": f"média; venceu {sum(d > 0 for d in mp)} de {len(mp)} pares",
         "marg": red, "marg_note": "média da triagem do Bloco 1 sobre blocos, kernel e padding",
         "evidence": [f"Em todos os {len(mp)} pares comparáveis (mesmos blocos, kernel e padding), max pooling venceu, por {pp(min(mp))} a {pp(max(mp))}.",
                      "Os dois lados de cada par têm **o mesmo número de parâmetros**: a diferença vem só da operação.",
                      f"A pior rede com pooling ({pct(worst_pool[V])}) supera a melhor com stride 2 ({pct(best_stride[V])})."],
         "why": "Com stride 2, a convolução é calculada em só 1 de cada 4 posições: se a borda ou textura que o filtro detecta cai entre elas, simplesmente "
                "não é vista, e deslocar o objeto 1 pixel muda a resposta. Com max pooling, a convolução é calculada em todas as posições e o pooling guarda a "
                "resposta mais forte de cada janela 2×2, então a feature sobrevive mesmo que o objeto se desloque um pouco. É invariância local à translação, "
                "exatamente a variação mais comum entre fotos da mesma classe."},
        {"id": "hp-kernel", "group": "Extração espacial", "name": "Tamanho do kernel", "nets": ["CNN"], "values": "3×3 · 5×5",
         "choice": "3×3", "verdict": "moderado",
         "effect": f"{pp(min(kp))} a {pp(max(kp))}", "effect_note": f"3×3 × 5×5 com max pooling, {sum(d > 0 for d in kp)} de {len(kp)} pares",
         "marg": ker, "marg_note": "média da triagem do Bloco 1, só max pooling",
         "evidence": [f"Com max pooling, 3×3 venceu 5×5 em {sum(d > 0 for d in kp)} de {len(kp)} pares, por {pp(min(kp))} a {pp(max(kp))}.",
                      f"Com stride 2 o resultado é misto ({pp(min(ks))} a {pp(max(ks))}): ali o gargalo é a subamostragem, não o kernel.",
                      "Um kernel 5×5 tem 25 pesos por canal de entrada contra 9 do 3×3 (≈ 2,8×)."],
         "why": "Em imagens 32×32, a hierarquia de blocos já amplia o campo receptivo: uma convolução 3×3, um pooling 2×2 e outra 3×3 cobrem 7×7 pixels da "
                "imagem original, mais que um único 5×5. O kernel maior acrescenta quase três vezes mais pesos por camada, e portanto mais capacidade de "
                "decorar, sem trazer informação que a profundidade não obteria de qualquer forma."},
        {"id": "hp-padding", "group": "Extração espacial", "name": "Padding", "nets": ["CNN"], "values": "same · valid",
         "choice": "same", "verdict": "pequeno",
         "effect": f"{pp(min(pdp))} a {pp(max(pdp))}", "effect_note": f"same × valid com max pooling, {sum(d > 0 for d in pdp)} de {len(pdp)} pares",
         "marg": pad, "marg_note": "média da triagem do Bloco 1, só max pooling",
         "evidence": [f"Com max pooling, same venceu em {sum(d > 0 for d in pdp)} de {len(pdp)} pares ({pp(min(pdp))} a {pp(max(pdp))}); considerando também stride 2, em {sum(d > 0 for d in pda)} de {len(pda)}.",
                      "4 das 24 combinações do Bloco 1 foram descartadas antes do treino: com valid e redes profundas ou kernel 5, o mapa chega a 0×0.",
                      "Nas 2 linhas e colunas externas de uma imagem 32×32 estão 240 dos 1.024 pixels (23%)."],
         "why": "Sem padding, cada convolução 3×3 corta 1 pixel de cada borda (32 → 30), e os pixels da borda participam de menos convoluções que os do centro. "
                "Em imagens tão pequenas, a borda é uma fração grande da informação, e objetos do CIFAR-10 frequentemente encostam nela. O efeito é pequeno "
                "porque o pooling reduz a resolução muito mais que o corte do valid; ele só vira decisivo quando os cortes se acumulam até zerar o mapa."},
        {"id": "hp-pooling", "group": "Extração espacial", "name": "Janela de pooling", "nets": ["CNN"], "values": "2×2 · 3×3 · 4×4",
         "choice": "2×2 com 3 blocos (mapa final 4×4)", "verdict": "moderado",
         "effect": pp(-pb["B3_pool3"]["delta"] + pb["B3_pool2"]["delta"]), "effect_note": f"3 blocos: pool 2 × pool 3 (5 folds)",
         "marg": pool, "marg_note": "Complementar B · 5 folds · rótulo mostra o mapa final",
         "evidence": [f"O melhor resultado ({pct(eb['B3_pool2'][V])}) veio de reduzir 32 → 16 → 8 → 4 em três passos de 2×2.",
                      f"Reduzir rápido demais perde: `3 blocos · pool 3` chega a 1×1 e cai para {pct(eb['B3_pool3'][V])}; `2 blocos · pool 4` chega a 2×2 e fica em {pct(eb['B2_pool4'][V])}.",
                      f"Reduzir de menos também perde: `2 blocos · pool 2` termina em 8×8 × 128 filtros = 8.192 entradas na camada densa ({mil(eb['B2_pool2']['num_parameters'])} parâmetros) e fica em {pct(eb['B2_pool2'][V])}."],
         "why": "A janela de pooling controla a velocidade com que a imagem é resumida. Com janela 3 ou 4, cada pooling descarta 8/9 ou 15/16 das posições "
                "de uma vez, e detalhes que só seriam combinados no bloco seguinte desaparecem antes. No outro extremo, um mapa final grande chega à camada "
                "densa com milhares de entradas que ainda carregam a posição exata de cada feature, e a camada densa usa esses pesos para decorar. "
                "O ponto ideal foi reduzir devagar até um mapa pequeno, mas não trivial."},
        # ---------------------------------------------------------------- otimização
        {"id": "hp-otimizador", "group": "Otimização", "name": "Otimizador: Adam × SGD com momentum", "nets": ["MLP", "CNN"], "values": "Adam · SGD (momentum 0,9)",
         "choice": "Adam (lr 3e-4 na MLP, 1e-3 na CNN)", "verdict": "empate na faixa útil",
         "effect": f"{pp(ma[V] - ms[V])} · {pp(ca[V] - cs[V])}", "effect_note": "Adam campeão × melhor SGD · MLP · CNN",
         "marg": [dict(x, label="MLP · " + x["label"]) for x in opt_m] + [dict(x, label="CNN · " + x["label"]) for x in opt_c],
         "marg_note": "Adam campeão e melhor taxa do SGD · triagem do Bloco 2",
         "evidence": [f"Com a taxa campeã do Adam e a melhor do SGD, os dois empatam: MLP {pct(ma[V])} × {pct(ms[V])}; CNN {pct(ca[V])} × {pct(cs[V])}.",
                      f"O Adam rende o melhor com lr 3e-4 a 1e-3 (na CNN, {pct(ca3[V])} × {pct(ca[V])}); o SGD precisa de {lrlab(ms)}, 10 a 30× maior.",
                      f"Com lr 1e-4, o SGD esgota as 40 épocas e para em {pct(sgdlow[V])} na MLP; o Adam com o mesmo lr chega a {pct(a4m[V])}."],
         "why": "O SGD dá passos proporcionais ao gradiente: onde o gradiente é pequeno, anda pouco, e por isso precisa de lr alto (o momentum 0,9 ainda "
                "acumula passos na mesma direção). O Adam divide o passo de cada peso pela magnitude recente do seu próprio gradiente, então o tamanho "
                "do passo fica perto do lr qualquer que seja a escala do gradiente. Isso explica os dois fatos: o Adam converge rápido com lr pequeno, e o SGD "
                "só chega ao mesmo lugar quando o lr compensa a escala do gradiente. Acertada a taxa de cada um, o algoritmo quase não importa."},
        {"id": "hp-lr", "group": "Otimização", "name": "Taxa de aprendizagem", "nets": ["MLP", "CNN"], "values": "1e-4 · 3e-4 · 1e-3 · 3e-3 · 1e-2 · 3e-2 · 1e-1",
         "choice": "Adam 3e-4 (MLP) · Adam 1e-3 (CNN)", "verdict": "risco alto se errado",
         "effect": pp(0.1 - a3m[V], 0), "effect_note": "Adam na MLP: lr 3e-4 → lr ≥ 3e-2 (colapso em 10%)",
         "marg": lr_adam_c, "marg_note": "CNN com Adam · triagem do Bloco 2",
         "evidence": [f"Com Adam, a MLP colapsa em 10% (acaso) a partir de 3e-2; a CNN já fica instável em 3e-3 ({pct(a3c[V])} ± {num(a3c['val/accuracy_std'] * 100)}) e colapsa em 1e-2 (média {pct(a2c[V])}: um fold chegou a 47%, os outros dois caíram para 10%).",
                      f"O SGD tolera melhor lr alto: com 1e-1, a MLP ainda chega a {pct(s1m[V])} e a CNN a {pct(s1c[V])}.",
                      f"Dentro da faixa útil a diferença é pequena, mas o custo em épocas não: com Adam, 1e-4 × 3e-4 muda {pp(a4m[V] - a3m[V])} na MLP e dobra a melhor época ({num(a3m['melhor_epoca_media'])} → {num(a4m['melhor_epoca_media'])}); na CNN, 1e-4 fica {num((ca[V] - a4c[V]) * 100)} p.p. abaixo."],
         "why": "Com Adam, cada peso anda aproximadamente lr por passo, qualquer que seja o gradiente. Com lr de 1e-2 ou mais, esses passos são grandes perto "
                "de pesos que valem da ordem de 0,01 a 0,1 na inicialização: a rede salta para uma região onde as saídas saturam e muitos ReLU deixam de ativar, "
                "os gradientes somem e ela passa a prever uma única classe. A acurácia cai para 10% sem que a loss vire NaN, por isso é colapso e não divergência. "
                "É o hiperparâmetro com o pior custo de erro do estudo, e o único em que um valor ruim destrói o modelo em vez de só piorá-lo."},
        {"id": "hp-epocas", "group": "Otimização", "name": "Épocas e paciência", "nets": ["MLP", "CNN"], "values": "40–150 épocas · paciência 5–15",
         "choice": "40 (B1–B2) · 50 (B3) · 100 (+A, aug CNN) · 150 (aug MLP)", "verdict": "verificado",
         "effect": pp(ea["do0.5_losscross_entropy"][V] - _one(b3m, dropout=0.5, loss_fn="cross_entropy")[V]),
         "effect_note": "dropout 0,5: 50 → 100 épocas", "marg": None,
         "evidence": [f"Com o dobro de épocas, dropout 0,5 foi de {pct(_one(b3m, dropout=0.5, loss_fn='cross_entropy')[V])} (folds 1–3) para {pct(ea['do0.5_losscross_entropy'][V])} (5 folds) e continuou perdendo para 0,2 em {pa['do0.5_losscross_entropy']['n'] - pa['do0.5_losscross_entropy']['wins']} de {pa['do0.5_losscross_entropy']['n']} folds.",
                      f"SGD com lr ≤ 3e-4 usou as 40 épocas inteiras (melhor época {num(sgdlow['melhor_epoca_media'], 0)}): limitado pelo orçamento.",
                      f"A MLP com augmentation teve melhor época {num(am['aug1_do0.2']['best_epoch'], 0)} num limite de 150: o 57,8% no teste é um piso."],
         "why": "O early stopping só interrompe quando a val/loss para de melhorar por várias épocas seguidas, então o limite de épocas só pesa para quem "
                "aprende devagar: lr baixo (passos pequenos), dropout alto (cada passo atualiza só uma sub-rede) e augmentation (cada época mostra imagens "
                "diferentes). Para essas configurações, o orçamento pode decidir o resultado, por isso foi testado com treinos mais longos em vez de assumido."},
        # ---------------------------------------------------------------- regularização
        {"id": "hp-dropout-mlp", "group": "Regularização", "name": "Dropout na MLP", "nets": ["MLP"], "values": "0 · 0,2 · 0,3 · 0,5",
         "choice": "0,2", "verdict": "moderado",
         "effect": pp(do_m[1]["mean"] - do_m[0]["mean"]), "effect_note": "0 → 0,2 com entropia cruzada",
         "marg": do_m, "marg_note": "triagem do Bloco 3 · entropia cruzada · rótulo à direita = melhor época",
         "evidence": [f"Validação por taxa: {' · '.join(x['label'].replace('dropout ', '') + ' = ' + pct(x['mean']) for x in do_m)}.",
                      f"A melhor época cresce com a taxa: {' → '.join(num(x['ep'], 0) for x in do_m)}.",
                      f"Com 100 épocas, 0,5 perdeu para 0,2 em {pa['do0.5_losscross_entropy']['n'] - pa['do0.5_losscross_entropy']['wins']} de {pa['do0.5_losscross_entropy']['n']} folds ({pp(pa['do0.5_losscross_entropy']['delta'], 2)}), com acurácia de treino menor ({pct(ea['do0.5_losscross_entropy']['train/accuracy_mean'])} × {pct(ea['do0.2_losscross_entropy']['train/accuracy_mean'])}): subajuste."],
         "why": "A cada passo, o dropout desliga uma fração aleatória dos neurônios. Nenhum neurônio pode contar com outro específico estar presente, então "
                "a informação precisa ser espalhada por caminhos redundantes; na prática, a rede treina um conjunto de sub-redes que compartilham pesos. "
                "Isso torna a memorização de imagens individuais mais lenta: com 0,2, a rede aprende por mais épocas antes de começar a decorar, e o gap "
                "treino–validação não diminui porque os dois sobem juntos. Com 0,5, cada camada de 256 neurônios treina com cerca de 128 ativos: a capacidade "
                "efetiva fica abaixo do necessário e a rede erra mais até no treino."},
        {"id": "hp-batchnorm", "group": "Regularização", "name": "BatchNorm", "nets": ["CNN"], "values": "sem · com",
         "choice": "com", "verdict": "decisivo",
         "effect": f"{pp(min(d for _, d in bn_pairs))} a {pp(max(d for _, d in bn_pairs))}", "effect_note": "com × sem BN, para cada taxa de dropout",
         "marg": bn, "marg_note": "média da triagem do Bloco 3 sobre as taxas de dropout",
         "evidence": ["Ganho em cada taxa de dropout: " + " · ".join(f"{num(d, 1)} → {pp(g)}" for d, g in bn_pairs) + ".",
                      f"A acurácia de treino também sobe ({pct(np.mean([r['train/accuracy_mean'] for r in bn0]))} → {pct(np.mean([r['train/accuracy_mean'] for r in bn1]))}): a rede otimiza melhor, não só generaliza melhor.",
                      "É o maior ganho de um único hiperparâmetro da CNN, e só com ele o dropout passa a fazer diferença."],
         "why": "A BatchNorm normaliza a saída de cada filtro para média 0 e desvio 1 dentro do mini-batch e depois reescala com dois parâmetros aprendidos. "
                "Sem ela, a escala das ativações de um bloco muda sempre que os pesos dos blocos anteriores mudam, e cada camada precisa se readaptar a "
                "entradas em movimento; com ela, cada bloco recebe entradas de escala estável e os gradientes chegam equilibrados aos primeiros blocos, "
                "então o mesmo lr rende passos mais produtivos. Os números mostram esse efeito na otimização (o treino sobe). Além disso, como média e desvio "
                "variam de um batch para outro, a normalização injeta um ruído que funciona como regularização leve."},
        {"id": "hp-dropout-cnn", "group": "Regularização", "name": "Dropout na camada densa da CNN", "nets": ["CNN"], "values": "0 · 0,3 · 0,5",
         "choice": "0,5", "verdict": "só com BatchNorm",
         "effect": f"{pp(g_bn)} com BN · {pp(g_nobn)} sem", "effect_note": "0 → 0,5",
         "marg": do_c_nobn + do_c_bn, "marg_note": "triagem do Bloco 3 · pooling 2×2",
         "evidence": [f"Sem BatchNorm, as três taxas ficam em {pct(min(x['mean'] for x in do_c_nobn))}–{pct(max(x['mean'] for x in do_c_nobn))}: diferença do tamanho do ruído.",
                      f"Com BatchNorm, 0 → 0,5 rende {pp(g_bn)} e reduz o gap de {num(gap_bn[0.0], 3)} para {num(gap_bn[0.5], 3)}.",
                      f"Com augmentation, 0,5 continuou à frente de 0,3 (4 blocos: {pct(ac['aug1_B4_do0.5']['val'])} × {pct(ac['aug1_B4_do0.3']['val'])})."],
         "why": f"O dropout foi aplicado só na camada densa de 512, que concentra boa parte dos parâmetros ({mil(dense(2048))} dos 2,6 M na rede de 4 blocos; "
                f"{mil(dense(4096))} dos 2,5 M na de 3 blocos). A leitura mais provável: sem BatchNorm, o limite da rede era a qualidade das features "
                "convolucionais (ela parava na época 4–5), e regularizar a camada densa não ajuda a extrair nada melhor. Com BatchNorm, as features melhoram "
                "e a camada densa passa a ser onde a rede decora; é exatamente esse tipo de overfitting que o dropout combate."},
        # ---------------------------------------------------------------- função de erro
        {"id": "hp-loss", "group": "Função de erro", "name": "Entropia cruzada × MSE", "nets": ["MLP"], "values": "cross_entropy · mse",
         "choice": "entropia cruzada", "verdict": "pequeno, consistente",
         "effect": pp(-pa["do0.2_lossmse"]["delta"], 2), "effect_note": f"pareado com dropout 0,2, {pa['do0.2_lossmse']['n'] - pa['do0.2_lossmse']['wins']} de {pa['do0.2_lossmse']['n']} folds",
         "marg": loss_m, "marg_note": "triagem do Bloco 3 · dropout 0,2",
         "evidence": [f"Com 100 épocas, a entropia cruzada venceu o MSE em {pa['do0.2_lossmse']['n'] - pa['do0.2_lossmse']['wins']} de {pa['do0.2_lossmse']['n']} folds ({pp(-pa['do0.2_lossmse']['delta'], 2)}).",
                      f"O MSE converge mais devagar: melhor época {num(ea['do0.2_lossmse']['melhor_epoca_media'], 0)} contra {num(ea['do0.2_losscross_entropy']['melhor_epoca_media'], 0)}.",
                      "O MSE foi calculado entre `softmax(logits)` e o vetor one-hot (`src/train.py`)."],
         "why": "Com softmax e entropia cruzada, o gradiente em relação a cada logit é simplesmente p − y: é grande exatamente quando a rede está confiante e "
                "errada. No MSE sobre as probabilidades, o gradiente passa pela derivada do softmax, proporcional a p(1 − p); quando a probabilidade da classe "
                "certa está perto de 0, esse fator também está, e os exemplos mais errados quase não corrigem a rede. Ela aprende mais devagar com os erros "
                "graves e termina um pouco pior, e a comparação pareada mostra que a diferença, embora pequena, não é ruído."},
        # ---------------------------------------------------------------- dados
        {"id": "hp-augmentation", "group": "Dados", "name": "Data augmentation (crop + flip)", "nets": ["MLP", "CNN"], "values": "sem · com",
         "choice": "com (bônus, fora dos blocos)", "verdict": "decisivo na CNN",
         "effect": f"{pp(pvc['aug1_B4_do0.5']['delta'])} · {pp(pvm['aug1_do0.2']['delta'])}", "effect_note": "pareado, 5 de 5 folds · CNN · MLP",
         "marg": aug_rows, "marg_note": "validação · rótulo à direita = melhor época",
         "evidence": [f"CNN: {pp(pvc['aug1_B4_do0.5']['delta'])} na validação, {pvc['aug1_B4_do0.5']['wins']} de {pvc['aug1_B4_do0.5']['n']} folds; o maior ganho do estudo.",
                      f"MLP: {pp(pvm['aug1_do0.2']['delta'])}, {pvm['aug1_do0.2']['wins']} de {pvm['aug1_do0.2']['n']} folds, com melhor época {num(am['aug0_do0.2']['best_epoch'], 0)} → {num(am['aug1_do0.2']['best_epoch'], 0)}.",
                      f"O gap cai nas duas redes: CNN {num(ac['aug0_B3_do0.5']['gap'], 3)} → {num(ac['aug1_B4_do0.5']['gap'], 3)}; MLP {num(am['aug0_do0.2']['gap'], 3)} → {num(am['aug1_do0.2']['gap'], 3)}."],
         "why": "Deslocar a imagem até 4 pixels e espelhá-la gera versões que continuam sendo da mesma classe, então a rede não consegue decorar cada imagem e "
                "precisa aprender o que é comum a todas as versões. A CNN aproveita isso quase de graça: a convolução usa os mesmos pesos em todas as posições, "
                "então um padrão aprendido num lugar vale nos outros, e o pooling absorve deslocamentos pequenos. A MLP tem um peso separado por pixel: para ela, "
                "uma imagem deslocada 2 pixels é um vetor de entrada diferente, e cada posição precisa ser aprendida separadamente. Por isso o ganho é menor e a "
                "convergência muito mais lenta."},
    ]
    return _tidy(H)
