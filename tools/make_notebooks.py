"""Gera Kaggle_MLP.ipynb e Kaggle_CNN.ipynb a partir dos grids em grids/*.json.

Uso: python tools/make_notebooks.py .
"""
import json
import os
import sys

ROOT = sys.argv[1]
REPO_URL = "https://github.com/diegoflyra/dfal-neural-networks.git"
REPO_DIR = "/tmp/dfal-neural-networks"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def spec(block):
    with open(os.path.join(ROOT, "grids", f"{block}.json"), encoding="utf-8") as f:
        return json.load(f)


def fmt_value(v):
    if isinstance(v, dict):
        label = v.get("_label", "")
        params = ", ".join(f"`{k}={x}`" for k, x in v.items() if k != "_label")
        return f"**{label}** ({params})"
    return f"`{v}`"


def grid_table(block):
    s = spec(block)
    lines = ["| Eixo | Valores |", "|---|---|"]
    for axis, values in s.get("grid", {}).items():
        lines.append(f"| `{axis}` | {', '.join(fmt_value(v) for v in values)} |")
    n = 1
    for values in s.get("grid", {}).values():
        n *= len(values)
    base = ", ".join(f"`{k}={v}`" for k, v in s.get("base", {}).items() if k not in ("seed",))
    inherited = f"Herdado do campeão de: {', '.join(f'`{b}`' for b in s['base_from'])}. " if s.get("base_from") else ""
    triage = (f"Triagem nos folds {s['screening_folds']} de {s['k_folds']}; as {s.get('finalists', 3)} melhores "
              f"completam os {s['k_folds']} folds." if len(s["screening_folds"]) < s["k_folds"]
              else f"Todas as configurações rodam os {s['k_folds']} folds.")
    extra = f" Limite: {s['max_parameters']:,} parâmetros." if s.get("max_parameters") else ""
    return (("\n".join(lines) + f"\n\n**{n} combinações.** " if s.get("grid") else "")
            + f"{inherited}{('Fixos no bloco: ' + base + '. ') if base else ''}{triage}{extra}")


def run_grid(block):
    return code(f"!python src/grid_search.py grids/{block}.json --workers_per_gpu $WORKERS_PER_GPU")


def backup():
    return code("# Backup parcial: CSV/JSON/PNG/logs (os .pth ficam em outputs/ na aba Output)\n"
                "!cd /kaggle/working && zip -qr outputs.zip outputs -x '*.pth' && ls -lh outputs.zip")


def analysis(title, questions):
    return md(f"### 📝 Análise — {title}\n\n" + "\n".join(f"- **{q}** _…_" for q in questions))


def grid_results(block, heatmap_call, show_discarded=False):
    cells = [
        md(f"#### Resultados da triagem — `{block}`\n\nTodas as configurações, ordenadas pela **validação** "
           f"(média ± desvio nos folds de triagem). As métricas de teste não aparecem aqui de propósito."),
        code(f'rep.grid_ranking("{block}", "triagem")'),
    ]
    if show_discarded:
        cells.append(code(f'rep.discarded_configs("{block}")  # combinações descartadas antes do treino e o motivo'))
    cells += [
        code(heatmap_call),
        code(f'rep.plot_grid_bars("{block}")'),
        md(f"#### Confirmação das finalistas e campeão — `{block}`\n\nAs finalistas completaram todos os folds; "
           f"o campeão é a maior `val/accuracy` média nos K folds."),
        code(f'rep.grid_ranking("{block}", "final")'),
        code(f'rep.show_champion("{block}")\nrep.plot_finalists("{block}")'),
    ]
    return cells


def intro(kind, time_1gpu, time_2gpu, workers):
    return md(f"""
# CIFAR-10 com {kind} — Relatório de experimentos (grid search em blocos)

**Metodologia.** Em vez de escolher um valor por vez, cada bloco executa um **grid search** e o vencedor é
decidido pelos dados. O campeão de um bloco é lido automaticamente pelo bloco seguinte, sem nenhum valor escolhido à mão:

| Bloco | Pergunta | Grid |
|---|---|---|
| 0 — Referência | Onde partimos? | {"regressão logística e MLP do notebook original" if kind == "MLP" else "LeNet-5 do notebook original"} |
| 1 — Topologia | Qual estrutura base absorve melhor o CIFAR-10? | {"camadas × neurônios" if kind == "MLP" else "blocos × kernel × padding × redução (pool/stride)"} |
| 2 — Otimização | Qual algoritmo e taxa de aprendizagem fazem a rede campeã convergir melhor? | otimizador × lr |
| Checagem | A escolha em blocos se sustenta? | 2º e 3º do Bloco 1 com o otimizador campeão |
| 3 — {"Regularização e erro" if kind == "MLP" else "Pooling e regularização"} | {"Como dropout e função de erro afetam a generalização?" if kind == "MLP" else "Como a janela de pooling, dropout e batch norm afetam a generalização?"} | {"dropout × função de erro" if kind == "MLP" else "pool × dropout × batch norm"} |

**Protocolo de avaliação.**
- **Validação cruzada estratificada em 5 folds** sobre as 50.000 imagens de treino, com as mesmas partições em todos os experimentos (comparações pareadas).
- **Triagem:** cada configuração roda os folds 1–3; as 3 melhores **completam os folds 4–5** (sem refazer nada) e o campeão é a maior média nos 5 folds.
- **Escolha sempre pela validação.** O conjunto de teste (10.000 imagens) só é revelado na seção final, para os campeões. Usá-lo para escolher tornaria o número final otimista (vazamento de dados).
- Diferenças menores que o desvio padrão entre folds não devem ser tratadas como melhora real.

**Registro dos resultados.** Cada experimento grava em `/kaggle/working/outputs/{{exp_name}}/`:

| Arquivo | Conteúdo |
|---|---|
| `parametros.json` | hiperparâmetros exatos, arquitetura resolvida, versões e commit |
| `historico_treino.csv` | uma linha por **fold × época**: loss, acurácia, precision/recall/F1 (gerais e por classe) de treino e validação, gap |
| `historico_treino_agregado.csv` | média e desvio padrão entre folds, por época |
| `resultados.json` | resultado de cada fold e médias (validação na melhor época e teste) |
| `melhor_modelo.pth` | pesos com a menor `val/loss` (melhor fold); cada fold em `folds/fold_k/` |

Cada bloco grava em `outputs/_grids/{{bloco}}/`: `configs.csv`, `ranking_triagem.csv`, `ranking_final.csv`, `campeao.json` e `logs/`.
Tabelas e figuras dos relatórios ficam em `outputs/_relatorio/`. Tudo é gravado a cada época, antes do envio ao Weights & Biases.

**Tempo estimado:** ~{time_1gpu} com 1 GPU T4 ou **~{time_2gpu} com 2× T4** (estimativa a partir de medições; varia ±30%).
Os experimentos são distribuídos entre as GPUs visíveis ({workers} por GPU, ajustável em `WORKERS_PER_GPU`).

**Antes de executar (Kaggle):**
1. *Settings → Accelerator*: **GPU T4 ×2**. *Settings → Internet*: **On**.
2. *Add-ons → Secrets*: `GITHUB_TOKEN` (obrigatório se o repositório for privado) e `WANDB_API_KEY`
   (opcional — sem ele tudo continua salvo localmente). Marque os dois como anexados a este notebook.
3. Use **Save Version → Save & Run All (Commit)**: `/kaggle/working` (com `outputs/` e `outputs.zip`) fica salvo na aba *Output*.
4. **Retomada:** tudo é retomável. Se a sessão cair, adicione o Output da versão anterior como *Input*, preencha `RESUME_FROM`
   e rode de novo: experimentos e folds concluídos são pulados.
""")


def setup(workers):
    return [
        md("## 0. Preparação do ambiente"),
        code(f"""
import os

REPO_URL = "{REPO_URL}"
REPO_DIR = "{REPO_DIR}"  # fora de /kaggle/working: código e dataset não poluem o Output

# Repositório privado: crie o secret GITHUB_TOKEN (Add-ons → Secrets) com um token de leitura do GitHub.
# O token fica só em /tmp (não vai para o Output) e nunca é impresso.
clone_url = REPO_URL
try:
    from kaggle_secrets import UserSecretsClient
    clone_url = REPO_URL.replace("https://", "https://" + UserSecretsClient().get_secret("GITHUB_TOKEN") + "@")
    print("GitHub: usando GITHUB_TOKEN")
except Exception:
    print("GitHub: sem GITHUB_TOKEN (funciona apenas se o repositório for público)")

if os.path.isdir(REPO_DIR):
    !git -C $REPO_DIR pull -q
else:
    !git clone -q $clone_url $REPO_DIR
%cd $REPO_DIR
!git log -1 --oneline
"""),
        code("!pip install -q -r requirements.txt"),
        code(f"""
import shutil
import sys

import pandas as pd

WORKERS_PER_GPU = {workers}  # experimentos simultâneos por GPU
RESUME_FROM = ""  # ex.: "/kaggle/input/<output-da-versao-anterior>/outputs" para retomar

# Onde os resultados são gravados (lido por run_experiment.py, grid_search.py e report_utils.py)
os.environ["EXP_OUTPUT_DIR"] = "/kaggle/working/outputs"
os.makedirs(os.environ["EXP_OUTPUT_DIR"], exist_ok=True)
if RESUME_FROM:
    shutil.copytree(RESUME_FROM, os.environ["EXP_OUTPUT_DIR"], dirs_exist_ok=True)
    print(f"Resultados anteriores copiados de {{RESUME_FROM}}; o que já foi concluído será pulado.")

# Weights & Biases via Kaggle Secrets; se falhar, os experimentos seguem apenas com o registro local.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    print("W&B: chave carregada.")
except Exception as exc:
    os.environ["WANDB_MODE"] = "disabled"
    print(f"W&B desativado ({{exc}}). Resultados continuam em {{os.environ['EXP_OUTPUT_DIR']}}.")

sys.path.insert(0, os.path.join(REPO_DIR, "src"))
import report_utils as rep
"""),
        code("""
# Verificações rápidas antes de gastar GPU (sem treino):
# GPUs, flags → arquitetura, carregamento em GPU/K-fold (baixa o CIFAR-10) e construção de todas as configurações dos grids
!nvidia-smi -L
!python tests/check_hyperparams.py
!python tests/check_data_loader.py
!python tests/check_grids.py
"""),
    ]


def final_section(prefix, blocks, last_block, per_class_note):
    return [
        md(f"""
## Resultado final — teste revelado

Até aqui todas as escolhas foram feitas pela validação cruzada. Agora o conjunto de teste é usado **uma única vez**
para os campeões de cada bloco e para as referências do Bloco 0. A tabela mostra média ± desvio do teste entre os
5 modelos (um por fold) de cada configuração.
"""),
        code(f"""
final = rep.final_report({blocks!r})
final
"""),
        md(f"Desempenho por classe no teste: referências do Bloco 0 × campeão final. {per_class_note}"),
        code(f"""
campeao_final = rep.show_champion("{last_block}")["exp_name"]
comparar = list(final.loc[final["papel"] == "referência", "exp_name"]) + [campeao_final]
rep.plot_per_class(comparar, metric="recall")
rep.plot_per_class(comparar, metric="precision")
pd.read_csv(os.path.join(os.environ["EXP_OUTPUT_DIR"], campeao_final, "matriz_confusao_teste.csv"), index_col=0)
"""),
        code("!cd /kaggle/working && zip -qr outputs.zip outputs -x '*.pth' && ls -lh outputs.zip"),
        md("## 📝 Conclusões\n\n"
           "- **Estrutura base que melhor absorveu o CIFAR-10 (Bloco 1) e por quê:** _…_\n"
           "- **Ganho de otimização (Bloco 2) e sensibilidade à taxa de aprendizagem:** _…_\n"
           "- **A checagem confirmou a escolha em blocos?** _…_\n"
           "- **Efeito da regularização/erro (Bloco 3):** _…_\n"
           "- **Ganho total sobre a referência (teste):** _…_\n"
           "- **Classes mais difíceis e hipótese:** _…_"),
    ]


# =============================================================================== MLP
mlp = [intro("MLP", "4 h", "2–2,5 h", 2), *setup(2),
    md(f"""
## Bloco 0 — Referências

Mesmo protocolo dos blocos seguintes (Adam, lr=1e-3, batch 128, early stopping com paciência 5, 5 folds), aplicado à
regressão logística (piso) e à MLP 64-128-64 do notebook original. Servem de ponto de comparação para todo o estudo.

{grid_table("mlp_b0_referencia")}
"""),
    run_grid("mlp_b0_referencia"),
    code('rep.grid_ranking("mlp_b0_referencia", "final")'),

    md(f"""
## Bloco 1 — Topologia

**Pergunta:** qual combinação de profundidade (camadas ocultas) e largura (neurônios por camada) absorve melhor o CIFAR-10?

**Por que primeiro:** a topologia define a capacidade do modelo; otimização e regularização são ajustadas sobre ela.
Todas as redes usam a mesma otimização padrão e **nenhuma regularização**, com early stopping na `val/loss`
para que redes grandes não sejam penalizadas por treinar além do ponto ótimo.

{grid_table("mlp_b1_topologia")}

**O que observar:** o heatmap camadas × neurônios, o `gap/accuracy` (overfitting) das redes maiores e se ganhos de
capacidade continuam aparecendo na validação.
"""),
    run_grid("mlp_b1_topologia"),
    *grid_results("mlp_b1_topologia", 'rep.heatmap("mlp_b1_topologia", row="mlp_layers", col="mlp_neurons")'),
    code('rep.heatmap("mlp_b1_topologia", row="mlp_layers", col="mlp_neurons", value="gap/accuracy_mean")'),
    analysis("Bloco 1", ["Profundidade ou largura: o que mais contribuiu?", "A partir de qual tamanho a validação estagna?",
                         "Como o gap cresce com a capacidade?", "Diferença do campeão para as referências (validação):"]),
    backup(),

    md(f"""
## Bloco 2 — Otimização

**Pergunta:** com a topologia campeã do Bloco 1 fixa, qual combinação de algoritmo e taxa de aprendizagem converge melhor?

**Base:** o campeão do Bloco 1 (abaixo), lido de `outputs/_grids/mlp_b1_topologia/campeao.json`. SGD usa momentum 0,9.
O grid de lr é comum aos dois algoritmos de propósito: ele mostra a faixa em que cada um funciona (lr baixos
não convergem no SGD; lr altos podem divergir no Adam — divergências são registradas sem perda de arquivos).

{grid_table("mlp_b2_otimizacao")}

**O que observar:** o heatmap otimizador × lr, a `melhor_epoca_media` (velocidade) e as curvas das finalistas.
"""),
    code('rep.show_champion("mlp_b1_topologia")'),
    run_grid("mlp_b2_otimizacao"),
    *grid_results("mlp_b2_otimizacao", 'rep.heatmap("mlp_b2_otimizacao", row="optimizer", col="lr")'),
    code('rep.heatmap("mlp_b2_otimizacao", row="optimizer", col="lr", value="melhor_epoca_media")'),
    analysis("Bloco 2", ["Faixa de lr útil para SGD e para Adam:", "Qual convergiu em menos épocas?",
                         "Houve divergência? Em quais combinações?", "Ganho sobre o Bloco 1 (validação):"]),

    md(f"""
### Checagem de interação

A busca em blocos assume que a melhor topologia com Adam lr=1e-3 continua sendo a melhor com o otimizador campeão.
Para verificar, o 2º e o 3º colocados do Bloco 1 são treinados com o otimizador/lr campeões do Bloco 2 (5 folds).
O Bloco 3 parte da melhor rede entre o campeão do Bloco 2 e esta checagem.

{grid_table("mlp_b2_checagem")}
"""),
    run_grid("mlp_b2_checagem"),
    code("""
pd.concat([rep.grid_ranking("mlp_b2_otimizacao", "final").head(1),
           rep.grid_ranking("mlp_b2_checagem", "final")], ignore_index=True)
"""),
    analysis("Checagem", ["A ordem das topologias se manteve com o novo otimizador?",
                          "Se mudou, o que isso indica sobre a interação topologia × otimização?"]),
    backup(),

    md(f"""
## Bloco 3 — Regularização e função de erro

**Pergunta:** com a rede convergindo bem, como a taxa de dropout e a função de erro afetam a generalização?

**Base:** a melhor rede entre o campeão do Bloco 2 e a checagem. Épocas e paciência aumentadas, pois dropout
retarda a convergência. A seleção é pela `val/accuracy`: a `val/loss` de MSE e de entropia cruzada estão em
escalas diferentes e não são comparáveis entre si.

{grid_table("mlp_b3_regularizacao")}

**O que observar:** o `gap/accuracy` diminuindo com dropout, o ponto em que dropout passa a causar subajuste
e a diferença de convergência entre MSE e entropia cruzada.
"""),
    run_grid("mlp_b3_regularizacao"),
    *grid_results("mlp_b3_regularizacao", 'rep.heatmap("mlp_b3_regularizacao", row="dropout", col="loss_fn")'),
    code('rep.heatmap("mlp_b3_regularizacao", row="dropout", col="loss_fn", value="gap/accuracy_mean")'),
    analysis("Bloco 3", ["Dropout reduziu o gap? A partir de qual taxa houve subajuste?",
                         "MSE vs entropia cruzada: diferença de acurácia e de velocidade de convergência:",
                         "Ganho sobre o Bloco 2 (validação):"]),
    backup(),
    *final_section("mlp", ["mlp_b0_referencia", "mlp_b1_topologia", "mlp_b2_otimizacao", "mlp_b2_checagem",
                           "mlp_b3_regularizacao"], "mlp_b3_regularizacao",
                   "Classes com baixo recall indicam confusões sistemáticas (veja a matriz de confusão)."),
]

# =============================================================================== CNN
cnn = [intro("CNN", "10 h", "5–6 h", 1), *setup(1),
    md(f"""
## Bloco 0 — Referência

Mesmo protocolo dos blocos seguintes (Adam, lr=1e-3, batch 128, early stopping com paciência 5, 5 folds), aplicado à
LeNet-5 adaptada do notebook original.

{grid_table("cnn_b0_referencia")}
"""),
    run_grid("cnn_b0_referencia"),
    code('rep.grid_ranking("cnn_b0_referencia", "final")'),

    md(f"""
## Bloco 1 — Topologia e extração espacial

**Pergunta:** qual estrutura convolucional absorve melhor o CIFAR-10: quantos blocos, qual janela de convolução,
qual padding e como reduzir a resolução espacial?

Cada bloco é `Conv2d → Ativação → redução`, com 64 filtros no primeiro bloco, dobrando a cada bloco.
A redução de resolução é tratada como **uma** variável — max pooling 2×2 (stride 1) **ou** convolução com stride 2
(sem pooling) —, pois combinar os dois reduz a imagem 4× por bloco e colapsa redes profundas. Combinações cujo
mapa de ativação chega a 0×0 são descartadas automaticamente antes do treino.
Sem regularização e com otimização padrão; early stopping na `val/loss`.

{grid_table("cnn_b1_topologia")}

**O que observar:** efeito da profundidade, kernel 3×3 vs 5×5 (campo receptivo × parâmetros), `same` vs `valid`
(perda de bordas) e max pooling vs stride (invariância local × redução aprendida).
"""),
    run_grid("cnn_b1_topologia"),
    *grid_results("cnn_b1_topologia",
                  'rep.heatmap("cnn_b1_topologia", row="conv_blocks", col=["kernel_size", "padding"], facet="reducao")',
                  show_discarded=True),
    code('rep.heatmap("cnn_b1_topologia", row="conv_blocks", col=["kernel_size", "padding"], facet="reducao", value="num_parameters")'),
    analysis("Bloco 1", ["Profundidade: quantos blocos foram úteis?", "Kernel 3×3 vs 5×5:", "Padding same vs valid:",
                         "Max pooling vs stride 2:", "Diferença do campeão para a referência (validação):"]),
    backup(),

    md(f"""
## Bloco 2 — Otimização

**Pergunta:** com a topologia campeã do Bloco 1 fixa, qual combinação de algoritmo e taxa de aprendizagem converge melhor?

**Base:** o campeão do Bloco 1 (abaixo). SGD usa momentum 0,9. O grid de lr é comum aos dois algoritmos de propósito,
para mostrar a faixa útil de cada um.

{grid_table("cnn_b2_otimizacao")}
"""),
    code('rep.show_champion("cnn_b1_topologia")'),
    run_grid("cnn_b2_otimizacao"),
    *grid_results("cnn_b2_otimizacao", 'rep.heatmap("cnn_b2_otimizacao", row="optimizer", col="lr")'),
    code('rep.heatmap("cnn_b2_otimizacao", row="optimizer", col="lr", value="melhor_epoca_media")'),
    analysis("Bloco 2", ["Faixa de lr útil para SGD e para Adam:", "Qual convergiu em menos épocas?",
                         "Houve divergência?", "Ganho sobre o Bloco 1 (validação):"]),

    md(f"""
### Checagem de interação

O 2º e o 3º colocados do Bloco 1 são treinados com o otimizador/lr campeões do Bloco 2 (5 folds).
O Bloco 3 parte da melhor rede entre o campeão do Bloco 2 e esta checagem.

{grid_table("cnn_b2_checagem")}
"""),
    run_grid("cnn_b2_checagem"),
    code("""
pd.concat([rep.grid_ranking("cnn_b2_otimizacao", "final").head(1),
           rep.grid_ranking("cnn_b2_checagem", "final")], ignore_index=True)
"""),
    analysis("Checagem", ["A ordem das topologias se manteve com o novo otimizador?"]),
    backup(),

    md(f"""
## Bloco 3 — Janela de pooling e regularização

**Pergunta:** com a rede convergindo bem, como a janela de max pooling, o dropout nas camadas densas e a
batch normalization (após cada convolução) afetam a generalização?

**Base:** a melhor rede entre o campeão do Bloco 2 e a checagem. `pool_size=0` remove o pooling; combinações que
colapsam o mapa espacial ou excedem o limite de parâmetros (ex.: sem pooling em uma rede que dependia dele) são
descartadas e listadas abaixo. Épocas e paciência aumentadas.

{grid_table("cnn_b3_regularizacao")}

**O que observar:** o `gap/accuracy` com dropout e batch norm, e o efeito de pooling mais agressivo (3×3).
"""),
    run_grid("cnn_b3_regularizacao"),
    *grid_results("cnn_b3_regularizacao",
                  'rep.heatmap("cnn_b3_regularizacao", row="cnn_dropout", col="cnn_batch_norm", facet="pool_size")',
                  show_discarded=True),
    code('rep.heatmap("cnn_b3_regularizacao", row="cnn_dropout", col="cnn_batch_norm", facet="pool_size", value="gap/accuracy_mean")'),
    analysis("Bloco 3", ["Pooling 2×2 vs 3×3 (ou sem pooling):", "Dropout e batch norm reduziram o gap?",
                         "Houve subajuste com regularização forte?", "Ganho sobre o Bloco 2 (validação):"]),
    backup(),
    *final_section("cnn", ["cnn_b0_referencia", "cnn_b1_topologia", "cnn_b2_otimizacao", "cnn_b2_checagem",
                           "cnn_b3_regularizacao"], "cnn_b3_regularizacao",
                   "Compare com as classes mais difíceis das MLPs: as convoluções resolveram as mesmas confusões?"),
]

metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
}

for name, cells in (("Kaggle_MLP.ipynb", mlp), ("Kaggle_CNN.ipynb", cnn)):
    nb = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}
    with open(os.path.join(ROOT, name), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote", name, len(cells), "cells")
