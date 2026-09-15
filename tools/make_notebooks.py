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
    return code("# Backup parcial: CSV/JSON/PNG/logs (os .pth ficam em outputs/)\n"
                "!cd {WORKDIR} && zip -qr outputs.zip outputs -x '*.pth' && ls -lh outputs.zip")


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
| {"Complementar A — Dropout longo" if kind == "MLP" else "Complementar B — Pooling"} | {"Os perdedores do Bloco 3 só precisavam de mais épocas?" if kind == "MLP" else "Qual janela de pooling é melhor, e como ela interage com a profundidade?"} | {"dropout × função de erro com 100 épocas" if kind == "MLP" else "blocos × janela de pooling"} |
| Bônus — Data augmentation | Quanto um pré-processamento (crop + flip) acrescenta à rede campeã? | {"augmentation × dropout" if kind == "MLP" else "augmentation × blocos × dropout"} |

O notebook executa **o estudo completo da {kind} em uma única execução** (Save & Run All): cada bloco herda o campeão do
anterior, e os complementares e o bônus partem do campeão mais recente.

**Protocolo de avaliação.**
- **Validação cruzada estratificada em 5 folds** sobre as 50.000 imagens de treino, com as mesmas partições em todos os experimentos (comparações pareadas).
- **Triagem:** cada configuração roda os folds 1–3; as 3 melhores **completam os folds 4–5** (sem refazer nada) e o campeão é a maior média nos 5 folds.
- **Escolha sempre pela validação.** O conjunto de teste (10.000 imagens) só é revelado na seção final, para os campeões. Usá-lo para escolher tornaria o número final otimista (vazamento de dados).
- Diferenças menores que o desvio padrão entre folds não devem ser tratadas como melhora real.

**Registro dos resultados.** Cada experimento grava em `outputs/{{exp_name}}/` (no Kaggle: `/kaggle/working/outputs`; no Colab: `/content/outputs`; no Jupyter local: a pasta do repositório):

| Arquivo | Conteúdo |
|---|---|
| `parametros.json` | hiperparâmetros exatos, arquitetura resolvida, versões e commit |
| `historico_treino.csv` | uma linha por **fold × época**: loss, acurácia, precision/recall/F1 (gerais e por classe) de treino e validação, gap |
| `historico_treino_agregado.csv` | média e desvio padrão entre folds, por época |
| `resultados.json` | resultado de cada fold e médias (validação na melhor época e teste) |
| `melhor_modelo.pth` | pesos com a menor `val/loss` (melhor fold); cada fold em `folds/fold_k/` |

Cada bloco grava em `outputs/_grids/{{bloco}}/`: `configs.csv`, `ranking_triagem.csv`, `ranking_final.csv`, `campeao.json` e `logs/`.
Tabelas e figuras dos relatórios ficam em `outputs/_relatorio/`. Tudo é gravado a cada época, antes do envio ao Weights & Biases (se houver chave; senão só o disco).

**Tempo estimado:** ~{time_1gpu} com 1 GPU T4 ou **~{time_2gpu} com 2× T4** (estimativa a partir de medições; varia ±30%).
Os experimentos são distribuídos entre as GPUs visíveis ({workers} por GPU, ajustável em `WORKERS_PER_GPU`). Sem GPU o código roda em CPU, mas o estudo deixa de ser praticável.

A célula seguinte descreve a **execução original no Kaggle** (como os resultados do relatório foram gerados) e como repetir o mesmo fluxo em Colab ou Jupyter local.
""")


def execucao_original(kind, time_1gpu, time_2gpu, workers):
    jobs = "1 experimento por GPU" if workers == 1 else f"{workers} experimentos por GPU"
    return md(f"""
## Execução original no Kaggle (e como reproduzir)

Os números do relatório desta {kind} saíram de um **Save Version → Save & Run All (Commit)** no Kaggle. Esse caminho **continua válido**: com os Settings abaixo, o notebook faz o mesmo estudo, nos mesmos blocos.

| Item | Como foi rodado |
|---|---|
| Acelerador | *Settings → Accelerator*: **GPU T4 ×2** |
| Internet | *Settings → Internet*: **On** (clone do GitHub, pip, W&B e fallback do CIFAR-10) |
| Dataset | *Add Input → Datasets*: **cifar10-python** (pasta `cifar-10-batches-py` ou arquivo `cifar-10-python.tar.gz`). Cópia conferida pelo MD5 oficial; sem ela o download de Toronto é muito lento |
| Código | clone de `https://github.com/diegoflyra/dfal-neural-networks.git` para `/tmp` (não vai para o Output) |
| `GITHUB_TOKEN` | *Add-ons → Secrets*, marcado como anexado. Obrigatório se o repositório for privado; o token nunca é impresso |
| `WANDB_API_KEY` | *Add-ons → Secrets*, anexado. **Opcional** — sem a chave o treino segue e tudo é gravado em disco; com ela cada fold vira uma run no W&B |
| Paralelismo | `WORKERS_PER_GPU = {workers}` ({jobs}) |
| Saída | `/kaggle/working/outputs/` e `outputs.zip` na aba *Output* da versão |
| Duração | **~{time_2gpu}** com 2× T4 (~{time_1gpu} com 1× T4; varia ±30%) |
| Retomada | anexar o Output da versão anterior como Input e preencher `RESUME_FROM`; folds já concluídos são pulados |

### No Colab, Jupyter local ou outra máquina

O **fluxo dos blocos é o mesmo** (referência → grids → campeão automático → teste só no final). Só o ambiente muda:

1. **GPU.** No Colab: *Runtime → Change runtime type → GPU* (em geral 1 T4, portanto mais lento). Ajuste `WORKERS_PER_GPU` se a memória apertar.
2. **Dataset.** Coloque `cifar-10-batches-py` ou `cifar-10-python.tar.gz` em `/content`, na pasta `data/` do repositório, ou em `CIFAR10_PATH`. Se não houver cópia, o notebook baixa do servidor oficial (MD5 conferido do mesmo jeito).
3. **Código.** Abra o notebook **dentro** de uma cópia do repositório (pastas `src/` e `grids/` no diretório atual) **ou** deixe clonar. Repo privado: defina `GITHUB_TOKEN` (Colab Secrets ou variável de ambiente). Repo público: clone sem chave.
4. **W&B.** Pode omitir. Sem `WANDB_API_KEY` o modo fica `disabled` e os CSV/JSON/PNG/`pth` saem em `outputs/` (`/content/outputs` no Colab; pasta do repositório no Jupyter local).
5. **Run All.** Igual ao Kaggle. Para retomar, aponte `RESUME_FROM` para a pasta `outputs` de uma execução anterior.

Não é preciso Kaggle, nem W&B, nem (com repo público ou cópia local) token do GitHub. O protocolo experimental não muda.
""")


def setup(workers):
    return [
        md("## 0. Preparação do ambiente"),
        code(f"""
import os
import shutil
import subprocess
import sys

REPO_URL = "{REPO_URL}"
CLONE_DIR = "{REPO_DIR}"


def _secret(name):
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        pass
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        pass
    return ""


def _is_repo(path):
    return os.path.isfile(os.path.join(path, "src", "grid_search.py")) and os.path.isdir(os.path.join(path, "grids"))


here = os.getcwd()
if _is_repo(here):
    REPO_DIR = here
    print(f"Código: cópia local em {{REPO_DIR}}")
else:
    REPO_DIR = CLONE_DIR
    token = _secret("GITHUB_TOKEN")
    clone_url = REPO_URL.replace("https://", f"https://x-access-token:{{token}}@", 1) if token else REPO_URL
    print("GitHub: usando GITHUB_TOKEN" if token else "GitHub: clone sem token")
    if os.path.isdir(REPO_DIR) and _is_repo(REPO_DIR):
        subprocess.run(["git", "-C", REPO_DIR, "pull", "-q"], check=False)
    else:
        if os.path.isdir(REPO_DIR):
            shutil.rmtree(REPO_DIR)
        cloned = subprocess.run(["git", "clone", "-q", clone_url, REPO_DIR])
        if cloned.returncode != 0 or not _is_repo(REPO_DIR):
            raise RuntimeError(
                "Falha ao clonar o repositório. Repositório privado exige GITHUB_TOKEN "
                "(Kaggle Secrets, Colab Secrets ou variável de ambiente), ou abra o notebook "
                "a partir de uma cópia local do projeto (pasta com src/ e grids/)."
            )
    print(f"Código: {{REPO_DIR}}")

os.chdir(REPO_DIR)
%cd {{REPO_DIR}}
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout.strip())
"""),
        code("""
import importlib.util
import subprocess
import sys

skip, pkgs = [], []
with open("requirements.txt", encoding="utf-8") as f:
    for line in f:
        pkg = line.strip()
        if not pkg or pkg.startswith("#"):
            continue
        name = pkg.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].split("[")[0].strip().lower()
        if name in {"torch", "torchvision"} and importlib.util.find_spec(name) is not None:
            skip.append(name)
            continue
        pkgs.append(pkg)
if skip:
    print("Já instalado, não reinstalar (preserva CUDA do ambiente):", ", ".join(skip))
if pkgs:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
"""),
        code(f"""
import glob
import hashlib
import os
import shutil
import sys
import tarfile

import pandas as pd
from torchvision.datasets import CIFAR10

WORKERS_PER_GPU = {workers}  # experimentos simultâneos por GPU
RESUME_FROM = ""  # pasta outputs de uma execução anterior, para retomar
CIFAR10_PATH = ""  # pasta cifar-10-batches-py ou arquivo .tar.gz; vazio = procurar / baixar

# CIFAR-10: usa uma cópia local se o MD5 for o oficial (os mesmos hashes que o torchvision
# usa para validar https://www.cs.toronto.edu/~kriz/cifar.html). Caso contrário baixa o original.
DATA_DIR = os.path.join(REPO_DIR, "data")
CIFAR_DIR = os.path.join(DATA_DIR, "cifar-10-batches-py")
OFFICIAL_MD5 = dict(CIFAR10.train_list + CIFAR10.test_list + [[CIFAR10.meta["filename"], CIFAR10.meta["md5"]]])
os.makedirs(DATA_DIR, exist_ok=True)


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cifar_is_official(folder):
    rows, ok = [], True
    for name, expected in OFFICIAL_MD5.items():
        path = os.path.join(folder, name)
        actual = md5(path) if os.path.isfile(path) else "arquivo ausente"
        ok &= actual == expected
        rows.append({{"arquivo": name, "md5 oficial": expected, "md5 da cópia": actual,
                     "status": "OK" if actual == expected else "DIFERENTE"}})
    display(pd.DataFrame(rows))
    return ok


def _unique(paths):
    seen, out = set(), []
    for path in paths:
        if path and os.path.exists(path) and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def find_cifar():
    folders, archives = [], []
    explicit = CIFAR10_PATH or os.environ.get("CIFAR10_PATH", "").strip()
    roots = [explicit, "/kaggle/input", "/content", DATA_DIR, os.getcwd()]
    for root in roots:
        if not root or not os.path.exists(root):
            continue
        if os.path.isdir(root) and os.path.isfile(os.path.join(root, "data_batch_1")):
            folders.append(root)
        if os.path.isfile(root) and root.endswith((".tar.gz", ".tgz")):
            archives.append(root)
        if os.path.isdir(root):
            # /content/drive no Colab é grande demais para glob recursivo.
            if os.path.abspath(root) == "/content":
                for base in (root, os.path.join(root, "data")):
                    folders.append(os.path.join(base, "cifar-10-batches-py"))
                    archives.append(os.path.join(base, "cifar-10-python.tar.gz"))
            else:
                folders.extend(glob.glob(os.path.join(root, "**", "cifar-10-batches-py"), recursive=True))
                archives.extend(glob.glob(os.path.join(root, "**", "cifar-10-python.tar.gz"), recursive=True))
    return _unique(folders), _unique(archives)


if not os.path.isdir(CIFAR_DIR):
    folders, archives = find_cifar()
    if folders:
        print(f"Cópia encontrada: {{folders[0]}}")
        if os.path.abspath(folders[0]) != os.path.abspath(CIFAR_DIR):
            shutil.copytree(folders[0], CIFAR_DIR)
    elif archives:
        print(f"Arquivo encontrado: {{archives[0]}} | md5 {{md5(archives[0])}} (oficial: {{CIFAR10.tgz_md5}})")
        with tarfile.open(archives[0]) as tar:
            tar.extractall(DATA_DIR)
    else:
        print("CIFAR-10 não encontrado localmente; baixando do servidor oficial (pode levar vários minutos).")
        CIFAR10(root=DATA_DIR, train=True, download=True)
        CIFAR10(root=DATA_DIR, train=False, download=True)

if os.path.isdir(CIFAR_DIR):
    if cifar_is_official(CIFAR_DIR):
        print("CIFAR-10 verificado: todos os arquivos são idênticos aos oficiais.")
    else:
        shutil.rmtree(CIFAR_DIR)
        print("CÓPIA REJEITADA: arquivos diferentes dos oficiais. Baixando do servidor original.")
        CIFAR10(root=DATA_DIR, train=True, download=True)
        CIFAR10(root=DATA_DIR, train=False, download=True)

if os.path.isdir("/kaggle/working"):
    WORKDIR = "/kaggle/working"
elif os.path.isdir("/content"):
    WORKDIR = "/content"
else:
    WORKDIR = REPO_DIR

os.environ["EXP_OUTPUT_DIR"] = os.path.join(WORKDIR, "outputs")
os.makedirs(os.environ["EXP_OUTPUT_DIR"], exist_ok=True)
print(f"Resultados em {{os.environ['EXP_OUTPUT_DIR']}}")
if RESUME_FROM:
    shutil.copytree(RESUME_FROM, os.environ["EXP_OUTPUT_DIR"], dirs_exist_ok=True)
    print(f"Resultados anteriores copiados de {{RESUME_FROM}}; o que já foi concluído será pulado.")

wandb_key = _secret("WANDB_API_KEY")
if wandb_key:
    os.environ["WANDB_API_KEY"] = wandb_key
    print("W&B: chave carregada.")
else:
    os.environ["WANDB_MODE"] = "disabled"
    print(f"W&B desativado (sem chave). Resultados continuam em {{os.environ['EXP_OUTPUT_DIR']}}.")

sys.path.insert(0, os.path.join(REPO_DIR, "src"))
import report_utils as rep
"""),
        code("""
# Verificações rápidas antes de gastar GPU (sem treino):
# GPUs, flags → arquitetura, carregamento em GPU/K-fold (baixa o CIFAR-10) e construção de todas as configurações dos grids
import shutil
if shutil.which("nvidia-smi"):
    !nvidia-smi -L
else:
    print("nvidia-smi não encontrado; o treino usa o device que o PyTorch enxergar (CPU se não houver GPU).")
!python tests/check_hyperparams.py
!python tests/check_data_loader.py
!python tests/check_grids.py
"""),
    ]


def complement_mlp():
    return [
        md(f"""
## Complementar A — Dropout com orçamento maior

**Pergunta:** configurações que atingiram a melhor época perto do limite de 50 épocas no Bloco 3 (tipicamente dropout alto)
mudariam de posição com mais tempo? A vantagem da entropia cruzada sobre MSE se mantém quando as duas convergem?

**Base:** o campeão do Bloco 3, com **100 épocas e paciência 10**. A configuração idêntica ao campeão é re-treinada no
novo orçamento para uma comparação justa.

{grid_table("mlp_b3b_dropout_longo")}

**O que observar:** `melhor_epoca_media` (agora com folga até 100) e a comparação **pareada por fold** contra a receita campeã.
"""),
        run_grid("mlp_b3b_dropout_longo"),
        code('rep.grid_ranking("mlp_b3b_dropout_longo", "final")'),
        code('rep.heatmap("mlp_b3b_dropout_longo", row="dropout", col="loss_fn")'),
        code('rep.heatmap("mlp_b3b_dropout_longo", row="dropout", col="loss_fn", value="melhor_epoca_media")'),
        code('rep.heatmap("mlp_b3b_dropout_longo", row="dropout", col="loss_fn", value="gap/accuracy_mean")'),
        code('rep.show_champion("mlp_b3b_dropout_longo")\nrep.plot_finalists("mlp_b3b_dropout_longo")'),
        md("#### Comparação pareada por fold\n\nReferência: a configuração idêntica à receita campeã do Bloco 3. "
           "Cada ponto é a diferença em um fold; “(4/5)” indica em quantos folds a configuração venceu a referência."),
        code('rep.paired_comparison(rep.base_config_name("mlp_b3b_dropout_longo"), "mlp_b3b_dropout_longo__*")'),
        analysis("Complementar A", ["Quantas épocas as configurações precisaram com o novo limite?",
                                    "Alguma configuração mudou de posição em relação ao Bloco 3?",
                                    "Entropia cruzada vs MSE fold a fold:"]),
        backup(),
    ]


def complement_cnn():
    return [
        md(f"""
## Complementar B — Janela de pooling

**Pergunta:** qual janela de max pooling extrai melhor as características espaciais, e como isso interage com a
profundidade? Janelas maiores reduzem a resolução mais rápido (menos parâmetros), mas descartam informação espacial mais cedo.
No Bloco 3, com a profundidade do campeão, várias janelas colapsavam o mapa espacial; aqui a profundidade também varia.

**Base:** o campeão do Bloco 3 (BatchNorm, dropout, otimizador). A combinação idêntica ao campeão é re-treinada e serve de
**checagem de reprodutibilidade**.

{grid_table("cnn_b3b_pooling")}

**O que observar:** o heatmap blocos × pooling, o tamanho final do mapa espacial (`cnn_shape_saida_por_bloco` em
`parametros.json`), o número de parâmetros e se a combinação igual ao campeão reproduz a validação do Bloco 3.
"""),
        run_grid("cnn_b3b_pooling"),
        code('rep.discarded_configs("cnn_b3b_pooling")'),
        code('rep.grid_ranking("cnn_b3b_pooling", "final")'),
        code('rep.heatmap("cnn_b3b_pooling", row="conv_blocks", col="pool_size")'),
        code('rep.heatmap("cnn_b3b_pooling", row="conv_blocks", col="pool_size", value="num_parameters")'),
        code('rep.heatmap("cnn_b3b_pooling", row="conv_blocks", col="pool_size", value="gap/accuracy_mean")'),
        code('rep.show_champion("cnn_b3b_pooling")\nrep.plot_finalists("cnn_b3b_pooling")'),
        md("#### Comparação pareada e reprodutibilidade\n\nReferência: a combinação idêntica ao campeão do Bloco 3. "
           "A tabela seguinte compara a validação dessa repetição com a execução original do Bloco 3 (mesmos folds e seed)."),
        code('rep.paired_comparison(rep.base_config_name("cnn_b3b_pooling"), "cnn_b3b_pooling__*")'),
        code("""
repeticao = rep.base_config_name("cnn_b3b_pooling")
pd.concat([rep.grid_ranking("cnn_b3_regularizacao", "final").head(1).assign(execucao="Bloco 3 (original)"),
           rep.grid_ranking("cnn_b3b_pooling", "final").query("exp_name == @repeticao").assign(execucao="Complementar B (repetição)")],
          ignore_index=True)[["execucao", "exp_name", "val/accuracy_mean", "val/accuracy_std", "melhor_epoca_media"]]
"""),
        analysis("Complementar B", ["Qual janela de pooling foi melhor? Depende da profundidade?",
                                    "Relação entre tamanho final do mapa, parâmetros e desempenho:",
                                    "A repetição reproduziu o campeão do Bloco 3? Qual o ruído entre execuções?"]),
        backup(),
    ]


def bonus(kind):
    block = f"{kind.lower()}_b4_augmentation"
    base_block = "mlp_b3b_dropout_longo" if kind == "MLP" else "cnn_b3b_pooling"
    hyp = ("Sem noção de vizinhança entre pixels, cada deslocamento é uma entrada nova para a MLP: o augmentation pode "
           "ensinar alguma invariância, mas o ganho deve ser pequeno e a convergência lenta."
           if kind == "MLP" else
           "Com mais diversidade de dados, (a) o overfitting deve cair, (b) redes mais profundas podem voltar a compensar e "
           "(c) a necessidade de dropout pode diminuir.")
    heat = ([code(f'rep.heatmap("{block}", row="dropout", col="augment")'),
             code(f'rep.heatmap("{block}", row="dropout", col="augment", value="gap/accuracy_mean")'),
             code(f'rep.heatmap("{block}", row="dropout", col="augment", value="melhor_epoca_media")')]
            if kind == "MLP" else
            [code(f'rep.heatmap("{block}", row="conv_blocks", col="cnn_dropout", facet="augment")'),
             code(f'rep.heatmap("{block}", row="conv_blocks", col="cnn_dropout", facet="augment", value="gap/accuracy_mean")'),
             code(f'rep.heatmap("{block}", row="conv_blocks", col="cnn_dropout", facet="augment", value="melhor_epoca_media")')])
    return [
        md(f"""
## Bônus — Data augmentation

Data augmentation é um **pré-processamento**, não um hiperparâmetro da rede: por isso fica fora da sequência principal
de blocos. Cria variações plausíveis de cada imagem de treino a cada época, aumentando a diversidade efetiva dos dados.

**Transformações** (apenas no treino, direto na GPU; validação e teste usam as imagens originais):
**RandomCrop 32 com padding 4** (desloca até 4 pixels) e **RandomHorizontalFlip** (espelha com probabilidade 0,5).

**Hipótese.** {hyp}

**Base:** o campeão de `{base_block}`, com mais épocas e paciência, pois augmentation retarda a convergência. A receita
campeã **sem** augmentation é re-treinada no mesmo bloco{" e sempre completa os 5 folds" if kind == "CNN" else ""}, como
referência pareada.

{grid_table(block)}

**O que observar:** o ganho pareado sobre a referência sem augmentation, a queda do `gap/accuracy` e se a
`melhor_epoca_media` ficou perto do limite de épocas (resultado possivelmente limitado pelo orçamento).
"""),
        run_grid(block),
        *([code(f'rep.grid_ranking("{block}", "triagem")')] if kind == "CNN" else []),
        *heat,
        code(f'rep.grid_ranking("{block}", "final")'),
        code(f'rep.show_champion("{block}")\nrep.plot_finalists("{block}")'),
        md("#### Comparação pareada por fold (validação)\n\nReferência: a receita campeã sem augmentation, treinada neste mesmo bloco."),
        code(f'rep.paired_comparison(rep.base_config_name("{block}"), "{block}__*")'),
        analysis("Bônus", ["Ganho do augmentation na validação (pareado):", "O gap treino-validação caiu?",
                           "Mudou a configuração ideal (profundidade/dropout)?" if kind == "CNN" else "Com augmentation, o dropout ainda é necessário?",
                           "Algum resultado ficou limitado pelo orçamento de épocas?"]),
        backup(),
    ]


def final_section(prefix, blocks, main_block, bonus_block, per_class_note, class_blocks, class_blocks_desc):
    return [
        md(f"""
## Resultado final — teste revelado

Até aqui todas as escolhas foram feitas pela validação cruzada. Agora o conjunto de teste é usado **uma única vez**
para os campeões de cada bloco e para as referências do Bloco 0. A tabela mostra média ± desvio do teste entre os
5 modelos (um por fold) de cada configuração. O campeão da sequência principal é o de `{main_block}`; o bônus mostra
quanto o augmentation acrescenta sobre ele.
"""),
        code(f"""
final = rep.final_report({blocks!r})
final
"""),
        md(f"Desempenho por classe no teste: referências do Bloco 0 × campeão principal × campeão com augmentation. {per_class_note}"),
        code(f"""
campeao_principal = rep.champion_name("{main_block}")
campeao_bonus = rep.champion_name("{bonus_block}")
comparar = list(final.loc[final["papel"] == "referência", "exp_name"]) + [campeao_principal, campeao_bonus]
rep.plot_per_class(comparar, metric="recall")
rep.plot_per_class(comparar, metric="precision")
pd.read_csv(os.path.join(os.environ["EXP_OUTPUT_DIR"], campeao_principal, "matriz_confusao_teste.csv"), index_col=0)
"""),
        md(f"""
#### Métricas gerais e por classe dos campeões

Para cada campeão: **acurácia, precision, recall e F1 gerais** (macro, no topo da figura) e **precision, recall e F1 de cada classe**
(tabela), no teste, em média ± desvio entre os 5 modelos (um por fold). Por classe, a acurácia é a fração das imagens daquela
classe classificadas corretamente, igual ao recall. {class_blocks_desc}
Figura e CSV: `outputs/_relatorio/metricas_por_classe_test_<experimento>.*`.
"""),
        *[code(f'rep.champion_class_metrics("{block}")') for block in class_blocks],
        code("!cd {WORKDIR} && zip -qr outputs.zip outputs -x '*.pth' && ls -lh outputs.zip"),
        md("## 📝 Conclusões\n\n"
           "- **Estrutura base que melhor absorveu o CIFAR-10 (Bloco 1) e por quê:** _…_\n"
           "- **Ganho de otimização (Bloco 2) e sensibilidade à taxa de aprendizagem:** _…_\n"
           "- **A checagem confirmou a escolha em blocos?** _…_\n"
           "- **Efeito da regularização/erro (Bloco 3):** _…_\n"
           "- **O que o experimento complementar corrigiu ou confirmou:** _…_\n"
           "- **Ganho total sobre a referência (teste):** _…_\n"
           "- **Bônus — quanto o augmentation acrescentou e por quê:** _…_\n"
           "- **Classes mais difíceis e hipótese:** _…_"),
    ]


# =============================================================================== MLP
mlp = [intro("MLP", "2 h", "1h–1h30", 2), execucao_original("MLP", "2 h", "1h–1h30", 2), *setup(2),
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
    *complement_mlp(),
    *bonus("MLP"),
    *final_section("mlp", ["mlp_b0_referencia", "mlp_b1_topologia", "mlp_b2_otimizacao", "mlp_b2_checagem",
                           "mlp_b3_regularizacao", "mlp_b3b_dropout_longo", "mlp_b4_augmentation"],
                   "mlp_b3b_dropout_longo", "mlp_b4_augmentation",
                   "Classes com baixo recall indicam confusões sistemáticas (veja a matriz de confusão).",
                   ["mlp_b3_regularizacao", "mlp_b4_augmentation"],
                   "Campeões do Bloco 3 (regularização) e do Bônus (augmentation)."),
]

# =============================================================================== CNN
cnn = [intro("CNN", "6 h", "3h–3h30", 1), execucao_original("CNN", "6 h", "3h–3h30", 1), *setup(1),
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
    *complement_cnn(),
    *bonus("CNN"),
    *final_section("cnn", ["cnn_b0_referencia", "cnn_b1_topologia", "cnn_b2_otimizacao", "cnn_b2_checagem",
                           "cnn_b3_regularizacao", "cnn_b3b_pooling", "cnn_b4_augmentation"],
                   "cnn_b3b_pooling", "cnn_b4_augmentation",
                   "Compare com as classes mais difíceis das MLPs: as convoluções resolveram as mesmas confusões?",
                   ["cnn_b3_regularizacao", "cnn_b3b_pooling", "cnn_b4_augmentation"],
                   "Campeões do Bloco 3 (regularização), do Complementar B (campeão principal da CNN) e do Bônus (augmentation)."),
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
