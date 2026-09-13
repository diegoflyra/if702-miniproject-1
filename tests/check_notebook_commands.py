"""Valida os notebooks Kaggle sem executá-los.

- Toda célula de código compila como Python (comandos ! e % do IPython são ignorados).
- Comandos `!python src/run_experiment.py`: parse com o argparse real + construção do modelo + forward.
- Comandos `!python src/grid_search.py grids/X.json`: o arquivo existe e os blocos dos quais ele
  depende (base_from, grid_from_ranking) são executados ANTES no mesmo notebook.
- Chamadas `rep.<função>("bloco", ...)`: o bloco existe e os eixos citados (row/col/facet) existem no grid.
Uso: python tests/check_notebook_commands.py [notebook.ipynb ...]
"""

import ast
import json
import os
import re
import shlex
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch

import grid_search as gs
from models import build_model, count_parameters
from run_experiment import parse_args

RUN_PREFIX = "!python src/run_experiment.py"
GRID_PREFIX = "!python src/grid_search.py"


def notebook_variables(cells):
    """Variáveis simples definidas nas células (ex.: WORKERS_PER_GPU = 2), usadas como $VAR nos comandos."""
    variables = {}
    for cell in cells:
        if cell["cell_type"] == "code":
            for match in re.finditer(r"^([A-Z_][A-Z0-9_]*)\s*=\s*([0-9.eE+-]+)\s*(?:#.*)?$", "".join(cell["source"]), re.M):
                variables[match.group(1)] = match.group(2)
    return variables


def python_source(source):
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        lines.append(f"{indent}pass" if stripped.startswith(("!", "%")) else line)
    return "\n".join(lines)


def check_notebook(path, specs):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    variables = notebook_variables(nb["cells"])
    executed_blocks, runs, grids, rep_calls = [], 0, 0, 0
    print(f"\n{os.path.basename(path)}")
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        raw = "".join(cell["source"])
        try:
            compile(python_source(raw), f"{os.path.basename(path)}[cell {index}]", "exec")
        except SyntaxError as exc:
            raise AssertionError(f"célula {index} não compila: {exc}") from exc

        source = re.sub(r"\$([A-Z_][A-Z0-9_]*)", lambda m: variables.get(m.group(1), m.group(0)),
                        raw.replace("\\\n", " "))
        for line in source.splitlines():
            line = line.strip()
            if line.startswith(RUN_PREFIX) and "--help" not in line:
                args = parse_args(shlex.split(line[len(RUN_PREFIX):]))
                with torch.no_grad():
                    out = build_model(args).eval()(torch.zeros(2, 3, 32, 32))
                assert out.shape == (2, 10)
                print(f"  OK run  {args.exp_name:<34} params={count_parameters(build_model(args)):,}")
                runs += 1
            elif line.startswith(GRID_PREFIX):
                argv = shlex.split(line[len(GRID_PREFIX):])
                spec_path = os.path.join(ROOT, argv[0])
                assert os.path.isfile(spec_path), f"grid inexistente: {argv[0]}"
                spec = gs.load_spec(spec_path)
                deps = list(spec.get("base_from") or [])
                if spec.get("grid_from_ranking"):
                    deps.append(spec["grid_from_ranking"]["block"])
                missing = [d for d in deps if d not in executed_blocks]
                assert not missing, f"{spec['block']} executado antes de suas dependências {missing}"
                executed_blocks.append(spec["block"])
                print(f"  OK grid {spec['block']:<34} depende de {deps or '-'}")
                grids += 1

        for match in re.finditer(r"rep\.(\w+)\((.*)\)", raw):
            func, arg_src = match.group(1), match.group(2)
            try:
                call = ast.parse(f"f({arg_src})").body[0].value
            except SyntaxError:
                continue
            assert hasattr(__import__("report_utils"), func), f"rep.{func} não existe"
            rep_calls += 1
            if not call.args or not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, str):
                continue
            block = call.args[0].value
            if block in specs:
                assert block in executed_blocks, f"rep.{func}('{block}') antes de executar o bloco"
                axes = set(gs.load_spec(specs[block])["grid"])
                for kw in call.keywords:
                    if kw.arg in ("row", "col", "facet"):
                        values = ast.literal_eval(kw.value)
                        for axis in [values] if isinstance(values, str) else values:
                            assert axis in axes, f"rep.{func}('{block}'): eixo '{axis}' não existe no grid {sorted(axes)}"
    return runs, grids, rep_calls


def main(paths):
    specs = {gs.load_spec(os.path.join(ROOT, "grids", f))["block"]: os.path.join(ROOT, "grids", f)
             for f in os.listdir(os.path.join(ROOT, "grids")) if f.endswith(".json")}
    totals = [0, 0, 0]
    for path in paths:
        for i, n in enumerate(check_notebook(path, specs)):
            totals[i] += n
    assert totals[1] > 0, "nenhum comando de grid encontrado"
    print(f"\nNotebooks validados: {totals[0]} runs isolados, {totals[1]} blocos de grid, {totals[2]} chamadas de relatório")


if __name__ == "__main__":
    default = [os.path.join(ROOT, "Kaggle_MLP.ipynb"), os.path.join(ROOT, "Kaggle_CNN.ipynb")]
    main(sys.argv[1:] or default)
