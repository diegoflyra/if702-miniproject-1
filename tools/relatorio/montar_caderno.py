"""Monta relatorio/caderno-cifar10.html a partir do modelo e dos dados extraídos.

Uso (na raiz do repositório):
    python tools/relatorio/extrair_dados.py
    python tools/relatorio/montar_caderno.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def payload(path):
    return json.dumps(json.load(open(path)), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")


def main():
    template = open(os.path.join(HERE, "caderno_template.html"), encoding="utf-8").read()
    html = (template.replace("__DATA__", payload(os.path.join("relatorio", "dados", "doc_data.json")))
                    .replace("__CURVES__", payload(os.path.join("relatorio", "dados", "curves.json"))))
    out = os.path.join("relatorio", "caderno-cifar10.html")
    open(out, "w", encoding="utf-8").write(html)
    print("caderno:", out, len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
