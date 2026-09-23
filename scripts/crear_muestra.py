"""Genera la muestra del proyecto desde el ZIP o CSV.GZ original de Kaggle.

Ejecutar desde la raíz del repositorio:
    python scripts/crear_muestra.py --archive data/lending-club.zip
    python scripts/crear_muestra.py --source data/accepted_2007_to_2018Q4.csv.gz
"""

import argparse
from pathlib import Path
import sys

# Permite usar el comando documentado sin instalar el proyecto como paquete.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import create_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--archive", type=Path, help="ZIP descargado de Kaggle")
    group.add_argument("--source", type=Path, help="CSV.GZ aceptados extraído del ZIP")
    parser.add_argument("--output", type=Path,
                        default=Path("data/lending_club_muestra.csv.gz"))
    parser.add_argument("--summary", type=Path,
                        default=Path("data/sample_summary.json"))
    parser.add_argument("--sample-size", type=int, default=115_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=25_000)
    args = parser.parse_args()
    create_sample(args.archive, args.source, args.output, args.summary,
                  args.sample_size, args.seed, args.chunksize)


if __name__ == "__main__":
    main()
