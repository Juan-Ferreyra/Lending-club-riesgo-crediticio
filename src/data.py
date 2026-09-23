"""Crea una muestra aleatoria reproducible de préstamos con resultado definitivo.

Uso:
    python scripts/crear_muestra.py --archive data/lending-club.zip
    python scripts/crear_muestra.py --source data/accepted_2007_to_2018Q4.csv.gz

La selección usa una prioridad aleatoria independiente por fila. Mantener las
115 000 prioridades menores equivale a un muestreo uniforme sin reemplazo de
los préstamos elegibles, sin cargar el archivo completo en memoria.
"""

from __future__ import annotations

from contextlib import contextmanager
import gzip
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


TARGET_STATUSES = ("Fully Paid", "Charged Off")


@contextmanager
def open_source(archive: Path | None, source: Path | None):
    if (archive is None) == (source is None):
        raise ValueError("Indique exactamente una opción: --archive o --source.")
    if archive is not None:
        with ZipFile(archive) as bundle:
            matches = [name for name in bundle.namelist()
                       if Path(name).name == "accepted_2007_to_2018Q4.csv.gz"]
            if len(matches) != 1:
                raise FileNotFoundError(
                    "El ZIP debe contener accepted_2007_to_2018Q4.csv.gz."
                )
            with bundle.open(matches[0]) as compressed:
                with gzip.GzipFile(fileobj=compressed) as stream:
                    yield stream
    else:
        if source is None or not source.is_file():
            raise FileNotFoundError(source)
        with gzip.open(source, "rb") as stream:
            yield stream


def create_sample(
    archive: Path | None,
    source: Path | None,
    output: Path,
    summary_path: Path,
    sample_size: int = 115_000,
    seed: int = 42,
    chunksize: int = 25_000,
) -> dict:
    if sample_size < 1 or chunksize < 1:
        raise ValueError("sample_size y chunksize deben ser positivos.")

    rng = np.random.default_rng(seed)
    sample = None
    source_rows = 0
    eligible_rows = 0
    source_status_counts = {status: 0 for status in TARGET_STATUSES}
    columns = None

    with open_source(archive, source) as stream:
        chunks = pd.read_csv(stream, chunksize=chunksize, dtype=str,
                             low_memory=False)
        for chunk in chunks:
            if "loan_status" not in chunk:
                raise KeyError("Falta la columna loan_status en el archivo fuente.")
            if columns is None:
                columns = list(chunk.columns)
            # Se generan claves para todas las filas: el resultado no depende
            # de la distribución de estados dentro de cada bloque.
            keys = rng.random(len(chunk))
            source_rows += len(chunk)
            status_counts = chunk["loan_status"].value_counts()
            for status in TARGET_STATUSES:
                source_status_counts[status] += int(status_counts.get(status, 0))
            eligible_mask = chunk["loan_status"].isin(TARGET_STATUSES).to_numpy()
            eligible = chunk.loc[eligible_mask].copy()
            eligible_rows += len(eligible)
            if eligible.empty:
                continue
            eligible["__sample_key"] = keys[eligible_mask]
            if sample is None:
                sample = eligible.nsmallest(sample_size, "__sample_key")
            else:
                sample = pd.concat((sample, eligible), ignore_index=True).nsmallest(
                    sample_size, "__sample_key"
                )
            print(f"\rFilas fuente: {source_rows:,} | elegibles: {eligible_rows:,}",
                  end="", flush=True)

    print()
    if sample is None or len(sample) < sample_size:
        raise ValueError(
            f"Solo se encontraron {eligible_rows:,} filas elegibles; "
            f"se solicitaron {sample_size:,}."
        )

    # La selección y el orden exportado quedan fijados por la prioridad aleatoria.
    sample = sample.drop(columns="__sample_key")
    sample = sample.loc[:, columns]
    output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output, index=False, compression="gzip")

    years = pd.to_datetime(sample["issue_d"], format="%b-%Y", errors="coerce").dt.year
    summary = {
        "source": "https://www.kaggle.com/datasets/wordsforthewise/lending-club/versions/3",
        "dataset_version": 3,
        "source_file": "accepted_2007_to_2018Q4.csv.gz",
        "license": "CC0: Public Domain (según la ficha de Kaggle)",
        "selection": "Muestreo aleatorio uniforme sin reemplazo entre Fully Paid y Charged Off",
        "seed": seed,
        "sample_size": len(sample),
        "source_rows": source_rows,
        "eligible_rows": eligible_rows,
        "source_status_counts": source_status_counts,
        "columns": len(columns),
        "loan_status_counts": {
            str(status): int(count)
            for status, count in sample["loan_status"].value_counts().items()
        },
        "issue_year_counts": {
            str(int(year)): int(count)
            for year, count in years.value_counts().sort_index().items()
        },
        "term_counts": {
            str(term): int(count)
            for term, count in sample["term"].value_counts().items()
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"Muestra: {output} ({len(sample):,} filas, {len(columns)} columnas)")
    print(f"Resumen: {summary_path}")
    return summary
