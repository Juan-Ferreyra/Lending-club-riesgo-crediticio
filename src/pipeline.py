"""Script principal del proyecto.

Dos modos:

- ``--mode sample`` (demo, minutos): usa solo ``data/lending_club_muestra.csv.gz`` incluida en el
  repositorio. Partición temporal: entrenamiento hasta 2013, selección en 2014 y una única
  evaluación final en 2015. Compara regresión logística (baseline), LightGBM y CatBoost con las mismas
  variables de originación. Sus cifras son una demostración ruidosa: no reproducen los resultados
  publicados, que usan el archivo completo.

- ``--mode full`` (horas): reproduce la evaluación V1 con el archivo original y después la Fase 4
  (técnicas del curso, solo desarrollo). V2 y V3 se reproducen con sus comandos propios (ver README).

Ejemplos:
    python -m src.pipeline --mode sample
    python -m src.pipeline --mode full --source data/accepted_2007_to_2018Q4.csv.gz
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def run_sample(output: Path, threads: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    from src.evaluate import metrics
    from src.models import LoanModel
    from src.utils import write_json, utc_now

    # La demo sobrescribe su salida en cada ejecución. El modo full no pasa por aquí:
    # scripts/run_experiments.py rechaza un directorio que ya tiene una evaluación externa.
    output.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    t0 = time.time()
    data = pd.read_csv(ROOT / "data/lending_club_muestra.csv.gz", dtype=str, low_memory=False)
    data["issue_date"] = pd.to_datetime(data.issue_d, format="%b-%Y")
    data["target"] = data.loan_status.eq("Charged Off").astype(int)
    year = data.issue_date.dt.year
    train, select, test = data[year <= 2013], data[year == 2014], data[year == 2015]
    print(f"muestra: train={len(train)} (<=2013), seleccion={len(select)} (2014), prueba={len(test)} (2015)")

    families = {
        "logistica_baseline": dict(family="logistic", extended=False),
        "lightgbm": dict(family="lightgbm", params=dict(n_estimators=500, learning_rate=0.03)),
        "catboost": dict(family="catboost", params=dict(iterations=600, eval_metric="Logloss")),
    }
    rows, fitted = [], {}
    for name, cfg in families.items():
        start = time.time()
        model = LoanModel(cfg["family"], cfg.get("params"), extended=cfg.get("extended", True),
                          seed=SEED, threads=threads).fit(train, train.target.to_numpy())
        p = model.predict_proba(select)[:, 1]
        m = metrics(select.target.to_numpy(), p)
        m.pop("confusion_matrix")
        rows.append({"model": name, "stage": "seleccion_2014", "seconds": round(time.time() - start, 1), **m})
        fitted[name] = model
        print(f"  {name}: AP 2014 = {m['AP']:.4f}  ROC-AUC = {m['ROC_AUC']:.4f}")
    board = pd.DataFrame(rows).sort_values("AP", ascending=False)
    board.to_csv(output / "selection_2014.csv", index=False)
    chosen = board.model.iloc[0]
    write_json(output / "selection_lock.json", {"chosen": chosen, "frozen_utc": utc_now(),
                                                "rule": "mayor AP en 2014; 2015 no se usó para decidir"})

    # Evaluación final única en 2015: el elegido y el baseline, mismas filas.
    final = {}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name in dict.fromkeys([chosen, "logistica_baseline"]):
        p = fitted[name].predict_proba(test)[:, 1]
        m = metrics(test.target.to_numpy(), p)
        final[name] = m
        prec, rec, _ = precision_recall_curve(test.target, p)
        ax.plot(rec, prec, label=f"{name} (AP={m['AP']:.3f})")
    ax.axhline(test.target.mean(), ls="--", c="gray", label=f"prevalencia={test.target.mean():.3f}")
    ax.set(xlabel="Recall", ylabel="Precisión", title="Demo con la muestra: prueba 2015")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "pr_curve_2015.png", dpi=130)
    write_json(output / "final_test.json", {"chosen": chosen, "test": "2015 (muestra)",
                                            "metrics": final, "seconds": round(time.time() - t0, 1),
                                            "warning": "Demo con la muestra; no comparable con los resultados del archivo completo"})
    print(f"elegido: {chosen}; AP 2015 = {final[chosen]['AP']:.4f} "
          f"(baseline {final['logistica_baseline']['AP']:.4f}). Resultados en {output}")


def run_full(source: str, output: str):
    def run(*args):
        print(">>", " ".join(map(str, args)), flush=True)
        subprocess.run([sys.executable, *map(str, args)], check=True, cwd=ROOT)
    run("scripts/run_experiments.py", "--source", source, "--output", output)
    run("-m", "fase4.experiment", "--source", source, "--output", f"{output}_v4")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sample", "full"], default="sample")
    ap.add_argument("--source", default="data/accepted_2007_to_2018Q4.csv.gz")
    ap.add_argument("--output", default=None)
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    if args.mode == "sample":
        run_sample(ROOT / (args.output or "outputs/sample_demo"), args.threads)
    else:
        run_full(args.source, args.output or "outputs/reproduction")
