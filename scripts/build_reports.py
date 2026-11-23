"""Consolida métricas en outputs/metrics.json y prepara reports/figures/ para el informe.

Solo lee resultados ya guardados; no entrena ni puntúa modelos.
    python scripts/build_reports.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import write_json, utc_now  # noqa: E402

OUT = ROOT / "outputs"
FIG = ROOT / "reports" / "figures"

COPY = {
    "outputs/figures/comparacion_modelos.png": "08_comparacion_v1.png",
    "outputs/figures/curvas_pr_roc.png": "10_curvas_pr_roc_v1.png",
    "outputs/figures/calibracion_temporal.png": "10_calibracion_v1.png",
    "outputs/figures/estabilidad_temporal.png": "10_estabilidad_v1.png",
    "outputs/figures/shap_global.png": "11_shap_global.png",
    "outputs/figures/importancia_permutacion.png": "11_permutacion.png",
    "outputs/figures/errores_segmentos.png": "12_errores_segmentos.png",
    "outputs/figures/casos_locales.png": "12_casos_locales.png",
    "outputs/research_v2/figures/cobertura.png": "03_cobertura.png",
    "outputs/research_v2/figures/disponibilidad_variables.png": "04_disponibilidad_variables.png",
    "outputs/research_v2/figures/tasas_y_limites.png": "05_tasas_y_limites.png",
    "outputs/research_v2/figures/presupuesto_igual.png": "08_presupuesto_igual.png",
    "outputs/research_v3/figures/01_ablaciones.png": "08_ablaciones_v3.png",
    "outputs/research_v3/figures/02_busqueda.png": "09_busqueda_v3.png",
    "outputs/research_v3/figures/04_benchmark_2015.png": "10_benchmark_2015_v3.png",
    "outputs/research_v3/figures/06_interpretabilidad.png": "11_interpretabilidad_v3.png",
    "outputs/figures/explanation/protocol.png": "06_protocolo.png",
}


def timeline_figure(path):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    blocks = [
        ("V1", 0, "2007-01", "2014-06", "entrenamiento", "#4C78A8"),
        ("V1", 0, "2014-07", "2014-09", "calib.", "#F58518"),
        ("V1", 0, "2014-10", "2014-12", "umbral", "#E45756"),
        ("V3", 1, "2012-01", "2014-06", "entrenamiento cand.", "#4C78A8"),
        ("V3", 1, "2014-07", "2014-09", "Q3", "#72B7B2"),
        ("V4", 2, "2012-01", "2014-09", "entrenamiento", "#4C78A8"),
        ("V4", 2, "2014-10", "2014-12", "Q4", "#72B7B2"),
    ]
    for _, row, a, b, label, color in blocks:
        start, end = pd.Timestamp(a), pd.Timestamp(b) + pd.offsets.MonthEnd(0)
        ax.barh(row, (end - start).days, left=start, color=color, edgecolor="white", height=0.6)
        vis = max(start, pd.Timestamp("2011-06-01"))
        ax.text(vis + (end - vis) / 2, row, label, ha="center", va="center", fontsize=7, color="white")
    for row in (0, 1):
        for a, b, label, color in [("2015-01", "2015-12", "benchmark 2015", "#54A24B"),
                                   ("2016-01", "2016-12", "estrés 2016", "#9D755D")]:
            start, end = pd.Timestamp(a), pd.Timestamp(b) + pd.offsets.MonthEnd(0)
            ax.barh(row, (end - start).days, left=start, color=color, height=0.6)
            ax.text(start + (end - start) / 2, row, label, ha="center", va="center", fontsize=7, color="white")
    ax.set_yticks([0, 1, 2], ["V1 (todos los plazos)", "V2/V3 (36 meses)", "V4 (36 meses, solo desarrollo)"])
    ax.set_xlim(pd.Timestamp("2011-06-01"), pd.Timestamp("2017-01-31"))
    ax.invert_yaxis()
    ax.set_title("Partición temporal por fecha de emisión (V1 entrena desde 2007; se muestra desde mediados de 2011)\nQ3/Q4 = selección de modelos", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def family_figure(path):
    v1 = pd.read_csv(OUT / "final/metrics.csv")
    v1 = v1[(v1.cohort == 2015) & (v1["scope"] == "36_month")][["model", "AP"]].assign(fase="V1")
    v3 = pd.read_csv(OUT / "research_v3/benchmark_metrics.csv")
    v3 = v3[(v3.cohort == 2015)][["model", "AP"]].assign(fase="V2/V3")
    d = pd.concat([v1, v3]).sort_values("AP")
    d["label"] = d.fase + " · " + d.model
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#E45756" if "combined_control" in m else "#4C78A8" if f == "V2/V3" else "#9D9D9D"
              for m, f in zip(d.model, d.fase)]
    ax.barh(d.label, d.AP, color=colors)
    ax.axvline(0.1486, ls="--", c="gray")
    ax.text(0.1495, -0.6, "prevalencia 0,149", fontsize=8, color="gray")
    ax.set_xlim(0.14, 0.29)
    ax.set_xlabel("Average Precision, 2015 a 36 meses (258.856 préstamos)")
    ax.set_title("Todas las familias quedan en un rango estrecho")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def v4_figures():
    board = OUT / "research_v4/leaderboard_q4.csv"
    if not board.exists():
        return None
    b = pd.read_csv(board)
    b = b[b.status.eq("OK")].sort_values("AP")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(b.model, b.AP, color=["#E45756" if m == "lightgbm_referencia" else "#4C78A8" for m in b.model])
    for y, (ap, n) in enumerate(zip(b.AP, b.train_n)):
        ax.text(ap + 0.001, y, f"{ap:.4f}" + (" (60k filas)" if n == 60000 else ""), va="center", fontsize=8)
    ax.set_xlim(max(0.0, b.AP.min() - 0.03), b.AP.max() + 0.03)
    ax.set_xlabel("Average Precision, selección Q4 2014 (36 meses)")
    ax.set_title("Fase 4: técnicas del curso y familias adicionales")
    fig.tight_layout()
    fig.savefig(FIG / "08_fase4_familias.png", dpi=140)
    plt.close(fig)
    lc_path = OUT / "research_v4/learning_curve.csv"
    if lc_path.exists():
        lc = pd.read_csv(lc_path)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(lc.train_n, lc.train_AP, "o-", label="entrenamiento (30k filas vistas)")
        ax.plot(lc.train_n, lc.selection_AP, "o-", label="selección Q4 2014")
        ax.set_xscale("log")
        ax.set(xlabel="Filas de entrenamiento (escala log)", ylabel="AP", title="Curva de aprendizaje (LightGBM)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "08_curva_aprendizaje.png", dpi=140)
        plt.close(fig)
    return b


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for src, dst in COPY.items():
        if (ROOT / src).exists():
            shutil.copy2(ROOT / src, FIG / dst)
    timeline_figure(FIG / "06_particion_temporal.png")
    family_figure(FIG / "08_familias_2015_36m.png")
    v4 = v4_figures()

    v1 = json.loads((OUT / "final/metrics.json").read_text(encoding="utf-8"))
    v3 = pd.read_csv(OUT / "research_v3/benchmark_metrics.csv")
    pick = lambda c, m: v3[(v3.cohort == c) & (v3.model == m)][["AP", "ROC_AUC", "Brier", "recall_at_top20"]].iloc[0].to_dict()
    consolidated = {
        "generated_utc": utc_now(),
        "primary_metric": "Average Precision (sklearn.metrics.average_precision_score)",
        "v1_confirmatory_test": {
            "note": "Única prueba no reutilizada: selección congelada con hashes antes de abrir 2015/2016",
            "model": "Ensamble 60% CatBoost + 40% LightGBM, calibración sigmoid",
            "source_file": "outputs/final/metrics.json",
            "metrics": v1 if isinstance(v1, dict) else {"rows": v1},
        },
        "v3_recommended_model_2015_36m": {
            "note": "Benchmark retrospectivo reutilizado (2015 ya observado en V1 y V2)",
            "catboost_combined_control_v3": pick(2015, "catboost_combined_control"),
            "v2_frozen": pick(2015, "v2_frozen"),
            "v1_ensemble_same_cohort": pick(2015, "original_ensemble"),
            "paired_delta_AP_vs_v2_ci95": [0.003767, 0.006188],
        },
        "v4_development_q4_2014": (v4.drop(columns=["notes"], errors="ignore").to_dict(orient="records")
                                   if v4 is not None else "no ejecutado"),
        "sample_demo": (json.loads((OUT / "sample_demo/final_test.json").read_text(encoding="utf-8"))
                        if (OUT / "sample_demo/final_test.json").exists() else "no ejecutado"),
    }
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, float) and x != x:
            return None
        return x
    write_json(OUT / "metrics.json", clean(consolidated))
    print("outputs/metrics.json y reports/figures/ actualizados:", len(list(FIG.glob("*.png"))), "figuras")


if __name__ == "__main__":
    main()
