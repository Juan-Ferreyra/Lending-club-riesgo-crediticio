"""Genera y ejecuta notebooks/08_tecnicas_curso.ipynb a partir de outputs/research_v4 (no entrena)."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 08. Fase 4: técnicas del curso y familias adicionales\n\n"
       "Este notebook **lee** los resultados de `python -m fase4.experiment` (no entrena). "
       "Protocolo: `experiments/protocol_v4.json`. Préstamos a 36 meses; entrenamiento 2012-01 a 2014-09; "
       "selección/diagnóstico en Q4 2014. **No se abre 2015 ni 2016.**"),
    code("import json, pandas as pd, matplotlib.pyplot as plt\nfrom pathlib import Path\n"
         "ROOT = Path.cwd() if (Path.cwd()/'outputs').exists() else Path.cwd().parent\n"
         "OUT = ROOT/'outputs/research_v4'\n"
         "print(json.dumps(json.loads((OUT/'data_scope.json').read_text()), indent=1, ensure_ascii=False)[:900])"),
    md("## 1. Comparación de familias (misma validación, mismas variables)"),
    code("board = pd.read_csv(OUT/'leaderboard_q4.csv')\n"
         "board[['model','topic','train_n','AP','ROC_AUC','Brier','recall_at_top20']].sort_values('AP', ascending=False).round(4)"),
    code("paired = pd.DataFrame(json.loads((OUT/'paired_vs_reference.json').read_text())).T\n"
         "paired[['delta','ci95_low','ci95_high']].astype(float).round(4).sort_values('delta', ascending=False)"),
    md("Ningún candidato supera a la referencia LightGBM; el boosting queda arriba, las familias lineales y la MLP ~0,014 por debajo, "
       "y PCA empeora a la logística. Ver informe §8.4."),
    md("## 2. Reducción del modelo con L1 (6.3.1)"),
    code("l1 = pd.DataFrame(json.loads((OUT/'l1_selected_features.json').read_text())['l1_coefficients'])\n"
         "print(len(l1), 'coeficientes no nulos de', int(board.loc[board.model=='logistica_l1','total_coefficients'].iloc[0]))\n"
         "l1.head(20).round(4)"),
    md("## 3. Sesgo algorítmico: con y sin `addr_state` (6.10.1)\nTPR y FPR al revisar el 20% de mayor riesgo."),
    code("f = pd.read_csv(OUT/'fairness_segments_top20.csv')\n"
         "f.pivot_table(index=['dimension','segment'], columns='model', values=['TPR','FPR']).round(3)"),
    md("Quitar la geografía casi no cambia nada. La disparidad grande es por ingreso (FPR Q1 ≈ 0,37 vs Q5 ≈ 0,05), "
       "que no depende de `addr_state`."),
    md("## 4. Curva de aprendizaje (6.3.2)"),
    code("lc = pd.read_csv(OUT/'learning_curve.csv'); display(lc.round(4))\n"
         "ax = lc.plot(x='train_n', y=['train_AP','selection_AP'], marker='o', logx=True, figsize=(6,4))\n"
         "ax.set_ylabel('AP'); ax.set_title('LightGBM: entrenamiento vs. selección Q4 2014'); plt.show()"),
    md("La brecha entrenamiento-selección indica varianza, y la curva de selección crece con rendimientos decrecientes."),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
path = ROOT/'notebooks/08_tecnicas_curso.ipynb'
NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(ROOT/'notebooks')}}).execute()
nbf.write(nb, path)
print("escrito", path)
