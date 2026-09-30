import os
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

warnings.filterwarnings("ignore")

MODEL_FILE = r"D:\333\Cohort2_模型训练\最终模型_锁定.pkl"
BG_FILE    = r"D:\333\Cohort2_数据处理\清洗后数据_Zscore.xlsx"
REF_FILE   = r"D:\333\Cohort2_数据处理\清洗后数据_原始浓度.xlsx"
RNG = 42

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "figure.dpi": 120,
})

print("Loading locked model ...")
lock = joblib.load(MODEL_FILE)
model      = lock["model"]
MODEL_NAME = lock["model_name"]
FEATURES   = lock["features"]
LOG2_MEAN  = np.asarray(lock["log2_mean"])
LOG2_STD   = np.asarray(lock["log2_std"])
CUTOFF     = float(lock["cutoff"])
print(f"Model: {MODEL_NAME} | Features: {FEATURES} | Cutoff: {CUTOFF:.3f}")

df_ref = pd.read_excel(REF_FILE)
FEAT_MEDIAN = df_ref[FEATURES].median().values

df_bg = pd.read_excel(BG_FILE)
X_bg = df_bg[FEATURES].values

import shap

def predict_prob(data):
    return model.predict_proba(np.asarray(data).reshape(-1, len(FEATURES)))[:, 1]

bg = shap.kmeans(X_bg, 10)
explainer = shap.KernelExplainer(predict_prob, bg)

def get_shap(z):
    sv = explainer.shap_values(np.asarray(z).reshape(1, -1), nsamples=200)
    return np.ravel(sv), float(explainer.expected_value)

def predict_from_conc(conc_values):
    conc = np.asarray(conc_values, dtype=float)
    if (conc <= 0).any():
        raise ValueError("Concentrations must be positive (log2 transform)")
    z = (np.log2(conc) - LOG2_MEAN) / LOG2_STD
    prob = float(predict_prob(z)[0])
    sv, base = get_shap(z)
    return prob, z, sv, base

class PredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Immunotherapy Response Predictor "
                   f"(Final model: {MODEL_NAME})")
        self.geometry("1080x560")
        self.resizable(False, False)
        self._build_ui()
        self.last_result = None

    def _build_ui(self):
        left = tk.Frame(self, padx=16, pady=14)
        left.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(left, text="Enter ELISA concentrations",
                 font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 8))

        self.entries = {}
        grid = tk.Frame(left)
        grid.pack(anchor="w")
        for i, f in enumerate(FEATURES):
            tk.Label(grid, text=f, font=("Arial", 10), width=10,
                     anchor="w").grid(row=i, column=0, pady=4, sticky="w")
            e = tk.Entry(grid, font=("Arial", 10), width=12)
            e.insert(0, f"{FEAT_MEDIAN[i]:.2f}")
            e.grid(row=i, column=1, pady=4, padx=(4, 0))
            self.entries[f] = e
            tk.Label(grid, text=f"(median {FEAT_MEDIAN[i]:.2f})",
                     font=("Arial", 8), fg="#888888"
                     ).grid(row=i, column=2, padx=(8, 0), sticky="w")

        btn_f = tk.Frame(left)
        btn_f.pack(anchor="w", pady=(12, 4))
        tk.Button(btn_f, text="Predict", font=("Arial", 11, "bold"),
                  bg="#3B6FB6", fg="white", width=10,
                  command=self.on_predict).pack(side=tk.LEFT)
        tk.Button(btn_f, text="Export PDF", font=("Arial", 9),
                  command=self.on_export).pack(side=tk.LEFT, padx=(10, 0))

        self.result_var = tk.StringVar(
            value="Enter concentrations, then click [Predict]")
        tk.Label(left, textvariable=self.result_var, font=("Arial", 10),
                 justify=tk.LEFT, wraplength=320).pack(anchor="w", pady=(10, 0))

        tk.Label(left, text=f"Decision cutoff (Youden): {CUTOFF:.3f}\n"
                            f"Probability >= cutoff -> NR (non-responder)\n"
                            f"Probability <  cutoff -> R  (responder)",
                 font=("Arial", 8), fg="#666666",
                 justify=tk.LEFT).pack(anchor="w", pady=(14, 0))

        right = tk.Frame(self, padx=8, pady=8)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig, self.ax = plt.subplots(figsize=(6.2, 4.6))
        self.ax.text(0.5, 0.5, "SHAP waterfall", ha="center", va="center",
                     color="#AAAAAA", fontsize=12)
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_predict(self):
        try:
            conc = [float(self.entries[f].get()) for f in FEATURES]
            prob, z, sv, base = predict_from_conc(conc)
        except ValueError as e:
            messagebox.showerror("Input error",
                                 f"All inputs must be positive numbers.\n({e})")
            return
        label = "NR (non-responder)" if prob >= CUTOFF else "R (responder)"
        self.result_var.set(
            f"Prediction: {label}\n"
            f"NR probability: {prob:.1%}\n"
            f"(Z-score: " +
            ", ".join(f"{f}={v:.2f}" for f, v in zip(FEATURES, z)) + ")")
        self.last_result = (prob, conc, z, sv, base)
        self.draw_waterfall(prob, conc, sv, base)

    def draw_waterfall(self, prob, conc, sv, base):
        self.ax.clear()
        self.ax.axis("on")
        order_i = np.argsort(np.abs(sv))[::-1]
        sv_o = sv[order_i]
        f_o = [FEATURES[t] for t in order_i]
        c_o = [conc[t] for t in order_i]
        cum = base + np.concatenate([[0], np.cumsum(sv_o)])
        nfeat = len(sv_o)
        for k in range(nfeat):
            c = "#D64B3B" if sv_o[k] > 0 else "#3B6FB6"
            self.ax.barh(nfeat - k, sv_o[k], left=cum[k], color=c,
                         edgecolor="black", lw=0.3, height=0.65, zorder=3)
            self.ax.text(cum[k] + sv_o[k] + (0.004 if sv_o[k] > 0 else -0.004),
                         nfeat - k, f"{sv_o[k]:+.3f}", va="center",
                         ha="left" if sv_o[k] > 0 else "right", fontsize=6)
        self.ax.set_yticks([nfeat - k for k in range(nfeat)])
        self.ax.set_yticklabels([f"{f_o[k]} = {c_o[k]:.2f}"
                                 for k in range(nfeat)], fontsize=7)
        self.ax.axvline(base, color="grey", lw=0.6, ls="--")
        self.ax.text(base, nfeat + 0.75, f"base={base:.2f}",
                     fontsize=6, ha="center", color="grey")
        self.ax.axvline(CUTOFF, color="black", lw=0.8, ls=":")
        self.ax.text(CUTOFF, 0.1, f"cutoff={CUTOFF:.2f}", fontsize=6,
                     ha="left", color="black")
        self.ax.set_title(f"P(NR) = {prob:.3f}  ->  "
                          f"{'NR' if prob >= CUTOFF else 'R'}", fontsize=9)
        self.ax.set_xlabel("Predicted probability of NR")
        self.ax.set_xlim(-0.05, 1.05)
        self.ax.set_ylim(0.2, nfeat + 1.3)
        for s in ["top", "right"]:
            self.ax.spines[s].set_visible(False)
        self.fig.tight_layout()
        self.canvas.draw()

    def on_export(self):
        if self.last_result is None:
            messagebox.showinfo("Note", "Please run a prediction first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF vector", "*.pdf")],
            initialfile="SHAP_waterfall_prediction.pdf")
        if path:
            self.fig.savefig(path, bbox_inches="tight", format="pdf")
            messagebox.showinfo("Done", f"Saved: {path}")

if __name__ == "__main__":
    app = PredictorApp()
    app.mainloop()
