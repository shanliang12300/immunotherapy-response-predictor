import os
import glob
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DATA_FILE  = r"C:\Users\Administrator\Desktop\444\Step1_数据处理\清洗后数据_Zscore.xlsx"
MODEL_DIR  = r"C:\Users\Administrator\Desktop\444\Step3_多模型比较"
OUT_DIR    = r"C:\Users\Administrator\Desktop\444\Step4_SHAP解释"
os.makedirs(OUT_DIR, exist_ok=True)
RNG = 42

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 300,
})
FIG_FORMAT = "pdf"
CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "bv", ["#3B6FB6", "#F5F5F5", "#D64B3B"])

def style_ax(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    return ax

def save_fig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name + "." + FIG_FORMAT),
                bbox_inches="tight", format=FIG_FORMAT)
    plt.close(fig)
    print(f"  [图] {name}.{FIG_FORMAT}")

def save_xls(df, name, **kw):
    df.to_excel(os.path.join(OUT_DIR, name + ".xlsx"), **kw)
    print(f"  [表] {name}.xlsx")

print("=" * 50)
print("Step 4.1 读取最佳模型与数据")
model_files = glob.glob(os.path.join(MODEL_DIR, "最佳模型_*.pkl"))
if not model_files:
    raise FileNotFoundError("未找到第三步保存的最佳模型pkl, 请先运行第三步")
model_path = model_files[0]
MODEL_NAME = os.path.basename(model_path).replace("最佳模型_", "").replace(".pkl", "")
FEATURES = joblib.load(os.path.join(MODEL_DIR, "建模特征.pkl"))
model = joblib.load(model_path)
print(f"  最佳模型: {MODEL_NAME}, 特征: {FEATURES}")

df = pd.read_excel(DATA_FILE)
X = df[FEATURES].values
y = df["label"].values
print(f"  样本数: {X.shape[0]} (R={(y == 0).sum()}, NR={(y == 1).sum()})")

print("=" * 50)
print("Step 4.2 计算SHAP值")
import shap

def predict_prob(data):
    return model.predict_proba(np.asarray(data))[:, 1]

if MODEL_NAME in ["RF", "XGBoost", "LightGBM"]:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    shap_values = sv[1] if isinstance(sv, list) else (
        sv[:, :, 1] if sv.ndim == 3 else sv)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.ravel(base_value)[-1]
elif MODEL_NAME == "LR":
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    base_value = explainer.expected_value
else:
    bg = shap.kmeans(X, 10)
    explainer = shap.KernelExplainer(predict_prob, bg)
    shap_values = explainer.shap_values(X, nsamples=200)
    base_value = float(explainer.expected_value)

shap_values = np.asarray(shap_values)
print(f"  SHAP矩阵: {shap_values.shape}, base_value={base_value:.4f}")
print("  (SHAP>0 表示推动预测为NR, SHAP<0 表示推动预测为R)")

sv_df = pd.DataFrame(shap_values, columns=[f"SHAP_{f}" for f in FEATURES])
sv_df.insert(0, "Group", df["Group"].values)
sv_df.insert(0, "Sample", df.index + 1)
save_xls(sv_df, "表1_SHAP值矩阵", index=False)

mean_abs = np.abs(shap_values).mean(axis=0)
imp_tbl = pd.DataFrame({"Protein": FEATURES, "MeanAbsSHAP": mean_abs})

direction = []
for j, f in enumerate(FEATURES):
    r = np.corrcoef(X[:, j], shap_values[:, j])[0, 1]
    direction.append("High expression -> NR" if r > 0 else "High expression -> R")
imp_tbl["Direction"] = direction
imp_tbl = imp_tbl.sort_values("MeanAbsSHAP", ascending=False)
save_xls(imp_tbl, "表2_SHAP重要性排序", index=False)
print("\nSHAP重要性排序:")
print(imp_tbl.to_string(index=False))

feat_order = imp_tbl["Protein"].tolist()
feat_idx = [FEATURES.index(f) for f in feat_order]

print("=" * 50)
print("Step 4.3 绘图")
fig, ax = plt.subplots(figsize=(2.8, 1.8))
plot_tbl = imp_tbl.iloc[::-1]
cols = ["#D64B3B" if "NR" in d else "#3B6FB6" for d in plot_tbl["Direction"]]
ax.barh(plot_tbl["Protein"], plot_tbl["MeanAbsSHAP"], color=cols,
        edgecolor="black", lw=0.4, height=0.65)
for i, (v, d) in enumerate(zip(plot_tbl["MeanAbsSHAP"],
                               plot_tbl["Direction"])):
    ax.text(v + 0.005, i, "NR" if "NR" in d else "R",
            va="center", fontsize=6)
ax.set_xlabel("Mean |SHAP value|")
ax.set_ylabel("")
handles = [mpl.patches.Patch(fc="#D64B3B", ec="black", lw=0.4,
                             label="High expr. -> NR"),
           mpl.patches.Patch(fc="#3B6FB6", ec="black", lw=0.4,
                             label="High expr. -> R")]
ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=5.5)
style_ax(ax)
save_fig(fig, "图1_SHAP全局重要性")

fig, ax = plt.subplots(figsize=(3.2, 2.2))
rng = np.random.default_rng(RNG)
bee_rows = []
for rank, (f, j) in enumerate(zip(feat_order, feat_idx)):
    yy = len(feat_order) - 1 - rank
    sv_j = shap_values[:, j]

    jit = rng.normal(0, 0.12, len(sv_j))
    v = X[:, j]
    sc = ax.scatter(sv_j, yy + jit, c=v, cmap=CMAP, s=7,
                    vmin=np.percentile(X, 2), vmax=np.percentile(X, 98),
                    edgecolor="none", alpha=0.9, zorder=3)
    for s_, v_ in zip(sv_j, v):
        bee_rows.append({"Protein": f, "SHAP": s_, "FeatureValue(Z)": v_})
ax.axvline(0, color="grey", lw=0.6, ls="--")
ax.set_yticks(range(len(feat_order)))
ax.set_yticklabels(feat_order[::-1])
ax.set_xlabel("SHAP value (impact on predicting NR)")
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Expression (Z-score)", fontsize=6)
cb.ax.tick_params(labelsize=5.5, width=0.5, length=2)
cb.outline.set_linewidth(0.5)
style_ax(ax)
save_fig(fig, "图2_SHAP蜂群图")
save_xls(pd.DataFrame(bee_rows), "图2_原始数据", index=False)

ncol = 3
nrow = int(np.ceil(len(FEATURES) / ncol))
fig, axes = plt.subplots(nrow, ncol,
                         figsize=(2.0 * ncol + 0.3, 1.9 * nrow + 0.3))
axes = np.atleast_1d(axes).ravel()
dep_rows = []
for k, (f, j) in enumerate(zip(feat_order, feat_idx)):
    ax = axes[k]
    v = X[:, j]
    s = shap_values[:, j]
    sc = ax.scatter(v, s, c=v, cmap=CMAP, s=9,
                    vmin=np.percentile(X, 2), vmax=np.percentile(X, 98),
                    edgecolor="black", lw=0.25, zorder=3)

    if len(np.unique(v)) > 5:
        zfit = np.polyfit(v, s, 2)
        xs = np.linspace(v.min(), v.max(), 100)
        ax.plot(xs, np.polyval(zfit, xs), color="black", lw=0.8, ls="--")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel(f"{f} (Z-score)")
    ax.set_ylabel(f"SHAP ({f})" if k % ncol == 0 else "")
    style_ax(ax)
    for v_, s_ in zip(v, s):
        dep_rows.append({"Protein": f, "FeatureValue(Z)": v_, "SHAP": s_})
for k in range(len(FEATURES), len(axes)):
    axes[k].axis("off")
fig.tight_layout()
save_fig(fig, "图3_SHAP依赖图")
save_xls(pd.DataFrame(dep_rows), "图3_原始数据", index=False)

pred_prob = predict_prob(X)
idx_nr = int(np.argmax(pred_prob))
idx_r  = int(np.argmin(pred_prob))
wf_rows = []

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
for ax, idx, tag in zip(axes, [idx_nr, idx_r],
                        [f"Typical NR (sample {idx_nr + 1})",
                         f"Typical R (sample {idx_r + 1})"]):
    sv_i = shap_values[idx]
    order_i = np.argsort(np.abs(sv_i))[::-1]
    sv_o = sv_i[order_i]
    f_o = [FEATURES[t] for t in order_i]
    cum = base_value + np.concatenate([[0], np.cumsum(sv_o)])
    for k in range(len(sv_o)):
        c = "#D64B3B" if sv_o[k] > 0 else "#3B6FB6"
        ax.barh(len(sv_o) - k, sv_o[k], left=cum[k], color=c,
                edgecolor="black", lw=0.3, height=0.65, zorder=3)
        ax.text(cum[k] + sv_o[k] + (0.004 if sv_o[k] > 0 else -0.004),
                len(sv_o) - k, f"{sv_o[k]:+.3f}",
                va="center", ha="left" if sv_o[k] > 0 else "right",
                fontsize=5.5)
        wf_rows.append({"Sample": idx + 1, "Type": tag, "Protein": f_o[k],
                        "FeatureValue(Z)": X[idx, order_i[k]],
                        "SHAP": sv_o[k]})
    ax.set_yticks([len(sv_o) - k for k in range(len(sv_o))])
    ax.set_yticklabels([f"{f_o[k]} = {X[idx, order_i[k]]:.2f}"
                        for k in range(len(sv_o))], fontsize=5.5)
    ax.axvline(base_value, color="grey", lw=0.6, ls="--")
    ax.text(base_value, len(sv_o) + 0.8, f"base={base_value:.2f}",
            fontsize=5.5, ha="center", color="grey")
    ax.axvline(cum[-1], color="black", lw=0.8)
    ax.set_title(f"{tag}\nP(NR) = {pred_prob[idx]:.3f}", fontsize=7)
    ax.set_xlabel("Predicted probability of NR")
    ax.set_xlim(-0.05, 1.05)
    style_ax(ax)
fig.tight_layout()
save_fig(fig, "图4_SHAP瀑布图")
save_xls(pd.DataFrame(wf_rows), "图4_原始数据", index=False)

import seaborn as sns
sv_plot = pd.DataFrame(shap_values[:, feat_idx], columns=feat_order)
row_colors = df["Group"].map({"R": "#3B6FB6", "NR": "#D64B3B"})
vmax = np.percentile(np.abs(shap_values), 98)
g = sns.clustermap(sv_plot, cmap=CMAP, center=0, vmin=-vmax, vmax=vmax,
                   row_cluster=True, col_cluster=True,
                   row_colors=row_colors,
                   figsize=(3.6, 4.2),
                   cbar_pos=(0.85, 0.15, 0.03, 0.15),
                   cbar_kws={"label": "SHAP value"},
                   xticklabels=True, yticklabels=False,
                   dendrogram_ratio=(0.15, 0.15))
g.ax_heatmap.set_xlabel("")
g.ax_heatmap.set_ylabel("Sample")
handles = [mpl.patches.Patch(fc=c, label=l) for l, c in
           [("R", "#3B6FB6"), ("NR", "#D64B3B")]]
g.ax_heatmap.legend(handles=handles, frameon=False, loc="upper left",
                    bbox_to_anchor=(1.15, 1.0), title="Group",
                    title_fontsize=7)
g.savefig(os.path.join(OUT_DIR, "图5_SHAP聚类热图." + FIG_FORMAT),
          bbox_inches="tight", format=FIG_FORMAT)
plt.close(g.fig)
print(f"  [图] 图5_SHAP聚类热图.{FIG_FORMAT}")
save_xls(sv_plot.assign(Group=df["Group"].values).reset_index(names="Sample"),
         "图5_原始数据", index=False)

report = imp_tbl.copy()
report["MeanSHAP_NRgroup"] = [shap_values[y == 1, FEATURES.index(f)].mean()
                              for f in report["Protein"]]
report["MeanSHAP_Rgroup"] = [shap_values[y == 0, FEATURES.index(f)].mean()
                             for f in report["Protein"]]
save_xls(report, "表3_SHAP汇总报告", index=False)

print("=" * 50)
print(f"第四步完成! 所有结果已保存至: {OUT_DIR}")
print(f"模型: {MODEL_NAME} | 特征重要性: "
      + " > ".join(imp_tbl["Protein"].tolist()))
