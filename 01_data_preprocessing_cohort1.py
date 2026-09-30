import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

INPUT_FILE = r"C:\Users\Administrator\Desktop\444\Olink 测定数据.xlsx"
OUT_DIR    = r"C:\Users\Administrator\Desktop\444\Step1_数据处理"
os.makedirs(OUT_DIR, exist_ok=True)

GROUP_MAP   = {0: "R", 1: "NR"}
GROUP_COLOR = {"R": "#3B6FB6", "NR": "#D64B3B"}
VAR_CUTOFF  = 0.01
REMOVE_OUTLIERS = False
RNG         = 42

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
print("Step 1.1 读取数据")
raw = pd.read_excel(INPUT_FILE, sheet_name=0)
raw.columns = ["label"] + list(raw.columns[1:])
print(f"  原始维度: {raw.shape[0]} 样本 x {raw.shape[1]} 列")

qc_cols = [c for c in raw.columns if c in ["Plate ID", "QC Warning"]]
df = raw.drop(columns=qc_cols)
df["Group"] = df["label"].map(GROUP_MAP)

prot_cols = [c for c in df.columns if c not in ["label", "Group"]]
X = df[prot_cols].astype(float)
y = df["label"].astype(int)
print(f"  剔除质控列: {qc_cols}")
print(f"  蛋白数: {len(prot_cols)};  分组: R={(y == 0).sum()}, NR={(y == 1).sum()}")

print("=" * 50)
print("Step 1.2 质量检查")
n_missing = int(X.isna().sum().sum())
n_dup     = int(X.duplicated().sum())
prot_var  = X.var()
low_var   = prot_var[prot_var < VAR_CUTOFF].index.tolist()
print(f"  缺失值: {n_missing};  重复行: {n_dup};  低方差蛋白(<{VAR_CUTOFF}): {len(low_var)}")

borderline = prot_var[(prot_var >= VAR_CUTOFF) & (prot_var < 0.05)]
if len(borderline):
    print(f"  提示: {len(borderline)}个蛋白方差偏低(0.01-0.05): {borderline.index.tolist()}")

keep_prots = [p for p in prot_cols if p not in low_var]
X = X[keep_prots]
print(f"  过滤后保留蛋白数: {len(keep_prots)}")

qc_summary = pd.DataFrame({
    "检查项": ["总样本数", "响应组R (label=0)", "非响应组NR (label=1)",
              "原始蛋白数", "缺失值总数", "重复样本数",
              f"低方差剔除(方差<{VAR_CUTOFF})", "保留蛋白数"],
    "结果":   [len(df), int((y == 0).sum()), int((y == 1).sum()),
              len(prot_cols), n_missing, n_dup,
              len(low_var), len(keep_prots)],
})
save_xls(qc_summary, "表1_数据质量检查汇总", index=False)

desc = pd.DataFrame({
    "Protein": keep_prots,
    "Mean": X.mean().values,
    "SD": X.std().values,
    "Variance": X.var().values,
    "Min": X.min().values,
    "Max": X.max().values,
    "Skewness": X.skew().values,
    "Missing": X.isna().sum().values,
})
save_xls(desc, "表2_蛋白描述统计", index=False)

print("=" * 50)
print("Step 1.3 Z-score标准化")
scaler = StandardScaler()
Z = pd.DataFrame(scaler.fit_transform(X), columns=keep_prots)
print(f"  标准化后均值范围: [{Z.mean().min():.2e}, {Z.mean().max():.2e}]")

print("=" * 50)
print("Step 1.4 绘图")
order = df.sort_values(["Group"]).index
fig, ax = plt.subplots(figsize=(7.0, 2.4))
pos_cols = [GROUP_COLOR[df.loc[i, "Group"]] for i in order]
bp = ax.boxplot([X.loc[i].values for i in order], positions=range(len(order)),
                widths=0.6, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", lw=0.8),
                whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
for patch, c in zip(bp["boxes"], pos_cols):
    patch.set(facecolor=c, alpha=0.75, edgecolor="black", lw=0.5)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([f"S{i + 1}" for i in range(len(order))], rotation=90)
ax.set_ylabel("NPX (log2 scale)")
ax.set_xlabel("Sample")
handles = [mpl.patches.Patch(fc=GROUP_COLOR[g], ec="black", lw=0.5, label=g)
           for g in ["R", "NR"]]
ax.legend(handles=handles, frameon=False, loc="upper right", ncol=2)
style_ax(ax)
save_fig(fig, "图1_标准化前样本NPX分布")
save_xls(X.loc[order].assign(Group=df.loc[order, "Group"]).reset_index(names="SampleID"),
         "图1_原始数据", index=False)

fig, ax = plt.subplots(figsize=(7.0, 2.4))
bp = ax.boxplot([Z.loc[i].values for i in order], positions=range(len(order)),
                widths=0.6, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", lw=0.8),
                whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
for patch, c in zip(bp["boxes"], pos_cols):
    patch.set(facecolor=c, alpha=0.75, edgecolor="black", lw=0.5)
ax.axhline(0, color="grey", lw=0.6, ls="--")
ax.set_xticks(range(len(order)))
ax.set_xticklabels([f"S{i + 1}" for i in range(len(order))], rotation=90)
ax.set_ylabel("Z-score")
ax.set_xlabel("Sample")
ax.legend(handles=handles, frameon=False, loc="upper right", ncol=2)
style_ax(ax)
save_fig(fig, "图2_标准化后样本分布")
save_xls(Z.loc[order].assign(Group=df.loc[order, "Group"]).reset_index(names="SampleID"),
         "图2_原始数据", index=False)

print("=" * 50)
print("Step 1.5 离群样本检测(PCA马氏距离)")
pca_full = PCA(n_components=min(10, len(keep_prots)), random_state=RNG)
scores = pca_full.fit_transform(Z)
mean_s = scores.mean(axis=0)
cov_s  = np.cov(scores.T)
inv_s  = np.linalg.pinv(cov_s)
md = np.sqrt(((scores - mean_s) @ inv_s * (scores - mean_s)).sum(axis=1))
cutoff = np.sqrt(stats.chi2.ppf(0.975, df=scores.shape[1]))
outlier_flag = md > cutoff
print(f"  马氏距离阈值(chi2, P=0.975): {cutoff:.2f}")
print(f"  离群样本数: {outlier_flag.sum()} -> "
      f"{np.where(outlier_flag)[0].tolist() if outlier_flag.any() else '无'}")

fig, ax = plt.subplots(figsize=(3.5, 2.4))
idx = np.arange(1, len(md) + 1)
cols = ["#D64B3B" if f else "#4D4D4D" for f in outlier_flag]
ax.bar(idx, md, color=cols, width=0.7, edgecolor="none")
ax.axhline(cutoff, color="black", lw=0.7, ls="--")
ax.text(len(md) * 0.98, cutoff * 1.02, f"Threshold = {cutoff:.2f}",
        ha="right", va="bottom", fontsize=6)
ax.set_xlabel("Sample index")
ax.set_ylabel("Mahalanobis distance")
style_ax(ax)
save_fig(fig, "图3_离群样本检测")
save_xls(pd.DataFrame({"SampleIndex": idx, "Sample": df.index + 1,
                       "Group": df["Group"], "MahalanobisDist": md,
                       "Threshold": cutoff, "Outlier": outlier_flag}),
         "图3_原始数据", index=False)

pca2 = PCA(n_components=2, random_state=RNG)
pc2 = pca2.fit_transform(Z)
ev = pca2.explained_variance_ratio_ * 100
print(f"  PC1={ev[0]:.1f}%, PC2={ev[1]:.1f}%")

fig, ax = plt.subplots(figsize=(2.8, 2.6))
for g in ["R", "NR"]:
    m = (df["Group"] == g).values
    ax.scatter(pc2[m, 0], pc2[m, 1], s=16, c=GROUP_COLOR[g],
               edgecolor="black", lw=0.4, label=f"{g} (n={m.sum()})", zorder=3)

for g in ["R", "NR"]:
    m = (df["Group"] == g).values
    cov = np.cov(pc2[m, 0], pc2[m, 1])
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * np.sqrt(vals * stats.chi2.ppf(0.95, 2))
    ax.add_patch(mpl.patches.Ellipse(pc2[m].mean(axis=0), w, h, angle=ang,
                 fc=GROUP_COLOR[g], alpha=0.12, ec=GROUP_COLOR[g], lw=0.6, zorder=2))
ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
ax.legend(frameon=False, loc="best", handletextpad=0.1, borderaxespad=0.2)
style_ax(ax)
save_fig(fig, "图4_PCA分组聚类")
save_xls(pd.DataFrame({"Sample": df.index + 1, "Group": df["Group"],
                       "PC1": pc2[:, 0], "PC2": pc2[:, 1]}),
         "图4_原始数据", index=False)

col_colors = df["Group"].map(GROUP_COLOR)
g = sns.clustermap(Z.T, cmap="RdBu_r", center=0, vmin=-2.5, vmax=2.5,
                   col_cluster=False, row_cluster=True,
                   col_colors=col_colors,
                   figsize=(7.2, 6.5),
                   cbar_pos=(0.02, 0.83, 0.03, 0.12),
                   cbar_kws={"label": "Z-score", "ticks": [-2, 0, 2]},
                   xticklabels=False, yticklabels=True,
                   dendrogram_ratio=(0.12, 0.001))
g.ax_heatmap.set_xlabel("Sample")
g.ax_heatmap.set_ylabel("")
g.ax_heatmap.tick_params(axis="y", labelsize=4.5)
for lbl in g.ax_heatmap.get_yticklabels():
    lbl.set_fontfamily("Arial")
handles = [mpl.patches.Patch(fc=GROUP_COLOR[gr], label=gr) for gr in ["R", "NR"]]
g.ax_heatmap.legend(handles=handles, frameon=False, loc="upper left",
                    bbox_to_anchor=(1.02, 1.0), title="Group", title_fontsize=7)
g.savefig(os.path.join(OUT_DIR, "图5_蛋白表达聚类热图." + FIG_FORMAT),
          bbox_inches="tight", format=FIG_FORMAT)
plt.close(g.fig)
print(f"  [图] 图5_蛋白表达聚类热图.{FIG_FORMAT}")
hm_data = Z.T.copy()
hm_data.columns = [f"S{i + 1}_{df.loc[i, 'Group']}" for i in Z.index]
hm_data.insert(0, "Protein", hm_data.index)
save_xls(hm_data, "图5_原始数据", index=False)

if REMOVE_OUTLIERS and outlier_flag.any():
    print(f"  REMOVE_OUTLIERS=True, 剔除样本: {(np.where(outlier_flag)[0] + 1).tolist()}")
    keep_mask = ~outlier_flag
    df = df.loc[keep_mask].reset_index(drop=True)
    X  = X.loc[keep_mask].reset_index(drop=True)
    y  = y.loc[keep_mask].reset_index(drop=True)
    scaler = StandardScaler()
    Z = pd.DataFrame(scaler.fit_transform(X), columns=keep_prots)
    print(f"  剔除后样本数: {len(df)} (R={(y == 0).sum()}, NR={(y == 1).sum()})")

print("=" * 50)
print("Step 1.6 保存处理后数据")
clean = pd.concat([df[["label", "Group"]].reset_index(drop=True),
                   X.reset_index(drop=True)], axis=1)
save_xls(clean, "清洗后数据_原始NPX", index=False)

zout = pd.concat([df[["label", "Group"]].reset_index(drop=True),
                  Z.reset_index(drop=True)], axis=1)
save_xls(zout, "清洗后数据_Zscore", index=False)

print("=" * 50)
print(f"第一步完成! 所有结果已保存至: {OUT_DIR}")
print("输出: 5张PDF矢量图 + 5份图原始数据Excel + 3份数据表")
