import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

INPUT_FILE = r"D:\333\Cohort 2 and 3.xlsx"
SHEET_NAME = "Cohort 2"
OUT_DIR    = r"D:\333\Cohort2_数据处理"
os.makedirs(OUT_DIR, exist_ok=True)

GROUP_MAP   = {0: "R", 1: "NR"}
GROUP_COLOR = {"R": "#3B6FB6", "NR": "#D64B3B"}
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
print("C2-Step1 读取数据")
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
df.columns = ["label"] + list(df.columns[1:])
df["Group"] = df["label"].map(GROUP_MAP)
prot_cols = list(df.columns[1:-1])
X = df[prot_cols].astype(float)
y = df["label"].astype(int)
print(f"  {len(df)}样本 x {len(prot_cols)}蛋白: {prot_cols}")
print(f"  分组: R={(y == 0).sum()}, NR={(y == 1).sum()}")
print(f"  缺失值: {X.isna().sum().sum()}, 重复行: {X.duplicated().sum()}, "
      f"<=0值: {(X <= 0).sum().sum()}")

qc = pd.DataFrame({
    "检查项": ["总样本数", "响应组R", "非响应组NR", "蛋白数",
              "缺失值", "重复样本", "<=0值(无法log转换)"],
    "结果": [len(df), int((y == 0).sum()), int((y == 1).sum()), len(prot_cols),
            int(X.isna().sum().sum()), int(X.duplicated().sum()),
            int((X <= 0).sum().sum())],
})
save_xls(qc, "表1_数据质量检查", index=False)

desc = pd.DataFrame({
    "Protein": prot_cols,
    "Min": X.min().values, "Median": X.median().values,
    "Mean": X.mean().values, "SD": X.std().values,
    "Max": X.max().values, "Skewness": X.skew().values,
})
save_xls(desc, "表2_原始浓度描述统计", index=False)

print("=" * 50)
print("C2-Step2 log2转换 + Z-score标准化")
Xlog = np.log2(X)
Z = pd.DataFrame(StandardScaler().fit_transform(Xlog), columns=prot_cols)
print("  完成(log2后偏度: " +
      ", ".join(f"{p}={Xlog[p].skew():.2f}" for p in prot_cols) + ")")

print("=" * 50)
print("C2-Step3 绘图")
fig, ax = plt.subplots(figsize=(4.6, 2.6))
data_list = [X[p].values for p in prot_cols]
bp = ax.boxplot(data_list, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", lw=0.8),
                whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                widths=0.6)
for patch in bp["boxes"]:
    patch.set(facecolor="#3B6FB6", alpha=0.6, edgecolor="black", lw=0.5)
rng_j = np.random.default_rng(RNG)
for i, p in enumerate(prot_cols):
    ax.scatter(i + 1 + rng_j.normal(0, 0.08, len(X)), X[p],
               s=3, c="white", edgecolor="black", lw=0.25, zorder=3, alpha=0.6)
ax.set_yscale("log")
ax.set_xticklabels(prot_cols, rotation=30, ha="right")
ax.set_ylabel("Concentration (log scale)")
style_ax(ax)
save_fig(fig, "图1_原始浓度分布")
save_xls(X.assign(Group=df["Group"]).reset_index(names="SampleID"),
         "图1_原始数据", index=False)

order = df.sort_values(["Group"]).index
pos_cols = [GROUP_COLOR[df.loc[i, "Group"]] for i in order]
fig, ax = plt.subplots(figsize=(7.0, 2.4))
bp = ax.boxplot([Z.loc[i].values for i in order], positions=range(len(order)),
                widths=0.6, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", lw=0.8),
                whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
for patch, c in zip(bp["boxes"], pos_cols):
    patch.set(facecolor=c, alpha=0.75, edgecolor="black", lw=0.5)
ax.axhline(0, color="grey", lw=0.6, ls="--")
ax.set_xticks(range(len(order)))
ax.set_xticklabels([f"S{i + 1}" for i in range(len(order))], rotation=90,
                   fontsize=4)
ax.set_ylabel("Z-score (log2)")
ax.set_xlabel("Sample")
handles = [mpl.patches.Patch(fc=GROUP_COLOR[g], ec="black", lw=0.5, label=g)
           for g in ["R", "NR"]]
ax.legend(handles=handles, frameon=False, loc="upper right", ncol=2)
style_ax(ax)
save_fig(fig, "图2_标准化后样本分布")
save_xls(Z.loc[order].assign(Group=df.loc[order, "Group"]).reset_index(
    names="SampleID"), "图2_原始数据", index=False)

print("=" * 50)
print("C2-Step4 离群样本检测(PCA马氏距离)")
pca_full = PCA(n_components=min(8, len(prot_cols)), random_state=RNG)
scores = pca_full.fit_transform(Z)
mean_s = scores.mean(axis=0)
inv_s = np.linalg.pinv(np.cov(scores.T))
md = np.sqrt(((scores - mean_s) @ inv_s * (scores - mean_s)).sum(axis=1))
cutoff = np.sqrt(stats.chi2.ppf(0.975, df=scores.shape[1]))
outlier_flag = md > cutoff
print(f"  阈值={cutoff:.2f}, 离群样本: {outlier_flag.sum()}个 -> "
      f"{(np.where(outlier_flag)[0] + 1).tolist() if outlier_flag.any() else '无'}")

fig, ax = plt.subplots(figsize=(4.6, 2.4))
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
save_xls(pd.DataFrame({"SampleIndex": idx, "Group": df["Group"],
                       "MahalanobisDist": md, "Threshold": cutoff,
                       "Outlier": outlier_flag}), "图3_原始数据", index=False)

pca2 = PCA(n_components=2, random_state=RNG)
pc2 = pca2.fit_transform(Z)
ev = pca2.explained_variance_ratio_ * 100
print(f"  PC1={ev[0]:.1f}%, PC2={ev[1]:.1f}%")

fig, ax = plt.subplots(figsize=(2.8, 2.6))
for g in ["R", "NR"]:
    m = (df["Group"] == g).values
    ax.scatter(pc2[m, 0], pc2[m, 1], s=14, c=GROUP_COLOR[g],
               edgecolor="black", lw=0.35, label=f"{g} (n={m.sum()})",
               zorder=3, alpha=0.85)
for g in ["R", "NR"]:
    m = (df["Group"] == g).values
    cov = np.cov(pc2[m, 0], pc2[m, 1])
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * np.sqrt(vals * stats.chi2.ppf(0.95, 2))
    ax.add_patch(mpl.patches.Ellipse(pc2[m].mean(axis=0), w, h, angle=ang,
                 fc=GROUP_COLOR[g], alpha=0.12, ec=GROUP_COLOR[g],
                 lw=0.6, zorder=2))
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
                   figsize=(7.2, 3.2),
                   cbar_pos=(0.02, 0.80, 0.03, 0.15),
                   cbar_kws={"label": "Z-score", "ticks": [-2, 0, 2]},
                   xticklabels=False, yticklabels=True,
                   dendrogram_ratio=(0.18, 0.001))
g.ax_heatmap.set_xlabel("Sample")
g.ax_heatmap.set_ylabel("")
g.ax_heatmap.tick_params(axis="y", labelsize=6)
for lbl in g.ax_heatmap.get_yticklabels():
    lbl.set_fontfamily("Arial")
handles = [mpl.patches.Patch(fc=GROUP_COLOR[gr], label=gr) for gr in ["R", "NR"]]
g.ax_heatmap.legend(handles=handles, frameon=False, loc="upper left",
                    bbox_to_anchor=(1.02, 1.0), title="Group", title_fontsize=7)
g.savefig(os.path.join(OUT_DIR, "图5_蛋白表达聚类热图." + FIG_FORMAT),
          bbox_inches="tight", format=FIG_FORMAT)
plt.close(g.fig)
print(f"  [图] 图5_蛋白表达聚类热图.{FIG_FORMAT}")
hm = Z.T.copy()
hm.columns = [f"S{i + 1}_{df.loc[i, 'Group']}" for i in Z.index]
hm.insert(0, "Protein", hm.index)
save_xls(hm, "图5_原始数据", index=False)

print("=" * 50)
print("C2-Step5 组间差异初筛(Mann-Whitney, 基于log2浓度)")
rows = []
for p in prot_cols:
    rv = Xlog.loc[y == 0, p]
    nv = Xlog.loc[y == 1, p]
    u, pv = stats.mannwhitneyu(rv, nv, alternative="two-sided")
    rows.append({"Protein": p,
                 "Median_R": np.round(2 ** rv.median(), 3),
                 "Median_NR": np.round(2 ** nv.median(), 3),
                 "Log2FC(NR/R)": nv.median() - rv.median(),
                 "P_value": pv})
diff_tbl = pd.DataFrame(rows).sort_values("P_value")

pv = diff_tbl["P_value"].values
op = np.argsort(pv)
qs = pv[op] * len(pv) / np.arange(1, len(pv) + 1)
qs = np.minimum.accumulate(qs[::-1])[::-1]
qq = np.empty_like(qs)
qq[op] = qs
diff_tbl["FDR_q"] = qq
diff_tbl["Sig(p<0.05)"] = diff_tbl["P_value"] < 0.05
save_xls(diff_tbl.round(6), "表3_组间差异初筛", index=False)
print(diff_tbl.round(4).to_string(index=False))

print("=" * 50)
print("C2-Step6 保存处理后数据")
save_xls(pd.concat([df[["label", "Group"]].reset_index(drop=True),
                    X.reset_index(drop=True)], axis=1),
         "清洗后数据_原始浓度", index=False)
save_xls(pd.concat([df[["label", "Group"]].reset_index(drop=True),
                    Xlog.reset_index(drop=True)], axis=1),
         "清洗后数据_log2", index=False)
save_xls(pd.concat([df[["label", "Group"]].reset_index(drop=True),
                    Z.reset_index(drop=True)], axis=1),
         "清洗后数据_Zscore", index=False)

print("=" * 50)
print(f"Cohort 2 数据处理完成! 结果保存至: {OUT_DIR}")
