import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from scipy import stats

warnings.filterwarnings("ignore")

INPUT_FILE = r"C:\Users\Administrator\Desktop\444\Step1_数据处理\清洗后数据_Zscore.xlsx"
OUT_DIR    = r"C:\Users\Administrator\Desktop\444\Step2_特征筛选"
DA_DIR     = os.path.join(OUT_DIR, "差异分析")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DA_DIR, exist_ok=True)

GROUP_COLOR   = {"R": "#3B6FB6", "NR": "#D64B3B"}
N_CV          = 5
N_BOOT        = 200
LASSO_C       = 0.5
FREQ_CUTOFF   = 0.5
IMP_SD        = 0.5
RFE_MAX_N     = 20
P_CUTOFF      = 0.05
TOP_N         = 20
CONSENSUS_MIN = 2
RNG           = 42

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

def save_fig(fig, name, out_dir=None):
    d = out_dir or OUT_DIR
    fig.savefig(os.path.join(d, name + "." + FIG_FORMAT),
                bbox_inches="tight", format=FIG_FORMAT)
    plt.close(fig)
    print(f"  [图] {name}.{FIG_FORMAT}")

def save_xls(df, name, out_dir=None, **kw):
    d = out_dir or OUT_DIR
    df.to_excel(os.path.join(d, name + ".xlsx"), **kw)
    print(f"  [表] {name}.xlsx")

def importance_barplot(imp_series, cutoff, selected_set, xlabel, fname, tname):
    tbl = pd.DataFrame({"Protein": imp_series.index,
                        "Importance": imp_series.values,
                        "Selected": imp_series.index.isin(selected_set)})
    save_xls(tbl, tname, index=False)
    top = tbl.head(TOP_N).iloc[::-1]
    fig, ax = plt.subplots(figsize=(2.8, 0.16 * TOP_N + 0.8))
    cols = ["#D64B3B" if s else "#B0B0B0" for s in top["Selected"]]
    ax.barh(top["Protein"], top["Importance"], color=cols,
            edgecolor="black", lw=0.4, height=0.7)
    ax.axvline(cutoff, color="black", lw=0.6, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    style_ax(ax)
    save_fig(fig, fname)
    return tbl

print("=" * 50)
print("Step 2.1 读取第一步清洗后数据(Z-score)")
df = pd.read_excel(INPUT_FILE)
prot_cols = [c for c in df.columns if c not in ["label", "Group"]]
X = df[prot_cols].values
y = df["label"].values
print(f"  样本数: {X.shape[0]}, 蛋白数: {X.shape[1]}")

cv = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=RNG)
selected = {}

print("=" * 50)
print("[A] 差异分析: Mann-Whitney U检验 + BH-FDR校正")
pvals, diffs = [], []
for j, p in enumerate(prot_cols):
    u, pv = stats.mannwhitneyu(X[y == 0, j], X[y == 1, j],
                               alternative="two-sided")
    pvals.append(pv)
    diffs.append(np.median(X[y == 1, j]) - np.median(X[y == 0, j]))
pvals = np.array(pvals)
order_p = np.argsort(pvals)
q_sorted = pvals[order_p] * len(pvals) / np.arange(1, len(pvals) + 1)
q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
qvals = np.empty_like(q_sorted)
qvals[order_p] = q_sorted
da_sel = [p for p, pv in zip(prot_cols, pvals) if pv < P_CUTOFF]
print(f"  名义p<{P_CUTOFF}: {len(da_sel)}个; FDR<0.05: {(qvals < 0.05).sum()}个; "
      f"FDR<0.10: {(qvals < 0.10).sum()}个")

da_tbl = pd.DataFrame({"Protein": prot_cols, "MedianDiff(NR-R)": diffs,
                       "P_value": pvals, "FDR_q": qvals,
                       "Sig(p<0.05)": pvals < P_CUTOFF,
                       "FDR<0.10": qvals < 0.10}).sort_values("P_value")
save_xls(da_tbl, "表A_Mann-Whitney差异分析", out_dir=DA_DIR, index=False)

fig, ax = plt.subplots(figsize=(3.4, 2.6))
neglog = -np.log10(pvals)
sig = pvals < P_CUTOFF
fdr10 = qvals < 0.10
ax.scatter(np.array(diffs)[~sig], neglog[~sig], s=8, c="#B0B0B0",
           edgecolor="none", alpha=0.7, label="NS")
ax.scatter(np.array(diffs)[sig & ~fdr10], neglog[sig & ~fdr10], s=10,
           c="#3B6FB6", edgecolor="black", lw=0.3, label=f"p < {P_CUTOFF}")
ax.scatter(np.array(diffs)[fdr10], neglog[fdr10], s=12,
           c="#D64B3B", edgecolor="black", lw=0.3, label="FDR < 0.10")
ax.axhline(-np.log10(P_CUTOFF), color="black", lw=0.6, ls="--")
for p, d, nl, f in zip(prot_cols, diffs, neglog, fdr10):
    if f:
        ax.annotate(p, (d, nl), fontsize=4.5,
                    xytext=(2, 2), textcoords="offset points")
ax.set_xlabel("Median difference (NR - R), Z-score")
ax.set_ylabel("-log10(p-value)")
ax.legend(frameon=False, loc="upper right", handletextpad=0.1)
style_ax(ax)
save_fig(fig, "图A_差异分析火山图", out_dir=DA_DIR)
save_xls(da_tbl[["Protein", "MedianDiff(NR-R)", "P_value", "FDR_q"]],
         "图A_原始数据", out_dir=DA_DIR, index=False)

print("=" * 50)
print("[B-1] LASSO稳定性选择")
Cs = np.logspace(-3, 2, 40)
lr_cv = LogisticRegressionCV(Cs=Cs, cv=cv, penalty="l1", solver="saga",
                             scoring="roc_auc", max_iter=10000,
                             random_state=RNG, n_jobs=-1)
lr_cv.fit(X, y)
mean_auc = lr_cv.scores_[1].mean(axis=0)
se_auc = lr_cv.scores_[1].std(axis=0) / np.sqrt(N_CV)
print(f"  最优C={Cs[mean_auc.argmax()]:.4f}, CV-AUC={mean_auc.max():.3f}")

fig, ax = plt.subplots(figsize=(3.2, 2.4))
ax.semilogx(Cs, mean_auc, "o-", color="#3B6FB6", lw=0.9, ms=3,
            mec="black", mew=0.4)
ax.fill_between(Cs, mean_auc - se_auc, mean_auc + se_auc,
                color="#3B6FB6", alpha=0.15, lw=0)
ax.axvline(LASSO_C, color="#D64B3B", lw=0.7, ls="--",
           label=f"C used = {LASSO_C}")
ax.set_xlabel("Regularization parameter C (log scale)")
ax.set_ylabel(f"Cross-validated AUC ({N_CV}-fold)")
ax.legend(frameon=False, loc="lower right")
style_ax(ax)
save_fig(fig, "图1_LASSO交叉验证曲线")
save_xls(pd.DataFrame({"C": Cs, "MeanAUC": mean_auc, "SE": se_auc}),
         "图1_原始数据", index=False)

rng = np.random.default_rng(RNG)
counts = pd.Series(0, index=prot_cols, dtype=float)
n_done = 0
for b in range(N_BOOT):
    idx = rng.integers(0, len(y), len(y))
    if len(np.unique(y[idx])) < 2:
        continue
    m = LogisticRegression(C=LASSO_C, penalty="l1", solver="saga",
                           max_iter=5000, random_state=b).fit(X[idx], y[idx])
    counts += (m.coef_[0] != 0).astype(int)
    n_done += 1
freq = (counts / n_done).sort_values(ascending=False)
lasso_sel = freq[freq >= FREQ_CUTOFF].index.tolist()
selected["LASSO"] = lasso_sel
print(f"  频率>={FREQ_CUTOFF} 选中{len(lasso_sel)}个")

save_xls(pd.DataFrame({"Protein": freq.index, "SelectionFreq": freq.values,
                       "Selected": freq.values >= FREQ_CUTOFF}),
         "表1_LASSO稳定性选择频率", index=False)

show_f = freq[freq >= 0.2].iloc[::-1]
fig, ax = plt.subplots(figsize=(2.8, max(2.0, 0.16 * len(show_f) + 0.6)))
cols = ["#D64B3B" if v >= FREQ_CUTOFF else "#B0B0B0" for v in show_f.values]
ax.barh(show_f.index, show_f.values, color=cols,
        edgecolor="black", lw=0.4, height=0.7)
ax.axvline(FREQ_CUTOFF, color="black", lw=0.6, ls="--")
ax.set_xlabel(f"Selection frequency ({N_BOOT} bootstraps)")
ax.set_ylabel("")
ax.set_xlim(0, 1)
style_ax(ax)
save_fig(fig, "图2_LASSO稳定性选择频率")

print("=" * 50)
print("[B-2] 随机森林特征重要性")
rf = RandomForestClassifier(n_estimators=500, random_state=RNG,
                            n_jobs=-1, class_weight="balanced")
rf.fit(X, y)
imp_rf = pd.Series(rf.feature_importances_, index=prot_cols).sort_values(ascending=False)
rf_cut = imp_rf.mean() + IMP_SD * imp_rf.std()
rf_sel = imp_rf[imp_rf > rf_cut].index.tolist()
selected["RF"] = rf_sel
print(f"  阈值={rf_cut:.4f}, 选中{len(rf_sel)}个")
importance_barplot(imp_rf, rf_cut, set(rf_sel), "Mean decrease in impurity",
                   "图3_随机森林重要性Top20", "表2_随机森林重要性")

print("=" * 50)
print("[B-3] XGBoost特征重要性")
try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", random_state=RNG, n_jobs=-1)
    xgb.fit(X, y)
    imp_xgb = pd.Series(xgb.feature_importances_,
                        index=prot_cols).sort_values(ascending=False)
    xgb_cut = imp_xgb.mean() + IMP_SD * imp_xgb.std()
    xgb_sel = imp_xgb[imp_xgb > xgb_cut].index.tolist()
    selected["XGBoost"] = xgb_sel
    print(f"  阈值={xgb_cut:.4f}, 选中{len(xgb_sel)}个")
    importance_barplot(imp_xgb, xgb_cut, set(xgb_sel),
                       "Gain (XGBoost importance)",
                       "图4_XGBoost重要性Top20", "表3_XGBoost重要性")
except ImportError:
    print("  XGBoost未安装, 跳过 (pip install xgboost)")

print("=" * 50)
print("[B-4] LightGBM特征重要性")
try:
    from lightgbm import LGBMClassifier
    lgb = LGBMClassifier(n_estimators=500, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=RNG, n_jobs=-1, verbose=-1)
    lgb.fit(X, y)
    imp_lgb = pd.Series(lgb.feature_importances_,
                        index=prot_cols).sort_values(ascending=False)
    lgb_cut = imp_lgb.mean() + IMP_SD * imp_lgb.std()
    lgb_sel = imp_lgb[imp_lgb > lgb_cut].index.tolist()
    selected["LightGBM"] = lgb_sel
    print(f"  阈值={lgb_cut:.4f}, 选中{len(lgb_sel)}个")
    importance_barplot(imp_lgb, lgb_cut, set(lgb_sel),
                       "Split count (LightGBM importance)",
                       "图5_LightGBM重要性Top20", "表4_LightGBM重要性")
except ImportError:
    print("  LightGBM未安装, 跳过 (pip install lightgbm)")

print("=" * 50)
print("[B-5] SVM-RFE (收紧版: 1-SE规则, 上限%d个特征)" % RFE_MAX_N)
svc = SVC(kernel="linear", random_state=RNG)
rfecv = RFECV(estimator=svc, step=1, cv=cv, scoring="roc_auc",
              min_features_to_select=3, n_jobs=-1)
rfecv.fit(X, y)
n_feats = rfecv.cv_results_["n_features"]
m_auc = rfecv.cv_results_["mean_test_score"]
s_auc = rfecv.cv_results_["std_test_score"]
best_auc = m_auc.max()
se_best = s_auc[m_auc.argmax()]

ok = np.where(m_auc >= best_auc - se_best)[0]
n_1se = int(n_feats[ok[0]])

n_final = min(n_1se, RFE_MAX_N)
ranks = pd.Series(rfecv.ranking_, index=prot_cols)
rfe_sel = ranks.nsmallest(n_final).index.tolist()
selected["SVM_RFE"] = rfe_sel
print(f"  原始CV最优: {rfecv.n_features_}个特征 (AUC={best_auc:.3f}) "
      f"<- 过于宽松")
print(f"  1-SE规则: {n_1se}个 -> 封顶后: {n_final}个")
print(f"  收紧后选中: {rfe_sel}")

rfe_tbl = pd.DataFrame({"Protein": prot_cols, "Ranking": rfecv.ranking_,
                        "Selected_tightened": ranks <= n_final,
                        "Selected_original": rfecv.support_}
                       ).sort_values("Ranking")
save_xls(rfe_tbl, "表5_SVM-RFE排序", index=False)

fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.plot(n_feats, m_auc, "o-", color="#3B6FB6", lw=0.9, ms=3,
        mec="black", mew=0.4)
ax.fill_between(n_feats, m_auc - s_auc, m_auc + s_auc,
                color="#3B6FB6", alpha=0.15, lw=0)
ax.axvline(rfecv.n_features_, color="#B0B0B0", lw=0.7, ls="--")
ax.annotate(f"CV-optimal (too loose)\nn={rfecv.n_features_}",
            (rfecv.n_features_, best_auc), fontsize=5.5,
            xytext=(-6, -34), textcoords="offset points",
            ha="right", color="#666666")
ax.axvline(n_final, color="#D64B3B", lw=0.8, ls="--")
auc_at_nfinal = m_auc[np.where(n_feats == n_final)[0][0]]
ax.annotate(f"Tightened (1-SE, cap {RFE_MAX_N})\nn={n_final}, "
            f"AUC={auc_at_nfinal:.3f}",
            (n_final, auc_at_nfinal), fontsize=5.5, color="#D64B3B",
            xytext=(6, -40), textcoords="offset points", ha="left")
ax.set_xlabel("Number of features selected")
ax.set_ylabel(f"Cross-validated AUC ({N_CV}-fold)")
style_ax(ax)
save_fig(fig, "图6_SVM-RFE交叉验证曲线")
save_xls(pd.DataFrame({"N_features": n_feats, "MeanAUC": m_auc,
                       "StdAUC": s_auc,
                       "Original_optimal": n_feats == rfecv.n_features_,
                       "Tightened_n": n_feats == n_final}),
         "图6_原始数据", index=False)

print("=" * 50)
print("[B-6] Boruta (全相关特征选择)")
try:
    from boruta import BorutaPy
    bt = BorutaPy(RandomForestClassifier(n_estimators=500, random_state=RNG,
                                         n_jobs=-1, class_weight="balanced"),
                  n_estimators="auto", random_state=RNG, max_iter=100)
    bt.fit(X, y)
    bt_sel = [p for p, s in zip(prot_cols, bt.support_) if s]
    bt_weak = [p for p, s in zip(prot_cols, bt.support_weak_) if s]
    selected["Boruta"] = bt_sel
    print(f"  确认{len(bt_sel)}个: {bt_sel}; 待定{len(bt_weak)}个")

    save_xls(pd.DataFrame({"Protein": prot_cols, "Ranking": bt.ranking_,
                           "Confirmed": bt.support_,
                           "Tentative": bt.support_weak_}
                          ).sort_values("Ranking"),
             "表6_Boruta结果", index=False)

    bt_imp = imp_rf.reindex(prot_cols)
    n_show = min(25, (bt.support_.sum() + bt.support_weak_.sum() + 8))
    show_idx = bt_imp.sort_values(ascending=False).head(n_show).index[::-1]
    decision = {p: ("Confirmed" if c else "Tentative" if t else "Rejected")
                for p, c, t in zip(prot_cols, bt.support_, bt.support_weak_)}
    dcolor = {"Confirmed": "#D64B3B", "Tentative": "#E8A33D",
              "Rejected": "#B0B0B0"}
    fig, ax = plt.subplots(figsize=(2.8, 0.16 * n_show + 0.8))
    ax.barh(show_idx, bt_imp[show_idx],
            color=[dcolor[decision[p]] for p in show_idx],
            edgecolor="black", lw=0.4, height=0.7)
    ax.set_xlabel("Importance (colored by Boruta decision)")
    ax.set_ylabel("")
    handles = [mpl.patches.Patch(fc=dcolor[d], ec="black", lw=0.4, label=d)
               for d in ["Confirmed", "Tentative", "Rejected"]]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=5.5)
    style_ax(ax)
    save_fig(fig, "图7_Boruta判定结果")
except ImportError:
    print("  Boruta未安装, 跳过 (pip install boruta)")

print("=" * 50)
print("[C] 方法间比较与共识特征")
methods = list(selected.keys())
mat = pd.DataFrame(0, index=prot_cols, columns=methods)
for m, feats in selected.items():
    mat.loc[feats, m] = 1
mat["N_ML_methods"] = mat.sum(axis=1)
mat["DiffAnalysis"] = 0
mat.loc[da_sel, "DiffAnalysis"] = 1
mat = mat.sort_values(["N_ML_methods", "DiffAnalysis"], ascending=False)

consensus  = mat[mat["N_ML_methods"] >= CONSENSUS_MIN].index.tolist()
core_feats = mat[mat["N_ML_methods"] >= 3].index.tolist()
both_evidence = mat[(mat["N_ML_methods"] >= CONSENSUS_MIN) &
                    (mat["DiffAnalysis"] == 1)].index.tolist()
print(f"  各ML方法选中数: { {m: len(v) for m, v in selected.items()} }")
print(f"  共识特征(>={CONSENSUS_MIN}种ML方法): {len(consensus)}个")
print(f"  核心特征(>=3种ML方法): {len(core_feats)}个")
print(f"  双重证据(ML共识+差异显著): {len(both_evidence)}个")

save_xls(mat.reset_index(names="Protein"), "表7_各方法选择结果矩阵", index=False)

show = mat[mat["N_ML_methods"] >= CONSENSUS_MIN]
show = show.sort_values(["N_ML_methods", "DiffAnalysis"],
                        ascending=[False, False])
ml_block = show[methods].values
da_block = show[["DiffAnalysis"]].values
fig, ax = plt.subplots(figsize=(2.2 + 0.5 * len(methods),
                                max(2.0, 0.16 * len(show) + 0.6)))
cmap_ml = mpl.colors.ListedColormap(["#F0F0F0", "#3B6FB6"])
cmap_da = mpl.colors.ListedColormap(["#F0F0F0", "#7B5EA7"])
nrow, ncol_ml = ml_block.shape
ax.imshow(ml_block, cmap=cmap_ml, aspect="auto", vmin=0, vmax=1,
          extent=[0, ncol_ml, 0, nrow])
ax.imshow(da_block, cmap=cmap_da, aspect="auto", vmin=0, vmax=1,
          extent=[ncol_ml + 0.4, ncol_ml + 1.4, 0, nrow])
ax.set_xticks([j + 0.5 for j in range(ncol_ml)] + [ncol_ml + 0.9])
ax.set_xticklabels(methods + ["Diff.\nanalysis"], rotation=30, ha="right")
ax.set_yticks([nrow - i - 0.5 for i in range(nrow)])
ax.set_yticklabels(show.index)
for i in range(nrow):
    for j in range(ncol_ml):
        if ml_block[i, j]:
            ax.text(j + 0.5, nrow - i - 0.5, chr(10003), ha="center",
                    va="center", fontsize=5, color="white")
    if da_block[i, 0]:
        ax.text(ncol_ml + 0.9, nrow - i - 0.5, chr(10003), ha="center",
                va="center", fontsize=5, color="white")
ax.axvline(ncol_ml + 0.2, color="black", lw=0.6)
ax.set_xlim(0, ncol_ml + 1.4)
ax.set_ylim(0, nrow)
ax.tick_params(length=0)
ax.set_title(f"Consensus features (>={CONSENSUS_MIN} ML methods): "
             f"n={len(consensus)}", fontsize=7)
for s in ax.spines.values():
    s.set_visible(False)
save_fig(fig, "图8_各方法选择结果热图")
full_show = mat[mat["N_ML_methods"] >= 1]
save_xls(full_show.reset_index(names="Protein"), "图8_原始数据", index=False)

if consensus:
    n = len(consensus)
    ncol = min(4, n)
    nrow_b = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow_b, ncol,
                             figsize=(1.5 * ncol + 0.5, 1.4 * nrow_b + 0.4))
    axes = np.atleast_1d(axes).ravel()
    q_map = da_tbl.set_index("Protein")["FDR_q"]
    rng_j = np.random.default_rng(RNG)
    for k, p in enumerate(consensus):
        ax = axes[k]
        for gi, g in enumerate(["R", "NR"]):
            vals = df.loc[df["Group"] == g, p].values
            bp = ax.boxplot(vals, positions=[gi], widths=0.55,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", lw=0.8),
                            whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
            bp["boxes"][0].set(facecolor=GROUP_COLOR[g], alpha=0.8,
                               edgecolor="black", lw=0.5)
            ax.scatter(gi + rng_j.normal(0, 0.06, len(vals)), vals,
                       s=4, c="white", edgecolor="black", lw=0.3, zorder=3)
        q = q_map[p]
        star = "***" if q < 0.001 else "**" if q < 0.01 else \
               "*" if q < 0.05 else "ns"
        ax.set_title(f"{p} ({star})", fontsize=7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["R", "NR"])
        if k % ncol == 0:
            ax.set_ylabel("Z-score")
        style_ax(ax)
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.tight_layout()
    save_fig(fig, "图9_共识特征表达箱线图")
    save_xls(df[["Group"] + consensus].copy(), "图9_原始数据", index=False)

print("=" * 50)
print("Step 2.3 保存筛选结果")
cons_tbl = pd.DataFrame({"Protein": consensus})
for m in methods:
    cons_tbl[m] = cons_tbl["Protein"].isin(selected[m]).astype(int)
cons_tbl["N_ML_methods"] = cons_tbl[methods].sum(axis=1)
cons_tbl["DiffAnalysis(p<0.05)"] = cons_tbl["Protein"].isin(da_sel).astype(int)
cons_tbl = cons_tbl.merge(
    pd.DataFrame({"Protein": prot_cols,
                  "RF_Importance": rf.feature_importances_}), on="Protein") \
    .merge(da_tbl[["Protein", "P_value", "FDR_q"]], on="Protein")
cons_tbl = cons_tbl.sort_values(["N_ML_methods", "RF_Importance"],
                                ascending=[False, False])
save_xls(cons_tbl, "表8_共识特征汇总", index=False)

out = pd.concat([df[["label", "Group"]].reset_index(drop=True),
                 df[consensus].reset_index(drop=True)], axis=1)
save_xls(out, "共识特征数据_Zscore", index=False)

print("=" * 50)
print(f"第二步(修订版v3)完成! 结果保存至: {OUT_DIR}")
print(f"  差异分析: {DA_DIR}")
print(f"  共识特征({len(consensus)}个): {consensus}")
