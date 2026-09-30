import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import joblib
from scipy import stats
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             brier_score_loss)

warnings.filterwarnings("ignore")

INPUT_FILE = r"D:\333\Cohort 2 and 3.xlsx"
SHEET_NAME = "Cohort 3"
MODEL_FILE = r"D:\333\Cohort2_模型训练\最终模型_锁定.pkl"
OUT_DIR    = r"D:\333\Cohort3_独立验证"
os.makedirs(OUT_DIR, exist_ok=True)

GROUP_COLOR = {"R": "#3B6FB6", "NR": "#D64B3B"}
N_BOOT_CI   = 2000
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

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - half, center + half

print("=" * 50)
print("C3-Step1 加载锁定模型(Cohort 2训练, 不修改)")
lock = joblib.load(MODEL_FILE)
model      = lock["model"]
MODEL_NAME = lock["model_name"]
FEATURES   = lock["features"]
LOG2_MEAN  = lock["log2_mean"]
LOG2_STD   = lock["log2_std"]
CUTOFF     = lock["cutoff"]
print(f"  模型: {MODEL_NAME} | 特征: {FEATURES} | 锁定阈值: {CUTOFF:.3f}")

print("=" * 50)
print("C3-Step2 读取Cohort 3并质控")
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
df.columns = ["label"] + list(df.columns[1:])
df["Group"] = df["label"].map({0: "R", 1: "NR"})
missing_f = [f for f in FEATURES if f not in df.columns]
if missing_f:
    raise ValueError(f"Cohort 3缺少特征: {missing_f}")
X_raw = df[FEATURES].astype(float)
y = df["label"].values
print(f"  {len(df)}样本 (R={(y == 0).sum()}, NR={(y == 1).sum()})")
print(f"  缺失值: {X_raw.isna().sum().sum()}, <=0值: {(X_raw <= 0).sum().sum()}")

qc = pd.DataFrame({
    "检查项": ["总样本数", "响应组R", "非响应组NR", "缺失值", "<=0值"],
    "结果": [len(df), int((y == 0).sum()), int((y == 1).sum()),
            int(X_raw.isna().sum().sum()), int((X_raw <= 0).sum().sum())],
})
save_xls(qc, "表1_数据质量检查", index=False)

desc = pd.DataFrame({
    "Protein": FEATURES,
    "Min": X_raw.min().values, "Median": X_raw.median().values,
    "Mean": X_raw.mean().values, "SD": X_raw.std().values,
    "Max": X_raw.max().values, "Skewness": X_raw.skew().values,
})
save_xls(desc.round(4), "表2_原始浓度描述统计", index=False)

print("=" * 50)
print("C3-Step3 标准化(使用Cohort 2锁定参数, 不重估)")
Xlog = np.log2(X_raw)
Z = (Xlog - LOG2_MEAN) / LOG2_STD
Z = pd.DataFrame(Z, columns=FEATURES)
print("  log2转换 + Z-score(锁定参数) 完成")

save_xls(pd.concat([df[["label", "Group"]].reset_index(drop=True),
                    X_raw.reset_index(drop=True)], axis=1),
         "清洗后数据_原始浓度", index=False)
save_xls(pd.concat([df[["label", "Group"]].reset_index(drop=True),
                    Z.reset_index(drop=True)], axis=1),
         "清洗后数据_Zscore(锁定参数)", index=False)

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
ax.set_xticks([])
ax.set_ylabel("Z-score (locked scaler)")
ax.set_xlabel("Sample (n=240)")
handles = [mpl.patches.Patch(fc=GROUP_COLOR[g], ec="black", lw=0.5, label=g)
           for g in ["R", "NR"]]
ax.legend(handles=handles, frameon=False, loc="upper right", ncol=2)
style_ax(ax)
save_fig(fig, "图1_标准化后样本分布")

print("=" * 50)
print("C3-Step4 盲态预测(锁定模型, 一次性评估)")
prob = model.predict_proba(Z.values)[:, 1]
pred = (prob >= CUTOFF).astype(int)

auc = roc_auc_score(y, prob)
rng = np.random.default_rng(RNG)
boot_aucs = []
for b in range(N_BOOT_CI):
    idx = rng.integers(0, len(y), len(y))
    if len(np.unique(y[idx])) < 2:
        continue
    boot_aucs.append(roc_auc_score(y[idx], prob[idx]))
auc_lo, auc_hi = np.percentile(boot_aucs, [2.5, 97.5])
print(f"  AUC = {auc:.3f} (95%CI {auc_lo:.3f}-{auc_hi:.3f})")

tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
sens = tp / (tp + fn); spec = tn / (tn + fp)
acc = (tp + tn) / len(y)
ppv = tp / (tp + fp); npv = tn / (tn + fn)
brier = brier_score_loss(y, prob)
sens_ci = wilson_ci(tp, tp + fn); spec_ci = wilson_ci(tn, tn + fp)
acc_ci = wilson_ci(tp + tn, len(y))
ppv_ci = wilson_ci(tp, tp + fp);  npv_ci = wilson_ci(tn, tn + fn)

perf_tbl = pd.DataFrame({
    "指标": ["AUC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV",
            "Brier score"],
    "点估计": [round(auc, 4), round(acc, 4), round(sens, 4), round(spec, 4),
              round(ppv, 4), round(npv, 4), round(brier, 4)],
    "95%CI下限": [round(auc_lo, 4), round(acc_ci[0], 4), round(sens_ci[0], 4),
                round(spec_ci[0], 4), round(ppv_ci[0], 4),
                round(npv_ci[0], 4), ""],
    "95%CI上限": [round(auc_hi, 4), round(acc_ci[1], 4), round(sens_ci[1], 4),
                round(spec_ci[1], 4), round(ppv_ci[1], 4),
                round(npv_ci[1], 4), ""],
    "备注": [f"bootstrap {N_BOOT_CI}次", "Wilson CI",
            f"锁定阈值={CUTOFF:.3f}", f"锁定阈值={CUTOFF:.3f}",
            "Wilson CI", "Wilson CI", ""],
})
save_xls(perf_tbl, "表3_独立验证性能汇总", index=False)
print(perf_tbl.to_string(index=False))

save_xls(pd.DataFrame({"Sample": df.index + 1, "Group": df["Group"],
                       "label": y, "Pred_prob": prob,
                       "Pred_label": pred}),
         "表4_逐样本预测结果", index=False)

fpr, tpr, _ = roc_curve(y, prob)
mean_fpr = np.linspace(0, 1, 100)
boot_tprs = []
for b in range(N_BOOT_CI):
    idx = rng.integers(0, len(y), len(y))
    if len(np.unique(y[idx])) < 2:
        continue
    f, t, _ = roc_curve(y[idx], prob[idx])
    boot_tprs.append(np.interp(mean_fpr, f, t))
tpr_lo, tpr_hi = np.percentile(boot_tprs, [2.5, 97.5], axis=0)
tpr_mean = np.interp(mean_fpr, fpr, tpr)

fig, ax = plt.subplots(figsize=(3.0, 3.0))
ax.plot(mean_fpr, tpr_mean, color="#D64B3B", lw=1.2,
        label=f"AUC = {auc:.3f} ({auc_lo:.3f}-{auc_hi:.3f})")
ax.fill_between(mean_fpr, tpr_lo, tpr_hi, color="#D64B3B",
                alpha=0.15, lw=0, label="95% CI")
ax.plot([0, 1], [0, 1], color="grey", lw=0.6, ls="--")

sens_at, spec_at = sens, spec
ax.scatter([1 - spec_at], [sens_at], s=30, c="black", zorder=5)
ax.annotate(f"Cutoff {CUTOFF:.2f}", (1 - spec_at, sens_at),
            xytext=(-40, -12), textcoords="offset points", fontsize=6)
ax.set_xlabel("1 - Specificity")
ax.set_ylabel("Sensitivity")
ax.set_title("Independent validation (Cohort 3)", fontsize=8)
ax.legend(frameon=False, loc="lower right", fontsize=6)
style_ax(ax)
save_fig(fig, "图2_ROC曲线_独立验证")
save_xls(pd.DataFrame({"FPR": mean_fpr, "TPR": tpr_mean,
                       "TPR_lo95": tpr_lo, "TPR_hi95": tpr_hi}),
         "图2_原始数据", index=False)

cm = confusion_matrix(y, pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(2.2, 2.0))
ax.imshow(cm, cmap="Blues", vmin=0)
labs = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{labs[i][j]}={cm[i, j]}", ha="center", va="center",
                fontsize=8,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred R", "Pred NR"])
ax.set_yticklabels(["True R", "True NR"])
ax.set_title(f"{MODEL_NAME} (Cohort 3, cutoff={CUTOFF:.2f})", fontsize=8)
save_fig(fig, "图3_混淆矩阵_独立验证")
save_xls(pd.DataFrame(cm, index=["True_R", "True_NR"],
                      columns=["Pred_R", "Pred_NR"]).reset_index(names=""),
         "图3_原始数据", index=False)

fig, ax = plt.subplots(figsize=(2.8, 2.8))
bins = np.unique(np.quantile(prob, np.linspace(0, 1, 11)))
digit = np.clip(np.digitize(prob, bins[1:-1]), 0, len(bins) - 2)
mp, op, nn = [], [], []
calib_rows = []
for b in range(len(bins) - 1):
    msk = digit == b
    if msk.sum() >= 5:
        mp.append(prob[msk].mean()); op.append(y[msk].mean())
        nn.append(msk.sum())
        calib_rows.append({"Bin": b + 1, "MeanPredProb": mp[-1],
                           "ObservedFreq": op[-1], "N": int(msk.sum())})
ax.plot(mp, op, "o-", color="#D64B3B", lw=1.0, ms=4,
        mec="black", mew=0.3, label=f"Cohort 3 (Brier={brier:.3f})")
ax.plot([0, 1], [0, 1], color="grey", lw=0.6, ls="--", label="Perfect")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed frequency (NR)")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_title("Calibration (independent validation)", fontsize=8)
ax.legend(frameon=False, loc="upper left", fontsize=6)
style_ax(ax)
save_fig(fig, "图4_校准曲线_独立验证")
save_xls(pd.DataFrame(calib_rows), "图4_原始数据", index=False)

hl_rows = pd.DataFrame(calib_rows)
chi2_hl = ((hl_rows["ObservedFreq"] * hl_rows["N"]
            - hl_rows["MeanPredProb"] * hl_rows["N"]) ** 2
           / (hl_rows["N"] * hl_rows["MeanPredProb"]
              * (1 - hl_rows["MeanPredProb"]))).sum()
hl_p = 1 - stats.chi2.cdf(chi2_hl, len(hl_rows) - 2)
print(f"  Hosmer-Lemeshow: chi2={chi2_hl:.2f}, P={hl_p:.4f} "
      f"(P>0.05提示校准良好)")

fig, ax = plt.subplots(figsize=(3.2, 2.8))
thresholds = np.linspace(0.01, 0.99, 99)
n = len(y)
prev = y.mean()
nb_model = []
dca_rows = []
for pt in thresholds:
    p_ = prob >= pt
    tp_ = ((p_ == 1) & (y == 1)).sum()
    fp_ = ((p_ == 1) & (y == 0)).sum()
    nbv = tp_ / n - fp_ / n * (pt / (1 - pt))
    nb_model.append(nbv)
    dca_rows.append({"Threshold": pt, "NetBenefit_model": nbv,
                     "NetBenefit_all": prev - (1 - prev) * pt / (1 - pt),
                     "NetBenefit_none": 0})
ax.plot(thresholds, nb_model, color="#D64B3B", lw=1.1,
        label=f"{MODEL_NAME} model")
ax.plot(thresholds, [d["NetBenefit_all"] for d in dca_rows],
        color="grey", lw=0.8, ls="--", label="Treat all")
ax.axhline(0, color="black", lw=0.8, ls=":", label="Treat none")
ax.axvline(CUTOFF, color="#3B6FB6", lw=0.7, ls="-.", alpha=0.7)
ax.text(CUTOFF + 0.01, ax.get_ylim()[0] + 0.02, f"cutoff={CUTOFF:.2f}",
        fontsize=6, color="#3B6FB6")
ax.set_xlabel("Threshold probability")
ax.set_ylabel("Net benefit")
ax.set_ylim(-0.15, prev + 0.1)
ax.set_title("Decision curve (independent validation)", fontsize=8)
ax.legend(frameon=False, loc="upper right", fontsize=6)
style_ax(ax)
save_fig(fig, "图5_DCA决策曲线_独立验证")
save_xls(pd.DataFrame(dca_rows), "图5_原始数据", index=False)

print("=" * 50)
report = pd.DataFrame({
    "项目": ["模型", "特征数", "特征", "锁定阈值(Youden, Cohort2)",
            "验证队列", "样本量", "AUC (95%CI)",
            "Sensitivity (95%CI)", "Specificity (95%CI)",
            "Accuracy (95%CI)", "Brier", "Hosmer-Lemeshow P"],
    "结果": [MODEL_NAME, len(FEATURES), ", ".join(FEATURES),
            round(CUTOFF, 4), "Cohort 3 (独立验证)", len(y),
            f"{auc:.3f} ({auc_lo:.3f}-{auc_hi:.3f})",
            f"{sens:.3f} ({sens_ci[0]:.3f}-{sens_ci[1]:.3f})",
            f"{spec:.3f} ({spec_ci[0]:.3f}-{spec_ci[1]:.3f})",
            f"{acc:.3f} ({acc_ci[0]:.3f}-{acc_ci[1]:.3f})",
            round(brier, 4), round(hl_p, 4)],
})
save_xls(report, "表5_独立验证报告", index=False)
print("=" * 50)
print(f"Cohort 3 独立验证完成! 结果保存至: {OUT_DIR}")
print(f"AUC={auc:.3f} ({auc_lo:.3f}-{auc_hi:.3f}) | "
      f"Sens={sens:.3f} | Spec={spec:.3f} | HL P={hl_p:.3f}")
