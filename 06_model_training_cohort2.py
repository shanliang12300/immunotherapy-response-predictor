import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                             f1_score, confusion_matrix, brier_score_loss)

warnings.filterwarnings("ignore")

DATA_DIR   = r"D:\333\Cohort2_数据处理"
OUT_DIR    = r"D:\333\Cohort2_模型训练"
os.makedirs(OUT_DIR, exist_ok=True)

FINAL_FEATURES = ["LAG3", "CCL23", "Gal-1", "CD70", "CXCL12"]

N_BOOT    = 200
N_SPLITS  = 5
N_REPEATS = 20
RNG       = 42

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
MODEL_COLORS = {"LR": "#3B6FB6", "SVM": "#D64B3B", "RF": "#2E8B57",
                "kNN": "#E8A33D", "NB": "#7B5EA7",
                "XGBoost": "#C4622D", "LightGBM": "#4D9DE0"}

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
print("[A] 8->5 特征筛选证据")
df_z = pd.read_excel(os.path.join(DATA_DIR, "清洗后数据_Zscore.xlsx"))
prot8 = [c for c in df_z.columns if c not in ["label", "Group"]]
X8 = df_z[prot8].values
y = df_z["label"].values
print(f"  {len(df_z)}样本, 8蛋白: {prot8}")

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RNG)

rng = np.random.default_rng(RNG)
cnt = pd.Series(0, index=prot8, dtype=float)
for b in range(N_BOOT):
    idx = rng.integers(0, len(y), len(y))
    if len(np.unique(y[idx])) < 2:
        continue
    m = LogisticRegression(C=0.5, penalty="l1", solver="saga",
                           max_iter=5000, random_state=b).fit(X8[idx], y[idx])
    cnt += (m.coef_[0] != 0)
lasso_freq = cnt / N_BOOT

rf = RandomForestClassifier(n_estimators=500, random_state=RNG,
                            n_jobs=-1, class_weight="balanced").fit(X8, y)
rf_imp = pd.Series(rf.feature_importances_, index=prot8)
try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", random_state=RNG,
                        n_jobs=-1).fit(X8, y)
    xgb_imp = pd.Series(xgb.feature_importances_, index=prot8)
except ImportError:
    xgb_imp = pd.Series(np.nan, index=prot8)
    print("  XGBoost未安装, 重要性列留空")

uni_file = os.path.join(DATA_DIR, "表3_组间差异初筛.xlsx")
uni = pd.read_excel(uni_file).set_index("Protein")["Log2FC(NR/R)"]
uni_abs = uni.abs()

rank_tbl = pd.DataFrame({
    "LASSO_freq": lasso_freq,
    "RF_importance": rf_imp,
    "XGB_importance": xgb_imp,
    "AbsLog2FC": uni_abs.reindex(prot8),
})
rank_tbl["Rank_LASSO"] = rank_tbl["LASSO_freq"].rank(ascending=False)
rank_tbl["Rank_RF"] = rank_tbl["RF_importance"].rank(ascending=False)
rank_tbl["Rank_XGB"] = rank_tbl["XGB_importance"].rank(ascending=False)
rank_tbl["Rank_Uni"] = rank_tbl["AbsLog2FC"].rank(ascending=False)
rank_tbl["MeanRank"] = rank_tbl[["Rank_LASSO", "Rank_RF",
                                 "Rank_XGB", "Rank_Uni"]].mean(axis=1)
rank_tbl = rank_tbl.sort_values("MeanRank")
rank_tbl["In_Final5"] = rank_tbl.index.isin(FINAL_FEATURES)
save_xls(rank_tbl.round(4).reset_index(names="Protein"),
         "表1_8蛋白筛选证据与排名", index=False)
print(rank_tbl.round(3).to_string())
print(f"  最终5特征(与Cohort3 panel一致): {FINAL_FEATURES}")
print(f"  注: 数据驱动排名前5为 {rank_tbl.index[:5].tolist()}, "
      f"CCL23因Cohort1证据/单变量效应/临床重要性入选")

fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.2))
metrics = [("LASSO_freq", "LASSO selection freq."),
           ("RF_importance", "RF importance"),
           ("XGB_importance", "XGBoost importance"),
           ("AbsLog2FC", "|Log2FC| (univariate)")]
plot_order = rank_tbl.index[::-1]
for ax, (col, xlabel) in zip(axes, metrics):
    vals = rank_tbl.loc[plot_order, col]
    cols = ["#D64B3B" if p in FINAL_FEATURES else "#B0B0B0"
            for p in plot_order]
    ax.barh(range(len(plot_order)), vals, color=cols,
            edgecolor="black", lw=0.4, height=0.65)
    ax.set_yticks(range(len(plot_order)))
    ax.set_yticklabels(plot_order if col == "LASSO_freq"
                       else [""] * len(plot_order))
    ax.set_xlabel(xlabel, fontsize=6.5)
    style_ax(ax)
handles = [mpl.patches.Patch(fc="#D64B3B", ec="black", lw=0.4,
                             label="Final 5"),
           mpl.patches.Patch(fc="#B0B0B0", ec="black", lw=0.4,
                             label="Not selected")]
fig.legend(handles=handles, frameon=False, loc="upper right",
           fontsize=6.5, ncol=2)
fig.tight_layout(rect=[0, 0, 0.94, 1])
save_fig(fig, "图1_8蛋白筛选证据")
save_xls(rank_tbl.round(4).reset_index(names="Protein"),
         "图1_原始数据", index=False)

print("=" * 50)
print("[B] 模型比较 (重复5折CV x 20次, 训练集性能)")
X5 = df_z[FINAL_FEATURES].values

models = {
    "LR":  LogisticRegression(C=1.0, max_iter=5000, random_state=RNG),
    "SVM": SVC(C=1.0, kernel="rbf", gamma="scale", probability=True,
               random_state=RNG),
    "RF":  RandomForestClassifier(n_estimators=500, random_state=RNG,
                                  n_jobs=-1, class_weight="balanced"),
    "kNN": KNeighborsClassifier(n_neighbors=7),
    "NB":  GaussianNB(),
}
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=RNG, n_jobs=-1)
except ImportError:
    pass
try:
    from lightgbm import LGBMClassifier
    models["LightGBM"] = LGBMClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RNG, n_jobs=-1, verbose=-1)
except ImportError:
    pass

per_repeat = []
oof_prob = {m: np.zeros((len(y), N_REPEATS)) for m in models}
for mname, model in models.items():
    for rep in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=RNG + rep)
        prob = cross_val_predict(model, X5, y, cv=skf,
                                 method="predict_proba")[:, 1]
        oof_prob[mname][:, rep] = prob
        pred = (prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        per_repeat.append({
            "Model": mname, "Repeat": rep + 1,
            "AUC": roc_auc_score(y, prob),
            "Accuracy": accuracy_score(y, pred),
            "Sensitivity": tp / (tp + fn),
            "Specificity": tn / (tn + fp),
            "F1": f1_score(y, pred),
            "Brier": brier_score_loss(y, prob),
        })
    print(f"  {mname:8s} 完成")

perf = pd.DataFrame(per_repeat)
perf_summary = perf.groupby("Model").agg(
    AUC_mean=("AUC", "mean"), AUC_sd=("AUC", "std"),
    ACC_mean=("Accuracy", "mean"), ACC_sd=("Accuracy", "std"),
    Sens_mean=("Sensitivity", "mean"), Sens_sd=("Sensitivity", "std"),
    Spec_mean=("Specificity", "mean"), Spec_sd=("Specificity", "std"),
    F1_mean=("F1", "mean"), F1_sd=("F1", "std"),
    Brier_mean=("Brier", "mean"),
).sort_values("AUC_mean", ascending=False).round(4)
perf_summary.insert(0, "Performance_type", "Training (Cohort 2, CV)")
save_xls(perf, "表2_各模型每次重复性能_训练集", index=False)
save_xls(perf_summary.reset_index(), "表3_模型性能汇总_训练集", index=False)
print("\n训练集性能(仅训练参考, 不可作为验证性能):")
print(perf_summary[["AUC_mean", "AUC_sd", "ACC_mean",
                    "Sens_mean", "Spec_mean"]].to_string())

best_name = perf_summary.index[0]
print(f"\n>>> 最佳模型: {best_name} "
      f"(训练CV-AUC={perf_summary.loc[best_name, 'AUC_mean']:.3f})")

oof_mean = oof_prob[best_name].mean(axis=1)
save_xls(pd.DataFrame({"Sample": df_z.index + 1, "Group": df_z["Group"],
                       "label": y, "OOF_prob": oof_mean}),
         "表4_最佳模型OOF预测概率", index=False)

fig, ax = plt.subplots(figsize=(3.0, 3.0))
mean_fpr = np.linspace(0, 1, 100)
roc_rows = []
for mname in perf_summary.index:
    tprs = []
    for rep in range(N_REPEATS):
        fpr, tpr, _ = roc_curve(y, oof_prob[mname][:, rep])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
    mt = np.mean(tprs, axis=0)
    auc_m = perf_summary.loc[mname, "AUC_mean"]
    auc_s = perf_summary.loc[mname, "AUC_sd"]
    ax.plot(mean_fpr, mt, color=MODEL_COLORS.get(mname, "#666666"),
            lw=1.0, label=f"{mname} ({auc_m:.3f}±{auc_s:.3f})")
    for f, t in zip(mean_fpr, mt):
        roc_rows.append({"Model": mname, "FPR": f, "MeanTPR": t})
ax.plot([0, 1], [0, 1], color="grey", lw=0.6, ls="--")
ax.set_xlabel("1 - Specificity")
ax.set_ylabel("Sensitivity")
ax.set_title("Training performance (Cohort 2)", fontsize=8)
ax.legend(frameon=False, loc="lower right", fontsize=5.5)
style_ax(ax)
save_fig(fig, "图2_各模型ROC_训练集")
save_xls(pd.DataFrame(roc_rows), "图2_原始数据", index=False)

metrics = ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1"]
fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.2))
mlist = perf_summary.index.tolist()
xpos = np.arange(len(mlist))
for k, met in enumerate(metrics):
    ax = axes[k]
    means = perf.groupby("Model")[met].mean().loc[mlist]
    sds = perf.groupby("Model")[met].std().loc[mlist]
    cols = [MODEL_COLORS.get(m, "#666666") for m in mlist]
    ax.bar(xpos, means, yerr=sds, color=cols, edgecolor="black",
           lw=0.4, width=0.7, capsize=1.5, error_kw=dict(lw=0.6))
    ax.set_xticks(xpos)
    ax.set_xticklabels(mlist, rotation=45, ha="right")
    ax.set_title(met, fontsize=7)
    ax.set_ylim(0, 1.12)
    style_ax(ax)
axes[0].set_ylabel("Score")
fig.suptitle("Training performance (Cohort 2)", fontsize=8, y=1.02)
fig.tight_layout()
save_fig(fig, "图3_模型性能比较_训练集")

print("=" * 50)
print("[C] 锁定最终模型")
final_model = models[best_name]
final_model.fit(X5, y)

fpr, tpr, thr = roc_curve(y, oof_mean)
CUTOFF = float(thr[np.argmax(tpr - fpr)])
print(f"  模型: {best_name}, Youden阈值: {CUTOFF:.3f}")

df_log = pd.read_excel(os.path.join(DATA_DIR, "清洗后数据_log2.xlsx"))
scaler_params = pd.DataFrame({
    "Protein": FINAL_FEATURES,
    "log2_mean": df_log[FINAL_FEATURES].mean().values,
    "log2_std": df_log[FINAL_FEATURES].std(ddof=0).values,
})
save_xls(scaler_params, "表5_标准化参数_Cohort3必须使用", index=False)

joblib.dump({"model": final_model,
             "model_name": best_name,
             "features": FINAL_FEATURES,
             "log2_mean": scaler_params["log2_mean"].values,
             "log2_std": scaler_params["log2_std"].values,
             "cutoff": CUTOFF},
            os.path.join(OUT_DIR, "最终模型_锁定.pkl"))
print("  最终模型已锁定保存: 最终模型_锁定.pkl")

pred_oof = (oof_mean >= CUTOFF).astype(int)
cm = confusion_matrix(y, pred_oof, labels=[0, 1])
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
ax.set_title(f"{best_name} OOF (training)", fontsize=8)
save_fig(fig, "图4_混淆矩阵_训练集")
save_xls(pd.DataFrame(cm, index=["True_R", "True_NR"],
                      columns=["Pred_R", "Pred_NR"]).reset_index(names=""),
         "图4_原始数据", index=False)

fig, ax = plt.subplots(figsize=(2.8, 2.8))
calib_rows = []
for mname in perf_summary.index[:4]:
    pm = oof_prob[mname].mean(axis=1)
    bins = np.unique(np.quantile(pm, np.linspace(0, 1, 9)))
    digit = np.clip(np.digitize(pm, bins[1:-1]), 0, len(bins) - 2)
    mp, op = [], []
    for b in range(len(bins) - 1):
        msk = digit == b
        if msk.sum() >= 5:
            mp.append(pm[msk].mean())
            op.append(y[msk].mean())
            calib_rows.append({"Model": mname, "MeanPredProb": mp[-1],
                               "ObservedFreq": op[-1], "N": int(msk.sum())})
    ax.plot(mp, op, "o-", color=MODEL_COLORS.get(mname, "#666666"),
            lw=0.9, ms=3.5, mec="black", mew=0.3,
            label=f"{mname} (Brier={perf_summary.loc[mname, 'Brier_mean']:.3f})")
ax.plot([0, 1], [0, 1], color="grey", lw=0.6, ls="--", label="Perfect")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed frequency (NR)")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_title("Calibration (training set)", fontsize=8)
ax.legend(frameon=False, loc="upper left", fontsize=5.5)
style_ax(ax)
save_fig(fig, "图5_校准曲线_训练集")
save_xls(pd.DataFrame(calib_rows), "图5_原始数据", index=False)

print("=" * 50)
print(f"Cohort 2 模型训练完成! 结果保存至: {OUT_DIR}")
print(f"锁定模型: {best_name} | 5特征: {FINAL_FEATURES} | 阈值: {CUTOFF:.3f}")
print("注意: 以上性能均为训练集性能, 最终验证请运行Cohort 3脚本")
