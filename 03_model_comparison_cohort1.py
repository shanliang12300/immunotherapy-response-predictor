import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                             recall_score, f1_score, confusion_matrix,
                             brier_score_loss)
from scipy import stats

warnings.filterwarnings("ignore")

INPUT_FILE = r"C:\Users\Administrator\Desktop\444\Step1_数据处理\清洗后数据_Zscore.xlsx"
OUT_DIR    = r"C:\Users\Administrator\Desktop\444\Step3_多模型比较"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES  = ["LAG3", "CCL23", "Gal-1", "CD70", "CXCL12"]
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
print("Step 3.1 读取数据")
df = pd.read_excel(INPUT_FILE)
missing_f = [f for f in FEATURES if f not in df.columns]
if missing_f:
    raise ValueError(f"特征不在数据中: {missing_f}")
X = df[FEATURES].values
y = df["label"].values
print(f"  样本数: {X.shape[0]}, 建模特征({len(FEATURES)}): {FEATURES}")
print(f"  分组: R={(y == 0).sum()}, NR={(y == 1).sum()}")

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
    print("  XGBoost 未安装, 跳过 (pip install xgboost 可启用)")
try:
    from lightgbm import LGBMClassifier
    models["LightGBM"] = LGBMClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RNG, n_jobs=-1, verbose=-1)
except ImportError:
    print("  LightGBM 未安装, 跳过 (pip install lightgbm 可启用)")

print("=" * 50)
print(f"Step 3.2 重复分层{N_SPLITS}折交叉验证 x {N_REPEATS}次")
rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                               random_state=RNG)

per_repeat = []
oof_prob = {m: np.zeros((len(y), N_REPEATS)) for m in models}

for mname, model in models.items():
    for rep in range(N_REPEATS):
        skf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=1,
                                      random_state=RNG + rep)
        prob = cross_val_predict(model, X, y, cv=skf,
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
    Brier_mean=("Brier", "mean"), Brier_sd=("Brier", "std"),
).sort_values("AUC_mean", ascending=False).round(4)
save_xls(perf, "表1_各模型每次重复性能", index=False)
save_xls(perf_summary.reset_index(), "表2_模型性能汇总", index=False)
print("\n模型性能汇总(按AUC排序):")
print(perf_summary[["AUC_mean", "AUC_sd", "ACC_mean",
                    "Sens_mean", "Spec_mean", "F1_mean"]].to_string())

best_model_name = perf_summary.index[0]
print(f"\n>>> 最佳模型: {best_model_name} "
      f"(AUC={perf_summary.loc[best_model_name, 'AUC_mean']:.3f})")

oof_mean = pd.DataFrame({"Sample": df.index + 1, "Group": df["Group"],
                         "label": y})
for m in models:
    oof_mean[f"Prob_{m}"] = oof_prob[m].mean(axis=1)
save_xls(oof_mean, "表3_OOF预测概率", index=False)

print("=" * 50)
print("Step 3.3 绘图")
fig, ax = plt.subplots(figsize=(3.0, 3.0))
mean_fpr = np.linspace(0, 1, 100)
roc_export = []
for mname in perf_summary.index:
    tprs = []
    for rep in range(N_REPEATS):
        fpr, tpr, _ = roc_curve(y, oof_prob[mname][:, rep])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
    mt = np.mean(tprs, axis=0)
    st = np.std(tprs, axis=0)
    auc_m = perf_summary.loc[mname, "AUC_mean"]
    auc_s = perf_summary.loc[mname, "AUC_sd"]
    ax.plot(mean_fpr, mt, color=MODEL_COLORS.get(mname, "#666666"),
            lw=1.0, label=f"{mname} ({auc_m:.3f}±{auc_s:.3f})")
    ax.fill_between(mean_fpr, mt - st, mt + st,
                    color=MODEL_COLORS.get(mname, "#666666"),
                    alpha=0.08, lw=0)
    for f, t in zip(mean_fpr, mt):
        roc_export.append({"Model": mname, "FPR": f, "MeanTPR": t})
ax.plot([0, 1], [0, 1], color="grey", lw=0.6, ls="--")
ax.set_xlabel("1 - Specificity")
ax.set_ylabel("Sensitivity")
ax.set_title("Repeated CV ROC", fontsize=8)
ax.legend(frameon=False, loc="lower right", fontsize=5.5)
style_ax(ax)
save_fig(fig, "图1_各模型ROC曲线")
save_xls(pd.DataFrame(roc_export), "图1_原始数据", index=False)

metrics = ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1"]
fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.2), sharey=False)
mlist = perf_summary.index.tolist()
xpos = np.arange(len(mlist))
for k, met in enumerate(metrics):
    ax = axes[k]
    means = perf.groupby("Model")[met].mean().loc[mlist]
    sds = perf.groupby("Model")[met].std().loc[mlist]
    cols = [MODEL_COLORS.get(m, "#666666") for m in mlist]
    ax.bar(xpos, means, yerr=sds, color=cols, edgecolor="black",
           lw=0.4, width=0.7, capsize=1.5,
           error_kw=dict(lw=0.6))
    ax.set_xticks(xpos)
    ax.set_xticklabels(mlist, rotation=45, ha="right")
    ax.set_title(met, fontsize=7)
    ax.set_ylim(0, 1.12)
    style_ax(ax)
axes[0].set_ylabel("Score")
fig.tight_layout()
save_fig(fig, "图2_模型性能比较")
save_xls(perf_summary.reset_index(), "图2_原始数据", index=False)

print("=" * 50)
print("Step 3.4 DeLong检验(最佳模型 vs 其他模型)")

def delong_test(y_true, prob1, prob2):
    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        T2 = np.empty(N)
        T2[J] = T
        return T2

    def fast_delong(preds_sorted, label_1_count):
        pos = preds_sorted[:label_1_count]
        neg = preds_sorted[label_1_count:]
        m, n = label_1_count, len(neg)
        tx = np.empty(m)
        ty = np.empty(n)
        for i in range(m):
            tx[i] = np.mean(pos[i] > neg) + 0.5 * np.mean(pos[i] == neg)
        for j in range(n):
            ty[j] = np.mean(neg[j] < pos) + 0.5 * np.mean(neg[j] == pos)
        return tx, ty

    order = (-y_true).argsort()
    n1 = int(y_true.sum())
    p1s = prob1[order]
    p2s = prob2[order]
    tx1, ty1 = fast_delong(p1s, n1)
    tx2, ty2 = fast_delong(p2s, n1)
    auc1, auc2 = tx1.mean(), tx2.mean()
    cov = np.cov(np.vstack([np.concatenate([tx1, ty1]),
                            np.concatenate([tx2, ty2])]))
    var = cov[0, 0] / n1 + cov[1, 1] / (len(y_true) - n1) \
        - 2 * cov[0, 1] / np.sqrt(n1 * (len(y_true) - n1))

    m_, n_ = n1, len(y_true) - n1
    v10_1 = tx1 - auc1
    v01_1 = ty1 - auc1
    v10_2 = tx2 - auc2
    v01_2 = ty2 - auc2
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    var = s10[0, 0] / m_ + s01[0, 0] / n_ + s10[1, 1] / m_ + s01[1, 1] / n_ \
        - 2 * (s10[0, 1] / m_ + s01[0, 1] / n_)
    if var <= 0:
        return auc1, auc2, np.nan
    z = (auc1 - auc2) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, p

delong_rows = []
pbest = oof_prob[best_model_name].mean(axis=1)
for mname in models:
    if mname == best_model_name:
        continue
    pm = oof_prob[mname].mean(axis=1)
    a1, a2, pv = delong_test(y, pbest, pm)
    delong_rows.append({"Model1": best_model_name, "Model2": mname,
                        "AUC_1": round(a1, 4), "AUC_2": round(a2, 4),
                        "DeLong_P": pv})
    print(f"  {best_model_name}(AUC={a1:.3f}) vs {mname}(AUC={a2:.3f}): "
          f"P={pv:.4f}" if not np.isnan(pv) else
          f"  {best_model_name} vs {mname}: P=NA")
save_xls(pd.DataFrame(delong_rows), "表4_DeLong检验", index=False)

prob_best = pbest
pred_best = (prob_best >= 0.5).astype(int)
cm = confusion_matrix(y, pred_best, labels=[0, 1])
fig, ax = plt.subplots(figsize=(2.2, 2.0))
im = ax.imshow(cm, cmap="Blues", vmin=0)
labs = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{labs[i][j]}={cm[i, j]}", ha="center", va="center",
                fontsize=8,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred R", "Pred NR"])
ax.set_yticklabels(["True R", "True NR"])
ax.set_title(f"{best_model_name} (OOF)", fontsize=8)
save_fig(fig, "图3_最佳模型混淆矩阵")
save_xls(pd.DataFrame(cm, index=["True_R", "True_NR"],
                      columns=["Pred_R", "Pred_NR"]).reset_index(names=""),
         "图3_原始数据", index=False)

fig, ax = plt.subplots(figsize=(2.8, 2.8))
calib_rows = []
for mname in perf_summary.index[:4]:
    pm = oof_prob[mname].mean(axis=1)
    bins = np.quantile(pm, np.linspace(0, 1, 7))
    bins = np.unique(bins)
    digit = np.clip(np.digitize(pm, bins[1:-1]), 0, len(bins) - 2)
    mp, op = [], []
    for b in range(len(bins) - 1):
        msk = digit == b
        if msk.sum() >= 3:
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
ax.legend(frameon=False, loc="upper left", fontsize=5.5)
style_ax(ax)
save_fig(fig, "图4_校准曲线")
save_xls(pd.DataFrame(calib_rows), "图4_原始数据", index=False)

fig, ax = plt.subplots(figsize=(3.2, 2.8))
thresholds = np.linspace(0.01, 0.99, 99)
n = len(y)
prev = y.mean()
dca_rows = []
for mname in perf_summary.index[:4]:
    pm = oof_prob[mname].mean(axis=1)
    nb = []
    for pt in thresholds:
        pred = pm >= pt
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        nb.append(tp / n - fp / n * (pt / (1 - pt)))
        dca_rows.append({"Model": mname, "Threshold": pt,
                         "NetBenefit": nb[-1]})
    ax.plot(thresholds, nb, color=MODEL_COLORS.get(mname, "#666666"),
            lw=1.0, label=mname)
nb_all = [prev - (1 - prev) * (pt / (1 - pt)) for pt in thresholds]
ax.plot(thresholds, nb_all, color="grey", lw=0.8, ls="--", label="Treat all")
ax.axhline(0, color="black", lw=0.8, ls=":", label="Treat none")
for pt, v in zip(thresholds, nb_all):
    dca_rows.append({"Model": "Treat all", "Threshold": pt, "NetBenefit": v})
    dca_rows.append({"Model": "Treat none", "Threshold": pt, "NetBenefit": 0})
ax.set_xlabel("Threshold probability")
ax.set_ylabel("Net benefit")
ax.set_ylim(-0.15, max(prev + 0.1, 0.6))
ax.legend(frameon=False, loc="upper right", fontsize=5.5)
style_ax(ax)
save_fig(fig, "图5_DCA决策曲线")
save_xls(pd.DataFrame(dca_rows), "图5_原始数据", index=False)

print("=" * 50)
print("Step 3.5 保存最佳模型(全数据拟合, 供第四步SHAP)")
import joblib
final_model = models[best_model_name]
final_model.fit(X, y)
joblib.dump(final_model, os.path.join(OUT_DIR, f"最佳模型_{best_model_name}.pkl"))
joblib.dump(FEATURES, os.path.join(OUT_DIR, "建模特征.pkl"))
print(f"  最佳模型 {best_model_name} 已保存为pkl")

if best_model_name == "LR":
    coef_tbl = pd.DataFrame({"Protein": FEATURES,
                             "Coefficient": final_model.coef_[0]})
    save_xls(coef_tbl, "表5_最佳模型系数", index=False)
elif hasattr(final_model, "feature_importances_"):
    fi_tbl = pd.DataFrame({"Protein": FEATURES,
                           "Importance": final_model.feature_importances_})
    save_xls(fi_tbl.sort_values("Importance", ascending=False),
             "表5_最佳模型特征重要性", index=False)

print("=" * 50)
print(f"第三步完成! 所有结果已保存至: {OUT_DIR}")
print(f"最佳模型: {best_model_name} -> 用于第四步SHAP解释")
