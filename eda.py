"""Exploratory data analysis and statistical tests. Run: python eda.py"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save figures to files, don't open windows
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils import load_data, NUM_COLS, CAT_COLS

sns.set_theme(style="whitegrid")
df = load_data()
log = []


def say(text=""):
    """Print a line and keep it for outputs/eda_summary.txt."""
    print(text)
    log.append(text)


# ---------- 1. Summary statistics ----------
say(f"Shape: {df.shape}")
say(f"Missing values:\n{df.isna().sum().to_string()}")
say(f"Duplicate rows: {df.duplicated().sum()}")
say(f"\nDescriptive statistics:\n{df.describe().T.round(2).to_string()}")

# Approval rate with a 95% confidence interval (normal approximation)
n, p = len(df), df["loan_status"].mean()
se = np.sqrt(p * (1 - p) / n)
say(f"\nApproval rate = {p:.3f}, 95% CI = ({p - 1.96 * se:.3f}, {p + 1.96 * se:.3f})")

# ---------- 2. Figures ----------
labels = df["loan_status"].map({1: "Approved", 0: "Rejected"})

plt.figure(figsize=(5, 4))
labels.value_counts().plot.bar(color=["#4c78a8", "#e45756"])
plt.title("Class distribution")
plt.tight_layout()
plt.savefig("outputs/class_distribution.png", dpi=150)
plt.close()

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for ax, col in zip(axes.ravel(), NUM_COLS):
    sns.histplot(data=df, x=col, hue="loan_status", kde=True, ax=ax,
                 palette={0: "#e45756", 1: "#4c78a8"})
plt.tight_layout()
plt.savefig("outputs/distributions.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x="loan_status", y="cibil_score", hue="loan_status",
            palette=["#e45756", "#4c78a8"], legend=False)
plt.xticks([0, 1], ["Rejected", "Approved"])
plt.title("CIBIL score vs loan status")
plt.tight_layout()
plt.savefig("outputs/cibil_boxplot.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig("outputs/correlation.png", dpi=150)
plt.close()

# ---------- 3. Statistical tests (alpha = 0.05) ----------
say("\n--- Statistical tests ---")

# Chi-square: are two categorical variables independent?
for col in CAT_COLS:
    chi2, pv, dof, _ = stats.chi2_contingency(pd.crosstab(df[col], df["loan_status"]))
    verdict = "associated with approval" if pv < 0.05 else "no significant association"
    say(f"Chi-square {col} vs status: chi2={chi2:.3f}, p={pv:.4f} -> {verdict}")

# Mann-Whitney U: do numeric features differ between approved and rejected?
for col in NUM_COLS:
    a = df.loc[df.loan_status == 1, col]
    b = df.loc[df.loan_status == 0, col]
    u, pv = stats.mannwhitneyu(a, b)
    say(f"Mann-Whitney U {col}: p={pv:.4g} "
        f"(median approved={a.median():.0f}, rejected={b.median():.0f})")

# Welch t-test + 95% CI for the difference in mean CIBIL score
a = df.loc[df.loan_status == 1, "cibil_score"]
b = df.loc[df.loan_status == 0, "cibil_score"]
t, pv = stats.ttest_ind(a, b, equal_var=False)
diff = a.mean() - b.mean()
se = np.sqrt(a.var() / len(a) + b.var() / len(b))
say(f"Welch t-test CIBIL: t={t:.2f}, p={pv:.3g}, mean difference={diff:.1f}, "
    f"95% CI=({diff - 1.96 * se:.1f}, {diff + 1.96 * se:.1f})")

with open("outputs/eda_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log))

print("\nSaved figures and eda_summary.txt in outputs/")