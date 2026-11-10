"""Loan Approval Predictor - Streamlit app. Run: streamlit run app.py"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from utils import add_features, load_data, SEED, NUM_COLS, CAT_COLS  # noqa: F401

st.set_page_config(page_title="Loan Approval Studio", page_icon="🏦", layout="wide")

# =============================================================== styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@500;600;700;800&display=swap');
.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp button, .stApp td, .stApp th {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .hero-title, .section-title, .g-num, .stat .v, .kpi .v {
    font-family: 'Poppins', 'Inter', sans-serif;
}
.stApp {
    background:
        radial-gradient(1100px 560px at 8% -8%, #1d2160 0%, transparent 60%),
        radial-gradient(900px 520px at 100% 0%, #43207a 0%, transparent 55%),
        #070b1a;
    background-attachment: fixed;
}
.stApp::before, .stApp::after {
    content: ""; position: fixed; width: 520px; height: 520px; border-radius: 50%;
    filter: blur(95px); opacity: .32; z-index: 0; pointer-events: none;
}
.stApp::before { background: #6d28d9; top: -140px; left: -120px; animation: float1 18s ease-in-out infinite; }
.stApp::after  { background: #0891b2; bottom: -180px; right: -140px; animation: float2 22s ease-in-out infinite; }
@keyframes float1 { 0%,100% { transform: translate(0,0); } 50% { transform: translate(170px,130px); } }
@keyframes float2 { 0%,100% { transform: translate(0,0); } 50% { transform: translate(-190px,-110px); } }
@keyframes fadeUp { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pop    { from { opacity: 0; transform: scale(.88); } to { opacity: 1; transform: scale(1); } }
@keyframes grow   { from { width: 0; } }
@keyframes shine  { 0% { transform: translateX(-100%); } 60%,100% { transform: translateX(100%); } }
@keyframes gradmove { to { background-position: 300% 0; } }
@keyframes gfill  { to { stroke-dashoffset: var(--off); } }
@keyframes pulse  { 0%,100% { box-shadow: 0 0 0 0 var(--glow); } 50% { box-shadow: 0 0 0 12px rgba(0,0,0,0); } }

.block-container { position: relative; z-index: 1; padding-top: 1.4rem; max-width: 1240px; }
header[data-testid="stHeader"] { background: transparent; }

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.10) !important;
    border-radius: 20px !important; backdrop-filter: blur(14px);
    box-shadow: 0 10px 30px rgba(0,0,0,.25); transition: border-color .3s, box-shadow .3s;
    animation: fadeUp .7s ease both;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(139,92,246,.55) !important; box-shadow: 0 14px 40px rgba(139,92,246,.22);
}
button[data-baseweb="tab"] { font-weight: 600; font-size: 1rem; }
img { border-radius: 14px; }

.hero {
    position: relative; overflow: hidden; border-radius: 26px; padding: 40px 46px; margin-bottom: 24px;
    background: linear-gradient(135deg, rgba(99,102,241,.38), rgba(139,92,246,.26) 45%, rgba(34,211,238,.22));
    border: 1px solid rgba(255,255,255,.15); backdrop-filter: blur(16px);
    box-shadow: 0 20px 60px rgba(76,29,149,.35); animation: fadeUp .8s ease both;
}
.hero::after {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background: linear-gradient(110deg, transparent 30%, rgba(255,255,255,.11) 50%, transparent 70%);
    transform: translateX(-100%); animation: shine 6s ease-in-out infinite;
}
.hero-title {
    font-size: 2.9rem; font-weight: 800; line-height: 1.15; margin: 0 0 10px 0;
    background: linear-gradient(90deg, #ffffff, #c4b5fd, #67e8f9, #ffffff); background-size: 300% 100%;
    -webkit-background-clip: text; background-clip: text; color: transparent;
    animation: gradmove 9s linear infinite;
}
.hero-sub { color: #c7cdf0; font-size: 1.05rem; max-width: 760px; margin: 0; }
.stats { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 24px; }
.stat {
    background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.15); border-radius: 16px;
    padding: 12px 20px; min-width: 130px; animation: pop .7s ease both;
}
.stat .v { font-size: 1.55rem; font-weight: 700; color: #fff; }
.stat .l { font-size: .72rem; letter-spacing: .1em; color: #aab2e0; font-weight: 600; }

.section-title { font-weight: 700; font-size: 1.05rem; color: #c4b5fd; margin: 0 0 6px 0; }
.muted { color: #94a3c8; font-size: .88rem; }

.gauge-card {
    text-align: center; border-radius: 24px; padding: 20px 18px 22px;
    background: rgba(255,255,255,.055); border: 1px solid rgba(255,255,255,.13);
    box-shadow: 0 18px 50px var(--glow); animation: fadeUp .6s ease both;
}
.gauge-fill { stroke-dasharray: 100; stroke-dashoffset: 100; animation: gfill 1.4s cubic-bezier(.22,1,.36,1) .15s forwards; }
.g-num { font-size: 34px; font-weight: 800; fill: #ffffff; }
.g-lbl { font-size: 9.5px; letter-spacing: 2px; fill: #94a3c8; font-weight: 600; }
.badge {
    display: inline-block; color: #fff; font-weight: 700; letter-spacing: .06em; font-size: .92rem;
    padding: 8px 22px; border-radius: 999px; background: var(--accent); animation: pulse 2.4s infinite;
}
.mrow {
    display: grid; grid-template-columns: 1fr auto auto; gap: 6px 10px; align-items: center;
    margin: 14px 0; animation: fadeUp .6s ease both;
}
.mname { grid-column: 1; grid-row: 1; font-weight: 600; font-size: .9rem; color: #e8ecf8; }
.tag { font-size: .68rem; color: #fbbf24; margin-left: 6px; }
.mtrack { grid-column: 1 / -1; grid-row: 2; height: 11px; border-radius: 999px; background: rgba(255,255,255,.09); overflow: hidden; }
.mfill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--c1), var(--c2)); animation: grow 1.2s cubic-bezier(.22,1,.36,1) both; }
.mval { grid-column: 2; grid-row: 1; font-weight: 700; font-size: .95rem; text-align: right; color: #fff; }
.chip { grid-column: 3; grid-row: 1; min-width: 74px; font-size: .72rem; font-weight: 700; padding: 4px 0; border-radius: 999px; text-align: center; }
.chip.ok  { background: rgba(52,211,153,.18); color: #6ee7b7; border: 1px solid rgba(52,211,153,.4); }
.chip.bad { background: rgba(251,113,133,.16); color: #fda4af; border: 1px solid rgba(251,113,133,.4); }

.winner {
    border-radius: 22px; padding: 22px 28px; margin: 6px 0 14px 0; animation: pop .7s ease both;
    background: linear-gradient(120deg, rgba(251,191,36,.20), rgba(244,114,182,.16), rgba(139,92,246,.20));
    border: 1px solid rgba(251,191,36,.45); box-shadow: 0 14px 44px rgba(251,191,36,.14);
}
.winner .t { font-family: 'Poppins', sans-serif; font-size: 1.5rem; font-weight: 700; color: #fff; }
.winner .s { color: #e2d8c0; font-size: .92rem; }

.kpi {
    border-radius: 18px; padding: 18px 20px; background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.12); border-left: 5px solid var(--c1); animation: pop .6s ease both;
}
.kpi .l { color: #94a3c8; font-size: .76rem; font-weight: 700; letter-spacing: .1em; }
.kpi .v { font-size: 2rem; font-weight: 700; color: #fff; }

.cm { display: grid; grid-template-columns: 110px 1fr 1fr; gap: 8px; align-items: stretch; }
.cm-h { color: #94a3c8; font-size: .78rem; font-weight: 600; display: flex; align-items: center; justify-content: center; text-align: center; }
.cm-c { border-radius: 14px; padding: 18px 8px; text-align: center; animation: pop .7s ease both; }
.cm-c .n { font-family: 'Poppins', sans-serif; font-size: 1.9rem; font-weight: 700; color: #fff; }
.cm-c .k { font-size: .72rem; color: #cbd5f5; }
.cm-c.good { background: rgba(52,211,153,.20); border: 1px solid rgba(52,211,153,.45); }
.cm-c.bad  { background: rgba(251,113,133,.17); border: 1px solid rgba(251,113,133,.45); }

.step { display: flex; gap: 14px; margin: 14px 0; animation: fadeUp .6s ease both; }
.step .n {
    flex: 0 0 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-weight: 700; color: #fff; background: linear-gradient(135deg, #8b5cf6, #22d3ee);
}
.footer { text-align: center; color: #7e89b5; font-size: .85rem; margin: 34px 0 10px 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =============================================================== data + models
best_name = json.loads(Path("outputs/best_model.json").read_text())["best_model"] \
    if Path("outputs/best_model.json").exists() else None
if best_name is None or not Path("models/best_model.joblib").exists():
    st.error("Model not found. Run `python train.py` first.")
    st.stop()

COLORS = {"Baseline (majority class)": ("#64748b", "#94a3b8"),
          "Logistic Regression": ("#22d3ee", "#38bdf8"),
          "Random Forest": ("#8b5cf6", "#c084fc"),
          "Gradient Boosting": ("#f472b6", "#fb7185")}


def colors(name):
    for key, val in COLORS.items():
        if key in name:
            return val
    return ("#8b5cf6", "#22d3ee")


@st.cache_resource
def load_models():
    path = Path("models/all_models.joblib")
    if path.exists():
        return joblib.load(path)
    return {best_name: joblib.load("models/best_model.joblib")}


@st.cache_data
def load_results():
    p = Path("outputs/model_results.csv")
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def get_test_set():
    """Same split as train.py, so these rows were never used for training."""
    df = load_data()
    X, y = df.drop(columns="loan_status"), df["loan_status"]
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    return X_te, y_te


@st.cache_data(show_spinner="Computing feature importance...")
def feature_importance(model_name, _model):
    X_te, y_te = get_test_set()
    r = permutation_importance(_model, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=SEED)
    return pd.Series(r.importances_mean, index=X_te.columns).sort_values(ascending=False)


def evaluate(models, X, y):
    rows = []
    for name, m in models.items():
        pred, proba = m.predict(X), m.predict_proba(X)[:, 1]
        try:
            auc = roc_auc_score(y, proba)
        except ValueError:
            auc = np.nan
        rows.append({"Model": name, "Accuracy": accuracy_score(y, pred),
                     "Precision": precision_score(y, pred, zero_division=0),
                     "Recall": recall_score(y, pred, zero_division=0),
                     "F1": f1_score(y, pred, zero_division=0), "ROC-AUC": auc})
    return pd.DataFrame(rows)


models = load_models()
results = load_results()
best_row = None
if results is not None and (results["model"] == best_name).any():
    best_row = results[results["model"] == best_name].iloc[0]

# =============================================================== html helpers
def mrow(name, value, text, delay=0.0, chip=None, tag=""):
    c1, c2 = colors(name)
    chip_html = f'<div class="chip {chip[0]}">{chip[1]}</div>' if chip else "<div></div>"
    return (f'<div class="mrow" style="animation-delay:{delay:.2f}s">'
            f'<div class="mname">{name}{tag}</div>'
            f'<div class="mtrack"><div class="mfill" style="width:{max(value, 0) * 100:.1f}%;'
            f'--c1:{c1};--c2:{c2};animation-delay:{delay:.2f}s"></div></div>'
            f'<div class="mval">{text}</div>{chip_html}</div>')


def gauge(prob, threshold, approved):
    import math
    accent, glow = ("#34d399", "rgba(52,211,153,.28)") if approved else ("#fb7185", "rgba(251,113,133,.28)")
    c2 = "#22d3ee" if approved else "#f472b6"
    off = 100 - prob * 100
    th = math.pi * (1 - threshold)
    x1, y1 = 110 + 78 * math.cos(th), 115 - 78 * math.sin(th)
    x2, y2 = 110 + 103 * math.cos(th), 115 - 103 * math.sin(th)
    verdict = "LIKELY APPROVED" if approved else "LIKELY REJECTED"
    return (f'<div class="gauge-card" style="--accent:{accent};--glow:{glow}">'
            f'<svg viewBox="0 0 220 140" width="100%">'
            f'<defs><linearGradient id="gg" x1="0" x2="1"><stop offset="0%" stop-color="{accent}"/>'
            f'<stop offset="100%" stop-color="{c2}"/></linearGradient></defs>'
            f'<path d="M20 115 A90 90 0 0 1 200 115" pathLength="100" fill="none" '
            f'stroke="rgba(255,255,255,.10)" stroke-width="14" stroke-linecap="round"/>'
            f'<path class="gauge-fill" d="M20 115 A90 90 0 0 1 200 115" pathLength="100" fill="none" '
            f'stroke="url(#gg)" stroke-width="14" stroke-linecap="round" style="--off:{off:.2f}"/>'
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#ffffff" '
            f'stroke-width="2.5" stroke-linecap="round"/>'
            f'<text x="110" y="92" text-anchor="middle" class="g-num">{prob:.1%}</text>'
            f'<text x="110" y="112" text-anchor="middle" class="g-lbl">APPROVAL PROBABILITY</text>'
            f'<text x="20" y="134" text-anchor="middle" class="g-lbl">0%</text>'
            f'<text x="200" y="134" text-anchor="middle" class="g-lbl">100%</text></svg>'
            f'<span class="badge">{verdict}</span>'
            f'<div class="muted" style="margin-top:10px">white tick = threshold {threshold:.0%}</div></div>')


# =============================================================== hero
stats = [(str(len(models)), "MODELS COMPARED"), ("11", "INPUT FEATURES")]
if best_row is not None:
    stats = [(f'{best_row["accuracy"]:.1%}', "BEST TEST ACCURACY"), (f'{best_row["f1"]:.3f}', "BEST F1-SCORE"),
             (f'{best_row["roc_auc"]:.3f}', "BEST ROC-AUC")] + stats
stat_html = "".join(f'<div class="stat" style="animation-delay:{0.15 * i:.2f}s"><div class="v">{v}</div>'
                    f'<div class="l">{l}</div></div>' for i, (v, l) in enumerate(stats))
st.markdown(
    '<div class="hero"><div class="hero-title">Loan Approval Studio</div>'
    '<p class="hero-sub">Enter an applicant, watch four machine-learning models decide, and compare '
    'their accuracy on real held-out data.</p>'
    f'<div class="stats">{stat_html}</div></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "⚔️ Model Arena", "📊 Performance", "ℹ️ About"])

# =============================================================== tab 1: predict
with tab1:
    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="section-title">👤 Applicant</div>', unsafe_allow_html=True)
            a1, a2 = st.columns(2)
            dependents = a1.number_input("Number of dependents", 0, 5, 2)
            education = a2.selectbox("Education", ["Graduate", "Not Graduate"])
            self_emp = a1.selectbox("Self employed", ["No", "Yes"])
            cibil = a2.slider("CIBIL score", 300, 900, 700)
        with st.container(border=True):
            st.markdown('<div class="section-title">💰 Loan details</div>', unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            income = b1.number_input("Annual income", 200000, 10000000, 5000000, step=100000)
            loan_amt = b2.number_input("Loan amount", 300000, 40000000, 15000000, step=100000)
            term = st.slider("Loan term (years)", 2, 20, 10)
        with st.container(border=True):
            st.markdown('<div class="section-title">🏠 Assets</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            res_a = c1.number_input("Residential assets", 0, 30000000, 5000000, step=100000)
            com_a = c2.number_input("Commercial assets", 0, 20000000, 3000000, step=100000)
            lux_a = c1.number_input("Luxury assets", 0, 40000000, 10000000, step=100000)
            bank_a = c2.number_input("Bank assets", 0, 15000000, 5000000, step=100000)

    row = pd.DataFrame([{
        "no_of_dependents": dependents, "education": education, "self_employed": self_emp,
        "income_annum": income, "loan_amount": loan_amt, "loan_term": term, "cibil_score": cibil,
        "residential_assets_value": res_a, "commercial_assets_value": com_a,
        "luxury_assets_value": lux_a, "bank_asset_value": bank_a}])
    probs = {name: float(m.predict_proba(row)[0, 1]) for name, m in models.items()}

    with right:
        # The baseline is a no-skill benchmark, so it can't be used to make a decision
        names = [n for n in models if not n.startswith("Baseline")]
        chosen = st.selectbox("Decision model", names, index=names.index(best_name) if best_name in names else 0)
        threshold = st.slider("Approval threshold", 0.10, 0.90, 0.50, 0.05,
                              help="Raise it to be stricter: fewer risky approvals, more rejections.")
        prob = probs[chosen]
        st.markdown(gauge(prob, threshold, prob >= threshold), unsafe_allow_html=True)
        st.write("")
        with st.container(border=True):
            st.markdown('<div class="section-title">🤖 What every model says</div>', unsafe_allow_html=True)
            rows_html = ""
            for i, (name, p) in enumerate(probs.items()):
                ok = p >= threshold
                rows_html += mrow(name, p, f"{p:.0%}", delay=0.08 * i,
                                  chip=("ok", "Approve") if ok else ("bad", "Reject"),
                                  tag=" ★" if name == chosen else (" · no-skill" if name.startswith("Baseline") else ""))
            st.markdown(rows_html, unsafe_allow_html=True)
            real = [p for n, p in probs.items() if not n.startswith("Baseline")]
            if real:
                yes = sum(p >= threshold for p in real)
                st.caption(f"Consensus: {yes} of {len(real)} trained models approve this applicant. "
                           "The baseline always predicts the majority class and is shown only for reference.")
        with st.container(border=True):
            st.markdown('<div class="section-title">Derived features</div>', unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Total assets", f"{res_a + com_a + lux_a + bank_a:,}")
            m2.metric("Loan-to-income", f"{loan_amt / income:.2f}")

# =============================================================== tab 2: arena
with tab2:
    st.markdown("### ⚔️ Model Arena")
    st.markdown('<p class="muted">Run every model on the same applicants and let the data decide which one '
                'is the most accurate. Use rows the models never trained on, or upload your own CSV.</p>',
                unsafe_allow_html=True)
    ctl, out = st.columns([1, 2], gap="large")
    X_arena = y_arena = None
    with ctl:
        with st.container(border=True):
            st.markdown('<div class="section-title">Setup</div>', unsafe_allow_html=True)
            source = st.radio("Data source", ["Held-out test sample", "Upload my own CSV"])
            if source == "Held-out test sample":
                X_te, y_te = get_test_set()
                n = st.slider("Number of applicants", 30, len(X_te), min(300, len(X_te)), step=10)
                if "seed" not in st.session_state:
                    st.session_state.seed = 42
                if st.button("🎲 Draw a new random sample"):
                    st.session_state.seed = int(np.random.randint(0, 10000))
                st.caption(f"Sample seed: {st.session_state.seed}")
                idx = X_te.sample(n, random_state=st.session_state.seed).index
                X_arena, y_arena = X_te.loc[idx], y_te.loc[idx]
            else:
                up = st.file_uploader("CSV in the same format as the Kaggle file (with loan_status)", type="csv")
                if up is not None:
                    try:
                        d = load_data(up)
                        missing = [c for c in NUM_COLS + CAT_COLS + ["loan_status"] if c not in d.columns]
                        if missing:
                            st.error(f"Missing columns: {', '.join(missing)}")
                        else:
                            X_arena, y_arena = d.drop(columns="loan_status"), d["loan_status"]
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Could not read the file: {exc}")
            metric = st.selectbox("Rank models by", ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC"])

    with out:
        if X_arena is None:
            st.info("Upload a CSV to start the arena.")
        else:
            res = evaluate(models, X_arena, y_arena).sort_values(metric, ascending=False).reset_index(drop=True)
            top = res.iloc[0]
            tied = res[np.isclose(res[metric], top[metric])]
            if len(tied) > 1:
                title = "🤝 Tie: " + " & ".join(tied["Model"])
            else:
                title = f"🏆 {top['Model']}"
            st.markdown(
                f'<div class="winner"><div class="t">{title}</div>'
                f'<div class="s">Best {metric.lower()} on {len(X_arena)} applicants: '
                f'<b>{top[metric]:.2%}</b>. Small samples are noisy, so redraw a few times.</div></div>',
                unsafe_allow_html=True)
            medals = ["🥇", "🥈", "🥉", "4️⃣"]
            html = "".join(mrow(f"{medals[min(i, 3)]} {r['Model']}", 0 if np.isnan(r[metric]) else r[metric],
                                f"{r[metric]:.1%}", delay=0.1 * i) for i, r in res.iterrows())
            with st.container(border=True):
                st.markdown(f'<div class="section-title">Leaderboard by {metric}</div>' + html,
                            unsafe_allow_html=True)
            fmt = st.column_config.NumberColumn
            st.dataframe(res, hide_index=True, column_config={
                c: fmt(c, format="%.4f") for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]})

# =============================================================== tab 3: performance
with tab3:
    names = list(models.keys())
    sel = st.selectbox("Choose a model to inspect", names, index=names.index(best_name) if best_name in names else 0)
    c1_, c2_ = colors(sel)
    if results is not None and (results["model"] == sel).any():
        r = results[results["model"] == sel].iloc[0]
        k = st.columns(4)
        for i, (col, (label, key)) in enumerate(zip(k, [("ACCURACY", "accuracy"), ("PRECISION", "precision"),
                                                         ("RECALL", "recall"), ("F1-SCORE", "f1")])):
            col.markdown(f'<div class="kpi" style="--c1:{c1_};animation-delay:{0.1 * i:.1f}s">'
                         f'<div class="l">{label}</div><div class="v">{r[key]:.3f}</div></div>',
                         unsafe_allow_html=True)
        st.write("")

    X_te, y_te = get_test_set()
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="section-title">Confusion matrix (held-out test set)</div>', unsafe_allow_html=True)
            tn, fp, fn, tp = confusion_matrix(y_te, models[sel].predict(X_te), labels=[0, 1]).ravel()
            st.markdown(
                '<div class="cm"><div class="cm-h"></div><div class="cm-h">Predicted<br>Rejected</div>'
                '<div class="cm-h">Predicted<br>Approved</div>'
                f'<div class="cm-h">Actual<br>Rejected</div><div class="cm-c good"><div class="n">{tn}</div>'
                '<div class="k">correct rejections</div></div>'
                f'<div class="cm-c bad"><div class="n">{fp}</div><div class="k">false approvals</div></div>'
                f'<div class="cm-h">Actual<br>Approved</div><div class="cm-c bad"><div class="n">{fn}</div>'
                '<div class="k">false rejections</div></div>'
                f'<div class="cm-c good"><div class="n">{tp}</div><div class="k">correct approvals</div></div></div>',
                unsafe_allow_html=True)
            st.caption("False approvals are the costly mistake for a bank.")
    with right:
        with st.container(border=True):
            st.markdown('<div class="section-title">What drives the decision (permutation importance)</div>',
                        unsafe_allow_html=True)
            imp = feature_importance(sel, models[sel])
            if imp.max() <= 0:
                st.caption("The baseline ignores all features, so every importance is zero.")
            else:
                top = imp.head(7)
                st.markdown("".join(mrow(n.replace("_", " ").title(), v / top.max(), f"{v:.3f}", delay=0.07 * i)
                                    for i, (n, v) in enumerate(top.items())), unsafe_allow_html=True)
                st.caption("Drop in ROC-AUC when the feature is shuffled. Bigger means more important.")

    left, right = st.columns([2, 3], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="section-title">ROC curves</div>', unsafe_allow_html=True)
            if Path("outputs/roc_curves.png").exists():
                st.image("outputs/roc_curves.png")
            else:
                st.caption("Run train.py to generate this figure.")
    with right:
        with st.container(border=True):
            st.markdown('<div class="section-title">Model comparison (held-out test set)</div>', unsafe_allow_html=True)
            if results is not None:
                fmt = st.column_config.NumberColumn
                st.dataframe(results.drop(columns="best_params", errors="ignore"), hide_index=True, column_config={
                    "model": st.column_config.TextColumn("Model"), "cv_f1": fmt("CV F1", format="%.4f"),
                    "accuracy": fmt("Accuracy", format="%.4f"), "precision": fmt("Precision", format="%.4f"),
                    "recall": fmt("Recall", format="%.4f"), "f1": fmt("F1", format="%.4f"),
                    "roc_auc": fmt("ROC-AUC", format="%.4f")})

# =============================================================== tab 4: about
with tab4:
    a, b = st.columns(2, gap="large")
    steps = [("Data", "4,000+ loan applications (Kaggle), cleaned and encoded"),
             ("Features", "total assets and loan-to-income added"),
             ("Pipeline", "scaling and one-hot encoding fitted inside cross-validation, so no leakage"),
             ("Models", "baseline, Logistic Regression, Random Forest, Gradient Boosting, tuned with GridSearchCV"),
             ("Selection", "best model chosen by cross-validated F1, reported on a held-out test set")]
    with a:
        with st.container(border=True):
            st.markdown('<div class="section-title">How it works</div>' + "".join(
                f'<div class="step" style="animation-delay:{0.1 * i:.1f}s"><div class="n">{i + 1}</div>'
                f'<div><b>{t}</b><br><span class="muted">{d}</span></div></div>'
                for i, (t, d) in enumerate(steps)), unsafe_allow_html=True)
    with b:
        with st.container(border=True):
            st.markdown("#### Limitations")
            st.markdown("- Educational project, **not** a real credit decision tool\n"
                        "- Small, very clean dataset; CIBIL score dominates the outcome\n"
                        "- No fairness audit and no real credit-bureau or repayment history\n"
                        "- False approvals are the costly error; use the threshold slider to be stricter")
    st.markdown('<div class="footer">Built by Namandeep Kaur · Python · scikit-learn · Streamlit · '
                'github.com/Namandeepk21/Loan_approval_prediction</div>', unsafe_allow_html=True)
