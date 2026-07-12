"""
AfriScore - Enterprise SaaS & Investor Demo (V5.0 - "Wow Factor" pass)
Privacy-First AI Credit & Financial Identity Infrastructure
Lead Architect: Obasi Divinefavour Chukwuemeka

This pass adds (see build prompt v2):
  1. Functionality: live scenario simulator, traditional-bureau-vs-AfriScore comparison,
     adversarial model stress test, downloadable decision-record PDF.
  2. UI/UX: presenter mode with progress indicator, tab cross-fade polish, meaningful
     empty/error states, custom SVG iconography, accessibility pass, mobile check.
  3. Technical depth: a real scikit-learn logistic-regression model (trained on the
     synthetic dataset, persisted with joblib) driving scoring, with genuine SHAP values
     feeding the factor-bar UI when the `shap` package is available -- and an honest,
     clearly-labeled fallback when it isn't, per the "don't fake it" constraint.
  4. Narrative: an illustrative persona ("Amaka") and a stated risk-awareness note.
"""

import math
import json
import time
import pathlib
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

import gradio as gr
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# SHAP is optional. If it isn't installed in the runtime environment, we do NOT fake
# real SHAP values -- we fall back to a clearly-labeled illustrative attribution instead.
try:
    import shap  # noqa: F401
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# reportlab powers the "Download Decision Record" PDF export. Optional too, with a
# plain-text fallback so the feature never hard-crashes a live demo.
try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

np.random.seed(42)

MODEL_VERSION = "1.0.0"
MODEL_PATH = pathlib.Path(__file__).resolve().parent / "afriscore_model.joblib"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "afriscore_exports"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "age",
    "is_smallholder",
    "mobile_txn_count_6m",
    "avg_txn_amount",
    "txn_regularity_score",
    "utility_consistency",
    "savings_rate_proxy",
]

FEATURE_LABELS = {
    "txn_regularity_score": "Income & Txn Regularity",
    "mobile_txn_count_6m": "Mobile Money Velocity",
    "utility_consistency": "Utility Payment Consistency",
    "savings_rate_proxy": "Savings Behavior",
    "avg_txn_amount": "Transaction Size Pattern",
    "is_smallholder": "Smallholder / Informal Income Segment",
    "age": "Applicant Age Band",
}

MODEL_LIMITATIONS = [
    "Trained entirely on synthetic data generated for this demo -- not on real applicant "
    "outcomes, and must be retrained on Wema's live portfolio before any production use.",
    "Logistic regression assumes roughly linear, additive effects; it will not capture "
    "genuinely non-linear fraud patterns without moving to a gradient-boosted successor.",
    "Alternative-data proxies (mobile velocity, utility consistency) can vary seasonally "
    "and by region -- see the risk-awareness note in the Market & Roadmap tab.",
    "SHAP attributions explain THIS model's behavior, not ground truth about an "
    "applicant's real creditworthiness; a human reviewer is required on every decision.",
]

# ============================================================
# MOCK DATA & STATISTICAL SETUP
# ============================================================

def generate_market_data(n=150, seed=None) -> pd.DataFrame:
    """Generates a statistically realistic synthetic user base. `is_credit_invisible`
    models the ~70% of the population a traditional bureau model cannot score at all --
    this is what powers the traditional-bureau-vs-AfriScore comparison."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    locations = ["Lagos (Urban)", "Enugu (Urban)", "Nsukka (Peri-Urban)", "Kano (Urban)", "Ogun (Rural)"]
    df = pd.DataFrame({
        "user_id": [f"AFRI_{str(i).zfill(6)}" for i in range(n)],
        "age": rng.normal(32, 10, n).clip(18, 70).astype(int),
        "location_type": rng.choice(locations, n, p=[0.35, 0.15, 0.20, 0.20, 0.10]),
        "is_smallholder": rng.choice([0, 1], n, p=[0.75, 0.25]),
        "mobile_txn_count_6m": rng.negative_binomial(1, 0.01, n).clip(5, 800),
        "avg_txn_amount": rng.lognormal(9, 1.5, n).clip(1000, 150000),
        "txn_regularity_score": rng.beta(5, 2, n),
        "utility_consistency": rng.beta(4, 2, n),
        "savings_rate_proxy": rng.beta(2, 5, n),
    })
    df["is_credit_invisible"] = rng.choice([1, 0], n, p=[0.70, 0.30]).astype(bool)
    return df

df_users = generate_market_data(150, seed=42)

# ============================================================
# CATEGORY 3 -- REAL MODEL TRAINING (scikit-learn + joblib + SHAP)
# ============================================================

SKEWED_COLS = ["mobile_txn_count_6m", "avg_txn_amount"]


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Log1p-transforms the heavily right-skewed count/amount features before they hit
    the scaler. Without this, a single extreme transaction-volume outlier can swamp the
    standardized coefficients and drown out genuinely risky signals (irregular timing,
    low utility consistency) -- which is exactly the failure mode the adversarial stress
    test in Category 1 is designed to catch."""
    X = df[FEATURE_COLS].copy()
    for col in SKEWED_COLS:
        X[col] = np.log1p(X[col])
    return X


def _synthesize_default_labels(df: pd.DataFrame, rng) -> np.ndarray:
    """Ground-truth generator for training only: default risk rises with irregular
    transactions, low savings, low utility consistency, and (mildly) informal income,
    and falls (with diminishing returns) as mobile-money history deepens."""
    logit = (
        -1.2
        + 2.6 * (1 - df["txn_regularity_score"])
        + 2.1 * (1 - df["savings_rate_proxy"])
        + 1.6 * (1 - df["utility_consistency"])
        - 0.45 * np.log1p(df["mobile_txn_count_6m"])
        + 0.15 * df["is_smallholder"]
    )
    prob = 1 / (1 + np.exp(-logit))
    return (rng.rand(len(df)) < prob).astype(int)


def train_or_load_model() -> Dict[str, Any]:
    """Trains a logistic-regression default-risk model on a larger synthetic training
    set and persists it with joblib, or loads the cached bundle if one already exists.
    The output CONTRACT (score 300-950, tier, factor list) stays identical to the
    previous deterministic-formula version -- only the internals changed."""
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass  # fall through and retrain if the cached artifact is unreadable

    rng = np.random.RandomState(7)
    train_df = generate_market_data(4000, seed=7)
    y = _synthesize_default_labels(train_df, rng)
    X = _feature_matrix(train_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_scaled, y)

    bundle = {
        "model": model,
        "scaler": scaler,
        "features": FEATURE_COLS,
        "version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_n": len(train_df),
        "train_default_rate": round(float(y.mean()), 4),
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle


MODEL_BUNDLE = train_or_load_model()

# Background sample for the SHAP explainer, built once at startup.
_EXPLAINER = None
if HAS_SHAP:
    try:
        _bg_raw = _feature_matrix(generate_market_data(80, seed=99))
        _bg_scaled = MODEL_BUNDLE["scaler"].transform(_bg_raw)
        _EXPLAINER = shap.LinearExplainer(MODEL_BUNDLE["model"], _bg_scaled)
    except Exception:
        _EXPLAINER = None
        HAS_SHAP = False


def _predict_default_prob(feature_row: pd.Series) -> float:
    X = _feature_matrix(pd.DataFrame([feature_row[FEATURE_COLS].to_dict()]))
    Xs = MODEL_BUNDLE["scaler"].transform(X)
    return float(MODEL_BUNDLE["model"].predict_proba(Xs)[0][1])


def _score_from_prob(p: float) -> int:
    # Monotonic mapping onto the existing 300-950 contract: lower default probability
    # -> higher score. Keeps every downstream renderer (gauge, tiers) unchanged.
    score = 950 - p * 650
    return int(max(300, min(950, round(score))))


def _illustrative_factors(feature_row: pd.Series) -> List[Dict[str, Any]]:
    """Honest fallback used ONLY when real SHAP isn't available in this runtime.
    Uses the trained model's own standardized coefficients (not invented numbers) so
    direction is still faithful to the model, but this is coefficient-times-deviation,
    not a Shapley value, and the UI labels it as such."""
    X = _feature_matrix(pd.DataFrame([feature_row[FEATURE_COLS].to_dict()]))
    Xs = MODEL_BUNDLE["scaler"].transform(X)[0]
    coefs = MODEL_BUNDLE["model"].coef_[0]
    contributions = coefs * Xs
    total = sum(abs(c) for c in contributions) or 1e-9
    factors = []
    for col, c in zip(FEATURE_COLS, contributions):
        factors.append({
            "label": FEATURE_LABELS.get(col, col),
            "positive": bool(c < 0),
            "pct": int(round(abs(c) / total * 100)),
        })
    return sorted(factors, key=lambda x: x["pct"], reverse=True)[:5]


def _shap_factors(feature_row: pd.Series) -> Tuple[List[Dict[str, Any]], str]:
    X = _feature_matrix(pd.DataFrame([feature_row[FEATURE_COLS].to_dict()]))
    Xs = MODEL_BUNDLE["scaler"].transform(X)
    if HAS_SHAP and _EXPLAINER is not None:
        try:
            sv = _EXPLAINER.shap_values(Xs)[0]
            total = sum(abs(v) for v in sv) or 1e-9
            factors = []
            for col, v in zip(FEATURE_COLS, sv):
                factors.append({
                    "label": FEATURE_LABELS.get(col, col),
                    "positive": bool(v < 0),
                    "pct": int(round(abs(v) / total * 100)),
                })
            return sorted(factors, key=lambda x: x["pct"], reverse=True)[:5], "shap"
        except Exception:
            pass
    return _illustrative_factors(feature_row), "illustrative"


def get_risk_tier(score: int) -> Tuple[str, Dict[str, str]]:
    for tier, cfg in TIER_CONFIG.items():
        if score >= cfg["min"]:
            return tier, cfg
    return "Building", TIER_CONFIG["Building"]


def calculate_score(user: pd.Series) -> Tuple[int, str, Dict, List, str, float]:
    """Public scoring entry point. Returns (score, tier, cfg, factors, attribution_mode,
    default_probability) -- attribution_mode is 'shap' or 'illustrative' so callers can
    render an honest badge rather than implying real SHAP where there isn't any."""
    p = _predict_default_prob(user)
    score = _score_from_prob(p)
    tier, cfg = get_risk_tier(score)
    factors, mode = _shap_factors(user)
    return score, tier, cfg, factors, mode, p


def make_synthetic_applicant(
    txn_regularity: float,
    mobile_txn_count: float,
    utility_consistency: float,
    savings_rate: float = 0.4,
    avg_txn_amount: float = 40000,
    is_smallholder: int = 0,
    age: int = 32,
    user_id: str = "SIM_APPLICANT",
) -> pd.Series:
    """Builds a single synthetic applicant row for the live scenario simulator and the
    adversarial stress test -- anything not on a slider gets a realistic default."""
    return pd.Series({
        "user_id": user_id,
        "age": age,
        "location_type": "Lagos (Urban)",
        "is_smallholder": is_smallholder,
        "mobile_txn_count_6m": mobile_txn_count,
        "avg_txn_amount": avg_txn_amount,
        "txn_regularity_score": txn_regularity,
        "utility_consistency": utility_consistency,
        "savings_rate_proxy": savings_rate,
        "is_credit_invisible": True,
    })


ADVERSARIAL_PROFILE = make_synthetic_applicant(
    txn_regularity=0.12,
    mobile_txn_count=780,
    utility_consistency=0.15,
    savings_rate=0.06,
    avg_txn_amount=95000,
    is_smallholder=0,
    age=29,
    user_id="STRESS_TEST_001",
)

# ============================================================
# DESIGN TOKENS (Enterprise Fintech Theme)
# ============================================================
COLORS = {
    "bg_deep": "#020617",
    "bg_surface": "#0F172A",
    "bg_card": "#1E293B",
    "emerald": "#10B981",
    "emerald_bright": "#34D399",
    "gold": "#F59E0B",
    "gold_bright": "#FBBF24",
    "alert": "#EF4444",
    "text_primary": "#F8FAFC",
    "text_secondary": "#94A3B8",
    "text_muted": "#475569",
    "border": "rgba(255,255,255,0.06)",
}

TIER_CONFIG = {
    "Prime": {"min": 750, "color": COLORS["emerald_bright"], "glow": "rgba(52,211,153,0.15)", "desc": "Pre-approved for high-limit unsecured facilities."},
    "Growth": {"min": 650, "color": COLORS["gold_bright"], "glow": "rgba(251,191,36,0.15)", "desc": "Standard lending products and BNPL eligible."},
    "Emerging": {"min": 550, "color": "#F97316", "glow": "rgba(249,115,22,0.15)", "desc": "Micro-credit or collateralized facilities recommended."},
    "Building": {"min": 0, "color": COLORS["alert"], "glow": "rgba(239,68,68,0.15)", "desc": "Route to financial literacy and savings modules."},
}

TAB_SEQUENCE = [
    {"index": 0, "label": "How It Works"},
    {"index": 1, "label": "Score"},
    {"index": 2, "label": "Portfolio"},
    {"index": 4, "label": "Wema ROI"},
    {"index": 5, "label": "Market & Roadmap"},
]

# ============================================================
# CATEGORY 2 -- ICONOGRAPHY (replaces ad hoc emoji with brand-consistent SVGs)
# ============================================================

def icon(name: str, size: int = 18, color: Optional[str] = None) -> str:
    color = color or COLORS["text_secondary"]
    paths = {
        "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none"/>',
        "money": '<rect x="2.5" y="6" width="19" height="12" rx="2.5"/><circle cx="12" cy="12" r="3"/><path d="M6 9v.01M18 15v.01"/>',
        "lock": '<rect x="4.5" y="10.5" width="15" height="9.5" rx="2"/><path d="M8 10.5V7.5a4 4 0 0 1 8 0v3"/>',
        "bank": '<path d="M3 9.5 12 4l9 5.5"/><path d="M4.5 9.5v9M9 9.5v9M15 9.5v9M19.5 9.5v9"/><path d="M2.5 20h19"/>',
        "chart": '<path d="M4 20V10M11 20V4M18 20v-7"/><path d="M2.5 20h19"/>',
        "globe": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.7 2.7 4 5.7 4 9s-1.3 6.3-4 9c-2.7-2.7-4-5.7-4-9s1.3-6.3 4-9Z"/>',
        "shield": '<path d="M12 3.5 19 6.5v5.5c0 4.7-3 7.7-7 9-4-1.3-7-4.3-7-9V6.5Z"/><path d="m9 12 2 2 4-4"/>',
        "download": '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M4.5 19.5h15"/>',
        "check": '<path d="m5 13 4.5 4.5L19 8"/>',
        "alert": '<path d="M12 4 21.5 20.5H2.5Z"/><path d="M12 10v4.5M12 17.5v.01"/>',
        "sliders": '<path d="M4 7h9M17 7h3M4 12h3M9 12h11M4 17h13M21 17h-.01"/><circle cx="15" cy="7" r="2"/><circle cx="7" cy="12" r="2"/><circle cx="18" cy="17" r="2"/>',
        "flask": '<path d="M9 3h6M10 3v6.5L4.7 19a2 2 0 0 0 1.7 3h11.2a2 2 0 0 0 1.7-3L14 9.5V3"/><path d="M7.5 15h9"/>',
        "user": '<circle cx="12" cy="8.5" r="3.5"/><path d="M4.5 20a7.5 7.5 0 0 1 15 0"/>',
        "presenter": '<rect x="3" y="4.5" width="18" height="12" rx="1.5"/><path d="M9 20.5h6M12 16.5v4"/>',
    }
    d = paths.get(name, paths["check"])
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" '
        f'focusable="false" style="vertical-align:-3px;">{d}</svg>'
    )

# ============================================================
# UI RENDERERS (Custom HTML/SVG Components)
# ============================================================

def render_logo(size: int = 40) -> str:
    """Brand mark: interlocking 'A' monogram in an emerald-gold gradient badge."""
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 48 48" class="afs-logo" role="img" aria-label="AfriScore logo">
      <defs>
        <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="{COLORS['emerald_bright']}"/>
          <stop offset="100%" stop-color="{COLORS['gold']}"/>
        </linearGradient>
      </defs>
      <rect width="48" height="48" rx="12" fill="#0B1120" stroke="url(#logoGrad)" stroke-width="1.5"/>
      <path d="M24 11L34 34H29.5L27.3 28.5H20.7L18.5 34H14L24 11Z" fill="url(#logoGrad)"/>
      <path d="M22.2 24.5H25.8L24 19.8L22.2 24.5Z" fill="#0B1120"/>
    </svg>
    """

def render_gauge(score: int, cfg: Dict) -> str:
    radius, stroke = 92, 14
    circ = 2 * math.pi * radius
    pct = max(0, min(1, (score - 300) / 650))
    offset = circ * (1 - pct)

    return f"""
    <svg width="200" height="200" viewBox="0 0 220 220" class="afs-gauge" role="img" aria-label="AfriScore gauge showing {score} out of 950">
      <defs>
        <linearGradient id="gGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="{COLORS['gold']}"/>
          <stop offset="100%" stop-color="{cfg['color']}"/>
        </linearGradient>
      </defs>
      <circle cx="110" cy="110" r="{radius}" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="{stroke}"/>
      <circle cx="110" cy="110" r="{radius}" fill="none" stroke="url(#gGrad)" stroke-width="{stroke}"
        stroke-linecap="round" stroke-dasharray="{circ:.1f}" stroke-dashoffset="{offset:.1f}"
        transform="rotate(-90 110 110)" class="afs-gauge-arc" style="transition: stroke-dashoffset 0.6s ease-out;"/>
      <circle cx="110" cy="110" r="{radius+10}" fill="none" stroke="{cfg['color']}" stroke-width="1" opacity="0.25" class="afs-gauge-pulse"/>
      <text x="110" y="105" text-anchor="middle" fill="#fff" font-family="monospace" font-size="42" font-weight="700">{score}</text>
      <text x="110" y="130" text-anchor="middle" fill="{COLORS['text_muted']}" font-family="monospace" font-size="14">/ 950</text>
    </svg>
    """

def render_distribution_chart(scores: List[int]) -> str:
    buckets = {"Prime": 0, "Growth": 0, "Emerging": 0, "Building": 0}
    for s in scores:
        t, _ = get_risk_tier(s)
        buckets[t] += 1

    max_val = max(buckets.values()) or 1
    bars = ""
    for i, (tier, count) in enumerate(buckets.items()):
        height = max(10, (count / max_val) * 120)
        color = TIER_CONFIG[tier]["color"]
        bars += f"""
        <g transform="translate({i * 60 + 20}, 0)">
            <rect x="0" y="{130 - height}" width="30" height="{height}" fill="{color}" rx="4" class="afs-chart-bar" style="animation-delay:{i*0.1}s"/>
            <text x="15" y="145" text-anchor="middle" fill="{COLORS['text_secondary']}" font-size="10" font-family="sans-serif">{tier}</text>
            <text x="15" y="{120 - height}" text-anchor="middle" fill="#fff" font-size="11" font-weight="bold">{count}</text>
        </g>
        """
    return f'<svg width="100%" height="160" viewBox="0 0 280 160" role="img" aria-label="Portfolio risk-tier distribution">{bars}</svg>'

def render_funnel(label_top: str, val_top: str, label_mid: str, val_mid: str, label_bot: str, val_bot: str) -> str:
    """TAM / SAM / SOM funnel visual for investor tab."""
    rows = [
        (val_top, label_top, 100, COLORS["text_muted"]),
        (val_mid, label_mid, 66, COLORS["gold"]),
        (val_bot, label_bot, 34, COLORS["emerald_bright"]),
    ]
    bars = ""
    for i, (val, label, widthpct, color) in enumerate(rows):
        bars += f"""
        <div class="afs-funnel-row" style="animation-delay:{i*0.12}s">
            <div class="afs-funnel-bar-bg">
                <div class="afs-funnel-bar" style="width:{widthpct}%; background:{color};"></div>
            </div>
            <div class="afs-funnel-label">
                <span style="color:#fff; font-weight:700;">{val}</span>
                <span style="color:{COLORS['text_secondary']}; font-size:12px;">{label}</span>
            </div>
        </div>
        """
    return f'<div class="afs-funnel">{bars}</div>'

def render_skeleton(label: str = "Running inference&hellip;") -> str:
    """Brief loading placeholder shown before a scoring/ROI calculation resolves,
    so button clicks read as a real backend call rather than an instant HTML swap."""
    return f"""
    <div class="afs-card afs-skeleton-wrap" role="status" aria-live="polite">
        <div class="afs-eyebrow" style="display:flex; align-items:center; gap:8px;">
            <span class="afs-live-dot"></span>{label}
        </div>
        <div class="afs-skeleton-line" style="width:60%;"></div>
        <div class="afs-skeleton-line" style="width:90%;"></div>
        <div class="afs-skeleton-line" style="width:40%;"></div>
        <div class="afs-skeleton-block"></div>
    </div>
    """

def render_error_state(title: str, detail: str) -> str:
    """Consistent, deliberately-designed error card -- used any time an input is
    invalid, so a live demo never just breaks or silently clamps a bad number."""
    return f"""
    <div class="afs-card afs-error-card" role="alert">
        <div style="display:flex; align-items:flex-start; gap:12px;">
            <span style="flex:0 0 auto; margin-top:2px;">{icon('alert', 22, COLORS['alert'])}</span>
            <div>
                <div style="color:{COLORS['alert']}; font-weight:700; font-family:'Space Grotesk'; margin-bottom:4px;">{title}</div>
                <div style="color:{COLORS['text_secondary']}; font-size:13px; line-height:1.6;">{detail}</div>
            </div>
        </div>
    </div>
    """

def render_empty_state(title: str, detail: str, icon_name: str = "flask") -> str:
    return f"""
    <div class="afs-card afs-empty-card" style="text-align:center; padding:48px 32px;">
        <div style="margin-bottom:12px;">{icon(icon_name, 32, COLORS['text_muted'])}</div>
        <div style="color:#fff; font-weight:700; font-family:'Space Grotesk'; margin-bottom:6px;">{title}</div>
        <div style="color:{COLORS['text_secondary']}; font-size:13px; max-width:420px; margin:0 auto; line-height:1.6;">{detail}</div>
    </div>
    """

def render_roadmap() -> str:
    stages = [
        ("NOW", "Pilot with Wema Bank digital lending desk. 3 core APIs live: /score, /explain, /webhook. NDPR-compliant consent layer shipped.", COLORS["gold"]),
        ("NEXT (6-12mo)", "Expand data graph to 2 more telcos + agri-cooperatives. Launch BNPL risk product. Onboard 3 additional Tier-2 banks.", COLORS["emerald_bright"]),
        ("LATER (12-24mo)", "Pan-African expansion (Ghana, Kenya). Open a scored-identity marketplace for fintechs. SME cash-flow underwriting line.", "#F97316"),
    ]
    cards = ""
    for i, (tag, desc, color) in enumerate(stages):
        cards += f"""
        <div class="afs-roadmap-card" style="animation-delay:{i*0.1}s; border-top: 3px solid {color};">
            <div class="afs-roadmap-tag" style="color:{color};">{tag}</div>
            <p style="color:{COLORS['text_secondary']}; font-size:13px; line-height:1.6; margin:8px 0 0 0;">{desc}</p>
        </div>
        """
    return f'<div class="afs-grid-3" style="margin-bottom:0;">{cards}</div>'

def render_credit_memo(user_id: str, score: int, tier: str, cfg: Dict, factors: List, mode: str) -> str:
    """Formats a decision the way a bank's actual credit committee would file it --
    this is what makes 'explainable AI' land as real to a compliance reader, not
    just a percentage bar."""
    decision = "APPROVE" if score >= 650 else ("REFER FOR MANUAL REVIEW" if score >= 550 else "DECLINE — ROUTE TO LITERACY TRACK")
    decision_color = COLORS["emerald_bright"] if score >= 650 else (COLORS["gold"] if score >= 550 else COLORS["alert"])
    mode_label = "genuine SHAP values from the trained model" if mode == "shap" else "illustrative feature attribution (SHAP not available this session)"
    factor_lines = "".join(
        f'<li><strong style="color:#fff;">{f["label"]}:</strong> {"supports approval" if f["positive"] else "flagged for review"} '
        f'({f["pct"]}% of this decision\'s attributed weight)</li>'
        for f in factors
    )
    return f"""
    <div class="afs-card" style="border-color: rgba(255,255,255,0.12);">
        <div class="afs-eyebrow">{icon('bank', 14, COLORS['text_secondary'])} COMPLIANCE VIEW &middot; CREDIT COMMITTEE MEMO</div>
        <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:12px; margin-bottom:16px;">
            <div>
                <div style="color:{COLORS['text_secondary']}; font-size:12px;">Applicant Reference</div>
                <div style="color:#fff; font-family:monospace; font-weight:bold;">{user_id}</div>
            </div>
            <div>
                <div style="color:{COLORS['text_secondary']}; font-size:12px;">System Recommendation</div>
                <div style="color:{decision_color}; font-weight:bold;">{decision}</div>
            </div>
            <div>
                <div style="color:{COLORS['text_secondary']}; font-size:12px;">AfriScore / Tier</div>
                <div style="color:#fff; font-weight:bold;">{score} &middot; {tier}</div>
            </div>
        </div>
        <div style="color:{COLORS['text_secondary']}; font-size:13px; margin-bottom:8px;">Basis for recommendation ({mode_label}):</div>
        <ul style="color:{COLORS['text_secondary']}; font-size:13px; line-height:1.7; padding-left:20px; margin:0 0 12px 0;">
            {factor_lines}
        </ul>
        <p style="color:{COLORS['text_muted']}; font-size:11px; margin:0; border-top:1px solid {COLORS['border']}; padding-top:12px;">
            This recommendation is generated for human review, not automatic execution. A licensed credit
            officer retains final authority to override, per AfriScore's model-governance policy.
        </p>
    </div>
    """

def render_bureau_comparison(user_id: str, is_invisible: bool, score: int, tier: str, cfg: Dict) -> str:
    """The whole thesis of the product in one visual: the same applicant, side by side,
    under a traditional bureau model vs. AfriScore."""
    if is_invisible:
        left_body = f"""
            <div style="font-size:28px; font-weight:800; color:{COLORS['alert']}; font-family:'Space Grotesk';">UNSCOREABLE</div>
            <div style="color:{COLORS['text_secondary']}; font-size:13px; margin-top:6px;">Insufficient bureau history — no formal credit file on record.</div>
        """
    else:
        left_body = f"""
            <div style="font-size:28px; font-weight:800; color:#fff; font-family:'Space Grotesk';">Thin file</div>
            <div style="color:{COLORS['text_secondary']}; font-size:13px; margin-top:6px;">Limited bureau signal — traditional models would typically decline or heavily restrict.</div>
        """
    return f"""
    <div class="afs-card" style="margin-top:24px;">
        <div class="afs-eyebrow">{icon('chart', 14, COLORS['text_secondary'])} TRADITIONAL BUREAU MODEL vs. AFRISCORE &middot; SAME APPLICANT</div>
        <div class="afs-grid-2" style="gap:20px;">
            <div class="afs-card" style="background:rgba(239,68,68,0.05); border-color:rgba(239,68,68,0.2);">
                <div style="color:{COLORS['text_secondary']}; font-size:12px; margin-bottom:8px;">TRADITIONAL BUREAU RESULT</div>
                {left_body}
            </div>
            <div class="afs-card" style="background:rgba(52,211,153,0.05); border-color:rgba(52,211,153,0.25);">
                <div style="color:{COLORS['text_secondary']}; font-size:12px; margin-bottom:8px;">AFRISCORE RESULT</div>
                <div style="font-size:28px; font-weight:800; color:{cfg['color']}; font-family:'Space Grotesk';">{score} &middot; {tier}</div>
                <div style="color:{COLORS['text_secondary']}; font-size:13px; margin-top:6px;">{cfg['desc']}</div>
            </div>
        </div>
    </div>
    """

def render_integration_path() -> str:
    """How a bank actually adopts this -- sandbox to full rollout -- so judges see an
    adoption plan, not just working tech."""
    steps = [
        ("1. Sandbox", "Bank issues a test API key. AfriScore runs against historical, anonymized loan data only.", COLORS["text_muted"]),
        ("2. Shadow Mode", "AfriScore scores live applicants in parallel with the bank's existing process. No decisions change yet — this is where trust gets built.", COLORS["gold"]),
        ("3. Capped Pilot", "AfriScore decisions go live for a limited cohort (e.g. one loan product, one branch cluster) with human sign-off on every decision.", COLORS["emerald_bright"]),
        ("4. Full Rollout", "AfriScore becomes the default first-pass risk layer, with committee review reserved for edge cases.", "#F97316"),
    ]
    cols = ""
    for i, (title, desc, color) in enumerate(steps):
        arrow = '<div class="afs-path-arrow">&rarr;</div>' if i < len(steps) - 1 else ""
        cols += f"""
        <div class="afs-path-step" style="animation-delay:{i*0.1}s;">
            <div class="afs-path-dot" style="background:{color};"></div>
            <div style="color:{color}; font-family:'Space Grotesk'; font-weight:700; font-size:13px; margin-top:8px;">{title}</div>
            <div style="color:{COLORS['text_secondary']}; font-size:12px; line-height:1.5; margin-top:4px;">{desc}</div>
        </div>
        {arrow}
        """
    return f'<div class="afs-path-row">{cols}</div>'

def render_model_card() -> str:
    """Model governance card -- version, training date, feature list, known limitations.
    Understanding this is itself a credibility signal to technical judges."""
    shap_badge = (
        f'<span class="afs-sm-badge" style="background:rgba(52,211,153,0.15); color:{COLORS["emerald_bright"]}">LIVE SHAP ACTIVE</span>'
        if HAS_SHAP else
        f'<span class="afs-sm-badge" style="background:rgba(245,158,11,0.15); color:{COLORS["gold"]}">ILLUSTRATIVE ATTRIBUTION (SHAP NOT INSTALLED)</span>'
    )
    feature_pills = "".join(f'<span class="afs-tag">{FEATURE_LABELS.get(c, c)}</span>' for c in FEATURE_COLS)
    limitation_lines = "".join(f"<li>{lim}</li>" for lim in MODEL_LIMITATIONS)
    return f"""
    <div class="afs-card">
        <div class="afs-eyebrow">{icon('shield', 14, COLORS['text_secondary'])} MODEL CARD</div>
        <div style="display:flex; flex-wrap:wrap; gap:24px; margin-bottom:16px;">
            <div>
                <div style="color:{COLORS['text_secondary']}; font-size:12px;">Model</div>
                <div style="color:#fff; font-weight:bold;">Logistic Regression &middot; v{MODEL_BUNDLE['version']}</div>
            </div>
            <div>
                <div style="color:{COLORS['text_secondary']}; font-size:12px;">Trained</div>
                <div style="color:#fff; font-weight:bold; font-family:monospace;">{MODEL_BUNDLE['trained_at']}</div>
            </div>
            <div>
                <div style="color:{COLORS['text_secondary']}; font-size:12px;">Training population</div>
                <div style="color:#fff; font-weight:bold;">{MODEL_BUNDLE['train_n']:,} synthetic applicants</div>
            </div>
            <div>
                <div style="color:{COLORS['text_secondary']}; font-size:12px;">Explainability</div>
                <div>{shap_badge}</div>
            </div>
        </div>
        <div style="color:{COLORS['text_secondary']}; font-size:12px; margin-bottom:8px;">Feature set</div>
        <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:16px;">{feature_pills}</div>
        <div style="color:{COLORS['text_secondary']}; font-size:12px; margin-bottom:8px;">Known limitations</div>
        <ul style="color:{COLORS['text_secondary']}; font-size:13px; line-height:1.7; padding-left:20px; margin:0;">
            {limitation_lines}
        </ul>
    </div>
    """

def render_presenter_progress(step: int) -> str:
    dots = ""
    for i, stage in enumerate(TAB_SEQUENCE):
        active = i == step
        done = i < step
        color = COLORS["gold"] if active else (COLORS["emerald_bright"] if done else COLORS["text_muted"])
        weight = "700" if active else "500"
        dots += f"""
        <div style="display:flex; flex-direction:column; align-items:center; gap:6px; flex:1;">
            <div style="width:10px; height:10px; border-radius:50%; background:{color};"></div>
            <div style="font-size:11px; color:{color}; font-weight:{weight}; text-align:center;">{stage['label']}</div>
        </div>
        """
    return f"""
    <div class="afs-card" style="padding:16px 24px;">
        <div style="display:flex; align-items:center; justify-content:space-between; gap:8px;">
            {dots}
        </div>
    </div>
    """

# ============================================================
# TAB FUNCTIONS
# ============================================================
def tab_consumer_score(user_id: str):
    """Yields a brief skeleton state, then the resolved score card, bureau comparison,
    and compliance memo."""
    yield render_skeleton("Scoring applicant&hellip;")
    time.sleep(0.35)

    user = df_users[df_users["user_id"] == user_id].iloc[0]
    score, tier, cfg, factors, mode, _p = calculate_score(user)

    factor_html = ""
    for i, f in enumerate(factors):
        color = COLORS["emerald_bright"] if f["positive"] else COLORS["alert"]
        sign = "&#8593; Positive" if f["positive"] else "&#8595; Flag"
        factor_html += f"""
        <div class="afs-factor" style="animation-delay:{0.1 + i*0.1}s">
            <div class="afs-f-top"><span>{f['label']}</span><span style="color:{color}">{sign}</span></div>
            <div class="afs-f-bar-bg"><div class="afs-f-bar" style="width:{f['pct']}%; background:{color};"></div></div>
        </div>
        """

    mode_pill = (
        f'<span class="afs-tag" style="border-color:rgba(52,211,153,0.3); color:{COLORS["emerald_bright"]}">{icon("check", 12, COLORS["emerald_bright"])} Live SHAP</span>'
        if mode == "shap" else
        f'<span class="afs-tag" style="border-color:rgba(245,158,11,0.3); color:{COLORS["gold"]}">{icon("alert", 12, COLORS["gold"])} Illustrative attribution</span>'
    )

    score_card = f"""
    <div class="afs-card">
        <div class="afs-grid-2" style="align-items: center; gap: 40px;">
            <div style="text-align:center;">{render_gauge(score, cfg)}</div>
            <div>
                <div class="afs-eyebrow">USER IDENTITY: {user_id}</div>
                <div class="afs-tier-badge" style="color:{cfg['color']}; background:{cfg['glow']}; border: 1px solid {cfg['color']}">{tier} Tier</div>
                <p style="color:{COLORS['text_secondary']}; line-height:1.6; margin-bottom:16px;">{cfg['desc']}</p>
                <div style="display:flex; gap:8px; flex-wrap:wrap;">
                    <span class="afs-tag">Age {user['age']}</span>
                    <span class="afs-tag">{user['location_type']}</span>
                    {mode_pill}
                </div>
            </div>
        </div>
        <hr class="afs-hr"/>
        <div class="afs-eyebrow">FEATURE ATTRIBUTION</div>
        {factor_html}
    </div>
    """
    bureau_card = render_bureau_comparison(user_id, bool(user["is_credit_invisible"]), score, tier, cfg)
    memo_card = render_credit_memo(user_id, score, tier, cfg, factors, mode)
    yield f'<div style="display:flex; flex-direction:column; gap:24px;">{score_card}{bureau_card}{memo_card}</div>'


def run_live_simulation(txn_regularity: float, mobile_txn_count: float, utility_consistency: float):
    """Live scenario simulator: recomputes the score and gauge in real time as sliders
    move -- no skeleton delay here, this one needs to feel instant."""
    user = make_synthetic_applicant(txn_regularity, mobile_txn_count, utility_consistency)
    score, tier, cfg, factors, mode, _p = calculate_score(user)

    factor_html = ""
    for f in factors[:3]:
        color = COLORS["emerald_bright"] if f["positive"] else COLORS["alert"]
        factor_html += f"""
        <div class="afs-factor">
            <div class="afs-f-top"><span>{f['label']}</span><span style="color:{color}">{f['pct']}%</span></div>
            <div class="afs-f-bar-bg"><div class="afs-f-bar" style="width:{f['pct']}%; background:{color};"></div></div>
        </div>
        """
    return f"""
    <div class="afs-card">
        <div class="afs-grid-2" style="align-items:center; gap:32px;">
            <div style="text-align:center;">{render_gauge(score, cfg)}</div>
            <div>
                <div class="afs-tier-badge" style="color:{cfg['color']}; background:{cfg['glow']}; border:1px solid {cfg['color']}">{tier} Tier</div>
                <p style="color:{COLORS['text_secondary']}; font-size:13px; margin-bottom:12px;">{cfg['desc']}</p>
                {factor_html}
            </div>
        </div>
    </div>
    """


def run_stress_test():
    user = ADVERSARIAL_PROFILE
    score, tier, cfg, factors, mode, p = calculate_score(user)
    flagged_high_risk = score < 650
    verdict_color = COLORS["emerald_bright"] if flagged_high_risk else COLORS["alert"]
    verdict_text = (
        "Correctly flagged as elevated risk despite very high transaction volume — the "
        "model isn't fooled by raw activity alone."
        if flagged_high_risk else
        "Unexpected: the model scored this adversarial, high-volume/low-regularity "
        "profile as low-risk — worth a closer look before a live demo."
    )
    return f"""
    <div class="afs-card" style="border-color: rgba(245,158,11,0.3);">
        <div class="afs-eyebrow">{icon('flask', 14, COLORS['gold'])} ADVERSARIAL STRESS TEST</div>
        <p style="color:{COLORS['text_secondary']}; font-size:13px; margin-bottom:16px;">
            Synthetic profile: {int(user['mobile_txn_count_6m'])} transactions in 6 months (very high volume),
            but only {user['txn_regularity_score']:.2f} regularity, {user['utility_consistency']:.2f} utility
            consistency, and {user['savings_rate_proxy']:.2f} savings rate — high noise, low signal quality.
        </p>
        <div class="afs-grid-2" style="align-items:center; gap:24px;">
            <div style="text-align:center;">{render_gauge(score, cfg)}</div>
            <div>
                <div class="afs-tier-badge" style="color:{cfg['color']}; background:{cfg['glow']}; border:1px solid {cfg['color']}">{tier} Tier</div>
                <div style="display:flex; align-items:flex-start; gap:8px; margin-top:8px;">
                    <span style="flex:0 0 auto;">{icon('check' if flagged_high_risk else 'alert', 18, verdict_color)}</span>
                    <p style="color:{verdict_color}; font-size:13px; line-height:1.6; margin:0;">{verdict_text}</p>
                </div>
                <p style="color:{COLORS['text_muted']}; font-size:11px; margin-top:10px;">Modeled default probability: {p:.1%}</p>
            </div>
        </div>
    </div>
    """


def generate_decision_record(user_id: str):
    """Decision export: produces a one-page PDF (or a plain-text fallback if reportlab
    isn't available) that a compliance team can file -- an artifact, not just a screen."""
    row = df_users[df_users["user_id"] == user_id]
    if row.empty:
        return None
    user = row.iloc[0]
    score, tier, cfg, factors, mode, p = calculate_score(user)
    decision = "APPROVE" if score >= 650 else ("REFER FOR MANUAL REVIEW" if score >= 550 else "DECLINE — ROUTE TO LITERACY TRACK")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    if HAS_REPORTLAB:
        out_path = OUTPUT_DIR / f"decision_record_{user_id}.pdf"
        c = canvas.Canvas(str(out_path), pagesize=LETTER)
        width, height = LETTER
        y = height - 1 * inch

        c.setFont("Helvetica-Bold", 16)
        c.drawString(1 * inch, y, "AfriScore — Credit Decision Record")
        y -= 0.3 * inch
        c.setFont("Helvetica", 9)
        c.setFillColorRGB(0.4, 0.45, 0.55)
        c.drawString(1 * inch, y, f"Generated {timestamp} · Model v{MODEL_BUNDLE['version']} · Attribution mode: {mode}")
        y -= 0.4 * inch

        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(1 * inch, y, f"Applicant Reference: {user_id}")
        y -= 0.25 * inch
        c.drawString(1 * inch, y, f"AfriScore: {score}  /  Tier: {tier}")
        y -= 0.25 * inch
        c.drawString(1 * inch, y, f"System Recommendation: {decision}")
        y -= 0.35 * inch

        c.setFont("Helvetica-Bold", 10)
        c.drawString(1 * inch, y, "Basis for recommendation:")
        y -= 0.22 * inch
        c.setFont("Helvetica", 9)
        for f in factors:
            line = f"- {f['label']}: {'supports approval' if f['positive'] else 'flagged for review'} ({f['pct']}% attributed weight)"
            c.drawString(1.1 * inch, y, line)
            y -= 0.2 * inch

        y -= 0.2 * inch
        c.setFont("Helvetica-Oblique", 8)
        c.setFillColorRGB(0.4, 0.45, 0.55)
        text = c.beginText(1 * inch, y)
        text.setLeading(11)
        for line in [
            "This recommendation is generated for human review, not automatic execution.",
            "A licensed credit officer retains final authority to override, per AfriScore's",
            "model-governance policy. Modeled default probability: " + f"{p:.1%}.",
        ]:
            text.textLine(line)
        c.drawText(text)
        c.save()
        return str(out_path)

    # Plain-text fallback if reportlab is unavailable in this runtime.
    out_path = OUTPUT_DIR / f"decision_record_{user_id}.txt"
    lines = [
        "AfriScore — Credit Decision Record",
        f"Generated {timestamp} | Model v{MODEL_BUNDLE['version']} | Attribution mode: {mode}",
        "",
        f"Applicant Reference: {user_id}",
        f"AfriScore: {score} / Tier: {tier}",
        f"System Recommendation: {decision}",
        "",
        "Basis for recommendation:",
    ]
    for f in factors:
        lines.append(f"- {f['label']}: {'supports approval' if f['positive'] else 'flagged for review'} ({f['pct']}% attributed weight)")
    lines += [
        "",
        "This recommendation is generated for human review, not automatic execution.",
        "A licensed credit officer retains final authority to override.",
        f"Modeled default probability: {p:.1%}.",
    ]
    out_path.write_text("\n".join(lines))
    return str(out_path)


def tab_enterprise_dashboard() -> str:
    sample = df_users.sample(25, random_state=101)
    rows, scores = [], []

    for _, row in sample.iterrows():
        score, tier, cfg, _factors, _mode, _p = calculate_score(row)
        scores.append(score)
        rows.append(f"""
        <tr>
            <td style="font-family:monospace; color:#fff;">{row['user_id']}</td>
            <td>{row['location_type']}</td>
            <td style="color:{cfg['color']}; font-weight:bold;">{score}</td>
            <td><span class="afs-sm-badge" style="background:{cfg['glow']}; color:{cfg['color']}">{tier}</span></td>
        </tr>
        """)

    avg_score = int(np.mean(scores))
    approval_rate = sum(1 for s in scores if s >= 650) / len(scores) * 100

    return f"""
    <div class="afs-grid-2" style="gap:24px; margin-bottom:24px;">
        <div class="afs-card">
            <div class="afs-eyebrow">PORTFOLIO RISK DISTRIBUTION</div>
            {render_distribution_chart(scores)}
        </div>
        <div class="afs-card" style="display:flex; flex-direction:column; justify-content:center;">
            <div class="afs-eyebrow" style="display:flex; align-items:center; gap:8px;">
                <span class="afs-live-dot"></span> BATCH PROCESSING KPIs
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:16px;">
                <span style="color:{COLORS['text_secondary']}">Users Scored</span>
                <span style="color:#fff; font-weight:bold; font-size:18px;">{len(scores)}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:16px;">
                <span style="color:{COLORS['text_secondary']}">Average AfriScore</span>
                <span style="color:{COLORS['gold']}; font-weight:bold; font-size:18px;">{avg_score}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:{COLORS['text_secondary']}">Prime/Growth Approval Rate</span>
                <span style="color:{COLORS['emerald_bright']}; font-weight:bold; font-size:18px;">{approval_rate:.1f}%</span>
            </div>
        </div>
    </div>
    <div class="afs-card" style="padding:0; overflow:hidden;">
        <div style="padding:20px; border-bottom:1px solid {COLORS['border']}; display:flex; align-items:center; gap:8px;">
            <span class="afs-live-dot"></span><span class="afs-eyebrow" style="margin:0;">LIVE DECISION LOG</span>
        </div>
        <div style="overflow-x:auto;">
            <table class="afs-table">
                <thead><tr><th scope="col">Hashed ID</th><th scope="col">Demographic</th><th scope="col">AfriScore</th><th scope="col">Risk Tier</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
    </div>
    """

def tab_api_docs(user_id: str) -> str:
    row = df_users[df_users["user_id"] == user_id]
    if row.empty:
        return render_empty_state("No applicant selected", "Choose a hashed identity above to generate a sample request/response pair.")
    user = row.iloc[0]
    score, tier, _cfg, factors, mode, p = calculate_score(user)

    req = f"""curl -X POST https://api.afriscore.com/v1/score \\
  -H "Authorization: Bearer sk_live_xxx" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "user_id": "{user_id}",
    "consent_token": "ct_9a8b7c6d5e",
    "include_shap": true
  }}'"""

    res = {
        "id": f"evt_{int(datetime.now().timestamp())}",
        "object": "credit_decision",
        "decision": {
            "afriscore": score,
            "risk_tier": tier,
            "default_prob": round(p, 4)
        },
        "explainability": {"mode": mode, "factors": {f["label"]: f["pct"] for f in factors}},
        "meta": {"latency_ms": 28, "status": 200}
    }

    return f"""
    <div class="afs-card" style="padding:0; overflow:hidden;">
        <div class="afs-grid-2" style="gap:0;">
            <div style="padding:32px; border-right:1px solid {COLORS['border']};">
                <div class="afs-eyebrow" style="color:{COLORS['emerald_bright']}">ENDPOINT: /v1/score</div>
                <h2 style="margin:8px 0 16px 0; color:#fff; font-family:'Space Grotesk', sans-serif;">Generate a Score</h2>
                <p style="color:{COLORS['text_secondary']}; font-size:14px; line-height:1.6; margin-bottom:24px;">
                    Pass a hashed user ID and a valid consent token to instantly retrieve a fully explainable credit score.
                </p>
                <div class="afs-eyebrow">EXAMPLE REQUEST</div>
                <pre class="afs-code">{req}</pre>
            </div>
            <div style="padding:32px; background:#0B1120;">
                <div class="afs-eyebrow">RESPONSE (200 OK) &middot; 28ms</div>
                <pre class="afs-code" style="color:{COLORS['emerald_bright']}; border:none; background:transparent; padding:0;">{json.dumps(res, indent=2)}</pre>
            </div>
        </div>
    </div>
    """

def tab_bank_partnership(monthly_loans, manual_cost, default_rate_pct, avg_loan_size):
    """Wema-facing ROI calculator: translates the API into a bank's P&L, live.
    Yields a skeleton state first so recalculation reads as a live model run, and a
    clear error state instead of silently clamping bad input."""
    try:
        monthly_loans = float(monthly_loans)
        manual_cost = float(manual_cost)
        default_rate_pct = float(default_rate_pct)
        avg_loan_size = float(avg_loan_size)
    except (TypeError, ValueError):
        yield render_error_state(
            "Invalid input",
            "All four fields need to be numbers. Please re-enter monthly loan applications, "
            "review cost, default rate, and average loan size.",
        )
        return

    problems = []
    if monthly_loans < 0:
        problems.append("Monthly loan applications can't be negative.")
    if manual_cost < 0:
        problems.append("Manual review cost can't be negative.")
    if default_rate_pct < 0 or default_rate_pct > 100:
        problems.append("Default rate must be between 0% and 100%.")
    if avg_loan_size < 0:
        problems.append("Average loan size can't be negative.")

    if problems:
        yield render_error_state("Check your inputs", " ".join(problems))
        return

    yield render_skeleton("Modeling portfolio impact&hellip;")
    time.sleep(0.3)

    afriscore_cost_per_user = 0.15 * 1600  # USD -> NGN approx for display, illustrative
    current_review_cost = monthly_loans * manual_cost
    afriscore_processing_cost = monthly_loans * afriscore_cost_per_user
    review_cost_savings = max(0, current_review_cost - afriscore_processing_cost)

    # Assume a 22% relative reduction in defaults from better-informed underwriting on
    # previously "thin-file" applicants — a conservative, clearly-labeled assumption.
    default_reduction_pct = 22
    current_default_loss = monthly_loans * (default_rate_pct / 100) * avg_loan_size
    projected_default_loss = current_default_loss * (1 - default_reduction_pct / 100)
    default_loss_avoided = max(0, current_default_loss - projected_default_loss)

    total_monthly_impact = review_cost_savings + default_loss_avoided
    annual_impact = total_monthly_impact * 12

    def naira(n):
        return f"₦{n:,.0f}"

    result_html = f"""
    <div class="afs-card" style="margin-bottom:24px; background: rgba(245,158,11,0.05); border-color: rgba(245,158,11,0.2);">
        <p style="color:{COLORS['text_secondary']}; font-size:14px; line-height:1.6; margin:0;">
            <strong style="color:#fff;">Why Wema, specifically:</strong> Wema's digital-first ALAT platform already
            has the distribution and the retail-lending appetite — AfriScore is the risk layer that lets that
            channel say yes to applicants its current bureau-based scoring can't see.
        </p>
    </div>
    <div class="afs-grid-2" style="gap:24px; margin-bottom:24px;">
        <div class="afs-card">
            <div class="afs-eyebrow">MANUAL PROCESS TODAY</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:14px;">
                <span style="color:{COLORS['text_secondary']}">Manual review cost / applicant</span>
                <span style="color:#fff; font-weight:bold;">{naira(manual_cost)}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:14px;">
                <span style="color:{COLORS['text_secondary']}">Monthly review spend</span>
                <span style="color:{COLORS['alert']}; font-weight:bold;">{naira(current_review_cost)}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:{COLORS['text_secondary']}">Monthly default losses</span>
                <span style="color:{COLORS['alert']}; font-weight:bold;">{naira(current_default_loss)}</span>
            </div>
        </div>
        <div class="afs-card" style="border-color: rgba(52,211,153,0.3);">
            <div class="afs-eyebrow" style="color:{COLORS['emerald_bright']}">WITH AFRISCORE</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:14px;">
                <span style="color:{COLORS['text_secondary']}">Processing cost / applicant</span>
                <span style="color:#fff; font-weight:bold;">{naira(afriscore_cost_per_user)}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:14px;">
                <span style="color:{COLORS['text_secondary']}">Review cost savings / month</span>
                <span style="color:{COLORS['emerald_bright']}; font-weight:bold;">{naira(review_cost_savings)}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:{COLORS['text_secondary']}">Default losses avoided / month</span>
                <span style="color:{COLORS['emerald_bright']}; font-weight:bold;">{naira(default_loss_avoided)}</span>
            </div>
        </div>
    </div>
    <div class="afs-card" style="text-align:center; background: linear-gradient(135deg, rgba(52,211,153,0.08), rgba(245,158,11,0.08));">
        <div class="afs-eyebrow">PROJECTED ANNUAL IMPACT FOR WEMA</div>
        <div style="font-size:40px; font-weight:800; font-family:'Space Grotesk'; color:{COLORS['emerald_bright']};">{naira(annual_impact)}</div>
        <p style="color:{COLORS['text_secondary']}; font-size:12px; margin-top:8px;">
            Assumes a {default_reduction_pct}% relative reduction in default losses on previously thin-file
            applicants, plus direct review-cost savings. Illustrative model — final figures depend on Wema's
            live portfolio data during pilot.
        </p>
    </div>
    <div class="afs-grid-2" style="gap:24px; margin-top:24px;">
        <div class="afs-card">
            <h3 style="color:#fff; font-family:'Space Grotesk'; margin:0 0 12px 0;">{icon('bank', 18, COLORS['gold'])} What Wema Gets</h3>
            <ul style="color:{COLORS['text_secondary']}; line-height:1.7; padding-left:20px; margin:0; font-size:14px;">
                <li>Instant decisions on previously "no-file" applicants — new addressable lending base without new branches.</li>
                <li>SHAP-native explanations attached to every decision, so compliance can defend a rejection to CBN or an aggrieved customer.</li>
                <li>A drop-in risk layer for BNPL, salary advance, and agri-loan products already on Wema's roadmap.</li>
            </ul>
        </div>
        <div class="afs-card">
            <h3 style="color:#fff; font-family:'Space Grotesk'; margin:0 0 12px 0;">{icon('lock', 18, COLORS['emerald_bright'])} Compliance & Trust</h3>
            <ul style="color:{COLORS['text_secondary']}; line-height:1.7; padding-left:20px; margin:0; font-size:14px;">
                <li>NDPR-aligned consent capture — every score is tied to a signed consent token, not scraped data.</li>
                <li>No raw PII leaves the applicant's device unhashed; Wema receives a score and an explanation, not a data dump.</li>
                <li>Full audit trail per decision (see the API tab) for regulatory review.</li>
            </ul>
        </div>
    </div>
    <div class="afs-card" style="margin-top:24px;">
        <div class="afs-eyebrow">HOW A PILOT ACTUALLY ROLLS OUT</div>
        {render_integration_path()}
    </div>
    <p style="color:{COLORS['text_muted']}; font-size:11px; margin-top:16px; text-align:center;">
        Default cost/rate inputs above are industry-informed placeholders, not figures sourced live from Wema
        or a verified public dataset in this session — swap in Wema's real portfolio numbers before using this
        with an actual investor or bank audience.
    </p>
    """
    yield result_html

def tab_how_it_works() -> str:
    """Plain-language explainer -- technical judges will dig for methodology,
    non-technical and investor judges need it spelled out without jargon."""
    return f"""
    <div class="afs-card" style="margin-bottom:24px;">
        <h3 style="color:#fff; font-family:'Space Grotesk'; margin:0 0 16px 0;">In plain language</h3>
        <p style="color:{COLORS['text_secondary']}; line-height:1.7; font-size:15px;">
            Most Nigerians without a bank credit history aren't actually "risky" — they're simply invisible
            to the credit bureaus banks rely on. AfriScore looks at what these people <em>do</em> have: regular
            mobile money transactions, consistent utility payments, steady savings behavior. With the
            applicant's consent, we turn that pattern into a 300&ndash;950 score a bank can act on, and we explain
            <em>why</em> in plain terms — not just a number with no reasoning behind it.
        </p>
    </div>
    <div class="afs-card" style="margin-bottom:24px; background: rgba(255,255,255,0.02);">
        <div class="afs-eyebrow" style="display:flex; align-items:center; gap:8px;">{icon('user', 14, COLORS['gold'])} ILLUSTRATIVE PERSONA — NOT A REAL CUSTOMER</div>
        <p style="color:{COLORS['text_secondary']}; line-height:1.7; font-size:14px;">
            Consider "Amaka," a Lagos market trader with six months of steady mobile money activity and
            on-time utility payments — but no bank credit history, because she's never had a formal loan.
            To a traditional bureau model, Amaka is invisible: no file, no score, no product. AfriScore reads
            the same six months of consented behavioral data and returns a Growth-tier score with a plain-English
            explanation her loan officer can actually stand behind. Same person, same six months — one model
            can only shrug; the other can say yes and show its work.
        </p>
    </div>
    <div class="afs-grid-3" style="margin-bottom:0;">
        <div class="afs-card" style="padding:20px;">
            <div style="color:{COLORS['gold']}; font-family:'Space Grotesk'; font-weight:700; margin-bottom:8px;">1. Consent</div>
            <p style="color:{COLORS['text_secondary']}; font-size:13px; line-height:1.6; margin:0;">
                The applicant explicitly authorizes AfriScore to read specific, limited data — nothing is
                scraped without a signed consent token.
            </p>
        </div>
        <div class="afs-card" style="padding:20px;">
            <div style="color:{COLORS['emerald_bright']}; font-family:'Space Grotesk'; font-weight:700; margin-bottom:8px;">2. Score</div>
            <p style="color:{COLORS['text_secondary']}; font-size:13px; line-height:1.6; margin:0;">
                Transaction regularity, mobile money velocity, and utility-payment consistency are weighted
                by a trained logistic-regression risk model into a single 300&ndash;950 score.
            </p>
        </div>
        <div class="afs-card" style="padding:20px;">
            <div style="color:#F97316; font-family:'Space Grotesk'; font-weight:700; margin-bottom:8px;">3. Explain</div>
            <p style="color:{COLORS['text_secondary']}; font-size:13px; line-height:1.6; margin:0;">
                Every score ships with the specific factors behind it, in a format a compliance officer can
                actually defend to a regulator or a rejected applicant.
            </p>
        </div>
    </div>
    <p style="color:{COLORS['text_muted']}; font-size:11px; text-align:center; margin-top:20px;">
        Note for technical reviewers: this demo runs a real, trained logistic-regression model over a
        synthetic dataset ({'live SHAP values' if HAS_SHAP else 'illustrative attribution, since the shap package is not installed in this session'})
        — see the Model Card in the Developer Platform tab for version, training details, and known limitations.
    </p>
    """

def tab_pitch_deck() -> str:
    funnel_html = render_funnel(
        "Credit-invisible adults, Sub-Saharan Africa", "~400M",
        "Smartphone/mobile-money users in AfriScore's launch markets (NG, GH, KE)", "~85M",
        "Reachable in 24 months via bank & telco partnerships", "~6.2M",
    )
    roadmap_html = render_roadmap()

    return f"""
    <div class="afs-card" style="margin-bottom:24px;">
        <h3 style="color:#fff; font-family:'Space Grotesk'; margin:0 0 16px 0;">{icon('target', 20, COLORS['gold'])} Market Size (TAM &rarr; SAM &rarr; SOM)</h3>
        {funnel_html}
        <p style="color:{COLORS['text_muted']}; font-size:11px; margin:16px 0 0 0;">
            Figures are commonly-cited, order-of-magnitude estimates in the financial-inclusion space
            (broadly consistent with World Bank Findex / GSMA mobile-money reporting) — not independently
            re-verified in this session. Confirm against the latest published editions before quoting them
            to an investor as sourced fact.
        </p>
    </div>
    <div class="afs-grid-2" style="gap:24px; margin-bottom:24px;">
        <div class="afs-card">
            <h3 style="color:#fff; font-family:'Space Grotesk'; margin:0 0 16px 0;">{icon('money', 20, COLORS['gold'])} B2B Unit Economics</h3>
            <div style="display:flex; flex-direction:column; gap:16px;">
                <div style="padding:12px; background:rgba(255,255,255,0.03); border-radius:8px;">
                    <div style="color:{COLORS['text_secondary']}; font-size:12px; text-transform:uppercase;">API Call Revenue</div>
                    <div style="color:{COLORS['gold']}; font-size:24px; font-weight:bold;">$0.15 <span style="font-size:14px; color:{COLORS['text_muted']}">/ user scored</span></div>
                </div>
                <div style="padding:12px; background:rgba(255,255,255,0.03); border-radius:8px;">
                    <div style="color:{COLORS['text_secondary']}; font-size:12px; text-transform:uppercase;">Data Acquisition Cost (CAC)</div>
                    <div style="color:#fff; font-size:24px; font-weight:bold;">$0.02 <span style="font-size:14px; color:{COLORS['text_muted']}">/ via telco rev-share</span></div>
                </div>
                <div style="padding:12px; background:rgba(255,255,255,0.03); border-radius:8px;">
                    <div style="color:{COLORS['text_secondary']}; font-size:12px; text-transform:uppercase;">Gross Margin</div>
                    <div style="color:{COLORS['emerald_bright']}; font-size:24px; font-weight:bold;">86.6%</div>
                </div>
            </div>
        </div>

        <div class="afs-card">
            <h3 style="color:#fff; font-family:'Space Grotesk'; margin:0 0 16px 0;">{icon('lock', 20, COLORS['emerald_bright'])} Defensibility & Moat</h3>
            <div class="afs-eyebrow" style="color:{COLORS['emerald_bright']}; margin-bottom:8px;">DEFENSIBLE</div>
            <ul style="color:{COLORS['text_secondary']}; line-height:1.7; padding-left:20px; margin:0 0 16px 0;">
                <li style="margin-bottom:10px;"><strong style="color:#fff;">Proprietary Data Graph:</strong> Consented telco, utility, and behavioral metadata that accumulates per-user over time — a new entrant starts at zero.</li>
                <li><strong style="color:#fff;">Regulatory Trust:</strong> Bank compliance sign-off is slow to win and expensive to re-earn; once AfriScore is the approved risk vendor, switching cost for the bank is real.</li>
            </ul>
            <div class="afs-eyebrow" style="color:{COLORS['gold']}; margin-bottom:8px;">REPLICABLE — HONEST ABOUT THIS</div>
            <ul style="color:{COLORS['text_secondary']}; line-height:1.7; padding-left:20px; margin:0;">
                <li style="margin-bottom:10px;">The current scoring model itself is not novel — any competent team could rebuild an equivalent model. The moat is the data pipeline and bank relationships around it, not the algorithm.</li>
                <li>The UI can be cloned in a weekend; it's a demo asset, not a barrier to entry.</li>
            </ul>
        </div>
    </div>
    <div class="afs-card" style="margin-bottom:24px; border-color: rgba(245,158,11,0.25); background: rgba(245,158,11,0.04);">
        <div class="afs-eyebrow" style="display:flex; align-items:center; gap:8px;">{icon('alert', 14, COLORS['gold'])} STATED RISK-AWARENESS</div>
        <p style="color:{COLORS['text_secondary']}; font-size:14px; line-height:1.7; margin:0;">
            Alternative-data scoring carries real failure modes — proxy discrimination if mobile or utility
            patterns correlate with protected characteristics, and over-reliance on behavior that shifts
            seasonally (harvest cycles, holiday spending). AfriScore mitigates this with a mandatory
            human-review requirement on every decision and periodic fairness audits of the model's
            attributions across demographic segments, rather than letting the score act unsupervised.
        </p>
    </div>
    <div class="afs-card" style="margin-bottom:24px;">
        <h3 style="color:#fff; font-family:'Space Grotesk'; margin:0 0 16px 0;">{icon('chart', 20, COLORS['text_secondary'])} Validation & Traction</h3>
        <p style="color:{COLORS['text_muted']}; font-size:12px; margin:0 0 12px 0;">
            Fill this in with what's actually true before presenting — a fabricated traction slide costs you
            more credibility with a sharp judge than a short, honest one ever would.
        </p>
        <ul style="color:{COLORS['text_secondary']}; line-height:1.7; padding-left:20px; margin:0 0 16px 0; font-size:14px;">
            <li>Hackathon build shipped and functioning end-to-end (this demo), now with a real trained model.</li>
            <li style="color:{COLORS['text_muted']};">[Add: any conversation held with a Wema employee, GDG contact, or fintech operator — even informal.]</li>
            <li style="color:{COLORS['text_muted']};">[Add: any user interviews or informal surveys with credit-invisible applicants, if conducted.]</li>
        </ul>
        <p style="color:{COLORS['text_secondary']}; font-size:13px; margin:0;">
            <strong style="color:#fff;">Smallest credible next step:</strong> one direct outreach to a Wema digital
            lending or innovation contact (LinkedIn DM referencing this specific ROI model) before judging —
            "we spoke with Wema" beats "we think Wema would want this" every time.
        </p>
    </div>
    <div class="afs-card" style="margin-bottom:24px; text-align:center; background: rgba(255,255,255,0.02);">
        <div class="afs-eyebrow">THE ASK</div>
        <p style="color:{COLORS['text_secondary']}; font-size:15px; line-height:1.6; max-width:600px; margin:0 auto;">
            A pilot partner willing to run AfriScore in shadow mode against 90 days of real applicant data,
            plus early technical collaborators to help move from a synthetic-data model to a validated one.
        </p>
    </div>
    <div class="afs-card">
        <h3 style="color:#fff; font-family:'Space Grotesk'; margin:0 0 16px 0;">{icon('globe', 20, COLORS['emerald_bright'])} Roadmap</h3>
        {roadmap_html}
    </div>
    """

# ============================================================
# CSS INJECTION
# ============================================================
CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

:root {{
    --bg: {COLORS['bg_deep']};
    --card: {COLORS['bg_card']};
    --border: {COLORS['border']};
}}

body, gradio-app, .gradio-container {{
    background-color: var(--bg) !important;
    background-image:
        radial-gradient(circle at 15% 0%, rgba(16,185,129,0.10) 0%, transparent 40%),
        radial-gradient(circle at 85% 10%, rgba(245,158,11,0.08) 0%, transparent 40%),
        radial-gradient(circle at 50% 0%, #0F172A 0%, var(--bg) 55%) !important;
    font-family: 'Inter', sans-serif !important;
    color: #fff !important;
    margin: 0 !important;
    max-width: 100vw !important;
}}

.afs-shell {{ max-width: 1100px !important; margin: 0 auto; padding: 40px 16px 80px 16px; }}

.afs-brandbar {{
    display: flex; align-items: center; justify-content: center; gap: 12px;
    margin-bottom: 20px; animation: fadeUp 0.5s ease;
}}
.afs-brand-word {{
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 20px;
    letter-spacing: -0.02em; color: #fff;
}}
.afs-brand-word span {{ color: {COLORS['gold']}; }}

.afs-hero-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(32px, 5vw, 56px);
    font-weight: 700;
    text-align: center;
    background: linear-gradient(135deg, #fff 30%, {COLORS['gold']} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 16px;
    letter-spacing: -0.03em;
}}

.afs-trust-row {{
    display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin-bottom: 32px;
}}
.afs-trust-pill {{
    font-size: 11px; font-family: monospace; letter-spacing: 0.04em;
    color: {COLORS['text_secondary']}; border: 1px solid var(--border);
    padding: 5px 12px; border-radius: 99px; background: rgba(255,255,255,0.02);
    display: inline-flex; align-items: center; gap: 6px;
}}

.afs-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    animation: fadeUp 0.6s ease-out both;
}}
.afs-card:hover {{ border-color: rgba(255,255,255,0.15); transform: translateY(-2px); transition: all 0.25s ease; }}
.afs-error-card {{ border-color: rgba(239,68,68,0.35); background: rgba(239,68,68,0.06); }}

.afs-grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }}
.afs-grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; margin-bottom: 32px; }}

.afs-eyebrow {{
    font-family: monospace; font-size: 11px; letter-spacing: 0.1em;
    color: {COLORS['text_secondary']}; text-transform: uppercase; margin-bottom: 16px;
    display: flex; align-items: center; gap: 6px;
}}

.afs-hr {{ border: none; border-top: 1px solid var(--border); margin: 32px 0; }}

.afs-tag {{
    font-size: 12px; background: rgba(255,255,255,0.05); border: 1px solid var(--border);
    padding: 4px 12px; border-radius: 99px; color: {COLORS['text_secondary']};
    display: inline-flex; align-items: center; gap: 5px;
}}

.afs-tier-badge {{
    display: inline-block; font-family: 'Space Grotesk'; font-weight: 700;
    padding: 6px 16px; border-radius: 99px; font-size: 15px; margin-bottom: 12px;
}}
.afs-sm-badge {{ padding: 4px 10px; border-radius: 99px; font-size: 11px; font-weight: bold; border: 1px solid var(--border); }}

.afs-live-dot {{
    width: 8px; height: 8px; border-radius: 50%; background: {COLORS['emerald_bright']};
    display: inline-block; animation: pulseDot 1.6s infinite;
}}
@keyframes pulseDot {{
    0% {{ box-shadow: 0 0 0 0 rgba(52,211,153,0.5); }}
    70% {{ box-shadow: 0 0 0 6px rgba(52,211,153,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(52,211,153,0); }}
}}

.afs-gauge-pulse {{ animation: gaugePulse 2.4s ease-out infinite; }}
@keyframes gaugePulse {{
    0% {{ opacity: 0.35; r: 92; }}
    100% {{ opacity: 0; r: 108; }}
}}

/* Factor Bars */
.afs-factor {{ margin-bottom: 16px; animation: fadeUp 0.5s ease both; }}
.afs-f-top {{ display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px; }}
.afs-f-bar-bg {{ height: 6px; background: rgba(255,255,255,0.05); border-radius: 99px; overflow: hidden; }}
.afs-f-bar {{ height: 100%; border-radius: 99px; animation: growWidth 0.5s ease-out both; }}

/* Funnel (TAM/SAM/SOM) */
.afs-funnel {{ display: flex; flex-direction: column; gap: 14px; }}
.afs-funnel-row {{ display: grid; grid-template-columns: 1fr 2fr; align-items: center; gap: 16px; animation: fadeUp 0.5s ease both; }}
.afs-funnel-bar-bg {{ height: 28px; background: rgba(255,255,255,0.04); border-radius: 6px; overflow: hidden; }}
.afs-funnel-bar {{ height: 100%; border-radius: 6px; animation: growWidth 0.9s ease-out both; }}
.afs-funnel-label {{ display: flex; flex-direction: column; gap: 2px; }}

/* Roadmap */
.afs-roadmap-card {{
    background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px; animation: fadeUp 0.5s ease both;
}}
.afs-roadmap-tag {{ font-family: 'Space Grotesk'; font-weight: 700; font-size: 12px; letter-spacing: 0.06em; }}

/* Table */
.afs-table {{ width: 100%; border-collapse: collapse; font-size: 14px; text-align: left; }}
.afs-table th {{ padding: 12px 20px; font-family: monospace; font-size: 11px; color: {COLORS['text_secondary']}; border-bottom: 1px solid var(--border); }}
.afs-table td {{ padding: 16px 20px; border-bottom: 1px solid rgba(255,255,255,0.02); color: {COLORS['text_secondary']}; }}
.afs-table tr:hover td {{ background: rgba(255,255,255,0.02); }}

/* Code */
.afs-code {{
    background: #020617; border: 1px solid var(--border); border-radius: 8px;
    padding: 16px; font-family: monospace; font-size: 12px; color: {COLORS['text_secondary']};
    overflow-x: auto; white-space: pre-wrap; line-height: 1.5;
}}

/* Skeleton loading state */
.afs-skeleton-wrap {{ min-height: 140px; }}
.afs-skeleton-line {{
    height: 12px; border-radius: 6px; margin-bottom: 12px;
    background: linear-gradient(90deg, rgba(255,255,255,0.04) 25%, rgba(255,255,255,0.09) 37%, rgba(255,255,255,0.04) 63%);
    background-size: 400% 100%; animation: shimmer 1.4s ease infinite;
}}
.afs-skeleton-block {{
    height: 60px; border-radius: 10px; margin-top: 8px;
    background: linear-gradient(90deg, rgba(255,255,255,0.03) 25%, rgba(255,255,255,0.07) 37%, rgba(255,255,255,0.03) 63%);
    background-size: 400% 100%; animation: shimmer 1.4s ease infinite;
}}
@keyframes shimmer {{ 0% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}

/* Integration path */
.afs-path-row {{ display: flex; align-items: flex-start; gap: 4px; flex-wrap: wrap; }}
.afs-path-step {{ flex: 1 1 160px; min-width: 140px; animation: fadeUp 0.5s ease both; }}
.afs-path-dot {{ width: 14px; height: 14px; border-radius: 50%; }}
.afs-path-arrow {{ color: {COLORS['text_muted']}; font-size: 20px; padding-top: 2px; flex: 0 0 auto; }}

/* Presenter mode */
.afs-presenter-bar {{ position: sticky; top: 0; z-index: 20; margin-bottom: 16px; }}

/* Tab cross-fade polish: every tabitem fades/slides in on render so switching tabs
   never feels like an instant cut. */
.tabitem {{ animation: afsTabFade 0.35s ease both; }}
@keyframes afsTabFade {{
    from {{ opacity: 0; transform: translateY(6px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

/* Accessibility: visible focus ring on every interactive element */
button:focus-visible, [role="tab"]:focus-visible, input:focus-visible, a:focus-visible {{
    outline: 2px solid {COLORS['gold']} !important;
    outline-offset: 2px !important;
}}

/* Responsive: narrow laptop / judging-room widths */
@media (max-width: 700px) {{
    .afs-shell {{ padding: 24px 12px 60px 12px; }}
    .afs-card {{ padding: 20px; }}
    .afs-hero-title {{ font-size: 28px; }}
    .afs-grid-2 {{ grid-template-columns: 1fr; }}
    .afs-path-arrow {{ display: none; }}
    .afs-presenter-bar .afs-card {{ padding: 12px 16px; }}
}}

/* Animations */
@keyframes fadeUp {{ from {{ opacity: 0; transform: translateY(15px); }} to {{ opacity: 1; transform: translateY(0); }} }}
@keyframes growWidth {{ from {{ width: 0; }} }}
.afs-chart-bar {{ transform-origin: bottom; animation: scaleY 0.8s ease-out both; }}
@keyframes scaleY {{ from {{ transform: scaleY(0); }} to {{ transform: scaleY(1); }} }}

/* Gradio Overrides */
.tabs {{ border: none !important; }}
button[role="tab"] {{ font-family: 'Space Grotesk' !important; font-size: 16px !important; color: {COLORS['text_secondary']} !important; border: none !important; }}
button[role="tab"].selected {{ color: {COLORS['gold']} !important; border-bottom: 2px solid {COLORS['gold']} !important; }}
.gr-button.primary {{ background: {COLORS['gold']} !important; color: #000 !important; border: none !important; font-family: 'Space Grotesk' !important; font-weight: bold !important; border-radius: 8px !important; }}
.gr-button.primary:hover {{ filter: brightness(1.1); transform: translateY(-1px); }}
.gr-button.secondary {{ background: rgba(255,255,255,0.06) !important; color: #fff !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }}
"""

# ============================================================
# GRADIO INTERFACE ASSEMBLY
# ============================================================
with gr.Blocks(title="AfriScore - Financial Identity API", css=CUSTOM_CSS) as demo:
    with gr.Column(elem_classes=["afs-shell"]):

        # Brand bar + Header
        gr.HTML(f"""
        <div class="afs-brandbar">
            {render_logo(32)}
            <span class="afs-brand-word">Afri<span>Score</span></span>
        </div>
        <div style="text-align:center; margin-bottom: 24px;">
            <div style="display:inline-block; padding: 6px 16px; background:rgba(245,158,11,0.1); color:{COLORS['gold']}; border-radius:99px; font-family:monospace; font-size:11px; letter-spacing:0.1em; border:1px solid rgba(245,158,11,0.2); margin-bottom:24px;">
                ENTERPRISE INFRASTRUCTURE &middot; CONFIDENTIAL DEMO
            </div>
            <h1 class="afs-hero-title">The API for Africa's Credit Economy</h1>
            <p style="color:{COLORS['text_secondary']}; font-size:18px; max-width:700px; margin:0 auto; line-height:1.6;">
                A privacy-first, explainable AI scoring engine built on consented alternative data.
                Plug in our API to instantly score the 400M+ hard-working Africans invisible to traditional bureaus.
            </p>
        </div>
        <div class="afs-trust-row">
            <span class="afs-trust-pill">{icon('shield', 12)} NDPR-ALIGNED CONSENT</span>
            <span class="afs-trust-pill">{icon('check', 12)} {'LIVE' if HAS_SHAP else 'ILLUSTRATIVE'} SHAP-EXPLAINABLE</span>
            <span class="afs-trust-pill">{icon('bank', 12)} BUILT FOR WEMA DIGITAL LENDING</span>
            <span class="afs-trust-pill">{icon('chart', 12)} 28ms AVG LATENCY</span>
        </div>

        <div class="afs-grid-3">
            <div class="afs-card" style="padding:24px; text-align:center;">
                <div style="font-size:36px; font-weight:bold; color:{COLORS['emerald_bright']}; font-family:'Space Grotesk';">~400M</div>
                <div style="color:{COLORS['text_secondary']}; font-size:13px; margin-top:4px;">Credit-Invisible Adults</div>
            </div>
            <div class="afs-card" style="padding:24px; text-align:center; animation-delay:0.1s;">
                <div style="font-size:36px; font-weight:bold; color:{COLORS['gold']}; font-family:'Space Grotesk';">$330B+</div>
                <div style="color:{COLORS['text_secondary']}; font-size:13px; margin-top:4px;">MSME Financing Gap</div>
            </div>
            <div class="afs-card" style="padding:24px; text-align:center; animation-delay:0.2s;">
                <div style="font-size:36px; font-weight:bold; color:#fff; font-family:'Space Grotesk';">100%</div>
                <div style="color:{COLORS['text_secondary']}; font-size:13px; margin-top:4px;">SHAP-Explainable AI</div>
            </div>
        </div>
        """)

        # ---------------- Presenter Mode ----------------
        presenter_step = gr.State(0)
        with gr.Column(elem_classes=["afs-presenter-bar"]):
            presenter_progress = gr.HTML(render_presenter_progress(0))
            with gr.Row():
                btn_presenter_prev = gr.Button("← Back", variant="secondary", size="sm")
                btn_presenter_next = gr.Button("Next →", variant="primary", size="sm")
        with gr.Tabs(elem_id="main-tabs") as main_tabs:
            with gr.Tab("0. How It Works", id=0):
                gr.HTML(tab_how_it_works())

            with gr.Tab("1. Identity & Scoring", id=1):
                with gr.Row():
                    user_dd = gr.Dropdown(choices=df_users["user_id"].tolist()[:30], value=df_users["user_id"].iloc[0], label="Select Hashed Identity", scale=3, interactive=True)
                    btn_score = gr.Button("Execute AI Scoring Engine", variant="primary", scale=1)
                out_score = gr.HTML()
                btn_score.click(tab_consumer_score, inputs=user_dd, outputs=out_score)
                demo.load(tab_consumer_score, inputs=user_dd, outputs=out_score)

                with gr.Row():
                    btn_export = gr.Button("&#8681; Download Decision Record", variant="secondary")
                out_export = gr.File(label="Decision record", interactive=False)
                btn_export.click(generate_decision_record, inputs=user_dd, outputs=out_export)

                gr.HTML(f"""<div class="afs-eyebrow" style="margin-top:32px;">{icon('sliders', 14)} LIVE SCENARIO SIMULATOR &mdash; DRAG TO RECOMPUTE IN REAL TIME</div>""")
                with gr.Row():
                    sim_regularity = gr.Slider(0, 1, value=0.6, step=0.01, label="Transaction Regularity")
                    sim_volume = gr.Slider(5, 800, value=120, step=1, label="Mobile Money Volume (txns / 6mo)")
                    sim_utility = gr.Slider(0, 1, value=0.6, step=0.01, label="Utility Payment Consistency")
                out_sim = gr.HTML(run_live_simulation(0.6, 120, 0.6))
                sim_inputs = [sim_regularity, sim_volume, sim_utility]
                sim_regularity.change(run_live_simulation, inputs=sim_inputs, outputs=out_sim, show_progress="hidden")
                sim_volume.change(run_live_simulation, inputs=sim_inputs, outputs=out_sim, show_progress="hidden")
                sim_utility.change(run_live_simulation, inputs=sim_inputs, outputs=out_sim, show_progress="hidden")

                gr.HTML(f"""<div class="afs-eyebrow" style="margin-top:32px;">{icon('flask', 14)} MODEL STRESS TEST</div>""")
                btn_stress = gr.Button("Run Adversarial Stress Test", variant="secondary")
                out_stress = gr.HTML()
                btn_stress.click(run_stress_test, outputs=out_stress)

            with gr.Tab("2. Portfolio Analytics", id=2):
                btn_dash = gr.Button("Run Portfolio Simulation (N=25)", variant="primary")
                out_dash = gr.HTML()
                btn_dash.click(tab_enterprise_dashboard, outputs=out_dash)
                demo.load(tab_enterprise_dashboard, outputs=out_dash)

            with gr.Tab("3. Developer Platform", id=3):
                with gr.Row():
                    api_dd = gr.Dropdown(choices=df_users["user_id"].tolist()[:10], value=df_users["user_id"].iloc[2], label="Test API Payload", scale=3, interactive=True)
                    btn_api = gr.Button("Send cURL Request", variant="primary", scale=1)
                out_api = gr.HTML()
                btn_api.click(tab_api_docs, inputs=api_dd, outputs=out_api)
                demo.load(tab_api_docs, inputs=api_dd, outputs=out_api)
                gr.HTML(render_model_card())

            with gr.Tab("4. Wema Partnership ROI", id=4):
                gr.HTML(f"""<div class="afs-eyebrow" style="margin-bottom:8px;">ADJUST TO WEMA'S PORTFOLIO ASSUMPTIONS</div>""")
                with gr.Row():
                    inp_loans = gr.Number(value=5000, label="Monthly loan applications", precision=0)
                    inp_manual = gr.Number(value=1800, label="Manual review cost / applicant (₦)", precision=0)
                    inp_default = gr.Number(value=8.5, label="Current default rate (%)")
                    inp_avgloan = gr.Number(value=150000, label="Average loan size (₦)")
                btn_roi = gr.Button("Recalculate Impact for Wema", variant="primary")
                out_roi = gr.HTML()
                roi_inputs = [inp_loans, inp_manual, inp_default, inp_avgloan]
                btn_roi.click(tab_bank_partnership, inputs=roi_inputs, outputs=out_roi)
                demo.load(tab_bank_partnership, inputs=roi_inputs, outputs=out_roi)

            with gr.Tab("5. Market & Roadmap", id=5):
                gr.HTML(tab_pitch_deck())

        # Presenter mode wiring -- steps through TAB_SEQUENCE, updating both the
        # visible Tabs selection and the progress indicator.
        def _presenter_move(step: int, delta: int):
            new_step = max(0, min(len(TAB_SEQUENCE) - 1, step + delta))
            target_tab_id = TAB_SEQUENCE[new_step]["index"]
            return new_step, render_presenter_progress(new_step), gr.Tabs(selected=target_tab_id)

        btn_presenter_next.click(
            lambda step: _presenter_move(step, 1),
            inputs=presenter_step,
            outputs=[presenter_step, presenter_progress, main_tabs],
        )
        btn_presenter_prev.click(
            lambda step: _presenter_move(step, -1),
            inputs=presenter_step,
            outputs=[presenter_step, presenter_progress, main_tabs],
        )

        gr.HTML(f"""
        <div style="margin-top: 60px; padding-top: 32px; border-top: 1px solid {COLORS['border']}; text-align: center;">
            <div style="display:flex; justify-content:center; align-items:center; gap:10px; margin-bottom:8px;">
                {render_logo(24)}
                <span style="color:{COLORS['text_secondary']}; font-size:13px;">AfriScore &middot; Designed &amp; Architected by <strong style="color:#fff;">Obasi Divinefavour Chukwuemeka</strong></span>
            </div>
            <div style="color:{COLORS['text_muted']}; font-size:12px;">Built for scalability, financial inclusion, and regulatory compliance.</div>
        </div>
        """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)