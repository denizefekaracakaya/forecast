"""E-Forecast App — Streamlit UI with auth, dark theme, bilingual support, and personalization."""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecast.config import DEFAULT_PARAMS, TICKER_GROUPS, settings
from forecast.data.fundamentals import compute_price_stats, fetch_fundamentals
from forecast.data.news import get_news_provider
from forecast.data.normalization import normalize_ohlcv
from forecast.data.providers.yahoo import DataFetchError, YahooProvider
from forecast.forecasting.model import (
    ModelError,
    currency_symbol,
    evaluate,
    forecast,
    naive_baseline_metrics,
    prepare_prophet_df,
    train_model,
    validate_ticker,
)
from forecast.i18n import _

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_TTL = settings.CACHE_TTL_SECONDS
GATEWAY_BASE = f"http://{settings.GATEWAY_HOST}:{settings.GATEWAY_PORT}"

DEFAULT_STATE = {
    "lang": "tr",
    "theme": "light",
    "auth_page": "login",
    "jwt": None,
    "user_id": None,
    "username": "",
    "email": "",
    "is_verified": False,
    "has_password": False,
    "last_login": None,
    "auth_error": None,
    "auth_success": None,
    "login_loading": False,
    "reg_loading": False,
    "reset_token": None,
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

L = lambda key, **kw: _(key, st.session_state.lang, **kw)

st.set_page_config(page_title="E-Forecast", page_icon="🔮", layout="wide")

# ── Dark Theme CSS ────────────────────────────────────────────────────────

_THEME_CSS = """
<style>
    :root {
        --bg: #ffffff;
        --bg-secondary: #f8f9fa;
        --text: #111827;
        --text-muted: #6b7280;
        --text-light: #9ca3af;
        --border: #f0f0f0;
        --card-bg: #f8f9fa;
        --primary: #2563eb;
        --error: #dc2626;
        --success: #16a34a;
        --sentiment-bullish: #16a34a;
        --sentiment-bearish: #dc2626;
        --sentiment-neutral: #6b7280;
        --chart-history: #2563eb;
        --chart-forecast: #16a34a;
        --chart-fill: rgba(22, 163, 74, 0.15);
    }
    .dark {
        --bg: #0e1117;
        --bg-secondary: #1e1e2e;
        --text: #e4e4e7;
        --text-muted: #a1a1aa;
        --text-light: #71717a;
        --border: #27272a;
        --card-bg: #1e1e2e;
        --primary: #60a5fa;
        --error: #f87171;
        --success: #4ade80;
        --sentiment-bullish: #4ade80;
        --sentiment-bearish: #f87171;
        --sentiment-neutral: #a1a1aa;
        --chart-history: #60a5fa;
        --chart-forecast: #4ade80;
        --chart-fill: rgba(74, 222, 128, 0.15);
    }
    section[data-testid="stAppViewBlockContainer"] { background-color: var(--bg); color: var(--text); }
    section[data-testid="stSidebar"] { background-color: var(--bg); }
    section[data-testid="stSidebar"] .st-emotion-cache-1gv3huu { background-color: var(--bg-secondary); }
    .stApp { background-color: var(--bg); color: var(--text); }
    .main-header { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.2rem; color: var(--text); }
    .main-header span { color: var(--text-muted); font-weight: 400; font-size: 1rem; }
    .news-item { padding: 0.5rem 0; border-bottom: 1px solid var(--border); }
    .news-item:last-child { border-bottom: none; }
    .news-title { font-weight: 600; font-size: 0.95rem; color: var(--text); }
    .news-meta { font-size: 0.75rem; color: var(--text-light); }
    .stTabs [data-baseweb="tab-list"] { gap: 0; }
    .stTabs [data-baseweb="tab"] { padding: 0.4rem 1rem; }
    div[data-testid="stMetric"] { background: var(--card-bg); border-radius: 8px; padding: 0.5rem; }
    div.stButton > button { width: 100%; }
    div.stTextInput > div > input { background: var(--bg-secondary); color: var(--text); border-color: var(--border); }
    div.stAlert { background: var(--bg-secondary); border: 1px solid var(--border); color: var(--text); border-radius: 6px; }
    div[data-testid="stSelectbox"] > div > div { background: var(--bg-secondary); color: var(--text); }
    div[data-testid="stSelectbox"] svg { fill: var(--text-muted); }
    div[data-testid="stDataFrame"] { background: var(--bg-secondary); color: var(--text); }
    button[data-baseweb="tab"] { color: var(--text-muted) !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: var(--text) !important; }
    div.stSlider > div > div > div { background: var(--primary); }
    div[data-testid="stRadio"] > div > label { color: var(--text); }
    div[data-testid="stCheckbox"] > label { color: var(--text); }
    div.st-emotion-cache-1n76uvr { color: var(--text); }
    button[data-testid="baseButton-header"] { color: var(--text-muted); }
    * { transition: background-color 0.2s ease, color 0.2s ease; }
    @media (max-width: 640px) {
        div[data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
    }
</style>
<script>
    (function() {
        var theme = sessionStorage.getItem('forecast-theme');
        if (theme === 'dark') { document.documentElement.classList.add('dark'); }
    })();
</script>
"""

st.markdown(_THEME_CSS, unsafe_allow_html=True)


# ── Price helper (for favorites cards) ────────────────────────────────────


@st.cache_data(ttl=60, show_spinner=False)
def cached_price_stats(ticker: str) -> dict | None:
    try:
        provider = YahooProvider()
        raw = provider.fetch_history(ticker)
        if raw is None or raw.empty:
            return None
        df = normalize_ohlcv(raw, ticker, source="yahoo")
        return compute_price_stats(df)
    except Exception:
        return None


# ── API helpers (return (data, error)) ────────────────────────────────────


def _api_get(path: str) -> tuple[dict | list | None, str | None]:
    try:
        headers = {}
        if st.session_state.jwt:
            headers["Authorization"] = f"Bearer {st.session_state.jwt}"
        resp = httpx.get(f"{GATEWAY_BASE}{path}", headers=headers, timeout=15)
        if resp.status_code == 401:
            return None, L("auth.error.invalid_credentials")
        if resp.status_code == 403:
            return None, resp.json().get("detail", "Access denied")
        if resp.status_code == 404:
            return None, "Not found"
        resp.raise_for_status()
        return resp.json(), None
    except httpx.ConnectError:
        return None, L("auth.error.network")
    except Exception as exc:
        return None, str(exc)


def _api_post(path: str, body: dict) -> tuple[dict | None, str | None]:
    try:
        headers = {"Content-Type": "application/json"}
        if st.session_state.jwt:
            headers["Authorization"] = f"Bearer {st.session_state.jwt}"
        resp = httpx.post(f"{GATEWAY_BASE}{path}", json=body, headers=headers, timeout=15)
        if resp.status_code in (400, 409):
            return resp.json(), resp.json().get("detail", "Bad request")
        if resp.status_code == 401:
            return None, L("auth.error.invalid_credentials")
        if resp.status_code == 403:
            detail = resp.json().get("detail", "")
            return None, detail
        if resp.status_code == 404:
            return None, resp.json().get("detail", "Not found")
        if resp.status_code == 410:
            detail = resp.json().get("detail", "Link expired")
            return None, detail
        resp.raise_for_status()
        return resp.json(), None
    except httpx.ConnectError:
        return None, L("auth.error.network")
    except Exception as exc:
        return None, str(exc)


def _api_put(path: str, body: dict) -> tuple[dict | None, str | None]:
    try:
        headers = {"Content-Type": "application/json"}
        if st.session_state.jwt:
            headers["Authorization"] = f"Bearer {st.session_state.jwt}"
        resp = httpx.put(f"{GATEWAY_BASE}{path}", json=body, headers=headers, timeout=15)
        if resp.status_code in (400, 409):
            return resp.json(), resp.json().get("detail", "Bad request")
        if resp.status_code == 401:
            return None, L("auth.error.invalid_credentials")
        if resp.status_code == 403:
            return None, resp.json().get("detail", "Access denied")
        if resp.status_code == 404:
            return None, resp.json().get("detail", "Not found")
        resp.raise_for_status()
        return resp.json(), None
    except httpx.ConnectError:
        return None, L("auth.error.network")
    except Exception as exc:
        return None, str(exc)


def _api_delete(path: str) -> tuple[dict | None, str | None]:
    try:
        headers = {}
        if st.session_state.jwt:
            headers["Authorization"] = f"Bearer {st.session_state.jwt}"
        resp = httpx.delete(f"{GATEWAY_BASE}{path}", headers=headers, timeout=15)
        if resp.status_code == 404:
            return None, "Not found"
        if resp.status_code == 401:
            return None, L("auth.error.invalid_credentials")
        if resp.status_code == 403:
            return None, resp.json().get("detail", "Access denied")
        resp.raise_for_status()
        return resp.json(), None
    except httpx.ConnectError:
        return None, L("auth.error.network")
    except Exception as exc:
        return None, str(exc)


# ── Auth helper functions ─────────────────────────────────────────────────


def _do_login(login_id: str, password: str):
    st.session_state.auth_error = None
    st.session_state.auth_success = None
    if not login_id or not password:
        st.session_state.auth_error = L("auth.error.invalid_credentials")
        return
    st.session_state.login_loading = True
    data, err = _api_post("/api/v1/auth/login", {"login": login_id, "password": password})
    st.session_state.login_loading = False
    if err:
        st.session_state.auth_error = err
        return
    st.session_state.jwt = data["token"]
    st.session_state.user_id = data["id"]
    st.session_state.username = data["username"]
    st.session_state.email = data.get("email", "") or ""
    st.session_state.is_verified = data.get("is_verified", False)
    st.session_state.has_password = True
    st.session_state.last_login = data.get("last_login")
    st.session_state.lang = data.get("language", st.session_state.lang)
    st.session_state.auth_page = "logged_in"
    st.session_state.auth_success = None


def _do_register(email: str, username: str, password: str):
    st.session_state.auth_error = None
    st.session_state.auth_success = None
    if not email or "@" not in email:
        st.session_state.auth_error = L("auth.error.email_required")
        return
    if not username or len(username) < 2:
        st.session_state.auth_error = L("auth.error.empty_username")
        return
    if len(password) < 6:
        st.session_state.auth_error = L("auth.error.weak_password")
        return
    st.session_state.reg_loading = True
    data, err = _api_post("/api/v1/auth/register", {
        "email": email, "username": username, "password": password, "language": st.session_state.lang,
    })
    st.session_state.reg_loading = False
    if err:
        st.session_state.auth_error = err
        return
    st.session_state.jwt = data["token"]
    st.session_state.user_id = data["id"]
    st.session_state.username = data["username"]
    st.session_state.email = data.get("email", "") or ""
    st.session_state.is_verified = data.get("is_verified", False)
    st.session_state.has_password = True
    st.session_state.auth_success = L("login.verify_sent")
    st.session_state.auth_page = "verify_notice"


def _do_logout():
    st.session_state.jwt = None
    st.session_state.user_id = None
    st.session_state.username = ""
    st.session_state.email = ""
    st.session_state.is_verified = False
    st.session_state.has_password = False
    st.session_state.last_login = None
    st.session_state.auth_page = "login"
    st.session_state.auth_error = None
    st.session_state.auth_success = None


# ── Check URL params for verify/reset tokens ─────────────────────────────


query_params = st.query_params
if "verify" in query_params:
    token = query_params["verify"]
    data, err = _api_post("/api/v1/auth/verify-email", {"token": token})
    if err:
        st.session_state.auth_success = None
        st.session_state.auth_error = L("login.verify_error")
    else:
        st.session_state.auth_error = None
        st.session_state.auth_success = L("login.verify_success")
        st.session_state.is_verified = True
    st.query_params.clear()
    st.rerun()

if "reset" in query_params:
    st.session_state.reset_token = query_params["reset"]
    st.session_state.auth_page = "reset_password"
    st.query_params.clear()
    st.rerun()


# ── Cached data functions ────────────────────────────────────────────────


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_forecast_pipeline(ticker: str, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    provider = YahooProvider()
    raw = provider.fetch_history(ticker)
    normalized = normalize_ohlcv(raw, ticker, source="yahoo")
    prophet_df = prepare_prophet_df(normalized)
    df = prophet_df
    params = DEFAULT_PARAMS
    model = train_model(df, params)
    metrics = evaluate(model, df)
    baseline = naive_baseline_metrics(df)
    pred = forecast(model, horizon, history_df=df)
    return prophet_df, normalized, pred, params, metrics, baseline


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_fundamentals(ticker: str) -> dict:
    return fetch_fundamentals(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_news(ticker: str, max_results: int = 10) -> list:
    provider = get_news_provider()
    news = provider.fetch_news(ticker, max_results=max_results)
    return [
        {
            "id": n.id, "title": n.title, "summary": n.summary, "source": n.source,
            "url": n.url, "published_at": n.published_at, "sentiment": n.sentiment,
            "sentiment_score": n.sentiment_score, "importance_score": n.importance_score,
            "sectors": n.sectors, "source_provider": n.source_provider,
        }
        for n in news
    ]


def build_chart(history: pd.DataFrame, prediction: pd.DataFrame, ticker: str, currency: str) -> go.Figure:
    is_dark = st.session_state.theme == "dark"
    hist_color = "#60a5fa" if is_dark else "#2563eb"
    fc_color = "#4ade80" if is_dark else "#16a34a"
    fc_fill = "rgba(74, 222, 128, 0.15)" if is_dark else "rgba(22, 163, 74, 0.15)"
    hist = history.copy(); hist["ds"] = pd.to_datetime(hist["ds"])
    pred = prediction.copy(); pred["ds"] = pd.to_datetime(pred["ds"])
    last_hist = hist["ds"].max()
    future = pred[pred["ds"] > last_hist]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["ds"], y=hist["y"], mode="lines", name=L("history.label"), line=dict(color=hist_color, width=2)))
    fig.add_trace(go.Scatter(x=future["ds"], y=future["yhat"], mode="lines", name=L("forecast.label"), line=dict(color=fc_color, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=pd.concat([future["ds"], future["ds"].iloc[::-1]]), y=pd.concat([future["yhat_upper"], future["yhat_lower"].iloc[::-1]]), fill="toself", fillcolor=fc_fill, line=dict(color="rgba(255,255,255,0)"), name=L("confidence_interval"), showlegend=True))
    fig.update_layout(title=f"{ticker} — {L('chart.title')}", xaxis_title=L("chart.xaxis"), yaxis_title=f"{L('chart.yaxis')} ({currency})", hovermode="x unified", height=420, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02), template="plotly_dark" if is_dark else "plotly", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e4e4e7" if is_dark else "#111827")
    return fig


def _fmt(val, suffix=""):
    if val is None: return "—"
    return f"{val:,.2f}{suffix}"

def _fmt_int(val):
    if val is None: return "—"
    if val >= 1_000_000_000: return f"{val / 1_000_000_000:.2f}B"
    if val >= 1_000_000: return f"{val / 1_000_000:.2f}M"
    if val >= 1_000: return f"{val / 1_000:.2f}K"
    return f"{val:,.0f}"


# ── Sidebar ──────────────────────────────────────────────────────────────

st.sidebar.markdown(f'<div class="main-header">🔮 E-Forecast <span>| {L("app.subtitle")}</span></div>', unsafe_allow_html=True)
st.sidebar.caption(L("app.description"))
st.sidebar.warning(L("app.warning"), icon="⚠️")
st.sidebar.divider()

# Language & Theme
lang_col, theme_col = st.sidebar.columns(2)
with lang_col:
    lang_sel = st.selectbox(L("language.label"), options=["tr", "en"],
        format_func=lambda x: L(f"language.{x}"), label_visibility="collapsed", key="lang_sel")
    if lang_sel != st.session_state.lang:
        st.session_state.lang = lang_sel
        st.session_state.auth_error = None
        st.session_state.auth_success = None
        st.rerun()
with theme_col:
    is_dark = st.session_state.theme == "dark"
    if st.button(L("theme.dark" if is_dark else "theme.light"), use_container_width=True):
        st.session_state.theme = "dark" if not is_dark else "light"
        st.session_state.auth_error = None
        st.session_state.auth_success = None
        st.markdown(f"<script>document.querySelector('html').classList.toggle('dark');sessionStorage.setItem('forecast-theme','{'dark' if not is_dark else 'light'}');</script>", unsafe_allow_html=True)
        st.rerun()

if st.session_state.theme == "dark":
    st.markdown("<script>document.querySelector('html').classList.add('dark');</script>", unsafe_allow_html=True)

st.sidebar.divider()

# ── Auth Section in Sidebar ──────────────────────────────────────────────

if st.session_state.auth_page == "guest":
    st.sidebar.caption(L("login.or_continue_guest"))
    if st.button(L("login.title"), use_container_width=True, key="sidebar_login"):
        st.session_state.auth_page = "login"; st.rerun()
    st.sidebar.divider()

elif st.session_state.auth_page in ("logged_in", "my_account"):
    # Compact user indicator
    st.sidebar.success(f"👤 {st.session_state.username}", icon=None)
    if st.session_state.last_login:
        try:
            dt = datetime.fromisoformat(st.session_state.last_login.replace("Z", "+00:00"))
            delta = datetime.now(timezone.utc) - dt
            mins = int(delta.total_seconds() / 60)
            if mins < 60:
                st.sidebar.caption(f"{L('login.last_login')} {mins} {'dk' if st.session_state.lang == 'tr' else 'min'}")
            else:
                st.sidebar.caption(f"{L('login.last_login')} {mins // 60} {'saat' if st.session_state.lang == 'tr' else 'hr'}")
        except Exception:
            pass
    if st.session_state.email and not st.session_state.is_verified:
        st.sidebar.warning("⚠️ " + L("login.verify_email"), icon=None)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button(L("login.my_account"), use_container_width=True, key="my_account_btn"):
            st.session_state.auth_page = "my_account"
            st.rerun()
    with col2:
        if st.button(L("logout.button") if st.session_state.lang == "tr" else "🚪 Logout", use_container_width=True, key="logout_btn"):
            _do_logout()
            st.rerun()

    # Compact favorites in sidebar
    if st.session_state.jwt:
        st.sidebar.divider()
        st.sidebar.markdown(f"**{L('favorites.title')}**")
        wl_data, wl_err = _api_get(f"/api/v1/users/{st.session_state.user_id}/watchlist")
        wl = wl_data if isinstance(wl_data, list) else []
        if wl:
            for w in wl:
                wc = st.sidebar.columns([3, 1])
                wc[0].caption(w)
                if wc[1].button("✕", key=f"wl_rm_{w}", help=L("watchlist.remove")):
                    _api_delete(f"/api/v1/users/{st.session_state.user_id}/watchlist/{w}")
                    st.rerun()
        else:
            st.sidebar.caption(L("watchlist.empty"))

        with st.sidebar.form("add_watchlist_form"):
            add_ticker = st.text_input("", placeholder=L("watchlist.placeholder"), label_visibility="collapsed")
            if st.form_submit_button(L("watchlist.add")):
                if add_ticker.strip():
                    _api_post(f"/api/v1/users/{st.session_state.user_id}/watchlist", {"ticker": add_ticker.strip().upper()})
                    st.rerun()

        with st.sidebar.expander(L("preferences.title")):
            prefs_data, _ = _api_get(f"/api/v1/users/{st.session_state.user_id}/preferences")
            prefs = prefs_data or {}
            sectors_list = ["technology", "finance", "healthcare", "energy", "automotive",
                "retail", "telecom", "realestate", "manufacturing", "defense", "media", "food", "mining", "transportation"]
            sector_labels = {s: L(f"sectors.{s}") for s in sectors_list}
            selected_sectors = st.multiselect(L("preferences.sectors"), options=sectors_list,
                format_func=lambda x: sector_labels.get(x, x), default=prefs.get("preferred_sectors", []))
            sent_bias = st.select_slider(L("preferences.sentiment"), options=["bearish", "neutral", "bullish"],
                value=prefs.get("sentiment_bias", "neutral"), format_func=lambda x: L(f"sentiment.{x}"))
            risk = st.select_slider(L("preferences.risk"), options=["low", "medium", "high"],
                value=prefs.get("risk_level", "medium"), format_func=lambda x: L(f"preferences.risk_{x}"))
            if st.form_submit_button(L("preferences.save")):
                _api_put(f"/api/v1/users/{st.session_state.user_id}/preferences", {
                    "preferred_sectors": selected_sectors, "sentiment_bias": sent_bias, "risk_level": risk,
                })
                st.rerun()

    # Add password prompt for legacy users
    if not st.session_state.has_password:
        st.sidebar.info(L("login.add_password_desc"), icon=None)
        with st.sidebar.form("add_password_form"):
            add_email = st.text_input(L("login.email"), key="add_email_input")
            add_pass = st.text_input(L("login.password"), type="password", key="add_pass_input")
            if st.form_submit_button(L("login.add_password")):
                data, err = _api_post("/api/v1/auth/add-password", {"email": add_email, "password": add_pass})
                if err:
                    st.error(err)
                else:
                    st.session_state.jwt = data["token"]
                    st.session_state.email = data["email"]
                    st.session_state.is_verified = data["is_verified"]
                    st.session_state.has_password = True
                    st.success(L("login.verify_sent"))
                    st.rerun()

else:
    # Not logged in — minimal hint in sidebar
    st.sidebar.caption(L("login.or_continue_guest"))

st.sidebar.divider()
st.sidebar.caption(f"v{settings.APP_TITLE} · {len(TICKER_GROUPS[list(TICKER_GROUPS.keys())[0]])} {L('app.footer')}")

# ── Market & Ticker Controls (always visible) ────────────────────────────

market = st.sidebar.radio(L("market.label"), options=list(TICKER_GROUPS.keys()), horizontal=True, label_visibility="collapsed")
ticker_input = st.sidebar.selectbox(L("symbol.label"), options=TICKER_GROUPS[market])
horizon = st.sidebar.slider(L("horizon.label"), min_value=1, max_value=30, value=5)
show_news = st.sidebar.checkbox(L("show_news"), value=True)
show_details = st.sidebar.checkbox(L("show_details"), value=True)
run = st.sidebar.button(L("generate_forecast"), type="primary", use_container_width=True)


# ── Main Content: Auth pages + Tabs ──────────────────────────────────────

is_authenticated = st.session_state.auth_page == "logged_in"
showing_profile = st.session_state.auth_page == "my_account"
is_guest = st.session_state.auth_page == "guest"

# Auth error/success banners (shown in main area when relevant)
if st.session_state.auth_error:
    st.error(st.session_state.auth_error)
if st.session_state.auth_success:
    st.success(st.session_state.auth_success)

# ── Auth forms (shown when not authenticated and not guest) ───────────────

if not is_authenticated and not showing_profile and not is_guest:

    if st.session_state.auth_page == "verify_notice":
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("📧 " + L("login.verify_sent"))
            st.caption(L("login.verify_email"))
            if st.button(L("login.button"), use_container_width=True):
                st.session_state.auth_page = "login"; st.rerun()

    elif st.session_state.auth_page == "forgot_password":
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title(L("login.reset_password"))
            fp_email = st.text_input(L("login.email"), key="fp_email")
            if st.button(L("login.reset_password"), use_container_width=True, type="primary"):
                data, err = _api_post("/api/v1/auth/forgot-password", {"email": fp_email})
                if err:
                    st.error(err)
                else:
                    st.success(L("login.reset_sent"))
                    st.session_state.auth_page = "login"
                    st.rerun()
            st.divider()
            if st.button(L("login.button")):
                st.session_state.auth_page = "login"; st.rerun()

    elif st.session_state.auth_page == "reset_password":
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title(L("login.reset_password"))
            with st.form("reset_form"):
                np = st.text_input(L("login.new_password"), type="password")
                np2 = st.text_input(L("login.confirm_password"), type="password")
                if st.form_submit_button(L("login.reset_password"), use_container_width=True, type="primary"):
                    if np != np2:
                        st.error(L("auth.error.passwords_dont_match"))
                    elif len(np) < 6:
                        st.error(L("auth.error.weak_password"))
                    else:
                        data, err = _api_post("/api/v1/auth/reset-password", {"token": st.session_state.reset_token, "password": np})
                        if err:
                            st.error(err)
                        else:
                            st.session_state.reset_token = None
                            st.session_state.auth_page = "login"
                            st.rerun()

    elif st.session_state.auth_page == "register":
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title(L("login.register_title"))
            with st.form("register_form"):
                reg_email = st.text_input(L("login.email"))
                reg_user = st.text_input(L("login.username"))
                reg_pass = st.text_input(L("login.password"), type="password")
                reg_pass2 = st.text_input(L("login.confirm_password"), type="password")
                if st.form_submit_button(L("login.register"), use_container_width=True, type="primary", disabled=st.session_state.reg_loading):
                    if reg_pass != reg_pass2:
                        st.session_state.auth_error = L("auth.error.passwords_dont_match")
                        st.rerun()
                    else:
                        _do_register(reg_email, reg_user, reg_pass)
                        st.rerun()
            st.divider()
            st.write(L("login.button") + "?")
            if st.button(L("login.button"), use_container_width=True):
                st.session_state.auth_page = "login"; st.session_state.auth_error = None; st.rerun()

    else:
        # Login form (default auth_page)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title("🔮 " + L("app.title"))
            st.caption(L("app.subtitle"))
            st.divider()
            st.subheader(L("login.title"))
            with st.form("login_form"):
                login_id = st.text_input(L("login.email") + " / " + L("login.username"))
                login_pass = st.text_input(L("login.password"), type="password")
                if st.form_submit_button(L("login.button"), use_container_width=True, type="primary", disabled=st.session_state.login_loading):
                    _do_login(login_id, login_pass)
                    st.rerun()
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button(L("login.forgot_password"), use_container_width=True):
                    st.session_state.auth_page = "forgot_password"; st.session_state.auth_error = None; st.rerun()
            with col_b:
                if st.button(L("login.register_title"), use_container_width=True):
                    st.session_state.auth_page = "register"; st.session_state.auth_error = None; st.rerun()
            st.divider()
            if st.button(L("login.or_continue_guest") + " 🔍", use_container_width=True):
                st.session_state.auth_page = "guest"
                st.session_state.auth_error = None
                st.session_state.auth_success = None
                st.rerun()
    st.divider()

if showing_profile:
    # ── Profile Page in main area ────────────────────────────────────────
    st.title(L("profile.title"))
    st.divider()

    col_profile_left, col_profile_right = st.columns([1, 2])

    with col_profile_left:
        st.markdown(f"**{L('login.logged_in_as')}**")
        st.caption(f"👤 {st.session_state.username}")
        if st.session_state.email:
            st.caption(f"📧 {st.session_state.email}")

    with col_profile_right:
        if st.session_state.email:
            if st.session_state.is_verified:
                st.success("✅ " + L("login.verify_success"))
            else:
                st.warning("⚠️ " + L("login.verify_email"))
                if st.button(L("login.verify_sent"), key="resend_verify"):
                    _api_post("/api/v1/auth/resend-verification", {"email": st.session_state.email})
                    st.success(L("login.verify_sent"))
                    st.rerun()

        if st.session_state.last_login:
            try:
                dt = datetime.fromisoformat(st.session_state.last_login.replace("Z", "+00:00"))
                st.caption(f"{L('login.last_login')} {dt.strftime('%d %b %Y %H:%M')}")
            except Exception:
                pass

    st.divider()

    with st.expander(L("profile.change_password")):
        with st.form("change_password_form"):
            cur_pass = st.text_input(L("login.password"), type="password", key="cur_pass")
            new_pass = st.text_input(L("login.new_password"), type="password", key="new_pass_profile")
            new_pass2 = st.text_input(L("login.confirm_password"), type="password", key="new_pass2_profile")
            if st.form_submit_button(L("profile.change_password"), use_container_width=True, type="primary"):
                if new_pass != new_pass2:
                    st.error(L("auth.error.passwords_dont_match"))
                elif len(new_pass) < 6:
                    st.error(L("auth.error.weak_password"))
                else:
                    data, err = _api_post("/api/v1/auth/change-password", {
                        "current_password": cur_pass, "new_password": new_pass,
                    })
                    if err:
                        st.error(err)
                    else:
                        st.success(L("login.reset_success"))
                        st.rerun()

    st.divider()
    col_act1, col_act2, col_act3 = st.columns(3)
    with col_act1:
        if st.button(L("login.switch_user"), use_container_width=True):
            _do_logout(); st.rerun()
    with col_act2:
        if st.button(L("logout.button") if st.session_state.lang == "tr" else "🚪 Logout", use_container_width=True):
            _do_logout(); st.rerun()
    with col_act3:
        if st.button("⬅️ " + L("forecast.tab"), use_container_width=True):
            st.session_state.auth_page = "logged_in"
            st.rerun()

if not showing_profile:
    # ── Forecast App with 5 tabs ─────────────────────────────────────────

    DEFAULT_TICKER = TICKER_GROUPS[market][0]
    if "ticker" not in st.session_state:
        st.session_state.ticker = DEFAULT_TICKER
        st.session_state.horizon = 5
        st.session_state.has_run = False

    if run:
        st.session_state.ticker = ticker_input
        st.session_state.horizon = horizon
        st.session_state.has_run = False

    ticker = st.session_state.ticker
    currency = currency_symbol(ticker)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        L("forecast.tab"), L("details.tab"), L("news.tab"),
        L("personalized.tab"), L("favorites.tab"),
    ])

    # ── Tab 1: Forecast ──────────────────────────────────────────────────

    with tab1:
        if not st.session_state.has_run:
            with st.spinner(L("loading")):
                try:
                    ticker_val = validate_ticker(ticker)
                    prophet_df, df, pred, params, metrics, baseline = cached_forecast_pipeline(ticker_val, horizon)
                    fundamentals = cached_fundamentals(ticker_val)
                    stats = compute_price_stats(df)
                    st.session_state.prophet_df = prophet_df
                    st.session_state.df = df
                    st.session_state.pred = pred
                    st.session_state.params = params
                    st.session_state.metrics = metrics
                    st.session_state.baseline = baseline
                    st.session_state.fundamentals = fundamentals
                    st.session_state.stats = stats
                    st.session_state.has_run = True
                except DataFetchError as exc:
                    st.error(f"{L('data_error')}: {exc}"); st.stop()
                except ModelError as exc:
                    st.error(f"{L('model_error')}: {exc}"); st.stop()
                except Exception as exc:
                    logger.exception("Unexpected error")
                    st.error(f"{L('unexpected_error')}: {exc}"); st.stop()

        if st.session_state.has_run:
            metrics_data = st.session_state.metrics
            baseline_data = st.session_state.baseline
            stats_data = st.session_state.stats
            fundamentals_data = st.session_state.fundamentals
            mape_val = metrics_data["mape"]
            b_mape = baseline_data["mape"]
            beats = mape_val < b_mape
            delta_color = "normal" if beats else "inverse"
            rsi_val = stats_data.get("rsi")
            rsi_label = L("overbought") if rsi_val is not None and rsi_val >= 70 else L("oversold") if rsi_val is not None and rsi_val <= 30 else L("neutral")
            pos_val = stats_data.get("week52_position_pct")

            # ⭐ Favorite toggle
            if st.session_state.jwt:
                fav_wl, _ = _api_get(f"/api/v1/users/{st.session_state.user_id}/watchlist")
                fav_list = fav_wl if isinstance(fav_wl, list) else []
                is_fav = ticker in fav_list
                star_cols = st.columns([5, 1])
                with star_cols[0]:
                    st.caption("")
                with star_cols[1]:
                    if st.button("⭐" if is_fav else "☆", key=f"fav_toggle_{ticker}", use_container_width=True,
                                 help=L("favorites.remove") if is_fav else L("favorites.add")):
                        if is_fav:
                            _api_delete(f"/api/v1/users/{st.session_state.user_id}/watchlist/{ticker}")
                        else:
                            _api_post(f"/api/v1/users/{st.session_state.user_id}/watchlist", {"ticker": ticker})
                        st.rerun()

            cols = st.columns(6)
            cols[0].metric(L("price.label"), _fmt(stats_data.get("last_close"), f" {currency}"))
            cols[1].metric(L("daily_change"), _fmt(stats_data.get("daily_change_pct"), " %"), delta=_fmt(stats_data.get("daily_change"), f" {currency}"))
            cols[2].metric(L("mape.label"), f"{mape_val:.2f}%", delta=f"{b_mape - mape_val:+.2f}%", delta_color=delta_color if beats else "off")
            cols[3].metric(L("rsi.label"), _fmt(rsi_val), delta=rsi_label, delta_color="off")
            cols[4].metric(L("volume.label"), _fmt_int(stats_data.get("last_volume")))
            cols[5].metric(L("week52_pos"), _fmt(pos_val, " %") if pos_val is not None else "—")

            st.plotly_chart(build_chart(st.session_state.prophet_df, st.session_state.pred, ticker, currency), use_container_width=True)

            last_hist = pd.to_datetime(st.session_state.prophet_df["ds"]).max()
            future_rows = st.session_state.pred[pd.to_datetime(st.session_state.pred["ds"]) > last_hist][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            if not future_rows.empty:
                future_rows.columns = [L("chart.xaxis"), L("forecast.label"), "Alt", "Üst"]
                future_rows[L("forecast.label")] = future_rows[L("forecast.label")].map(lambda x: f"{currency}{x:.2f}")
                future_rows["Alt"] = future_rows["Alt"].map(lambda x: f"{currency}{x:.2f}")
                future_rows["Üst"] = future_rows["Üst"].map(lambda x: f"{currency}{x:.2f}")
                st.caption(f"{L('forecast.table.title')} — {ticker}  ·  {len(future_rows)} {L('forecast.business_days')}")
                st.dataframe(future_rows, hide_index=True, use_container_width=True)

    # ── Tab 2: Details ───────────────────────────────────────────────────

    with tab2:
        if st.session_state.has_run and show_details:
            metrics_data = st.session_state.metrics; baseline_data = st.session_state.baseline
            stats_data = st.session_state.stats; fundamentals_data = st.session_state.fundamentals
            b_mape = baseline_data["mape"]; mape_val = metrics_data["mape"]
            if mape_val < b_mape:
                st.success(L("model_comparison.better", points=b_mape - mape_val))
            else:
                st.warning(L("model_comparison.worse", points=mape_val - b_mape))
            col_a, col_b = st.columns(2)
            col_a.metric(L("model_mape"), f"{mape_val:.2f}%"); col_b.metric(L("naive_mape"), f"{b_mape:.2f}%")
            col_c, col_d = st.columns(2)
            col_c.metric(L("model_rmse"), f"{currency}{metrics_data['rmse']:.2f}"); col_d.metric(L("naive_rmse"), f"{currency}{baseline_data['rmse']:.2f}")
            st.divider(); st.subheader(L("stock_overview"))

            row_a = st.columns(4)
            for col, (label, val, delta) in zip(row_a, [
                (L("price.label"), _fmt(stats_data.get("last_close"), f" {currency}"), None),
                (L("daily_change"), _fmt(stats_data.get("daily_change"), f" {currency}"), _fmt(stats_data.get("daily_change_pct"), " %")),
                ("Market Cap", _fmt_int(fundamentals_data.get("marketCap")), None),
                ("P/E", _fmt(fundamentals_data.get("trailingPE")), None),
            ]):
                col.metric(label, val, delta=delta) if val != "—" else col.empty()

            row_b = st.columns(4)
            for col, (label, val, delta) in zip(row_b, [
                (L("volume.label"), _fmt_int(stats_data.get("last_volume")), _fmt(stats_data.get("volume_vs_avg_pct"), " %")),
                ("52H Yüksek", _fmt(stats_data.get("week52_high"), f" {currency}"), None),
                ("Beta", _fmt(fundamentals_data.get("beta")), None),
                ("Temettü", _fmt(fundamentals_data.get("dividendYield"), " %"), _fmt(fundamentals_data.get("forwardPE"), " ileri P/E")),
            ]):
                col.metric(label, val, delta=delta) if val != "—" else col.empty()

            st.divider(); st.subheader(L("technical_indicators"))
            tech_cols = st.columns(5)
            for col, (label, val, delta) in zip(tech_cols, [
                (L("rsi.label"), _fmt(stats_data.get("rsi")), rsi_label),
                ("SMA 20", _fmt(stats_data.get("sma_20"), f" {currency}"), None),
                ("SMA 50", _fmt(stats_data.get("sma_50"), f" {currency}"), None),
                ("Momentum 5g", _fmt(stats_data.get("momentum_5"), " %"), None),
                ("Hacim/30g Ort", _fmt(stats_data.get("volume_vs_avg_pct"), " %"), None),
            ]):
                col.metric(label, val, delta=delta) if val != "—" else col.empty()

            wk_pos = stats_data.get("week52_position_pct")
            if wk_pos is not None:
                st.caption(f"52 Haftalık Konum: en düşük seviyenin %{wk_pos:.0f} üzerinde ({_fmt(stats_data.get('week52_low'), f' {currency}')} → {_fmt(stats_data.get('week52_high'), f' {currency}')})")
                st.progress(min(max(wk_pos / 100, 0.0), 1.0))
            cross = stats_data.get("sma_cross_pct")
            if cross is not None:
                st.caption(f"SMA 20 / SMA 50 kesişimi: %{cross:.2f} ({'📈 yukarı' if cross > 0 else '📉 aşağı'} yönlü)")
            st.divider()
            with st.expander(L("model_config")):
                st.json(st.session_state.params)
        elif not show_details:
            st.info(L("details_disabled"))
        else:
            st.info(L("details_no_forecast"))

    # ── Tab 3: News ──────────────────────────────────────────────────────

    with tab3:
        if show_news:
            news_items = cached_news(ticker) if st.session_state.has_run else []
            st.subheader(f"📰 {L('news.title')} — {ticker if st.session_state.has_run else L('news.market')}")
            st.caption(L("news.subtitle"))
            if not news_items:
                with st.spinner(L("news.subtitle")):
                    try:
                        provider = get_news_provider()
                        news_items = [{"id": n.id, "title": n.title, "summary": n.summary, "source": n.source, "url": n.url, "published_at": n.published_at, "sentiment": n.sentiment, "sentiment_score": n.sentiment_score, "importance_score": n.importance_score} for n in provider.fetch_news(ticker, max_results=settings.NEWS_MAX_RESULTS)]
                    except Exception:
                        news_items = []
            if news_items:
                for news in news_items:
                    sent = news["sentiment"]
                    sent_label = L(f"sentiment.{sent}")
                    sent_color = {"bullish": "var(--sentiment-bullish)", "bearish": "var(--sentiment-bearish)", "neutral": "var(--sentiment-neutral)"}.get(sent, "var(--sentiment-neutral)")
                    pub_time = news["published_at"]
                    time_str = pub_time.strftime("%d %b %H:%M") if hasattr(pub_time, "strftime") else str(pub_time)[:16]
                    st.markdown(f"<div class='news-item'><div style='display:flex; align-items:center; gap:0.4rem;'><span style='color:{sent_color}; font-size:0.6rem;'>●</span><a href='{news['url']}' class='news-title'>{news['title']}</a></div><div class='news-meta'>{news['source']} · {time_str} · {sent_label}</div></div>", unsafe_allow_html=True)
                    if st.session_state.jwt:
                        fb_cols = st.columns([1, 1, 10])
                        with fb_cols[0]:
                            if st.button(L("feedback.like"), key=f"like_{news.get('id', '')}_{news['title'][:20]}", use_container_width=True):
                                _api_post(f"/api/v1/users/{st.session_state.user_id}/feedback", {"news_id": news.get("id", ""), "ticker": ticker, "rating": 1})
                        with fb_cols[1]:
                            if st.button(L("feedback.dislike"), key=f"dislike_{news.get('id', '')}_{news['title'][:20]}", use_container_width=True):
                                _api_post(f"/api/v1/users/{st.session_state.user_id}/feedback", {"news_id": news.get("id", ""), "ticker": ticker, "rating": -1})
            else:
                st.info(L("news.empty"))
        else:
            st.info(L("news_disabled"))

    # ── Tab 4: Personalized News ─────────────────────────────────────────

    with tab4:
        if st.session_state.jwt:
            st.subheader(L("personalized.title"))
            pdata, _ = _api_get(f"/api/v1/users/{st.session_state.user_id}/news?max_results=20")
            if pdata and pdata.get("data"):
                for item in pdata["data"]:
                    sent = item.get("sentiment", "neutral")
                    sent_color = {"bullish": "var(--sentiment-bullish)", "bearish": "var(--sentiment-bearish)", "neutral": "var(--sentiment-neutral)"}.get(sent, "var(--sentiment-neutral)")
                    score = item.get("personalized_score", 0)
                    st.markdown(f"<div class='news-item'><div style='display:flex; align-items:center; gap:0.4rem;'><span style='color:{sent_color}; font-size:0.6rem;'>●</span><a href='{item['url']}' class='news-title'>{item['title']}</a></div><div class='news-meta'>{item['source']} · {item['ticker']} · {L(f'sentiment.{sent}')} · {L('personalized.title')}: {score:.2f}</div></div>", unsafe_allow_html=True)
            else:
                st.info(L("personalized.empty"))
        else:
            st.info(L("personalized.empty"))

    # ── Tab 5: Favorite Shares ───────────────────────────────────────────

    with tab5:
        st.subheader("⭐ " + L("favorites.title"))
        wl_data, wl_err = _api_get(f"/api/v1/users/{st.session_state.user_id}/watchlist")
        wl = wl_data if isinstance(wl_data, list) else []
        if not wl:
            st.info(L("favorites.empty"))
        else:
            for i in range(0, len(wl), 3):
                row = wl[i:i+3]
                cols = st.columns(3)
                for col_idx, fav_ticker in enumerate(row):
                    with cols[col_idx]:
                        fav_stats = cached_price_stats(fav_ticker)
                        fav_currency = currency_symbol(fav_ticker)
                        fav_price = _fmt(fav_stats.get("last_close") if fav_stats else None, f" {fav_currency}")
                        fav_change = fav_stats.get("daily_change_pct") if fav_stats else None
                        fav_change_str = _fmt(fav_change, " %") if fav_change is not None else "—"
                        fav_color = "var(--sentiment-bullish)" if fav_change is not None and fav_change >= 0 else "var(--sentiment-bearish)" if fav_change is not None and fav_change < 0 else "var(--text-muted)"
                        st.markdown(f"""
                        <div style="background:var(--card-bg); border-radius:10px; padding:1rem; margin-bottom:0.5rem; border:1px solid var(--border); text-align:center;">
                            <div style="font-size:1.2rem; font-weight:700; margin-bottom:0.3rem;">{fav_ticker}</div>
                            <div style="font-size:1.8rem; font-weight:700; margin:0.3rem 0;">{fav_price}</div>
                            <div style="color:{fav_color}; font-weight:600;">{fav_change_str}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        btn_col1, btn_col2 = st.columns(2)
                        with btn_col1:
                            if st.button(L("favorites.forecast"), key=f"fc_{fav_ticker}", use_container_width=True):
                                st.session_state.ticker = fav_ticker
                                st.session_state.has_run = False
                                st.rerun()
                        with btn_col2:
                            if st.button("✕", key=f"fav_rm_{fav_ticker}", use_container_width=True, help=L("favorites.remove")):
                                _api_delete(f"/api/v1/users/{st.session_state.user_id}/watchlist/{fav_ticker}")
                                st.rerun()
