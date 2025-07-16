
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yaml, glob, os, time, requests, psutil
# import plotly.express as px
# from datetime import datetime
# from streamlit_autorefresh import st_autorefresh
# from streamlit_lottie import st_lottie

# # ─── CONFIG ─────────────────────────────────────────────────────────────
# TRADE_LOG = "trade_log.csv"
# MODEL_DIR = r"C:\Users\jordd\OneDrive\Desktop\Code\StockBot\StockFactory2\models"
# STATUS_LOG = "status.log"
# REFRESH_DEFAULT = 20  # seconds
# LIVE_METRICS_URL = "http://localhost:8000/metrics"
# CONFIG_PATH = r"C:\Users\jordd\OneDrive\Desktop\Code\StockBot\StockFactory2\config\coins.yaml"

# st.set_page_config(page_title="🤖 Botfarm Dashboard v0.4", layout="wide")
# import sys
# sys.path.append(r"C:\Users\jordd\OneDrive\Desktop\Code\StockBot\StockFactory2")

# from utils import utils  # Now usable as utils.some_function()
# # ─── LOAD ANIMATIONS ─────────────────────────────────────────────────────
# def load_lottie_url(url: str):
#     try:
#         return requests.get(url).json()
#     except:
#         return {}

# # Example URLs (replace with your own)
# lottie_train = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_j1adxtyb.json")
# lottie_trade = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_cgikwtux.json")
# lottie_tourney = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_touohxv0.json")

# # ─── SIDEBAR CONTROLS ───────────────────────────────────────────────────
# st.sidebar.title("Controls")
# refresh = st.sidebar.slider("Auto-refresh interval (s)", 5, 60, REFRESH_DEFAULT, 5)
# if refresh > 0:
#     st_autorefresh(interval=refresh * 1000, key="refresh")

# # Coin filter
# df_coins = yaml.safe_load(open(CONFIG_PATH))['coins']
# coin_symbols = [c['symbol'] for c in df_coins]
# selected_coins = st.sidebar.multiselect("Coins", options=coin_symbols, default=coin_symbols)

# # ─── DATA LOAD & METRICS ─────────────────────────────────────────────────
# def load_trades():
#     try:
#         return pd.read_csv(TRADE_LOG, parse_dates=["ts"])
#     except:
#         return pd.DataFrame(columns=["ts","symbol","side","qty","price","mode","cap_before","cap_after","pnl"])

# @st.cache_data(ttl=refresh)
# def fetch_metrics():
#     try:
#         data = requests.get(LIVE_METRICS_URL, timeout=2).json()
#         return data.get("performance", {}), data.get("proxies", {}), data.get("survival_stats", {})
#     except:
#         return {}, {}, {}

# @st.cache_data(ttl=30)
# def list_models():
#     rows = []
#     for path in glob.glob(f"{MODEL_DIR}/*.pkl"):
#         name = os.path.basename(path)[:-4]
#         sym = name.split("_")[0]
#         rows.append({"symbol": sym, "file": name})
#     return pd.DataFrame(rows)

# # ─── HELPERS ────────────────────────────────────────────────────────────
# def equity_curve(df):
#     if df.empty: return pd.DataFrame()
#     eq = {s:100.0 for s in coin_symbols}
#     recs = []
#     for _,r in df.sort_values("ts").iterrows():
#         eq[r.symbol] = r.cap_after
#         recs.append({"ts":r.ts, **eq})
#     return pd.DataFrame(recs).set_index("ts")

# # ─── LOAD DATA ──────────────────────────────────────────────────────────
# trades = load_trades()
# metrics, proxies, survival = fetch_metrics()
# if selected_coins:
#     trades = trades[trades.symbol.isin(selected_coins)]
# symbols = trades.symbol.unique().tolist()

# # ─── HEADER ─────────────────────────────────────────────────────────────
# st.title("🤖 Kraken Botfarm Dashboard v0.4")
# col_anim, c1, c2, c3, c4 = st.columns([0.5,1,1,1,1])
# # Animation based on last action
# last_action = proxies.get("last_action", "idle")
# if last_action == "train":
#     st_lottie(lottie_train, height=60, key="anim")
# elif last_action == "trade":
#     st_lottie(lottie_trade, height=60, key="anim")
# elif last_action == "tourney":
#     st_lottie(lottie_tourney, height=60, key="anim")
# else:
#     col_anim.write("")

# # Training / Trading / Tournament counts
# c1.metric("Training Jobs", proxies.get("train_jobs", 0))
# c2.metric("Trading Threads", proxies.get("trading_threads", 0))
# c3.metric("Tournament Runs", proxies.get("tourney_runs", 0))

# # System CPU
# cpu = psutil.cpu_percent()
# c4.metric("CPU Usage", f"{cpu:.0f}%")

# # Top Coin
# top = max(metrics.items(), key=lambda x: x[1].get('total_return', -1), default=(None,{}))[0]
# top_ret = metrics.get(top, {}).get('total_return', None)
# st.metric("Top Coin", top or "N/A", f"{top_ret:+.2%}" if top_ret is not None else "")

# # ─── % CHANGE ROW ───────────────────────────────────────────────────────
# st.subheader("🔄 Coin Equity % Change Since Start")
# if trades.shape[0] > 1:
#     eq = equity_curve(trades)
#     first = eq[symbols].iloc[0]
#     last = eq[symbols].iloc[-1]
#     pct = ((last/first)-1).fillna(0)
#     cols = st.columns(len(symbols))
#     for i,sym in enumerate(symbols):
#         cols[i].metric(sym, f"{pct[sym]:+.2%}")
# else:
#     st.info("Not enough data for percent changes.")

# # ─── TABS ────────────────────────────────────────────────────────────────
# tab1, tab2, tab3, tab4 = st.tabs(["Overview","Coin Detail","Models","Logs & Alerts"])

# with tab1:
#     st.subheader("📊 Live Performance by Coin")
#     if metrics:
#         dfp = pd.DataFrame.from_dict(metrics, orient='index').rename_axis('Symbol').reset_index()
#         dfp['total_return'] = dfp['total_return'].apply(lambda x: f"{x:+.2%}")
#         dfp['win_rate'] = dfp['win_rate'].apply(lambda x: f"{x:.0%}")
#         st.dataframe(dfp.set_index('Symbol'), use_container_width=True)
#         # PnL distribution
#         fig = px.violin(dfp, x='Symbol', y='total_return', title='PnL Distribution')
#         st.plotly_chart(fig, use_container_width=True)
#         # Sparklines
#         spark_df = equity_curve(trades)[symbols]
#         st.line_chart(spark_df)
#     else:
#         st.info("No performance data.")

# with tab2:
#     st.subheader("🔍 Coin Detail Explorer")
#     coin = st.selectbox("Select Coin", options=symbols) if symbols else None
#     if coin:
#         dfc = trades[trades.symbol==coin]
#         # date slider
#         dates = dfc.ts.sort_values()
#         start, end = st.slider("Date Range", min_value=dates.min(), max_value=dates.max(), value=(dates.min(), dates.max()))
#         dfc = dfc[(dfc.ts>=start)&(dfc.ts<=end)]
#         if not dfc.empty:
#             m = metrics.get(coin,{})
#             c1,c2,c3 = st.columns(3)
#             c1.metric("Return", f"{m.get('total_return',0):+.2%}")
#             c2.metric("Win Rate", f"{m.get('win_rate',0):.0%}")
#             c3.metric("Trades", m.get('total_trades',0))
#             # equity play-back
#             playback = st.button("Play Equity Curve")
#             chart = st.empty()
#             ec = equity_curve(dfc).loc[start:end, coin]
#             if playback:
#                 for ts, val in ec.iteritems():
#                     chart.line_chart(pd.DataFrame({coin:[val]}, index=[ts]))
#                     time.sleep(0.1)
#             else:
#                 st.line_chart(ec)
#             # radar chart
#             radar = pd.DataFrame([{
#                 'Metric':'Return','Value':m.get('total_return',0)},
#                 {'Metric':'Win Rate','Value':m.get('win_rate',0)},
#                 {'Metric':'Trades','Value':m.get('total_trades',0)/100},
#                 {'Metric':'Avg Gain','Value':m.get('avg_gain',0)}
#             ])
#             fig = px.line_polar(radar, r='Value', theta='Metric', line_close=True)
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info(f"No trades for {coin} in selected range.")
#     else:
#         st.info("No coins selected.")

# with tab3:
#     st.subheader("🗄️ Models & Config Usage")
#     mdf = list_models()
#     if not mdf.empty:
#         st.dataframe(mdf, use_container_width=True)
#         # champion vs challengers
#         from utils.model_factory import get_champion, list_models
#         champ = get_champion(coin) if coin else None
#         challengers = [p for p in list_models(coin) if p!=champ][:3] if coin else []
#         cols = st.columns(4)
#         cols[0].metric("Champion", champ or "—")
#         for i,ch in enumerate(challengers,1): cols[i].metric(f"Challenger {i}", ch.stem)
#         # heatmap stub
#         heat = np.random.rand(3,3)
#         fig = px.density_heatmap(heat, title="Hyperparam Grid (stub)")
#         st.plotly_chart(fig, use_container_width=True)

#         if survival:
#             st.subheader("🧬 Tournament Survival Stats")
#             surv_df = (
#                 pd.DataFrame(survival)
#                 .T.fillna(0)
#                 .astype(int)
#                 .rename_axis("Symbol")
#                 .reset_index()
#             )
#             st.dataframe(surv_df.set_index("Symbol"), use_container_width=True)
#     else:
#         st.info("No models found.")

# with tab4:
#     st.subheader("📜 Logs & Alerts")
#     if os.path.exists(STATUS_LOG):
#         logs = open(STATUS_LOG).read().splitlines()[-20:]
#         st.text("\n".join(logs))
#     else:
#         st.info("No status.log found.")
#     losses = {s:sum(trades[trades.symbol==s].pnl<0) for s in symbols}
#     alerts = [f"{s}: {c} losses" for s,c in losses.items() if c>5]
#     if alerts:
#         # carousel
#         idx = st.session_state.get('alert_idx',0) % len(alerts)
#         st.error(alerts[idx])
#         st.session_state['alert_idx'] = idx+1
#     else:
#         st.success("No alerts.")

# # Note: tab resets on refresh, Streamlit limitation.
# # cd C:\Users\jordd\OneDrive\Desktop\Code\StockBot\StockFactory2
# # Run: streamlit run feedback/dashboard.py




import streamlit as st
import pandas as pd
import numpy as np
import yaml, glob, os, time, requests, psutil
import plotly.express as px
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from streamlit_lottie import st_lottie

# ─── CONFIG ─────────────────────────────────────────────────────────────
TRADE_LOG = "trade_log.csv"
MODEL_DIR = r"C:\Users\jordd\OneDrive\Desktop\Code\StockBot\StockFactory2\models"
STATUS_LOG = "status.log"
REFRESH_DEFAULT = 20  # seconds
LIVE_METRICS_URL = "http://localhost:8000/metrics"
CONFIG_PATH = r"C:\Users\jordd\OneDrive\Desktop\Code\StockBot\StockFactory2\config\coins.yaml"

st.set_page_config(page_title="🤖 Botfarm Dashboard v1.2", layout="wide")
import sys
sys.path.append(r"C:\Users\jordd\OneDrive\Desktop\Code\StockBot\StockFactory2")

from utils import utils  # Now usable as utils.some_function()

# ─── LOAD ANIMATIONS ─────────────────────────────────────────────────────
def load_lottie_url(url: str):
    try:
        return requests.get(url).json()
    except:
        return {}

# Example Lottie animation URLs – replace with actual animated visuals
lottie_train = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_j1adxtyb.json")   # training
lottie_trade = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_cgikwtux.json")   # trading
lottie_tourney = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_touohxv0.json") # tournament
lottie_gather = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_5ngs2ksb.json")  # data gathering
lottie_load = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jtbfg2nb.json")    # loading
lottie_prune = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_7yyhhiep.json")   # gore/pruning
lottie_champion = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_q5pk6p1k.json")# champion celebration

# ─── SIDEBAR CONTROLS ───────────────────────────────────────────────────
st.sidebar.title("Controls")
refresh = st.sidebar.slider("Auto-refresh interval (s)", 5, 60, REFRESH_DEFAULT, 5)
if refresh > 0:
    st_autorefresh(interval=refresh * 1000, key="refresh")

# Manual actions (for demonstration; hook into backend as needed)
if st.sidebar.button("Gather Data"):
    utils.trigger_action("gather_data")
if st.sidebar.button("Load Models"):
    utils.trigger_action("load_models")
if st.sidebar.button("Trigger Training"):
    utils.trigger_action("train")
if st.sidebar.button("Start Tournament"):
    utils.trigger_action("tourney")

# Coin filter
df_coins = yaml.safe_load(open(CONFIG_PATH))['coins']
coin_symbols = [c['symbol'] for c in df_coins]
selected_coins = st.sidebar.multiselect("Coins", options=coin_symbols, default=coin_symbols)

# ─── DATA LOAD & METRICS ─────────────────────────────────────────────────
def load_trades():
    try:
        return pd.read_csv(TRADE_LOG, parse_dates=["ts"])
    except:
        return pd.DataFrame(columns=["ts","symbol","side","qty","price","mode","cap_before","cap_after","pnl"])

@st.cache_data(ttl=refresh)
def fetch_metrics():
    try:
        data = requests.get(LIVE_METRICS_URL, timeout=2).json()
        # Now expecting: performance, proxies, survival_stats, tournament
        return data.get("performance", {}), data.get("proxies", {}), data.get("survival_stats", {}), data.get("tournament", {})
    except:
        return {}, {}, {}, {}

@st.cache_data(ttl=30)
def list_models():
    rows = []
    for path in glob.glob(f"{MODEL_DIR}/*.pkl"):
        name = os.path.basename(path)[:-4]
        sym = name.split("_")[0]
        rows.append({"symbol": sym, "file": name})
    return pd.DataFrame(rows)

# ─── HELPERS ────────────────────────────────────────────────────────────
def equity_curve(df):
    if df.empty: return pd.DataFrame()
    eq = {s:100.0 for s in coin_symbols}
    recs = []
    for _,r in df.sort_values("ts").iterrows():
        eq[r.symbol] = r.cap_after
        recs.append({"ts":r.ts, **eq})
    return pd.DataFrame(recs).set_index("ts")

# ─── LOAD DATA ──────────────────────────────────────────────────────────
trades = load_trades()
metrics, proxies, survival, tournament = fetch_metrics()
if selected_coins:
    trades = trades[trades.symbol.isin(selected_coins)]
symbols = trades.symbol.unique().tolist()

# ─── HEADER ─────────────────────────────────────────────────────────────
st.title("🤖 Kraken Botfarm Dashboard v1.2")
col_anim, c1, c2, c3, c4 = st.columns([0.5,1,1,1,1])

# Animation based on last action
last_action = proxies.get("last_action", "idle")
if last_action == "gather_data":
    st_lottie(lottie_gather, height=60, key="anim")
elif last_action == "load_models":
    st_lottie(lottie_load, height=60, key="anim")
elif last_action == "train":
    st_lottie(lottie_train, height=60, key="anim")
elif last_action == "trade":
    st_lottie(lottie_trade, height=60, key="anim")
elif last_action == "tourney":
    st_lottie(lottie_tourney, height=60, key="anim")
elif last_action == "prune":
    st_lottie(lottie_prune, height=60, key="anim")
elif last_action == "champion":
    st_lottie(lottie_champion, height=60, key="anim")
else:
    col_anim.write("")

# Training / Trading / Tournament counts
c1.metric("Training Jobs", proxies.get("train_jobs", 0))
c2.metric("Trading Threads", proxies.get("trading_threads", 0))
c3.metric("Tournament Runs", proxies.get("tourney_runs", 0))

# System CPU
cpu = psutil.cpu_percent()
c4.metric("CPU Usage", f"{cpu:.0f}%")

# Top Coin
top = max(metrics.items(), key=lambda x: x[1].get('total_return', -1), default=(None,{}))[0]
top_ret = metrics.get(top, {}).get('total_return', None)
st.metric("Top Coin", top or "N/A", f"{top_ret:+.2%}" if top_ret is not None else "")

# ─── % CHANGE ROW ───────────────────────────────────────────────────────
st.subheader("🔄 Coin Equity % Change Since Start")
if trades.shape[0] > 1:
    eq = equity_curve(trades)
    first = eq[symbols].iloc[0]
    last = eq[symbols].iloc[-1]
    pct = ((last/first)-1).fillna(0)
    cols = st.columns(len(symbols))
    for i,sym in enumerate(symbols):
        cols[i].metric(sym, f"{pct[sym]:+.2%}")
else:
    st.info("Not enough data for percent changes.")

# ─── TABS ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview","Coin Detail","Models","Tournament","Logs & Alerts"])

with tab1:
    st.subheader("📊 Live Performance by Coin")
    if metrics:
        dfp = pd.DataFrame.from_dict(metrics, orient='index').rename_axis('Symbol').reset_index()
        dfp['total_return'] = dfp['total_return'].apply(lambda x: f"{x:+.2%}")
        dfp['win_rate'] = dfp['win_rate'].apply(lambda x: f"{x:.0%}")
        st.dataframe(dfp.set_index('Symbol'), use_container_width=True)

        # PnL distribution
        fig = px.violin(dfp, x='Symbol', y='total_return', title='PnL Distribution')
        st.plotly_chart(fig, use_container_width=True)

        # Sparklines
        spark_df = equity_curve(trades)[symbols]
        st.line_chart(spark_df)
    else:
        st.info("No performance data.")

with tab2:
    st.subheader("🔍 Coin Detail Explorer")
    coin = st.selectbox("Select Coin", options=symbols) if symbols else None
    if coin:
        dfc = trades[trades.symbol==coin]
        # date slider
        dates = dfc.ts.sort_values()
        start, end = st.slider("Date Range", min_value=dates.min(), max_value=dates.max(), value=(dates.min(), dates.max()))
        dfc = dfc[(dfc.ts>=start)&(dfc.ts<=end)]
        if not dfc.empty:
            m = metrics.get(coin,{})
            c1,c2,c3 = st.columns(3)
            c1.metric("Return", f"{m.get('total_return',0):+.2%}")
            c2.metric("Win Rate", f"{m.get('win_rate',0):.0%}")
            c3.metric("Trades", m.get('total_trades',0))

            # equity play-back
            playback = st.button("Play Equity Curve")
            chart = st.empty()
            ec = equity_curve(dfc).loc[start:end, coin]
            if playback:
                for ts, val in ec.iteritems():
                    chart.line_chart(pd.DataFrame({coin:[val]}, index=[ts]))
                    time.sleep(0.1)
            else:
                st.line_chart(ec)

            # radar chart
            radar = pd.DataFrame([{ 'Metric':'Return','Value':m.get('total_return',0)},
                                  {'Metric':'Win Rate','Value':m.get('win_rate',0)},
                                  {'Metric':'Trades','Value':m.get('total_trades',0)/100},
                                  {'Metric':'Avg Gain','Value':m.get('avg_gain',0)}])
            fig = px.line_polar(radar, r='Value', theta='Metric', line_close=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No trades for {coin} in selected range.")
    else:
        st.info("No coins selected.")

with tab3:
    st.subheader("🗄️ Models & Config Usage")
    mdf = list_models()
    if not mdf.empty:
        st.dataframe(mdf, use_container_width=True)

        # champion vs challengers
        from utils.model_factory import get_champion, list_models
        champ = get_champion(coin) if coin else None
        challengers = [p for p in list_models(coin) if p!=champ][:3] if coin else []

        cols = st.columns(4)
        cols[0].metric("Champion", champ or "—")
        for i,ch in enumerate(challengers,1):
            cols[i].metric(f"Challenger {i}", ch.stem)

        # heatmap stub (hyperparam grid)
        heat = np.random.rand(3,3)
        fig = px.density_heatmap(heat, title="Hyperparam Grid (stub)")
        st.plotly_chart(fig, use_container_width=True)

        if survival:
            st.subheader("🧬 Tournament Survival Stats")
            surv_df = (
                pd.DataFrame(survival)
                .T.fillna(0)
                .astype(int)
                .rename_axis("Symbol")
                .reset_index()
            )
            st.dataframe(surv_df.set_index("Symbol"), use_container_width=True)
    else:
        st.info("No models found.")

with tab4:
    st.subheader("🏆 Live Tournament Tracker")
    if tournament:
        # Display bracket or ladder if available
        ladder_df = pd.DataFrame(tournament.get('ladder', []))
        if not ladder_df.empty:
            st.dataframe(ladder_df)
        else:
            st.info("Tournament data unavailable.")

        # Show pruning gore animation when last_action is prune
        if last_action == "prune":
            st_lottie(lottie_prune, height=200)
    else:
        st.info("No live tournament data.")

with tab5:
    st.subheader("📜 Logs & Alerts")
    if os.path.exists(STATUS_LOG):
        logs = open(STATUS_LOG).read().splitlines()[-20:]
        st.text("\n".join(logs))
    else:
        st.info("No status.log found.")
    losses = {s:sum(trades[trades.symbol==s].pnl<0) for s in symbols}
    alerts = [f"{s}: {c} losses" for s,c in losses.items() if c>5]
    if alerts:
        # carousel of alerts
        idx = st.session_state.get('alert_idx',0) % len(alerts)
        st.error(alerts[idx])
        st.session_state['alert_idx'] = idx+1
    else:
        st.success("No alerts.")

# Note: tab resets on refresh, Streamlit limitation.
# cd C:\Users\jordd\OneDrive\Desktop\Code\StockBot\StockFactory2
# Run: streamlit run dashboard.py
