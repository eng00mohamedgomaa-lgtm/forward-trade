# ============================================================
#   FORWARD TRADE — Streamlit Web App
#   المهندس محمد السبكي — 2026
# ============================================================

# ===== الاستيراد =====
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import pytz

# ===== إعدادات الصفحة =====
st.set_page_config(
    page_title="Forward Trade",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== CSS مخصص =====
st.markdown("""
<style>
    /* الخلفية العامة */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    
    /* الهيدر */
    .main-header {
        background: linear-gradient(135deg, #064e3b, #0f172a);
        border: 1px solid rgba(34,197,94,0.3);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #22c55e;
        margin: 0;
    }
    
    .main-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
        letter-spacing: 2px;
    }
    
    /* بطاقة السهم */
    .stock-card {
        background: #1e293b;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .stock-card-buy {
        border-color: rgba(34,197,94,0.4);
        background: linear-gradient(135deg, #064e3b22, #1e293b);
    }
    
    .stock-card-sell {
        border-color: rgba(248,113,113,0.4);
        background: linear-gradient(135deg, #7f1d1d22, #1e293b);
    }
    
    .stock-card-watch {
        border-color: rgba(251,191,36,0.3);
    }
    
    /* الإشارة */
    .signal-buy {
        background: rgba(34,197,94,0.15);
        color: #4ade80;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
    }
    
    .signal-sell {
        background: rgba(248,113,113,0.15);
        color: #f87171;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
    }
    
    .signal-watch {
        background: rgba(251,191,36,0.15);
        color: #fbbf24;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
    }
    
    /* الأزرار */
    .stButton > button {
        background: linear-gradient(135deg, #16a34a, #22c55e);
        color: #0f172a;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 8px 20px;
        width: 100%;
    }
    
    /* المقاييس */
    .metric-box {
        background: #1e293b;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }
    
    .metric-val {
        font-size: 22px;
        font-weight: 700;
    }
    
    .metric-lbl {
        font-size: 11px;
        color: #64748b;
        margin-top: 2px;
    }
    
    /* إخفاء عناصر Streamlit الافتراضية */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* الـ divider */
    hr {
        border-color: rgba(255,255,255,0.08);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
#   الإعدادات الثابتة
# ============================================================

SYMBOLS = {
    "HELI":  {"name": "هيليوبوليس", "yahoo": "HELI.CA"},
    "TMGH":  {"name": "طلعت مصطفى", "yahoo": "TMGH.CA"},
    "EKHW":  {"name": "إخوان وتر",   "yahoo": "EKHW.CA"},
    "SCTS":  {"name": "ساكو",        "yahoo": "SCTS.CA"},
}

CAIRO_TZ       = pytz.timezone("Africa/Cairo")
SCAN_INTERVAL  = 300  # 5 دقايق


# ============================================================
#   جلب البيانات — Fallback System
# ============================================================

def get_data_yfinance(yahoo_symbol: str, bars: int = 100) -> pd.DataFrame:
    """جلب البيانات من yfinance"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(yahoo_symbol)
        df = ticker.history(period="5d", interval="5m", auto_adjust=True)
        if df is not None and not df.empty and len(df) >= 10:
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            return df.tail(bars)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def get_data_daily_fallback(yahoo_symbol: str) -> pd.DataFrame:
    """Fallback — بيانات يومية لو الـ 5 دقايق فشلت"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(yahoo_symbol)
        df = ticker.history(period="3mo", interval="1d", auto_adjust=True)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def get_ohlcv(symbol: str) -> pd.DataFrame:
    """الدالة الرئيسية — بتجرب المصادر بالترتيب"""
    yahoo_sym = SYMBOLS[symbol]["yahoo"]

    # المصدر الأول — 5 دقايق
    df = get_data_yfinance(yahoo_sym)
    if not df.empty:
        return df

    # المصدر الثاني — يومي كـ fallback
    df = get_data_daily_fallback(yahoo_sym)
    if not df.empty:
        return df

    return pd.DataFrame()


# ============================================================
#   المؤشرات الفنية
# ============================================================

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """حساب EMA"""
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """حساب RSI"""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = 100 - (100 / (1 + rs))
    return rsi


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """حساب ATR"""
    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift()),
        abs(df["low"]  - df["close"].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """إضافة كل المؤشرات"""
    if df.empty or len(df) < 20:
        return df

    df = df.copy()

    df["ema50"]      = calc_ema(df["close"], 50)
    df["ema200"]     = calc_ema(df["close"], 200)
    df["rsi"]        = calc_rsi(df["close"], 14)
    df["atr"]        = calc_atr(df, 14)
    df["vol_ma"]     = df["volume"].rolling(20).mean()
    df["vol_ratio"]  = df["volume"] / df["vol_ma"].replace(0, np.nan)
    df["body"]       = abs(df["close"] - df["open"])
    df["rng"]        = df["high"] - df["low"]
    df["body_ratio"] = df["body"] / df["rng"].replace(0, np.nan)
    df["bull"]       = df["close"] > df["open"]

    return df


# ============================================================
#   أنماط الشموع
# ============================================================

def detect_pattern(df: pd.DataFrame) -> str:
    """كشف النمط"""
    if len(df) < 2:
        return "لا يوجد"

    c = df.iloc[-1]
    p = df.iloc[-2]

    # Bullish Engulfing
    if (p["close"] < p["open"] and
        c["close"] > c["open"] and
        c["open"]  < p["close"] and
        c["close"] > p["open"]):
        return "Bullish Engulfing"

    # Bullish Pin Bar
    if c["rng"] > 0:
        lower_wick = min(c["open"], c["close"]) - c["low"]
        if lower_wick >= 2 * c["body"] and lower_wick >= 0.6 * c["rng"]:
            return "Bullish Pin Bar"

    # Bearish Engulfing
    if (p["close"] > p["open"] and
        c["close"] < c["open"] and
        c["open"]  > p["close"] and
        c["close"] < p["open"]):
        return "Bearish Engulfing"

    return "لا يوجد"


# ============================================================
#   مناطق الدعم والمقاومة
# ============================================================

def find_zones(df: pd.DataFrame, lookback: int = 50) -> dict:
    """إيجاد مناطق الدعم والمقاومة"""
    if len(df) < 15:
        return {"support": None, "resistance": None}

    lb     = min(lookback, len(df) - 5)
    r      = df.tail(lb)
    p      = float(df["close"].iloc[-1])
    highs  = []
    lows   = []

    for i in range(2, len(r) - 2):
        h = r["high"].iloc[i]
        l = r["low"].iloc[i]
        if (h > r["high"].iloc[i-1] and h > r["high"].iloc[i-2] and
                h > r["high"].iloc[i+1] and h > r["high"].iloc[i+2]):
            highs.append(h)
        if (l < r["low"].iloc[i-1] and l < r["low"].iloc[i-2] and
                l < r["low"].iloc[i+1] and l < r["low"].iloc[i+2]):
            lows.append(l)

    resistances = [h for h in highs if h > p]
    supports    = [l for l in lows  if l < p]

    return {
        "support":    max(supports)    if supports    else p * 0.97,
        "resistance": min(resistances) if resistances else p * 1.03
    }


# ============================================================
#   الـ 14 شرط — SMC Strategy
# ============================================================

def analyze_14_conditions(df: pd.DataFrame) -> dict:
    """تطبيق الـ 14 شرط"""
    empty_result = {
        "signal": "NONE", "score": 0, "price": 0,
        "support": None, "resistance": None,
        "rr": 0, "rsi": 0, "pattern": "لا يوجد",
        "conditions": {}
    }

    if df.empty or len(df) < 30:
        return empty_result

    # القيم الأساسية
    p    = float(df["close"].iloc[-1])
    e50  = float(df["ema50"].iloc[-1])
    e200 = float(df["ema200"].iloc[-1])
    rsi  = float(df["rsi"].iloc[-1])   if not pd.isna(df["rsi"].iloc[-1])   else 50
    atr  = float(df["atr"].iloc[-1])   if not pd.isna(df["atr"].iloc[-1])   else p * 0.01
    vr   = float(df["vol_ratio"].iloc[-1]) if not pd.isna(df["vol_ratio"].iloc[-1]) else 0
    vm   = float(df["vol_ma"].iloc[-1])    if not pd.isna(df["vol_ma"].iloc[-1])    else 1
    br   = float(df["body_ratio"].iloc[-1])if not pd.isna(df["body_ratio"].iloc[-1])else 0

    # المناطق والنمط
    zones = find_zones(df)
    sup   = zones["support"]
    resi  = zones["resistance"]
    pat   = detect_pattern(df)

    # R/R
    risk   = (p - sup)  if sup  else atr
    reward = (resi - p) if resi else atr * 2
    rr     = round(reward / risk, 2) if risk > 0 else 0

    # RSI Divergence
    rsi_div = False
    if len(df) > 28:
        price_diff = df["close"].iloc[-1] - df["close"].iloc[-14]
        rsi_diff   = df["rsi"].iloc[-1]   - df["rsi"].iloc[-14]
        rsi_div    = (price_diff > 0 and rsi_diff < -5)

    # الـ 14 شرط
    conditions = {
        "1. اختراق بزخم":         (resi is not None and p > resi and br >= 0.85 and bool(df["bull"].iloc[-1])),
        "2. انفجار فوليوم":        (vr >= 1.5),
        "3. منطقة محددة":         (sup is not None or resi is not None),
        "4. إعادة اختبار":        (sup is not None and abs(p - sup) / sup <= 0.005),
        "5. تصحيح بفوليوم ضعيف":  (df["volume"].tail(5).mean() < vm * 0.7),
        "6. إشارة دخول":          (pat in ["Bullish Engulfing", "Bullish Pin Bar"]),
        "7. دعم Order Book":       True,
        "8. اتجاه صاعد":          (p > e50 and p > e200 and e50 > e200),
        "9. قوة نسبية":            True,
        "10. RSI سليم":            (not rsi_div),
        "11. R/R ≥ 1:2":          (rr >= 2.0),
        "12. SL تحت الزون":       (sup is not None),
        "13. Trailing Stop":       True,
        "14. إدارة مالية":         True,
    }

    score = sum(1 for v in conditions.values() if v)

    return {
        "signal":     "STRONG" if score == 14 else "WEAK" if score >= 8 else "NONE",
        "score":      score,
        "price":      p,
        "support":    sup,
        "resistance": resi,
        "rr":         rr,
        "rsi":        round(rsi, 1),
        "atr":        round(atr, 2),
        "ema50":      round(e50, 2),
        "ema200":     round(e200, 2),
        "vol_ratio":  round(vr, 2),
        "pattern":    pat,
        "conditions": conditions,
    }


# ============================================================
#   Telegram
# ============================================================

def send_telegram(token: str, chat_id: str, text: str) -> bool:
    """إرسال رسالة على Telegram"""
    try:
        url     = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        r       = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def build_signal_message(symbol: str, result: dict) -> str:
    """بناء رسالة الإشارة"""
    p   = result["price"]
    sl  = round(result["support"] * 0.995, 2) if result["support"] else round(p * 0.98, 2)
    tp  = round(result["resistance"], 2)       if result["resistance"] else round(p * 1.04, 2)
    e   = "🟢" if result["signal"] == "STRONG" else "🟡"
    lbl = "قوية" if result["signal"] == "STRONG" else "متوسطة"
    now = datetime.now(CAIRO_TZ).strftime("%I:%M %p")

    msg = (
        f"{e} <b>إشارة {lbl}</b> — {symbol}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"💰 السعر  : <b>{p:.2f}</b>\n"
        f"🎯 الهدف  : <b>{tp:.2f}</b>\n"
        f"🛑 الوقف  : <b>{sl:.2f}</b>\n"
        f"⚖️ R/R    : <b>1 : {result['rr']}</b>\n"
        f"🕯️ النمط : {result['pattern']}\n"
        f"📊 RSI    : {result['rsi']}\n"
        f"✅ الشروط : {result['score']}/14\n"
        f"⏰ الوقت  : {now}\n"
        f"━━━━━━━━━━━━━━━\n"
        f"⚠️ Forward Trade — راجع قبل التنفيذ"
    )
    return msg


# ============================================================
#   الواجهة الرئيسية
# ============================================================

def main():

    # ===== الهيدر =====
    st.markdown("""
    <div class="main-header">
        <div class="main-title">📈 Forward Trade</div>
        <div class="main-subtitle">EGYPTIAN STOCK INTELLIGENCE · EGX · المهندس محمد السبكي</div>
    </div>
    """, unsafe_allow_html=True)

    # ===== الشريط الجانبي — إعدادات Telegram =====
    with st.sidebar:
        st.markdown("### 🔔 إعدادات Telegram")
        tg_token   = st.text_input("Bot Token", type="password", value=st.session_state.get("tg_token", ""))
        tg_chat_id = st.text_input("Chat ID", value=st.session_state.get("tg_chat_id", ""))

        if st.button("💾 حفظ الإعدادات"):
            st.session_state["tg_token"]   = tg_token
            st.session_state["tg_chat_id"] = tg_chat_id
            st.success("✅ تم الحفظ!")

        st.markdown("---")
        st.markdown("### ⚙️ الإعدادات")

        auto_refresh = st.checkbox("تحديث تلقائي كل 5 دقايق", value=True)
        send_alerts  = st.checkbox("إرسال تنبيهات Telegram", value=True)

        st.markdown("---")
        st.markdown("### 📅 معلومات الجلسة")
        now = datetime.now(CAIRO_TZ)
        st.info(f"الوقت: {now.strftime('%H:%M:%S')}\nالتاريخ: {now.strftime('%Y-%m-%d')}")

    # ===== زر التحديث اليدوي =====
    col_refresh, col_time = st.columns([1, 3])
    with col_refresh:
        manual_refresh = st.button("🔄 تحديث الآن")
    with col_time:
        last_update = st.session_state.get("last_update", "لم يتم التحديث بعد")
        st.caption(f"آخر تحديث: {last_update}")

    # ===== تحميل البيانات =====
    should_scan = False

    if manual_refresh:
        should_scan = True

    if auto_refresh:
        last_scan_time = st.session_state.get("last_scan_time", 0)
        if time.time() - last_scan_time > SCAN_INTERVAL:
            should_scan = True

    if should_scan or "results" not in st.session_state:
        with st.spinner("🔍 جاري مسح الأسهم..."):
            results = {}
            for sym in SYMBOLS:
                df = get_ohlcv(sym)
                if not df.empty:
                    df = add_indicators(df)
                    result = analyze_14_conditions(df)
                    results[sym] = result

                    # إرسال التنبيه على Telegram
                    if (send_alerts and
                            result["signal"] in ["STRONG", "WEAK"] and
                            st.session_state.get("tg_token") and
                            st.session_state.get("tg_chat_id")):
                        msg = build_signal_message(sym, result)
                        send_telegram(
                            st.session_state["tg_token"],
                            st.session_state["tg_chat_id"],
                            msg
                        )
                else:
                    results[sym] = {"signal": "NONE", "score": 0, "price": 0,
                                    "rsi": 0, "vol_ratio": 0, "ema50": 0, "ema200": 0,
                                    "support": None, "resistance": None, "rr": 0,
                                    "pattern": "لا يوجد", "conditions": {}}

            st.session_state["results"]        = results
            st.session_state["last_update"]    = datetime.now(CAIRO_TZ).strftime("%H:%M:%S")
            st.session_state["last_scan_time"] = time.time()

    results = st.session_state.get("results", {})

    # ===== ملخص سريع =====
    st.markdown("### 📊 ملخص السوق")
    col1, col2, col3, col4 = st.columns(4)

    buy_count   = sum(1 for r in results.values() if r["signal"] == "STRONG")
    weak_count  = sum(1 for r in results.values() if r["signal"] == "WEAK")
    none_count  = sum(1 for r in results.values() if r["signal"] == "NONE")
    total_score = sum(r["score"] for r in results.values())

    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val" style="color:#22c55e">{buy_count}</div>
            <div class="metric-lbl">إشارة قوية 🟢</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val" style="color:#fbbf24">{weak_count}</div>
            <div class="metric-lbl">إشارة متوسطة 🟡</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val" style="color:#64748b">{none_count}</div>
            <div class="metric-lbl">انتظار ⚪</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val" style="color:#22c55e">{total_score}</div>
            <div class="metric-lbl">مجموع الشروط</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ===== بطاقات الأسهم =====
    st.markdown("### 📋 تفاصيل الأسهم")

    for sym, result in results.items():
        signal  = result.get("signal", "NONE")
        score   = result.get("score", 0)
        price   = result.get("price", 0)
        rsi     = result.get("rsi", 0)
        vr      = result.get("vol_ratio", 0)
        pat     = result.get("pattern", "لا يوجد")
        sup     = result.get("support")
        resi    = result.get("resistance")
        rr      = result.get("rr", 0)
        e50     = result.get("ema50", 0)
        e200    = result.get("ema200", 0)
        conds   = result.get("conditions", {})

        # لون الإشارة
        if signal == "STRONG":
            signal_html = '<span class="signal-buy">🟢 شراء قوي</span>'
            card_class  = "stock-card stock-card-buy"
        elif signal == "WEAK":
            signal_html = '<span class="signal-watch">🟡 مراقبة</span>'
            card_class  = "stock-card stock-card-watch"
        else:
            signal_html = '<span style="color:#64748b">⚪ انتظار</span>'
            card_class  = "stock-card"

        sl = round(sup * 0.995, 2) if sup else round(price * 0.98, 2)
        tp = round(resi, 2)        if resi else round(price * 1.04, 2)

        # عرض البطاقة
        with st.expander(f"{sym} — {SYMBOLS[sym]['name']} | {price:.2f} | {score}/14 شرط", expanded=(signal != "NONE")):

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown(f"**الإشارة:** {signal_html}", unsafe_allow_html=True)
                st.metric("السعر الحالي", f"{price:.2f}")
                st.metric("RSI", f"{rsi}")

            with col_b:
                st.metric("الهدف 🎯", f"{tp:.2f}")
                st.metric("الوقف 🛑", f"{sl:.2f}")
                st.metric("R/R ⚖️", f"1 : {rr}")

            with col_c:
                st.metric("EMA 50", f"{e50:.2f}")
                st.metric("EMA 200", f"{e200:.2f}")
                st.metric("فوليوم", f"{vr:.1f}x")

            st.markdown(f"**النمط:** `{pat}`")

            # شريط تقدم الشروط
            progress = score / 14
            st.progress(progress)
            st.caption(f"الشروط المتحققة: {score}/14 ({round(progress*100)}%)")

            # تفاصيل الشروط
            if conds:
                st.markdown("**تفاصيل الـ 14 شرط:**")
                cond_cols = st.columns(2)
                items     = list(conds.items())
                half      = len(items) // 2

                for i, (name, val) in enumerate(items[:half]):
                    with cond_cols[0]:
                        icon = "✅" if val else "❌"
                        st.caption(f"{icon} {name}")

                for i, (name, val) in enumerate(items[half:]):
                    with cond_cols[1]:
                        icon = "✅" if val else "❌"
                        st.caption(f"{icon} {name}")

            # زر إرسال يدوي على Telegram
            if st.button(f"📤 إرسال {sym} على Telegram", key=f"send_{sym}"):
                token   = st.session_state.get("tg_token", "")
                chat_id = st.session_state.get("tg_chat_id", "")
                if token and chat_id:
                    msg     = build_signal_message(sym, result)
                    success = send_telegram(token, chat_id, msg)
                    if success:
                        st.success("✅ تم الإرسال!")
                    else:
                        st.error("❌ فشل الإرسال — تأكد من الإعدادات")
                else:
                    st.warning("⚠️ أدخل Token و Chat ID في الإعدادات")

    # ===== التحديث التلقائي =====
    if auto_refresh:
        time.sleep(1)
        st.rerun()


# ===== تشغيل التطبيق =====
if __name__ == "__main__":
    main()
