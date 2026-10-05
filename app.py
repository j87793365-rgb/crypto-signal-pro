import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
import feedparser
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

BINANCE_REST = "https://data-api.binance.vision"
BYBIT_REST = "https://api.bybit.com"
CONTEXT_CACHE = {}
CONTEXT_CACHE_TTL = 90
DERIV_CACHE = {}
DERIV_CACHE_TTL = 60
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
RADAR_CACHE = {}
RADAR_CACHE_TTL = 120

def valid_symbol(symbol):
    return bool(re.fullmatch(r"[A-Z0-9]{2,18}USDT", symbol or ""))

NEWS_FEEDS = [
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
]

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100/(1+rs))).fillna(50)

def atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def add_indicators(df):
    x = df.copy()
    x["ema5"] = ema(x["close"], 5)
    x["ema10"] = ema(x["close"], 10)
    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)
    x["ema100"] = ema(x["close"], 100)
    x["ema200"] = ema(x["close"], 200)
    x["rsi"] = rsi(x["close"], 14)

    e12 = ema(x["close"], 12)
    e26 = ema(x["close"], 26)
    x["macd"] = e12 - e26
    x["macd_signal"] = ema(x["macd"], 9)
    x["macd_hist"] = x["macd"] - x["macd_signal"]

    x["atr"] = atr(x, 14)
    x["vol_ma20"] = x["volume"].rolling(20).mean()
    x["volume_ratio"] = x["volume"] / x["vol_ma20"]
    x["ret5"] = x["close"].pct_change(5)

    # Bollinger Bands
    x["bb_mid"] = x["close"].rolling(20).mean()
    bb_std = x["close"].rolling(20).std()
    x["bb_upper"] = x["bb_mid"] + 2 * bb_std
    x["bb_lower"] = x["bb_mid"] - 2 * bb_std

    # Session-style cumulative VWAP over fetched window
    typical = (x["high"] + x["low"] + x["close"]) / 3
    pv = typical * x["volume"]
    vol_cum = x["volume"].cumsum().replace(0, np.nan)
    x["vwap"] = pv.cumsum() / vol_cum

    return x

def fetch_klines(symbol, interval, limit=500):
    r = requests.get(
        BINANCE_REST + "/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=12,
    )
    r.raise_for_status()
    rows = r.json()

    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","tb_base","tb_quote","ignore"
    ]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return add_indicators(df)

def timeframe_signal(df):
    row = df.iloc[-1]
    score = 0.0

    if row["ema20"] > row["ema50"] > row["ema200"]:
        score += 2.0
    elif row["ema20"] < row["ema50"] < row["ema200"]:
        score -= 2.0
    else:
        score += 0.7 if row["ema20"] > row["ema50"] else -0.7
        score += 0.5 if row["close"] > row["ema200"] else -0.5

    if 55 <= row["rsi"] <= 70:
        score += 0.8
    elif 30 <= row["rsi"] <= 45:
        score -= 0.8

    score += 0.7 if row["macd_hist"] > 0 else -0.7
    score += 0.5 if row["ret5"] > 0 else -0.5

    if pd.notna(row["volume_ratio"]) and row["volume_ratio"] > 1.25:
        score += 0.5 if row["ret5"] > 0 else -0.5

    return {
        "score": round(float(score), 2),
        "rsi": round(float(row["rsi"]), 2),
        "volume_ratio": round(float(row["volume_ratio"]) if pd.notna(row["volume_ratio"]) else 1.0, 2),
        "atr": float(row["atr"]) if pd.notna(row["atr"]) else 0.0,
        "support": float(df.tail(80)["low"].min()),
        "resistance": float(df.tail(80)["high"].max()),
        "close": float(row["close"]),
    }

def mode_config(mode):
    if mode == "scalp":
        return {
            "weights": {"15m": 0.50, "1h": 0.35, "4h": 0.15},
            "hold": "30 分鐘～4 小時",
            "label": "短線",
        }
    if mode == "swing":
        return {
            "weights": {"15m": 0.10, "1h": 0.30, "4h": 0.60},
            "hold": "1～5 天",
            "label": "波段",
        }
    return {
        "weights": {"15m": 0.25, "1h": 0.40, "4h": 0.35},
        "hold": "2～12 小時",
        "label": "當沖",
    }

def strictness_threshold(strictness):
    return {
        "normal": 1.6,
        "conservative": 2.1,
        "strict": 2.6,
    }.get(strictness, 1.6)

def build_trade_plan(symbol, mode="day", strictness="normal", min_rr=2.0, kline_limit=500):
    t15 = timeframe_signal(fetch_klines(symbol, "15m", limit=kline_limit))
    t1h = timeframe_signal(fetch_klines(symbol, "1h", limit=kline_limit))
    t4h = timeframe_signal(fetch_klines(symbol, "4h", limit=kline_limit))

    cfg = mode_config(mode)
    w = cfg["weights"]
    weighted = t15["score"]*w["15m"] + t1h["score"]*w["1h"] + t4h["score"]*w["4h"]

    price = t15["close"]
    atr1h = max(t1h["atr"], price*0.002)
    threshold = strictness_threshold(strictness)

    align_long = t1h["score"] > 0 and t4h["score"] > 0
    align_short = t1h["score"] < 0 and t4h["score"] < 0

    action = "WAIT"
    entry_low = entry_high = stop = tp1 = tp2 = rr = None

    if weighted >= threshold and align_long:
        action = "LONG"
        entry_low = price - 0.35*atr1h
        entry_high = price - 0.08*atr1h
        entry_mid = (entry_low+entry_high)/2
        stop = min(t1h["support"] - 0.15*atr1h, entry_mid - 1.10*atr1h)
        risk = max(entry_mid-stop, atr1h*0.6)
        tp1 = entry_mid + 1.5*risk
        tp2 = entry_mid + max(min_rr, 2.5)*risk
        rr = (tp2-entry_mid)/risk

    elif weighted <= -threshold and align_short:
        action = "SHORT"
        entry_low = price + 0.08*atr1h
        entry_high = price + 0.35*atr1h
        entry_mid = (entry_low+entry_high)/2
        stop = max(t1h["resistance"] + 0.15*atr1h, entry_mid + 1.10*atr1h)
        risk = max(stop-entry_mid, atr1h*0.6)
        tp1 = entry_mid - 1.5*risk
        tp2 = entry_mid - max(min_rr, 2.5)*risk
        rr = (entry_mid-tp2)/risk

    if action != "WAIT" and rr is not None and rr < min_rr:
        action = "WAIT"
        entry_low = entry_high = stop = tp1 = tp2 = rr = None

    abs_score = abs(weighted)
    if action != "WAIT" and abs_score >= 3.0:
        grade = "A"
    elif action != "WAIT" and abs_score >= 2.1:
        grade = "B"
    else:
        grade = "C"

    warnings = []
    if t15["rsi"] > 72:
        warnings.append("15m RSI 過熱，追多風險增加")
    if t15["rsi"] < 28:
        warnings.append("15m RSI 超賣，追空風險增加")
    if t1h["volume_ratio"] < 0.75:
        warnings.append("1H 量能偏低，突破可信度較弱")
    if action == "WAIT":
        warnings.append("目前條件未達你的設定門檻")

    confidence = int(max(0, min(100, 50 + weighted*11)))

    return {
        "symbol": symbol,
        "price": price,
        "action": action,
        "grade": grade,
        "score": round(weighted, 2),
        "confidence": confidence,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "rr": round(rr,2) if rr is not None else None,
        "warnings": warnings,
        "timeframes": {"15m": t15, "1h": t1h, "4h": t4h},
        "mode_label": cfg["label"],
        "holding_time": cfg["hold"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }



def _cache_get(cache, key, ttl):
    item = cache.get(key)
    if item and time.time() - item["time"] < ttl:
        return item["data"]
    return None


def _cache_set(cache, key, data):
    cache[key] = {"time": time.time(), "data": data}
    return data


def get_derivatives_snapshot(symbol):
    cached = _cache_get(DERIV_CACHE, symbol, DERIV_CACHE_TTL)
    if cached is not None:
        return cached

    data = {
        "available": False,
        "funding_rate": None,
        "funding_pct": None,
        "open_interest": None,
        "oi_change_pct": None,
        "oi_label": "暫無資料",
        "funding_label": "暫無資料",
    }

    try:
        ticker = requests.get(
            BYBIT_REST + "/v5/market/tickers",
            params={"category":"linear", "symbol":symbol},
            timeout=8,
        )
        ticker.raise_for_status()
        payload = ticker.json()
        rows = payload.get("result", {}).get("list", [])
        if rows:
            row = rows[0]
            funding = float(row.get("fundingRate") or 0)
            oi = float(row.get("openInterest") or 0)
            data["funding_rate"] = funding
            data["funding_pct"] = round(funding * 100, 4)
            data["open_interest"] = oi

            if funding >= 0.0015:
                data["funding_label"] = "多頭偏擁擠"
            elif funding <= -0.0015:
                data["funding_label"] = "空頭偏擁擠"
            elif funding > 0:
                data["funding_label"] = "小幅正費率"
            elif funding < 0:
                data["funding_label"] = "小幅負費率"
            else:
                data["funding_label"] = "中性"

        oi_resp = requests.get(
            BYBIT_REST + "/v5/market/open-interest",
            params={"category":"linear", "symbol":symbol, "intervalTime":"1h", "limit":4},
            timeout=8,
        )
        oi_resp.raise_for_status()
        oi_rows = oi_resp.json().get("result", {}).get("list", [])
        if len(oi_rows) >= 2:
            ordered = sorted(oi_rows, key=lambda x: int(x.get("timestamp", 0)))
            old = float(ordered[0].get("openInterest") or 0)
            new = float(ordered[-1].get("openInterest") or 0)
            if old > 0:
                change = (new / old - 1) * 100
                data["oi_change_pct"] = round(change, 2)
                if change >= 2:
                    data["oi_label"] = "OI 明顯增加"
                elif change >= 0.5:
                    data["oi_label"] = "OI 增加"
                elif change <= -2:
                    data["oi_label"] = "OI 明顯下降"
                elif change <= -0.5:
                    data["oi_label"] = "OI 下降"
                else:
                    data["oi_label"] = "OI 平穩"

        data["available"] = data["funding_rate"] is not None or data["oi_change_pct"] is not None
    except Exception:
        pass

    return _cache_set(DERIV_CACHE, symbol, data)


def classify_regime():
    try:
        df1 = fetch_klines("BTCUSDT", "1h", limit=240)
        df4 = fetch_klines("BTCUSDT", "4h", limit=240)
        s1 = timeframe_signal(df1)
        s4 = timeframe_signal(df4)

        last4 = df4.iloc[-1]
        atr_pct = float(last4["atr"] / last4["close"] * 100) if last4["close"] else 0
        atr_series = (df4["atr"] / df4["close"] * 100).dropna().tail(80)
        median_atr = float(atr_series.median()) if len(atr_series) else atr_pct

        same_trend = (s1["score"] > 1.2 and s4["score"] > 1.2) or (s1["score"] < -1.2 and s4["score"] < -1.2)
        if same_trend:
            regime = "趨勢盤"
        else:
            regime = "震盪盤"

        if median_atr > 0 and atr_pct > median_atr * 1.35:
            volatility = "高波動"
        elif median_atr > 0 and atr_pct < median_atr * 0.75:
            volatility = "低波動"
        else:
            volatility = "正常波動"

        btc_score = round(s1["score"] * 0.4 + s4["score"] * 0.6, 2)
        if btc_score >= 2.0:
            btc_bias = "強多"
        elif btc_score >= 0.7:
            btc_bias = "偏多"
        elif btc_score <= -2.0:
            btc_bias = "強空"
        elif btc_score <= -0.7:
            btc_bias = "偏空"
        else:
            btc_bias = "中性"

        return {
            "btc_score": btc_score,
            "btc_bias": btc_bias,
            "regime": regime,
            "volatility": volatility,
            "btc_1h": s1["score"],
            "btc_4h": s4["score"],
            "atr_pct": round(atr_pct, 2),
        }
    except Exception:
        return {"btc_score":0,"btc_bias":"未知","regime":"未知","volatility":"未知","btc_1h":0,"btc_4h":0,"atr_pct":0}


def market_context(candidate_count=16, force=False):
    key = f"market_{candidate_count}"
    if not force:
        cached = _cache_get(CONTEXT_CACHE, key, CONTEXT_CACHE_TTL)
        if cached is not None:
            return cached

    base = classify_regime()
    candidates = fetch_top_usdt_symbols(candidate_count)
    long_count = short_count = neutral_count = 0

    def one(symbol):
        try:
            a = timeframe_signal(fetch_klines(symbol, "1h", limit=220))["score"]
            b = timeframe_signal(fetch_klines(symbol, "4h", limit=220))["score"]
            return a * 0.4 + b * 0.6
        except Exception:
            return 0

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(one, s) for s,_ in candidates]
        for f in as_completed(futures):
            score = f.result()
            if score > 0.7:
                long_count += 1
            elif score < -0.7:
                short_count += 1
            else:
                neutral_count += 1

    total = max(1, long_count + short_count + neutral_count)
    breadth_score = (long_count - short_count) / total
    if breadth_score >= 0.35:
        breadth_label = "市場明顯偏多"
    elif breadth_score >= 0.12:
        breadth_label = "市場偏多"
    elif breadth_score <= -0.35:
        breadth_label = "市場明顯偏空"
    elif breadth_score <= -0.12:
        breadth_label = "市場偏空"
    else:
        breadth_label = "市場分歧／中性"

    base.update({
        "long_count": long_count,
        "short_count": short_count,
        "neutral_count": neutral_count,
        "breadth_score": round(breadth_score, 2),
        "breadth_label": breadth_label,
        "scanned": total,
    })
    return _cache_set(CONTEXT_CACHE, key, base)


def context_alignment_bonus(action, context):
    if action == "WAIT":
        return 0, "中性"
    direction = 1 if action == "LONG" else -1
    btc = float(context.get("btc_score", 0))
    breadth = float(context.get("breadth_score", 0))
    combined = direction * (btc * 0.65 + breadth * 3.0 * 0.35)
    if combined >= 1.4:
        return 10, "市場支持"
    if combined >= 0.3:
        return 4, "小幅支持"
    if combined <= -1.4:
        return -14, "市場逆風"
    if combined <= -0.3:
        return -6, "略逆風"
    return 0, "市場中性"


def derivative_bonus(action, deriv):
    if not deriv or not deriv.get("available") or action == "WAIT":
        return 0, "衍生品資料不足"

    bonus = 0
    notes = []
    oi = deriv.get("oi_change_pct")
    funding = deriv.get("funding_rate")

    if oi is not None:
        if oi >= 2:
            bonus += 7
            notes.append("OI明顯增加")
        elif oi >= 0.5:
            bonus += 4
            notes.append("OI增加")
        elif oi <= -2:
            bonus -= 5
            notes.append("OI明顯下降")
        elif oi <= -0.5:
            bonus -= 2
            notes.append("OI下降")

    if funding is not None:
        if action == "LONG":
            if funding >= 0.0015:
                bonus -= 9
                notes.append("多頭過度擁擠")
            elif funding < 0:
                bonus += 3
                notes.append("Funding對多方較有利")
            elif abs(funding) < 0.0005:
                bonus += 2
                notes.append("Funding正常")
        else:
            if funding <= -0.0015:
                bonus -= 9
                notes.append("空頭過度擁擠")
            elif funding > 0:
                bonus += 3
                notes.append("Funding對空方較有利")
            elif abs(funding) < 0.0005:
                bonus += 2
                notes.append("Funding正常")

    return bonus, "、".join(notes) if notes else "衍生品中性"

def fetch_top_usdt_symbols(limit=16):
    r = requests.get(BINANCE_REST + "/api/v3/ticker/24hr", timeout=12)
    r.raise_for_status()
    items = r.json()
    stable_bases = {"USDC","FDUSD","TUSD","USDP","BUSD","DAI","EUR","EURC","TRY","BRL","GBP","AUD"}
    leveraged_suffixes = ("UP","DOWN","BULL","BEAR")
    rows = []
    for item in items:
        symbol = str(item.get("symbol", "")).upper()
        if not valid_symbol(symbol):
            continue
        base = symbol[:-4]
        if base in stable_bases or base.endswith(leveraged_suffixes):
            continue
        try:
            qv = float(item.get("quoteVolume", 0) or 0)
        except Exception:
            qv = 0.0
        if qv > 0:
            rows.append((symbol, qv))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:max(5, min(int(limit), 30))]

def radar_entry_state(plan):
    if plan.get("action") == "WAIT" or not plan.get("entry_low") or not plan.get("entry_high"):
        return {"key":"wait","label":"等待","distance_pct":None,"bonus":-10}
    price = float(plan["price"])
    low = min(float(plan["entry_low"]), float(plan["entry_high"]))
    high = max(float(plan["entry_low"]), float(plan["entry_high"]))
    action = plan.get("action")
    if low <= price <= high:
        return {"key":"in_zone","label":"正在進場區","distance_pct":0.0,"bonus":18}
    if action == "LONG":
        if price > high:
            pct = (price/high - 1)*100
            if pct <= 0.25:
                return {"key":"near","label":"剛離開進場區","distance_pct":pct,"bonus":8}
            return {"key":"missed","label":"已錯過／不建議追價","distance_pct":pct,"bonus":-18}
        pct = (low/price - 1)*100
        return {"key":"waiting","label":"等待回到進場區","distance_pct":pct,"bonus":5}
    if price < low:
        pct = (low/price - 1)*100
        if pct <= 0.25:
            return {"key":"near","label":"剛離開進場區","distance_pct":pct,"bonus":8}
        return {"key":"missed","label":"已錯過／不建議追空","distance_pct":pct,"bonus":-18}
    pct = (price/high - 1)*100
    return {"key":"waiting","label":"等待回到進場區","distance_pct":abs(pct),"bonus":5}

def opportunity_score(plan, context=None, deriv=None):
    if plan.get("action") == "WAIT":
        return 0, radar_entry_state(plan), {"market":"中性","derivatives":"中性"}

    abs_score = min(abs(float(plan.get("score", 0))), 4.5)
    strength = (abs_score / 4.5) * 35
    grade_bonus = {"A":18, "B":12, "C":4}.get(plan.get("grade"), 0)

    tfs = plan.get("timeframes", {})
    direction = 1 if plan.get("action") == "LONG" else -1
    aligned = sum(1 for tf in ("15m","1h","4h") if direction * float(tfs.get(tf, {}).get("score", 0)) > 0.7)
    alignment_bonus = aligned * 5

    vr15 = float(tfs.get("15m", {}).get("volume_ratio", 0) or 0)
    vr1h = float(tfs.get("1h", {}).get("volume_ratio", 0) or 0)
    volume_bonus = min(10, max(vr15, vr1h) * 4)

    rr = float(plan.get("rr") or 0)
    rr_bonus = min(10, rr * 3)

    state = radar_entry_state(plan)
    market_bonus, market_label = context_alignment_bonus(plan.get("action"), context or {})
    deriv_bonus, deriv_label = derivative_bonus(plan.get("action"), deriv or {})

    regime_bonus = 0
    regime = (context or {}).get("regime")
    volatility = (context or {}).get("volatility")
    if regime == "趨勢盤":
        regime_bonus += 4
    if volatility == "高波動":
        regime_bonus -= 4
    elif volatility == "低波動":
        regime_bonus -= 2

    score = strength + grade_bonus + alignment_bonus + volume_bonus + rr_bonus + state["bonus"] + market_bonus + deriv_bonus + regime_bonus
    meta = {"market":market_label,"derivatives":deriv_label}
    return int(max(0, min(100, round(score)))), state, meta


def scan_one_symbol(symbol, quote_volume, mode, strictness, min_rr, context):
    try:
        plan = build_trade_plan(symbol, mode, strictness, min_rr, kline_limit=240)
        deriv = get_derivatives_snapshot(symbol)
        opp, state, meta = opportunity_score(plan, context=context, deriv=deriv)
        if plan.get("action") == "WAIT":
            return None
        return {
            "symbol": symbol,
            "action": plan.get("action"),
            "grade": plan.get("grade"),
            "opportunity": opp,
            "score": plan.get("score"),
            "rr": plan.get("rr"),
            "price": plan.get("price"),
            "entry_low": plan.get("entry_low"),
            "entry_high": plan.get("entry_high"),
            "state": state,
            "quote_volume": quote_volume,
            "holding_time": plan.get("holding_time"),
            "mode_label": plan.get("mode_label"),
            "derivatives": deriv,
            "market_fit": meta["market"],
            "derivative_fit": meta["derivatives"],
        }
    except Exception:
        return None


def scan_market(mode="day", strictness="normal", min_rr=2.0, direction="all", candidate_count=16):
    cache_key = (mode, strictness, round(float(min_rr), 2), direction, int(candidate_count), "v2")
    cached = RADAR_CACHE.get(cache_key)
    now = time.time()
    if cached and now - cached["time"] < RADAR_CACHE_TTL:
        return cached["data"]

    candidates = fetch_top_usdt_symbols(candidate_count)
    context = market_context(candidate_count)
    results = []

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(scan_one_symbol, symbol, qv, mode, strictness, min_rr, context) for symbol, qv in candidates]
        for f in as_completed(futures):
            item = f.result()
            if not item:
                continue
            if direction == "long" and item["action"] != "LONG":
                continue
            if direction == "short" and item["action"] != "SHORT":
                continue
            results.append(item)

    results.sort(key=lambda x: (x["opportunity"], x["quote_volume"]), reverse=True)
    data = {
        "scanned": len(candidates),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "market_context": context,
        "results": results[:8],
    }
    RADAR_CACHE[cache_key] = {"time": now, "data": data}
    return data

HIGH_IMPACT_WORDS = [
    "fed","federal reserve","cpi","inflation","interest rate","sec",
    "etf","hack","exploit","liquidation","tariff","war","regulation"
]


NEWS_TRANSLATION_CACHE = {}

def translate_title_zh(title):
    """Translate an English news title to Traditional Chinese. Falls back to original on failure."""
    if not title:
        return title
    if title in NEWS_TRANSLATION_CACHE:
        return NEWS_TRANSLATION_CACHE[title]
    try:
        r = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={
                "client": "gtx",
                "sl": "auto",
                "tl": "zh-TW",
                "dt": "t",
                "q": title,
            },
            timeout=6,
            headers={"User-Agent":"Mozilla/5.0"},
        )
        r.raise_for_status()
        data = r.json()
        zh = "".join(part[0] for part in data[0] if part and part[0]).strip()
        if not zh:
            zh = title
    except Exception:
        zh = title
    NEWS_TRANSLATION_CACHE[title] = zh
    return zh

def news_items(limit=40):
    out, seen = [], set()
    for source, url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:25]:
                title = getattr(e, "title", "").strip()
                if not title or title in seen:
                    continue
                seen.add(title)
                link = getattr(e, "link", "").strip()
                published = getattr(e, "published", "") or getattr(e, "updated", "")
                low = title.lower()
                title_zh = translate_title_zh(title)
                out.append({
                    "source": source,
                    "title": title_zh,
                    "title_original": title,
                    "link": link,
                    "published": published,
                    "high_impact": any(w in low for w in HIGH_IMPACT_WORDS)
                })
        except Exception:
            pass
    return out[:limit]

HTML = r"""
<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#090a0d">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>Crypto Signal Pro</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.2/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{--bg:#090a0d;--card:#12141a;--line:#252936;--text:#f5f6f8;--muted:#9298a8;--gold:#d9b45b;--green:#47d18c;--red:#ff666e;--yellow:#f2c96b}
*{box-sizing:border-box}
body{margin:0;background:#090a0d;color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding-bottom:90px;padding-top:env(safe-area-inset-top)}
.wrap{max-width:760px;margin:auto;padding:16px}
.top{display:block;margin-bottom:14px;padding-top:4px}
h1{font-size:28px;line-height:1.08;margin:0}.sub{font-size:12px;color:var(--muted);margin-top:6px}
.coinPicker{margin-top:14px;display:flex;gap:8px;overflow-x:auto;padding-bottom:4px;-webkit-overflow-scrolling:touch;scrollbar-width:none}.coinPicker::-webkit-scrollbar{display:none}
.coinBtn{flex:0 0 auto;min-width:76px;border-radius:14px;padding:10px 14px;border:1px solid var(--line);background:#12151b;color:#9aa1af;font-weight:750}.coinBtn.active{background:#20242d;color:#fff;border-color:#6f7685;box-shadow:inset 0 0 0 1px rgba(217,180,91,.25)}
#symbol{display:none}
select,button,input{background:#171a22;color:#fff;border:1px solid var(--line);border-radius:12px;padding:10px 12px}
.hero{background:linear-gradient(145deg,#171920,#101217);border:1px solid var(--line);border-radius:22px;padding:18px}
.heroTop{display:flex;justify-content:space-between;gap:10px}
.label{font-size:12px;color:var(--muted)}.price{font-size:34px;font-weight:850;margin-top:5px}
.action{font-size:24px;font-weight:850}.long{color:var(--green)}.short{color:var(--red)}.wait{color:var(--yellow)}
.badge{display:inline-block;margin-top:6px;padding:5px 9px;border-radius:999px;background:#222631;font-size:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:14px}
.big{font-size:21px;font-weight:800;margin-top:5px}.section{margin-top:12px}.sectionTitle{font-size:15px;font-weight:800;margin:0 0 8px 2px}
.row{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:11px 0;border-bottom:1px solid #20232c;font-size:14px}.row:last-child{border-bottom:0}
.tfrow{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.tfbox{background:#111318;border:1px solid var(--line);border-radius:14px;padding:10px}
.muted{font-size:12px;color:var(--muted);line-height:1.5}.news{display:flex;flex-direction:column;gap:8px}
.newsitem{background:#111318;border:1px solid var(--line);border-radius:14px;padding:12px}.newsitem a{color:#fff;text-decoration:none;font-size:14px;line-height:1.4}
.tag{font-size:10px;padding:3px 6px;border-radius:7px;background:#472127;color:#ffb3b7;margin-left:6px}
.page{display:none}.page.active{display:block}.bar{height:9px;background:#242832;border-radius:999px;overflow:hidden;margin-top:8px}.barin{height:100%;width:0;background:linear-gradient(90deg,#947334,#e1c46d)}
.bottom{position:fixed;left:50%;transform:translateX(-50%);bottom:0;width:100%;max-width:760px;background:rgba(14,16,21,.96);border-top:1px solid var(--line);display:grid;grid-template-columns:repeat(5,1fr);padding:8px 8px calc(8px + env(safe-area-inset-bottom));backdrop-filter:blur(16px)}
.nav{border:0;background:transparent;color:#969dab;font-size:11px}.nav.active{color:var(--gold)}.ico{font-size:20px;display:block;margin-bottom:2px}
.toggle{width:48px;height:28px;border-radius:999px;background:#2a2d36;position:relative;flex:0 0 auto}.toggle.on{background:#8d7136}.knob{position:absolute;width:22px;height:22px;border-radius:50%;background:#fff;top:3px;left:3px;transition:.2s}.toggle.on .knob{left:23px}
.settingSelect{min-width:145px}.notice{color:#f0c867;line-height:1.55}
.chartWrap{height:280px;margin-top:10px}.chartWrap.small{height:190px}.chartTools{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px}
.statusGood{color:var(--green)}.statusWarn{color:var(--yellow)}.statusBad{color:var(--red)}
.tradeHint{margin-top:10px;padding:12px;border-radius:14px;background:#101217;border:1px solid var(--line);font-weight:750;line-height:1.45}
.tradeHint.good{color:var(--green);border-color:rgba(71,209,140,.35)}
.tradeHint.warn{color:var(--yellow);border-color:rgba(242,201,107,.35)}
.tradeHint.bad{color:var(--red);border-color:rgba(255,102,110,.35)}
.candleBox{height:390px;margin-top:10px;border-radius:12px;overflow:hidden;background:#0d0f14}
.chartLegend{display:flex;gap:8px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-top:10px}.indicatorControls{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 4px}.indicatorBtn{padding:7px 10px;border-radius:999px;border:1px solid var(--line);background:#111318;color:#9298a8;font-size:12px}.indicatorBtn.on{color:#fff;border-color:#6b707d;background:#20242d}.indicatorGroupTitle{width:100%;font-size:11px;color:#777e8c;margin-top:4px}.miniActionRow{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
.radarControls{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px}.radarBtn{width:100%;font-weight:800;background:#2a2416;border-color:#6f5926;color:#f2c96b}.radarList{display:flex;flex-direction:column;gap:10px}.radarCard{background:#111318;border:1px solid var(--line);border-radius:16px;padding:14px}.radarTop{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}.radarSymbol{font-size:20px;font-weight:850}.radarScore{font-size:24px;font-weight:900}.radarMeta{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}.radarChip{font-size:11px;padding:5px 8px;border-radius:999px;background:#20232c;color:#c7ccd7}.radarOpen{margin-top:10px;width:100%;background:#171a22}.rank{font-size:20px;margin-right:6px}.radarEmpty{text-align:center;padding:24px;color:var(--muted)}
.marketGrid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.marketCell{background:#111318;border:1px solid var(--line);border-radius:14px;padding:11px}.marketCell strong{display:block;margin-top:5px;font-size:16px}.derivGrid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div><h1>Crypto Signal Pro</h1><div class="sub">即時價格 · 多週期 · 新聞風險</div></div>
    <select id="symbol" aria-hidden="true">
      <option>BTCUSDT</option><option>ETHUSDT</option><option>SOLUSDT</option><option>BNBUSDT</option><option>XRPUSDT</option>
    </select>
    <div class="coinPicker" id="coinPicker">
      <button class="coinBtn active" data-symbol="BTCUSDT">BTC</button>
      <button class="coinBtn" data-symbol="ETHUSDT">ETH</button>
      <button class="coinBtn" data-symbol="SOLUSDT">SOL</button>
      <button class="coinBtn" data-symbol="BNBUSDT">BNB</button>
      <button class="coinBtn" data-symbol="XRPUSDT">XRP</button>
    </div>
  </div>

  <section id="page-home" class="page active">
    <div class="hero">
      <div class="heroTop">
        <div><div class="label">即時價格</div><div class="price" id="price">--</div><div class="muted" id="tick">連線中...</div></div>
        <div style="text-align:right"><div class="label">目前建議</div><div class="action wait" id="action">--</div><div class="badge" id="grade">訊號 --</div></div>
      </div>
      <div class="grid2">
        <div class="card"><div class="label">綜合分數</div><div class="big" id="score">--</div><div class="bar"><div id="scorebar" class="barin"></div></div></div>
        <div class="card"><div class="label">風報比</div><div class="big" id="rr">--</div></div>
      </div>
    </div>

    <div class="section">
      <div class="sectionTitle">市場環境</div>
      <div class="card">
        <div class="marketGrid">
          <div class="marketCell"><div class="label">BTC 大盤</div><strong id="btcBias">--</strong></div>
          <div class="marketCell"><div class="label">市場型態</div><strong id="marketRegime">--</strong></div>
          <div class="marketCell"><div class="label">整體多空</div><strong id="marketBreadth">--</strong></div>
          <div class="marketCell"><div class="label">波動</div><strong id="marketVolatility">--</strong></div>
        </div>
        <div class="derivGrid">
          <div class="marketCell"><div class="label">Funding</div><strong id="fundingData">--</strong><div class="muted" id="fundingLabel"></div></div>
          <div class="marketCell"><div class="label">OI 變化</div><strong id="oiData">--</strong><div class="muted" id="oiLabel"></div></div>
        </div>
      </div>
    </div>

    <div class="section">
      <div class="sectionTitle">交易規劃</div>
      <div class="card">
        <div class="row"><span>交易模式</span><strong id="modeLabel">--</strong></div>
        <div class="row"><span>建議持倉</span><strong id="holding">--</strong></div>
        <div class="row"><span>進場區</span><strong id="entry">等待</strong></div>
        <div class="row"><span>止損</span><strong id="stop">--</strong></div>
        <div class="row"><span>TP1</span><strong id="tp1">--</strong></div>
        <div class="row"><span>TP2</span><strong id="tp2">--</strong></div>
        <div class="row"><span>TP1 建議平倉</span><strong id="partialText">40%</strong></div>
      </div>
    </div>

    <div class="section">
      <div class="sectionTitle">多週期</div>
      <div class="tfrow" id="timeframes"></div>
    </div>

    <div class="section">
      <div class="sectionTitle">進場狀態</div>
      <div class="card">
        <div class="row"><span>目前位置</span><strong id="entryStatus">--</strong></div>
        <div class="row"><span>距離進場區</span><strong id="entryDistance">--</strong></div>
      </div>
    </div>

    <div class="section">
      <div class="sectionTitle">技術圖表</div>
      <div class="card">
        <div class="chartTools">
          <div class="muted">K 線 / EMA / 進場區 / Stop / TP</div>
          <select id="chartInterval" style="min-width:110px">
            <option value="15m">15m</option>
            <option value="1h" selected>1h</option>
            <option value="4h">4h</option>
          </select>
        </div>

        <div id="tradeHint" class="tradeHint warn">讀取進場狀態中...</div>

        <div class="indicatorControls" id="indicatorControls">
          <div class="indicatorGroupTitle">主圖線條（點一下顯示 / 隱藏）</div>
          <button class="indicatorBtn" data-ind="ema5">EMA5</button>
          <button class="indicatorBtn" data-ind="ema10">EMA10</button>
          <button class="indicatorBtn on" data-ind="ema20">EMA20</button>
          <button class="indicatorBtn on" data-ind="ema50">EMA50</button>
          <button class="indicatorBtn" data-ind="ema100">EMA100</button>
          <button class="indicatorBtn on" data-ind="ema200">EMA200</button>
          <button class="indicatorBtn" data-ind="vwap">VWAP</button>
          <button class="indicatorBtn" data-ind="bb">布林通道</button>

          <div class="indicatorGroupTitle">交易線</div>
          <button class="indicatorBtn on" data-ind="entry">Entry</button>
          <button class="indicatorBtn on" data-ind="stop">Stop</button>
          <button class="indicatorBtn on" data-ind="tp1">TP1</button>
          <button class="indicatorBtn on" data-ind="tp2">TP2</button>
        </div>

        <div class="miniActionRow">
          <button id="onlyPriceBtn" class="indicatorBtn">只看 K 線</button>
          <button id="resetIndicatorsBtn" class="indicatorBtn">恢復預設</button>
        </div>

        <div id="candleChart" class="candleBox"></div>

        <div class="chartLegend">
          <span>點上面的按鈕決定要看哪些線，不用全部一起開。</span>
        </div>

        <div class="indicatorControls">
          <div class="indicatorGroupTitle">副圖</div>
          <button class="indicatorBtn on" id="toggleMacd">MACD</button>
          <button class="indicatorBtn on" id="toggleVolume">成交量</button>
        </div>

        <div id="macdWrap" class="chartWrap small">
          <canvas id="macdChart"></canvas>
        </div>

        <div id="volumeWrap" class="chartWrap small">
          <canvas id="volumeChart"></canvas>
        </div>
      </div>
    </div>

    <div class="section">
      <div class="sectionTitle">風險提醒</div>
      <div class="card muted" id="warnings">讀取中...</div>
    </div>
  </section>

  <section id="page-radar" class="page">
    <div class="sectionTitle">⚡ 市場雷達</div>
    <div class="card">
      <div class="radarControls">
        <select id="radarDirection"><option value="all">全部方向</option><option value="long">只找做多</option><option value="short">只找做空</option></select>
        <select id="radarCount"><option value="12">掃描前 12 名</option><option value="16" selected>掃描前 16 名</option><option value="20">掃描前 20 名</option></select>
      </div>
      <button id="scanMarketBtn" class="radarBtn">⚡ 掃描市場</button>
      <div class="muted" style="margin-top:8px">先抓 24H 高成交量 USDT 幣，再依多週期、量能、風報比與進場位置排序。</div>
    </div>
    <div class="section" id="radarSummary"></div>
    <div id="radarList" class="radarList"><div class="radarEmpty">按「掃描市場」開始找機會</div></div>
  </section>

  <section id="page-news" class="page">
    <div class="sectionTitle">最新新聞</div>
    <div class="news" id="newsbox"><div class="card muted">讀取中...</div></div>
  </section>

  <section id="page-alerts" class="page">
    <div class="sectionTitle">通知</div>
    <div class="card">
      <div class="row"><span>LONG / SHORT / WAIT 變化通知</span><div id="notifyToggle" class="toggle"><div class="knob"></div></div></div>
      <div class="row"><span>只提醒 A 級訊號</span><div id="aOnlyToggle" class="toggle"><div class="knob"></div></div></div>
      <button style="width:100%;margin-top:12px" onclick="requestNotify()">開啟瀏覽器通知權限</button>
    </div>
    <div class="card section muted">
      iPhone 瀏覽器通知能力會依 Safari / PWA 狀態而異。即使通知沒跳，首頁訊號仍會正常更新。
    </div>
  </section>

  <section id="page-settings" class="page">
    <div class="sectionTitle">設定</div>
    <div class="card">
      <div class="row">
        <span>交易模式</span>
        <select id="mode" class="settingSelect">
          <option value="scalp">短線</option>
          <option value="day">當沖</option>
          <option value="swing">波段</option>
        </select>
      </div>
      <div class="row">
        <span>訊號嚴格度</span>
        <select id="strictness" class="settingSelect">
          <option value="normal">一般</option>
          <option value="conservative">保守</option>
          <option value="strict">嚴格</option>
        </select>
      </div>
      <div class="row">
        <span>最低風報比</span>
        <select id="minRR" class="settingSelect">
          <option value="1.5">1 : 1.5</option>
          <option value="2.0">1 : 2</option>
          <option value="2.5">1 : 2.5</option>
          <option value="3.0">1 : 3</option>
        </select>
      </div>
      <div class="row">
        <span>TP1 平倉比例</span>
        <select id="partial" class="settingSelect">
          <option value="30">30%</option>
          <option value="40">40%</option>
          <option value="50">50%</option>
          <option value="60">60%</option>
        </select>
      </div>
    </div>

    <div class="card section notice">
      設定會存在這台手機的瀏覽器。訊號越嚴格，出現 LONG / SHORT 的次數會越少；不是勝率保證。
    </div>
  </section>
</div>

<nav class="bottom">
  <button class="nav active" data-page="home"><span class="ico">⌂</span>首頁</button>
  <button class="nav" data-page="radar"><span class="ico">⚡</span>雷達</button>
  <button class="nav" data-page="news"><span class="ico">📰</span>新聞</button>
  <button class="nav" data-page="alerts"><span class="ico">🔔</span>通知</button>
  <button class="nav" data-page="settings"><span class="ico">⚙️</span>設定</button>
</nav>

<script>
let ws=null,currentSymbol="BTCUSDT",lastAction=null,lastGrade=null;
let candleChart=null,candleSeries=null;
let indicatorSeries={};
let tradePriceLines=[];
let macdChart=null,volumeChart=null;
let latestAnalysis=null;
let latestChartData=null;

const defaultIndicatorState={
  ema5:false,ema10:false,ema20:true,ema50:true,ema100:false,ema200:true,
  vwap:false,bb:false,entry:true,stop:true,tp1:true,tp2:true,
  macd:true,volume:true
};

function getIndicatorState(){
  return {...defaultIndicatorState,...JSON.parse(localStorage.getItem("indicatorState")||"{}")};
}
function saveIndicatorState(s){
  localStorage.setItem("indicatorState",JSON.stringify(s));
}
function syncIndicatorButtons(){
  const s=getIndicatorState();
  document.querySelectorAll("[data-ind]").forEach(btn=>{
    btn.classList.toggle("on",!!s[btn.dataset.ind]);
  });
  document.getElementById("toggleMacd").classList.toggle("on",!!s.macd);
  document.getElementById("toggleVolume").classList.toggle("on",!!s.volume);
  document.getElementById("macdWrap").style.display=s.macd?"block":"none";
  document.getElementById("volumeWrap").style.display=s.volume?"block":"none";
}

const defaults={
  mode:"day",
  strictness:"normal",
  minRR:"2.0",
  partial:"40",
  notify:false,
  aOnly:false
};

function getSettings(){
  return {...defaults,...JSON.parse(localStorage.getItem("cryptoSettings")||"{}")};
}
function saveSettings(){
  const s={
    mode:document.getElementById("mode").value,
    strictness:document.getElementById("strictness").value,
    minRR:document.getElementById("minRR").value,
    partial:document.getElementById("partial").value,
    notify:document.getElementById("notifyToggle").classList.contains("on"),
    aOnly:document.getElementById("aOnlyToggle").classList.contains("on")
  };
  localStorage.setItem("cryptoSettings",JSON.stringify(s));
  document.getElementById("partialText").textContent=s.partial+"%";
  loadAnalysis();
}
function applySettings(){
  const s=getSettings();
  document.getElementById("mode").value=s.mode;
  document.getElementById("strictness").value=s.strictness;
  document.getElementById("minRR").value=s.minRR;
  document.getElementById("partial").value=s.partial;
  document.getElementById("partialText").textContent=s.partial+"%";
  document.getElementById("notifyToggle").classList.toggle("on",!!s.notify);
  document.getElementById("aOnlyToggle").classList.toggle("on",!!s.aOnly);
}
function fmt(n){
  if(n===null||n===undefined) return "--";
  const x=Number(n), digits=x>=1000?2:(x>=1?4:6);
  return x.toLocaleString(undefined,{maximumFractionDigits:digits});
}
function connectWS(){
  if(ws) try{ws.close()}catch(e){}
  currentSymbol=document.getElementById("symbol").value;
  ws=new WebSocket("wss://stream.binance.com:9443/ws/"+currentSymbol.toLowerCase()+"@trade");
  ws.onopen=()=>document.getElementById("tick").textContent="即時串流已連線";
  ws.onmessage=e=>{
    const d=JSON.parse(e.data);
    document.getElementById("price").textContent=fmt(d.p);
    document.getElementById("tick").textContent="最新成交";
  };
  ws.onerror=()=>document.getElementById("tick").textContent="即時串流異常";
  ws.onclose=()=>setTimeout(()=>{if(currentSymbol===document.getElementById("symbol").value)connectWS()},2500);
}

function chartBaseOptions(){
  return {
    responsive:true,
    maintainAspectRatio:false,
    interaction:{mode:"index",intersect:false},
    plugins:{
      legend:{labels:{color:"#f5f6f8",boxWidth:12}},
      tooltip:{enabled:true}
    },
    scales:{
      x:{ticks:{color:"#9298a8",maxTicksLimit:6},grid:{color:"#252936"}},
      y:{ticks:{color:"#9298a8"},grid:{color:"#252936"}}
    }
  };
}

function updateEntryStatus(d){
  latestAnalysis=d;

  const statusEl=document.getElementById("entryStatus");
  const distEl=document.getElementById("entryDistance");
  const hint=document.getElementById("tradeHint");

  const liveText=document.getElementById("price").textContent.replaceAll(",","");
  const p=Number(liveText) || d.price;

  if(!d.entry_low || !d.entry_high || d.action==="WAIT"){
    statusEl.textContent="等待訊號";
    statusEl.className="wait";
    distEl.textContent="--";
    hint.textContent="目前沒有符合條件的高品質進場訊號，先等待。";
    hint.className="tradeHint warn";
    return;
  }

  const low=Math.min(d.entry_low,d.entry_high);
  const high=Math.max(d.entry_low,d.entry_high);

  if(p>=low && p<=high){
    statusEl.textContent="已進場區";
    statusEl.className="statusGood";
    distEl.textContent="價格在區間內";
    hint.textContent=`已進入 ${d.action} 建議進場區，可依你的風控規則評估進場。`;
    hint.className="tradeHint good";
    return;
  }

  if(d.action==="LONG"){
    if(p>high){
      const pct=(p/high-1)*100;
      if(pct<0.25){
        statusEl.textContent="剛離開進場區";
        statusEl.className="statusWarn";
        distEl.textContent=`高於上緣 ${pct.toFixed(2)}%`;
        hint.textContent="剛離開最佳進場區，若要進場要特別注意不要追價。";
        hint.className="tradeHint warn";
      }else{
        statusEl.textContent="已錯過";
        statusEl.className="statusBad";
        distEl.textContent=`高於上緣 ${pct.toFixed(2)}%`;
        hint.textContent="已明顯離開最佳做多進場區，不建議追價，等待回踩或下一個訊號。";
        hint.className="tradeHint bad";
      }
    }else{
      const pct=(low/p-1)*100;
      statusEl.textContent="尚未進場";
      statusEl.className="statusWarn";
      distEl.textContent=`低於下緣 ${pct.toFixed(2)}%`;
      hint.textContent="價格尚未回到建議做多進場區，等待條件成立。";
      hint.className="tradeHint warn";
    }
    return;
  }

  if(d.action==="SHORT"){
    if(p<low){
      const pct=(low/p-1)*100;
      if(pct<0.25){
        statusEl.textContent="剛離開進場區";
        statusEl.className="statusWarn";
        distEl.textContent=`低於下緣 ${pct.toFixed(2)}%`;
        hint.textContent="剛離開最佳做空進場區，若要進場要避免追空。";
        hint.className="tradeHint warn";
      }else{
        statusEl.textContent="已錯過";
        statusEl.className="statusBad";
        distEl.textContent=`低於下緣 ${pct.toFixed(2)}%`;
        hint.textContent="已明顯離開最佳做空進場區，不建議追空，等待反彈或下一個訊號。";
        hint.className="tradeHint bad";
      }
    }else{
      const pct=(p/high-1)*100;
      statusEl.textContent="尚未進場";
      statusEl.className="statusWarn";
      distEl.textContent=`高於上緣 ${Math.abs(pct).toFixed(2)}%`;
      hint.textContent="價格尚未回到建議做空進場區，等待條件成立。";
      hint.className="tradeHint warn";
    }
  }
}

function clearTradeLines(){
  if(!candleSeries) return;
  tradePriceLines.forEach(line=>{
    try{ candleSeries.removePriceLine(line); }catch(e){}
  });
  tradePriceLines=[];
}

function addTradeLines(d){
  clearTradeLines();
  if(!d || !candleSeries) return;

  const s=getIndicatorState();

  function add(price,title,color,style=2){
    if(price===null || price===undefined) return;
    const line=candleSeries.createPriceLine({
      price:Number(price),
      color:color,
      lineWidth:1,
      lineStyle:style,
      axisLabelVisible:true,
      title:title
    });
    tradePriceLines.push(line);
  }

  if(s.entry && d.entry_low && d.entry_high){
    add(d.entry_low,"Entry L","#f2c96b",2);
    add(d.entry_high,"Entry H","#f2c96b",2);
  }
  if(s.stop) add(d.stop,"STOP","#ff666e",0);
  if(s.tp1) add(d.tp1,"TP1","#47d18c",0);
  if(s.tp2) add(d.tp2,"TP2","#29b6f6",0);
}

function setLineSeriesVisible(){
  const s=getIndicatorState();
  Object.entries(indicatorSeries).forEach(([key,series])=>{
    const visible = key==="bb_upper" || key==="bb_mid" || key==="bb_lower" ? s.bb : !!s[key];
    try{series.applyOptions({visible});}catch(e){}
  });
  addTradeLines(latestAnalysis);
  syncIndicatorButtons();
}

async function loadCharts(){
  const sym=document.getElementById("symbol").value;
  const interval=document.getElementById("chartInterval").value;

  try{
    const r=await fetch(`/api/chart/${sym}?interval=${interval}`);
    const d=await r.json();
    if(!r.ok) throw new Error(d.error||"圖表讀取失敗");

    // Main candlestick chart
    const el=document.getElementById("candleChart");
    if(candleChart){
      candleChart.remove();
      candleChart=null;
    }

    candleChart=LightweightCharts.createChart(el,{
      width:el.clientWidth,
      height:390,
      layout:{background:{color:"#0d0f14"},textColor:"#9298a8"},
      grid:{vertLines:{color:"#20232c"},horzLines:{color:"#20232c"}},
      rightPriceScale:{borderColor:"#252936"},
      timeScale:{borderColor:"#252936",timeVisible:true,secondsVisible:false},
      crosshair:{mode:LightweightCharts.CrosshairMode.Normal}
    });

    candleSeries=candleChart.addCandlestickSeries({
      upColor:"#47d18c",
      downColor:"#ff666e",
      borderVisible:false,
      wickUpColor:"#47d18c",
      wickDownColor:"#ff666e"
    });

    indicatorSeries={};
    indicatorSeries.ema5=candleChart.addLineSeries({color:"#b388ff",lineWidth:1,priceLineVisible:false,lastValueVisible:false});
    indicatorSeries.ema10=candleChart.addLineSeries({color:"#42a5f5",lineWidth:1,priceLineVisible:false,lastValueVisible:false});
    indicatorSeries.ema20=candleChart.addLineSeries({color:"#f2c96b",lineWidth:2,priceLineVisible:false,lastValueVisible:false});
    indicatorSeries.ema50=candleChart.addLineSeries({color:"#47d18c",lineWidth:2,priceLineVisible:false,lastValueVisible:false});
    indicatorSeries.ema100=candleChart.addLineSeries({color:"#ff9f43",lineWidth:1,priceLineVisible:false,lastValueVisible:false});
    indicatorSeries.ema200=candleChart.addLineSeries({color:"#ff666e",lineWidth:2,priceLineVisible:false,lastValueVisible:false});
    indicatorSeries.vwap=candleChart.addLineSeries({color:"#26c6da",lineWidth:2,priceLineVisible:false,lastValueVisible:false});
    indicatorSeries.bb_upper=candleChart.addLineSeries({color:"#8d95a5",lineWidth:1,priceLineVisible:false,lastValueVisible:false,lineStyle:2});
    indicatorSeries.bb_mid=candleChart.addLineSeries({color:"#606775",lineWidth:1,priceLineVisible:false,lastValueVisible:false,lineStyle:2});
    indicatorSeries.bb_lower=candleChart.addLineSeries({color:"#8d95a5",lineWidth:1,priceLineVisible:false,lastValueVisible:false,lineStyle:2});

    candleSeries.setData(d.candles);
    indicatorSeries.ema5.setData(d.ema5_points);
    indicatorSeries.ema10.setData(d.ema10_points);
    indicatorSeries.ema20.setData(d.ema20_points);
    indicatorSeries.ema50.setData(d.ema50_points);
    indicatorSeries.ema100.setData(d.ema100_points);
    indicatorSeries.ema200.setData(d.ema200_points);
    indicatorSeries.vwap.setData(d.vwap_points);
    indicatorSeries.bb_upper.setData(d.bb_upper_points);
    indicatorSeries.bb_mid.setData(d.bb_mid_points);
    indicatorSeries.bb_lower.setData(d.bb_lower_points);

    latestChartData=d;
    setLineSeriesVisible();
    candleChart.timeScale().fitContent();

    // MACD
    if(macdChart) macdChart.destroy();
    macdChart=new Chart(document.getElementById("macdChart"),{
      data:{
        labels:d.labels,
        datasets:[
          {type:"line",label:"MACD",data:d.macd,borderColor:"#ffffff",borderWidth:1.8,pointRadius:0,tension:0.15},
          {type:"line",label:"Signal",data:d.macd_signal,borderColor:"#f2c96b",borderWidth:1.5,pointRadius:0,tension:0.15},
          {type:"bar",label:"Hist",data:d.macd_hist,backgroundColor:d.macd_hist.map(v=>v>=0?"rgba(71,209,140,.55)":"rgba(255,102,110,.55)")}
        ]
      },
      options:chartBaseOptions()
    });

    // Volume
    if(volumeChart) volumeChart.destroy();
    volumeChart=new Chart(document.getElementById("volumeChart"),{
      type:"bar",
      data:{
        labels:d.labels,
        datasets:[
          {label:"Volume",data:d.volume,backgroundColor:d.volume_colors}
        ]
      },
      options:chartBaseOptions()
    });

    window.addEventListener("resize",()=>{
      if(candleChart){
        candleChart.applyOptions({width:el.clientWidth});
      }
    },{once:true});

  }catch(e){
    console.log("chart error",e);
  }
}


async function loadContext(){
  const sym=document.getElementById("symbol").value;
  try{
    const r=await fetch(`/api/context/${sym}`);
    const d=await r.json();
    if(!r.ok) throw new Error(d.error||"context failed");
    const c=d.market_context||{};
    const der=d.derivatives||{};
    document.getElementById("btcBias").textContent=c.btc_bias||"--";
    document.getElementById("marketRegime").textContent=c.regime||"--";
    document.getElementById("marketBreadth").textContent=`${c.long_count||0}多 / ${c.short_count||0}空 / ${c.neutral_count||0}中`;
    document.getElementById("marketVolatility").textContent=c.volatility||"--";
    document.getElementById("fundingData").textContent=der.funding_pct===null||der.funding_pct===undefined?"--":`${der.funding_pct}%`;
    document.getElementById("fundingLabel").textContent=der.funding_label||"";
    document.getElementById("oiData").textContent=der.oi_change_pct===null||der.oi_change_pct===undefined?"--":`${der.oi_change_pct}%`;
    document.getElementById("oiLabel").textContent=der.oi_label||"";
  }catch(e){console.log(e);}
}

async function loadAnalysis(){
  const s=getSettings();
  const sym=document.getElementById("symbol").value;
  const url=`/api/analysis/${sym}?mode=${encodeURIComponent(s.mode)}&strictness=${encodeURIComponent(s.strictness)}&min_rr=${encodeURIComponent(s.minRR)}`;

  try{
    const r=await fetch(url);
    const d=await r.json();
    if(!r.ok) throw new Error(d.error||"分析失敗");

    let shownAction=d.action;
    if(s.aOnly && d.grade!=="A") shownAction="WAIT";

    const a=document.getElementById("action");
    a.textContent=shownAction;
    a.className="action "+(shownAction==="LONG"?"long":shownAction==="SHORT"?"short":"wait");

    document.getElementById("grade").textContent="訊號 "+d.grade;
    document.getElementById("score").textContent=d.score;
    document.getElementById("scorebar").style.width=Math.max(0,Math.min(100,d.confidence))+"%";
    document.getElementById("rr").textContent=d.rr?"1 : "+d.rr:"--";
    document.getElementById("modeLabel").textContent=d.mode_label;
    document.getElementById("holding").textContent=d.holding_time;
    document.getElementById("entry").textContent=d.entry_low?fmt(d.entry_low)+" ~ "+fmt(d.entry_high):"等待";
    document.getElementById("stop").textContent=fmt(d.stop);
    document.getElementById("tp1").textContent=fmt(d.tp1);
    document.getElementById("tp2").textContent=fmt(d.tp2);
    updateEntryStatus(d);
    addTradeLines(d);

    const tf=document.getElementById("timeframes"); tf.innerHTML="";
    for(const [name,x] of Object.entries(d.timeframes)){
      let dir=x.score>1.8?"強多":x.score>0.7?"偏多":x.score<-1.8?"強空":x.score<-0.7?"偏空":"中性";
      let cls=(dir==="強多"||dir==="偏多")?"long":(dir==="強空"||dir==="偏空")?"short":"wait";
      tf.innerHTML+=`<div class="tfbox"><div class="label">${name}</div><div class="${cls}" style="font-size:17px;font-weight:800;margin-top:4px">${dir}</div><div class="muted">Score ${x.score}<br>RSI ${x.rsi}<br>量 ${x.volume_ratio}x</div></div>`;
    }

    document.getElementById("warnings").innerHTML=(d.warnings||[]).map(x=>"⚠ "+x).join("<br>")||"目前無額外警示";

    if(s.notify && shownAction!==lastAction){
      if(!s.aOnly || d.grade==="A"){
        notify(`Crypto Signal ${sym}`, `訊號變化：${shownAction}｜等級 ${d.grade}`);
      }
    }
    lastAction=shownAction;
    lastGrade=d.grade;
  }catch(e){
    document.getElementById("warnings").textContent="分析更新失敗："+e.message;
  }
}
async function loadNews(){
  try{
    const r=await fetch("/api/news");
    const items=await r.json();
    const box=document.getElementById("newsbox"); box.innerHTML="";
    items.slice(0,30).forEach(n=>{
      const tag=n.high_impact?`<span class="tag">高影響</span>`:"";
      const original=n.title_original&&n.title_original!==n.title?`<div class="muted" style="margin-top:5px;font-size:11px;line-height:1.35">原文：${n.title_original}</div>`:"";
      box.innerHTML+=`<div class="newsitem"><a href="${n.link}" target="_blank">${n.title}</a>${tag}${original}<div class="muted" style="margin-top:6px">${n.source} ${n.published||""}</div></div>`;
    });
  }catch(e){}
}

function fmtMoneyCompact(n){const x=Number(n||0);if(x>=1e9)return(x/1e9).toFixed(1)+"B";if(x>=1e6)return(x/1e6).toFixed(1)+"M";return x.toLocaleString();}
function ensureSymbolOption(symbol){const s=document.getElementById("symbol");if(![...s.options].some(o=>o.value===symbol)){const o=document.createElement("option");o.value=symbol;o.textContent=symbol;s.appendChild(o);}}
function ensureCoinButton(symbol){const p=document.getElementById("coinPicker");if([...p.querySelectorAll(".coinBtn")].some(b=>b.dataset.symbol===symbol))return;const b=document.createElement("button");b.className="coinBtn";b.dataset.symbol=symbol;b.textContent=symbol.replace("USDT","");b.addEventListener("click",()=>switchToSymbol(symbol));p.appendChild(b);}
function switchToSymbol(symbol){ensureSymbolOption(symbol);ensureCoinButton(symbol);document.getElementById("symbol").value=symbol;syncCoinButtons();connectWS();loadAnalysis();loadCharts();loadContext();document.querySelectorAll(".nav").forEach(x=>x.classList.remove("active"));document.querySelector('.nav[data-page="home"]').classList.add("active");document.querySelectorAll(".page").forEach(x=>x.classList.remove("active"));document.getElementById("page-home").classList.add("active");window.scrollTo({top:0,behavior:"smooth"});}
async function scanMarket(){const btn=document.getElementById("scanMarketBtn"),list=document.getElementById("radarList"),summary=document.getElementById("radarSummary"),s=getSettings(),direction=document.getElementById("radarDirection").value,count=document.getElementById("radarCount").value;btn.disabled=true;btn.textContent="掃描中…";list.innerHTML='<div class="radarEmpty">正在分析 15m / 1H / 4H，請稍候…</div>';summary.innerHTML="";try{const url=`/api/radar?mode=${encodeURIComponent(s.mode)}&strictness=${encodeURIComponent(s.strictness)}&min_rr=${encodeURIComponent(s.minRR)}&direction=${direction}&count=${count}`;const r=await fetch(url),d=await r.json();if(!r.ok)throw new Error(d.error||"掃描失敗");const mc=d.market_context||{};summary.innerHTML=`<div class="card"><div class="label">市場環境</div><div class="big">BTC ${mc.btc_bias||"--"} · ${mc.regime||"--"}</div><div class="muted">${mc.breadth_label||""}｜${mc.long_count||0}多 / ${mc.short_count||0}空 / ${mc.neutral_count||0}中｜${mc.volatility||""}</div></div>`;if(!d.results||!d.results.length){list.innerHTML='<div class="radarEmpty">目前沒有符合門檻的 LONG / SHORT。</div>';return;}const medals=["🥇","🥈","🥉"];list.innerHTML=d.results.map((x,i)=>{const cls=x.action==="LONG"?"long":"short",stateCls=x.state.key==="in_zone"?"statusGood":x.state.key==="missed"?"statusBad":"statusWarn",dist=x.state.distance_pct===null?"":` · ${Number(x.state.distance_pct).toFixed(2)}%`;return `<div class="radarCard"><div class="radarTop"><div><div class="radarSymbol"><span class="rank">${medals[i]||"#"+(i+1)}</span>${x.symbol}</div><div class="${cls}" style="font-weight:850;margin-top:5px">${x.action} · ${x.grade}級</div></div><div style="text-align:right"><div class="label">機會分數</div><div class="radarScore">${x.opportunity}</div></div></div><div class="radarMeta"><span class="radarChip">風報 1:${x.rr||"--"}</span><span class="radarChip">${x.mode_label}</span><span class="radarChip">24H量 ${fmtMoneyCompact(x.quote_volume)}</span><span class="radarChip">${x.market_fit}</span><span class="radarChip">Funding ${x.derivatives&&x.derivatives.funding_pct!=null?x.derivatives.funding_pct+"%":"--"}</span><span class="radarChip">OI ${x.derivatives&&x.derivatives.oi_change_pct!=null?x.derivatives.oi_change_pct+"%":"--"}</span></div><div class="${stateCls}" style="font-weight:750;margin-top:10px">${x.state.label}${dist}</div><button class="radarOpen" onclick="switchToSymbol('${x.symbol}')">查看 ${x.symbol.replace("USDT","")} K 線</button></div>`;}).join("");}catch(e){list.innerHTML=`<div class="radarEmpty">掃描失敗：${e.message}</div>`;}finally{btn.disabled=false;btn.textContent="⚡ 重新掃描市場";}}
document.getElementById("scanMarketBtn").addEventListener("click",scanMarket);

function requestNotify(){
  if("Notification" in window) Notification.requestPermission();
}
function notify(title,body){
  if("Notification" in window && Notification.permission==="granted"){
    new Notification(title,{body});
  }
}

document.querySelectorAll(".nav").forEach(btn=>btn.addEventListener("click",()=>{
  document.querySelectorAll(".nav").forEach(x=>x.classList.remove("active"));
  btn.classList.add("active");
  document.querySelectorAll(".page").forEach(x=>x.classList.remove("active"));
  document.getElementById("page-"+btn.dataset.page).classList.add("active");
}));

["mode","strictness","minRR","partial"].forEach(id=>{
  document.getElementById(id).addEventListener("change",saveSettings);
});
document.getElementById("notifyToggle").addEventListener("click",function(){this.classList.toggle("on");saveSettings();});
document.getElementById("aOnlyToggle").addEventListener("click",function(){this.classList.toggle("on");saveSettings();});

document.querySelectorAll("[data-ind]").forEach(btn=>{
  btn.addEventListener("click",()=>{
    const s=getIndicatorState();
    s[btn.dataset.ind]=!s[btn.dataset.ind];
    saveIndicatorState(s);
    setLineSeriesVisible();
  });
});

document.getElementById("toggleMacd").addEventListener("click",()=>{
  const s=getIndicatorState();
  s.macd=!s.macd;
  saveIndicatorState(s);
  syncIndicatorButtons();
});
document.getElementById("toggleVolume").addEventListener("click",()=>{
  const s=getIndicatorState();
  s.volume=!s.volume;
  saveIndicatorState(s);
  syncIndicatorButtons();
});

document.getElementById("onlyPriceBtn").addEventListener("click",()=>{
  const s=getIndicatorState();
  ["ema5","ema10","ema20","ema50","ema100","ema200","vwap","bb","entry","stop","tp1","tp2"].forEach(k=>s[k]=false);
  saveIndicatorState(s);
  setLineSeriesVisible();
});

document.getElementById("resetIndicatorsBtn").addEventListener("click",()=>{
  saveIndicatorState(defaultIndicatorState);
  setLineSeriesVisible();
});

function syncCoinButtons(){
  const current=document.getElementById("symbol").value;
  document.querySelectorAll(".coinBtn").forEach(btn=>btn.classList.toggle("active",btn.dataset.symbol===current));
}

document.querySelectorAll(".coinBtn").forEach(btn=>{btn.addEventListener("click",()=>switchToSymbol(btn.dataset.symbol));});

document.getElementById("symbol").addEventListener("change",()=>{connectWS();loadAnalysis();loadCharts();loadContext();});
document.getElementById("chartInterval").addEventListener("change",loadCharts);

applySettings();
syncIndicatorButtons();
syncCoinButtons();
connectWS();
loadAnalysis();
loadCharts();
loadContext();
loadNews();

setInterval(loadAnalysis,60000);
setInterval(loadNews,30000);
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/api/analysis/<symbol>")
def analysis(symbol):
    symbol = symbol.upper()
    if not valid_symbol(symbol):
        return jsonify({"error":"unsupported symbol"}), 400

    mode = request.args.get("mode", "day")
    strictness = request.args.get("strictness", "normal")

    try:
        min_rr = float(request.args.get("min_rr", "2.0"))
    except Exception:
        min_rr = 2.0

    try:
        return jsonify(build_trade_plan(symbol, mode, strictness, min_rr))
    except Exception as e:
        return jsonify({"error":str(e)}), 500



@app.route("/api/context/<symbol>")
def context_api(symbol):
    symbol = symbol.upper()
    if not valid_symbol(symbol):
        return jsonify({"error":"unsupported symbol"}), 400
    try:
        return jsonify({
            "market_context": market_context(16),
            "derivatives": get_derivatives_snapshot(symbol),
        })
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/radar")
def radar():
    mode = request.args.get("mode", "day")
    strictness = request.args.get("strictness", "normal")
    direction = request.args.get("direction", "all")
    if direction not in ("all","long","short"):
        direction = "all"
    try:
        min_rr = float(request.args.get("min_rr","2.0"))
    except Exception:
        min_rr = 2.0
    try:
        count = int(request.args.get("count","16"))
    except Exception:
        count = 16
    count = max(8, min(count, 20))
    try:
        return jsonify(scan_market(mode, strictness, min_rr, direction, count))
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/chart/<symbol>")
def chart_data(symbol):
    symbol = symbol.upper()
    if not valid_symbol(symbol):
        return jsonify({"error":"unsupported symbol"}), 400

    interval = request.args.get("interval","1h")
    if interval not in ["15m","1h","4h"]:
        interval = "1h"

    try:
        df = fetch_klines(symbol, interval, limit=220).tail(120).copy()

        labels = []
        candles = []
        ema5_points = []
        ema10_points = []
        ema20_points = []
        ema50_points = []
        ema100_points = []
        ema200_points = []
        vwap_points = []
        bb_upper_points = []
        bb_mid_points = []
        bb_lower_points = []
        volume_colors = []

        for _, row in df.iterrows():
            ts = int(int(row["open_time"]) / 1000)
            label = pd.to_datetime(int(row["open_time"]), unit="ms", utc=True).tz_convert("Asia/Taipei").strftime("%m/%d %H:%M")
            labels.append(label)

            candles.append({
                "time": ts,
                "open": round(float(row["open"]), 6),
                "high": round(float(row["high"]), 6),
                "low": round(float(row["low"]), 6),
                "close": round(float(row["close"]), 6),
            })

            if pd.notna(row["ema5"]):
                ema5_points.append({"time":ts,"value":round(float(row["ema5"]),6)})
            if pd.notna(row["ema10"]):
                ema10_points.append({"time":ts,"value":round(float(row["ema10"]),6)})
            if pd.notna(row["ema20"]):
                ema20_points.append({"time":ts,"value":round(float(row["ema20"]),6)})
            if pd.notna(row["ema50"]):
                ema50_points.append({"time":ts,"value":round(float(row["ema50"]),6)})
            if pd.notna(row["ema100"]):
                ema100_points.append({"time":ts,"value":round(float(row["ema100"]),6)})
            if pd.notna(row["ema200"]):
                ema200_points.append({"time":ts,"value":round(float(row["ema200"]),6)})
            if pd.notna(row["vwap"]):
                vwap_points.append({"time":ts,"value":round(float(row["vwap"]),6)})
            if pd.notna(row["bb_upper"]):
                bb_upper_points.append({"time":ts,"value":round(float(row["bb_upper"]),6)})
            if pd.notna(row["bb_mid"]):
                bb_mid_points.append({"time":ts,"value":round(float(row["bb_mid"]),6)})
            if pd.notna(row["bb_lower"]):
                bb_lower_points.append({"time":ts,"value":round(float(row["bb_lower"]),6)})

            volume_colors.append(
                "rgba(71,209,140,.55)" if float(row["close"]) >= float(row["open"])
                else "rgba(255,102,110,.55)"
            )

        def vals(col, digits=4):
            out=[]
            for x in df[col]:
                if pd.isna(x):
                    out.append(None)
                else:
                    out.append(round(float(x), digits))
            return out

        return jsonify({
            "symbol":symbol,
            "interval":interval,
            "labels":labels,
            "candles":candles,
            "ema5_points":ema5_points,
            "ema10_points":ema10_points,
            "ema20_points":ema20_points,
            "ema50_points":ema50_points,
            "ema100_points":ema100_points,
            "ema200_points":ema200_points,
            "vwap_points":vwap_points,
            "bb_upper_points":bb_upper_points,
            "bb_mid_points":bb_mid_points,
            "bb_lower_points":bb_lower_points,
            "macd":vals("macd",4),
            "macd_signal":vals("macd_signal",4),
            "macd_hist":vals("macd_hist",4),
            "volume":vals("volume",2),
            "volume_colors":volume_colors,
        })
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/news")
def news():
    return jsonify(news_items())

@app.route("/api/health")
def health():
    return jsonify({"ok":True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
