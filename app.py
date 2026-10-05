import os
import time
import threading
from datetime import datetime, timezone

import requests
import feedparser
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template_string
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

BINANCE_REST = "https://data-api.binance.vision"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

NEWS_FEEDS = [
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
]

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
ENABLE_TELEGRAM = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

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
        df["high"]-df["low"],
        (df["high"]-prev_close).abs(),
        (df["low"]-prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def add_indicators(df):
    x = df.copy()
    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)
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
    return x

def fetch_klines(symbol, interval, limit=500):
    r = requests.get(
        BINANCE_REST + "/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=10
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
    reasons = []

    if row["ema20"] > row["ema50"] > row["ema200"]:
        score += 2.0
        reasons.append("EMA 多頭排列")
    elif row["ema20"] < row["ema50"] < row["ema200"]:
        score -= 2.0
        reasons.append("EMA 空頭排列")
    else:
        if row["ema20"] > row["ema50"]:
            score += 0.7
            reasons.append("EMA20 > EMA50")
        else:
            score -= 0.7
            reasons.append("EMA20 < EMA50")
        if row["close"] > row["ema200"]:
            score += 0.5
            reasons.append("價格在 EMA200 上方")
        else:
            score -= 0.5
            reasons.append("價格在 EMA200 下方")

    if 55 <= row["rsi"] <= 70:
        score += 0.8
        reasons.append("RSI 偏強")
    elif 30 <= row["rsi"] <= 45:
        score -= 0.8
        reasons.append("RSI 偏弱")
    elif row["rsi"] > 75:
        reasons.append("RSI 過熱")
    elif row["rsi"] < 25:
        reasons.append("RSI 超賣")

    if row["macd_hist"] > 0:
        score += 0.7
        reasons.append("MACD 動能偏多")
    else:
        score -= 0.7
        reasons.append("MACD 動能偏空")

    if row["ret5"] > 0:
        score += 0.5
        reasons.append("近 5 根 K 動能向上")
    else:
        score -= 0.5
        reasons.append("近 5 根 K 動能向下")

    if pd.notna(row["volume_ratio"]) and row["volume_ratio"] > 1.25:
        if row["ret5"] > 0:
            score += 0.5
            reasons.append("量價偏多")
        else:
            score -= 0.5
            reasons.append("量價偏空")

    return {
        "score": round(float(score), 2),
        "rsi": round(float(row["rsi"]), 2),
        "volume_ratio": round(float(row["volume_ratio"]) if pd.notna(row["volume_ratio"]) else 1.0, 2),
        "atr": float(row["atr"]) if pd.notna(row["atr"]) else 0.0,
        "support": float(df.tail(80)["low"].min()),
        "resistance": float(df.tail(80)["high"].max()),
        "close": float(row["close"]),
        "reasons": reasons
    }

def build_trade_plan(symbol):
    t15 = timeframe_signal(fetch_klines(symbol, "15m"))
    t1h = timeframe_signal(fetch_klines(symbol, "1h"))
    t4h = timeframe_signal(fetch_klines(symbol, "4h"))

    weighted = t15["score"]*0.20 + t1h["score"]*0.35 + t4h["score"]*0.45
    price = t15["close"]
    atr1h = max(t1h["atr"], price*0.002)

    align_long = t1h["score"] > 0 and t4h["score"] > 0
    align_short = t1h["score"] < 0 and t4h["score"] < 0

    if weighted >= 1.6 and align_long:
        action = "LONG"
        entry_low = price - 0.35*atr1h
        entry_high = price - 0.08*atr1h
        entry_mid = (entry_low+entry_high)/2
        stop = min(t1h["support"] - 0.15*atr1h, entry_mid - 1.10*atr1h)
        risk = max(entry_mid-stop, atr1h*0.6)
        tp1 = entry_mid + 1.5*risk
        tp2 = entry_mid + 2.5*risk
        rr = (tp2-entry_mid)/risk
    elif weighted <= -1.6 and align_short:
        action = "SHORT"
        entry_low = price + 0.08*atr1h
        entry_high = price + 0.35*atr1h
        entry_mid = (entry_low+entry_high)/2
        stop = max(t1h["resistance"] + 0.15*atr1h, entry_mid + 1.10*atr1h)
        risk = max(stop-entry_mid, atr1h*0.6)
        tp1 = entry_mid - 1.5*risk
        tp2 = entry_mid - 2.5*risk
        rr = (entry_mid-tp2)/risk
    else:
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
        warnings.append("1H 量能偏低")
    if action == "WAIT":
        warnings.append("多週期尚未形成一致方向")

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
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

HIGH_IMPACT_WORDS = [
    "fed","federal reserve","cpi","inflation","interest rate","sec",
    "etf","hack","exploit","liquidation","tariff","war","regulation"
]

def news_items(limit=50):
    out = []
    seen = set()
    for source, url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:30]:
                title = getattr(e, "title", "").strip()
                if not title or title in seen:
                    continue
                seen.add(title)
                link = getattr(e, "link", "").strip()
                published = getattr(e, "published", "") or getattr(e, "updated", "")
                low = title.lower()
                out.append({
                    "source": source,
                    "title": title,
                    "link": link,
                    "published": published,
                    "high_impact": any(w in low for w in HIGH_IMPACT_WORDS)
                })
        except Exception:
            pass
    return out[:limit]

HTML = r'''<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0b0c10">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Crypto Signal">

<title>Crypto Signal Pro</title>
<style>
:root{
  --bg:#090a0d; --card:#12141a; --card2:#171a22; --line:#262a36;
  --text:#f6f7f9; --muted:#9aa0ad; --gold:#d9b45b; --green:#4cd38d;
  --red:#ff666e; --yellow:#f0c867;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
body{padding-bottom:calc(78px + env(safe-area-inset-bottom));}
.app{max-width:760px;margin:auto;min-height:100vh;background:linear-gradient(180deg,#090a0d,#0d0f14);}
.header{position:sticky;top:0;z-index:20;background:rgba(9,10,13,.92);backdrop-filter:blur(16px);padding:calc(12px + env(safe-area-inset-top)) 16px 12px;border-bottom:1px solid #171922}
.headrow{display:flex;justify-content:space-between;gap:10px;align-items:center}
.title{font-size:22px;font-weight:800}.muted{color:var(--muted);font-size:12px}
select,.btn{background:#161920;color:#fff;border:1px solid var(--line);border-radius:12px;padding:10px 12px;font-size:14px}
.main{padding:14px}
.hero{background:linear-gradient(145deg,#171920,#101217);border:1px solid var(--line);border-radius:22px;padding:18px}
.hero-top{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}
.price{font-size:34px;font-weight:850;letter-spacing:-.6px;margin-top:5px}
.action{font-size:23px;font-weight:850}.long{color:var(--green)}.short{color:var(--red)}.wait{color:var(--yellow)}
.badge{display:inline-flex;padding:5px 9px;border-radius:999px;background:#222631;font-size:12px;margin-top:6px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:14px}
.label{font-size:12px;color:var(--muted)}.big{font-size:21px;font-weight:800;margin-top:6px}
.section{margin-top:12px}
.section-title{font-size:15px;font-weight:750;margin:0 0 9px 2px}
.plan-row{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #20232c;font-size:14px}
.plan-row:last-child{border-bottom:none}
.tfrow{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.tfbox{background:#111318;border:1px solid var(--line);border-radius:14px;padding:11px}
.tfdir{font-size:17px;font-weight:800;margin-top:4px}
.news{display:flex;flex-direction:column;gap:8px}
.newsitem{background:#111318;border:1px solid var(--line);border-radius:14px;padding:12px}
.newsitem a{color:#fff;text-decoration:none;font-size:14px;line-height:1.45}
.tag{font-size:10px;padding:3px 6px;border-radius:7px;background:#272b36;color:#c9ced8;margin-left:6px}
.danger{background:#472127;color:#ffb3b7}
.bottom{position:fixed;left:50%;transform:translateX(-50%);bottom:0;z-index:30;width:100%;max-width:760px;background:rgba(14,16,21,.95);backdrop-filter:blur(18px);border-top:1px solid var(--line);padding:8px 8px calc(8px + env(safe-area-inset-bottom));display:grid;grid-template-columns:repeat(4,1fr)}
.nav{border:none;background:transparent;color:#9097a7;font-size:11px;padding:6px 2px}.nav.active{color:var(--gold)}.ico{display:block;font-size:20px;margin-bottom:2px}
.page{display:none}.page.active{display:block}
.alert{font-size:13px;color:#d7d9df;line-height:1.55}.warn{color:#f2c96b}
.bar{height:9px;background:#242832;border-radius:999px;overflow:hidden;margin-top:8px}.barin{height:100%;width:0;background:linear-gradient(90deg,#947334,#e1c46d)}
.install{margin-top:10px;width:100%}
@media(min-width:640px){.main{padding:20px}.hero{padding:22px}}
</style>
</head>
<body>
<div class="app">
  <header class="header">
    <div class="headrow">
      <div>
        <div class="title">Crypto Signal Pro</div>
        <div class="muted">即時價格 · 多週期 · 新聞風險</div>
      </div>
      <select id="symbol">
        <option>BTCUSDT</option><option>ETHUSDT</option><option>SOLUSDT</option><option>BNBUSDT</option><option>XRPUSDT</option>
      </select>
    </div>
  </header>

  <main class="main">
    <section class="page active" id="page-home">
      <div class="hero">
        <div class="hero-top">
          <div>
            <div class="label">即時價格</div>
            <div class="price" id="price">--</div>
            <div class="muted" id="tick">WebSocket 連線中</div>
          </div>
          <div style="text-align:right">
            <div class="label">目前建議</div>
            <div class="action wait" id="action">--</div>
            <div class="badge" id="grade">訊號 --</div>
          </div>
        </div>
        <div class="grid2">
          <div class="card"><div class="label">綜合分數</div><div class="big" id="score">--</div><div class="bar"><div class="barin" id="scorebar"></div></div></div>
          <div class="card"><div class="label">風報比</div><div class="big" id="rr">--</div><div class="muted">以 TP2 / Stop 估算</div></div>
        </div>
      </div>

      <div class="section">
        <div class="section-title">交易規劃</div>
        <div class="card">
          <div class="plan-row"><span>進場區</span><strong id="entry">等待</strong></div>
          <div class="plan-row"><span>止損</span><strong id="stop">--</strong></div>
          <div class="plan-row"><span>TP1</span><strong id="tp1">--</strong></div>
          <div class="plan-row"><span>TP2</span><strong id="tp2">--</strong></div>
        </div>
      </div>

      <div class="section">
        <div class="section-title">多週期</div>
        <div class="tfrow" id="timeframes"></div>
      </div>

      <div class="section">
        <div class="section-title">風險提醒</div>
        <div class="card alert" id="warnings">讀取中...</div>
      </div>
    </section>

    <section class="page" id="page-news">
      <div class="section-title">最新新聞</div>
      <div class="news" id="news"><div class="card muted">讀取中...</div></div>
    </section>

    <section class="page" id="page-alerts">
      <div class="section-title">通知</div>
      <div class="card">
        <div class="alert">
          A 級訊號出現時，可用瀏覽器通知提醒。<br><br>
          iPhone 若加入主畫面後使用，體驗會更接近 App。
        </div>
        <button class="btn install" onclick="requestNotify()">開啟通知權限</button>
      </div>
      <div class="card section">
        <div class="label">高影響新聞</div>
        <div class="alert" style="margin-top:8px">
          ETF、SEC、Fed、CPI、清算、駭客事件等會被標記為高影響。
        </div>
      </div>
    </section>

    <section class="page" id="page-settings">
      <div class="section-title">設定</div>
      <div class="card">
        <div class="plan-row"><span>行情來源</span><strong>Binance</strong></div>
        <div class="plan-row"><span>分析週期</span><strong>15m / 1H / 4H</strong></div>
        <div class="plan-row"><span>新聞來源</span><strong>CoinDesk / Cointelegraph</strong></div>
      </div>
      <div class="card section">
        <div class="alert warn">
          此工具是交易決策輔助，不代表保證獲利。高波動行情仍可能快速穿越止損或改變結構。
        </div>
      </div>
    </section>
  </main>

  <nav class="bottom">
    <button class="nav active" data-page="home"><span class="ico">⌂</span>首頁</button>
    <button class="nav" data-page="news"><span class="ico">📰</span>新聞</button>
    <button class="nav" data-page="alerts"><span class="ico">🔔</span>通知</button>
    <button class="nav" data-page="settings"><span class="ico">⚙️</span>設定</button>
  </nav>
</div>

<script>
let ws=null,lastAction="",lastGrade="",currentSymbol="BTCUSDT";

function fmt(n){
  if(n===null || n===undefined) return "--";
  let x=Number(n), digits=x>=1000?2:(x>=1?4:6);
  return x.toLocaleString(undefined,{maximumFractionDigits:digits});
}
function dirClass(score){
  if(score>0.7) return ["偏多","long"];
  if(score<-0.7) return ["偏空","short"];
  return ["中性","wait"];
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
  ws.onerror=()=>document.getElementById("tick").textContent="串流異常";
  ws.onclose=()=>setTimeout(()=>{if(currentSymbol===document.getElementById("symbol").value)connectWS()},2500);
}
async function loadAnalysis(){
  const sym=document.getElementById("symbol").value;
  try{
    const r=await fetch("/api/analysis/"+sym);
    const d=await r.json();
    if(!r.ok) throw new Error(d.error||"分析失敗");

    const a=document.getElementById("action");
    a.textContent=d.action;
    a.className="action "+(d.action==="LONG"?"long":d.action==="SHORT"?"short":"wait");
    document.getElementById("grade").textContent="訊號 "+d.grade;
    document.getElementById("score").textContent=d.score;
    document.getElementById("scorebar").style.width=Math.max(0,Math.min(100,d.confidence))+"%";
    document.getElementById("rr").textContent=d.rr ? "1 : "+d.rr : "--";
    document.getElementById("entry").textContent=d.entry_low?fmt(d.entry_low)+" ~ "+fmt(d.entry_high):"等待";
    document.getElementById("stop").textContent=fmt(d.stop);
    document.getElementById("tp1").textContent=fmt(d.tp1);
    document.getElementById("tp2").textContent=fmt(d.tp2);

    const tf=document.getElementById("timeframes"); tf.innerHTML="";
    for(const [name,x] of Object.entries(d.timeframes)){
      const [dir,cls]=dirClass(x.score);
      tf.innerHTML+=`<div class="tfbox"><div class="label">${name}</div><div class="tfdir ${cls}">${dir}</div><div class="muted">Score ${x.score}<br>RSI ${x.rsi}<br>量 ${x.volume_ratio}x</div></div>`;
    }
    document.getElementById("warnings").innerHTML=(d.warnings||[]).map(x=>"⚠ "+x).join("<br>") || "目前無額外警示";

    if(d.grade==="A" && d.action!=="WAIT" && (d.action!==lastAction || d.grade!==lastGrade)){
      notify(`A級 ${sym} ${d.action}`, `分數 ${d.score}｜進場 ${fmt(d.entry_low)} ~ ${fmt(d.entry_high)}`);
    }
    lastAction=d.action; lastGrade=d.grade;
  }catch(e){
    document.getElementById("warnings").textContent="分析更新失敗："+e.message;
  }
}
async function loadNews(){
  try{
    const r=await fetch("/api/news");
    const items=await r.json();
    const box=document.getElementById("news"); box.innerHTML="";
    items.slice(0,40).forEach(n=>{
      const tag=n.high_impact?`<span class="tag danger">高影響</span>`:"";
      box.innerHTML+=`<div class="newsitem"><a href="${n.link}" target="_blank">${n.title}</a>${tag}<div class="muted" style="margin-top:6px">${n.source} ${n.published||""}</div></div>`;
    });
  }catch(e){}
}
function requestNotify(){ if("Notification" in window) Notification.requestPermission(); }
function notify(title,body){ if("Notification" in window && Notification.permission==="granted") new Notification(title,{body}); }

document.getElementById("symbol").addEventListener("change",()=>{connectWS();loadAnalysis();});
document.querySelectorAll(".nav").forEach(btn=>btn.addEventListener("click",()=>{
  document.querySelectorAll(".nav").forEach(x=>x.classList.remove("active"));
  btn.classList.add("active");
  document.querySelectorAll(".page").forEach(x=>x.classList.remove("active"));
  document.getElementById("page-"+btn.dataset.page).classList.add("active");
}));



connectWS(); loadAnalysis(); loadNews();
setInterval(loadAnalysis,60000);
setInterval(loadNews,30000);
</script>
</body>
</html>
'''

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/api/analysis/<symbol>")
def analysis(symbol):
    symbol = symbol.upper()
    if symbol not in SYMBOLS:
        return jsonify({"error": "unsupported symbol"}), 400
    try:
        return jsonify(build_trade_plan(symbol))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/news")
def news():
    return jsonify(news_items())

@app.route("/api/health")
def health():
    return jsonify({"ok": True, "telegram": ENABLE_TELEGRAM})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
