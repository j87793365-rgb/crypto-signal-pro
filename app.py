import os
from datetime import datetime, timezone

import requests
import feedparser
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

BINANCE_REST = "https://data-api.binance.vision"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

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

def build_trade_plan(symbol, mode="day", strictness="normal", min_rr=2.0):
    t15 = timeframe_signal(fetch_klines(symbol, "15m"))
    t1h = timeframe_signal(fetch_klines(symbol, "1h"))
    t4h = timeframe_signal(fetch_klines(symbol, "4h"))

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

HIGH_IMPACT_WORDS = [
    "fed","federal reserve","cpi","inflation","interest rate","sec",
    "etf","hack","exploit","liquidation","tariff","war","regulation"
]

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

HTML = r"""
<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#090a0d">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>Crypto Signal Pro</title>
<style>
:root{--bg:#090a0d;--card:#12141a;--line:#252936;--text:#f5f6f8;--muted:#9298a8;--gold:#d9b45b;--green:#47d18c;--red:#ff666e;--yellow:#f2c96b}
*{box-sizing:border-box}
body{margin:0;background:#090a0d;color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding-bottom:90px}
.wrap{max-width:760px;margin:auto;padding:16px}
.top{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:14px}
h1{font-size:22px;margin:0}.sub{font-size:12px;color:var(--muted);margin-top:4px}
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
.bottom{position:fixed;left:50%;transform:translateX(-50%);bottom:0;width:100%;max-width:760px;background:rgba(14,16,21,.96);border-top:1px solid var(--line);display:grid;grid-template-columns:repeat(4,1fr);padding:8px 8px calc(8px + env(safe-area-inset-bottom));backdrop-filter:blur(16px)}
.nav{border:0;background:transparent;color:#969dab;font-size:11px}.nav.active{color:var(--gold)}.ico{font-size:20px;display:block;margin-bottom:2px}
.toggle{width:48px;height:28px;border-radius:999px;background:#2a2d36;position:relative;flex:0 0 auto}.toggle.on{background:#8d7136}.knob{position:absolute;width:22px;height:22px;border-radius:50%;background:#fff;top:3px;left:3px;transition:.2s}.toggle.on .knob{left:23px}
.settingSelect{min-width:145px}.notice{color:#f0c867;line-height:1.55}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div><h1>Crypto Signal Pro</h1><div class="sub">即時價格 · 多週期 · 新聞風險</div></div>
    <select id="symbol">
      <option>BTCUSDT</option><option>ETHUSDT</option><option>SOLUSDT</option><option>BNBUSDT</option><option>XRPUSDT</option>
    </select>
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
      <div class="sectionTitle">風險提醒</div>
      <div class="card muted" id="warnings">讀取中...</div>
    </div>
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
  <button class="nav" data-page="news"><span class="ico">📰</span>新聞</button>
  <button class="nav" data-page="alerts"><span class="ico">🔔</span>通知</button>
  <button class="nav" data-page="settings"><span class="ico">⚙️</span>設定</button>
</nav>

<script>
let ws=null,currentSymbol="BTCUSDT",lastAction=null,lastGrade=null;

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
      box.innerHTML+=`<div class="newsitem"><a href="${n.link}" target="_blank">${n.title}</a>${tag}<div class="muted" style="margin-top:6px">${n.source} ${n.published||""}</div></div>`;
    });
  }catch(e){}
}
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
document.getElementById("symbol").addEventListener("change",()=>{connectWS();loadAnalysis();});

applySettings();
connectWS();
loadAnalysis();
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
    if symbol not in SYMBOLS:
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

@app.route("/api/news")
def news():
    return jsonify(news_items())

@app.route("/api/health")
def health():
    return jsonify({"ok":True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
