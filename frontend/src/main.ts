const POLL_INTERVAL = 60_000; // ms

// ---- Types ----------------------------------------------------------------

interface ObsRec {
  dt: string;
  t: number;
  dp: number | null;
  ws: number;
  wd: number;
  p: number;
}

interface LstmResult {
  mu: number;
  sigma: number;
  ci_low: number;
  ci_high: number;
  n_obs: number;
  last_t: number;
  last_dt: string;
}

interface MosResult {
  mu: number;
  sigma: number;
  ci_low_95: number;
  ci_high_95: number;
  confidence: "HIGH" | "MEDIUM" | "LOW";
  nwp_bias_at_12: number;
  reasons: string[];
  error?: string;
}

interface ApiData {
  server_time: string;
  today: string;
  obs_err: string | null;
  nwp_err: string | null;
  current: ObsRec | null;
  trend: { delta: number; label: string } | null;
  nws_recs: ObsRec[];
  lstm: LstmResult | null;
  mos: MosResult | null;
}

// ---- DOM refs -------------------------------------------------------------

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const elServerTime  = $("server-time");
const elObsTime     = $("obs-time");
const elTemp        = $("temp");
const elDewpt       = $("dewpt");
const elWind        = $("wind");
const elTrend       = $("trend");
const elTrendDelta  = $("trend-delta");
const elObsErr      = $("obs-err");

const elLstmMu      = $("lstm-mu");
const elLstmPi      = $("lstm-pi");
const elLstmSigma   = $("lstm-sigma");
const elLstmInfo    = $("lstm-info");
const elLstmWait    = $("lstm-wait");

const elMosMu       = $("mos-mu");
const elMosPi       = $("mos-pi");
const elMosConf     = $("mos-conf");
const elMosBias     = $("mos-bias");
const elMosReasons  = $("mos-reasons");
const elMosWait     = $("mos-wait");

const elStatus      = $("status");
const canvas        = $<HTMLCanvasElement>("chart");
const ctx           = canvas.getContext("2d")!;

// ---- Fetch ----------------------------------------------------------------

async function fetchData(): Promise<ApiData | null> {
  try {
    const res = await fetch("/api/data");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json() as ApiData;
  } catch (e) {
    setStatus(`Fetch error: ${e}`, "error");
    return null;
  }
}

// ---- Render ---------------------------------------------------------------

function show(el: HTMLElement) { el.style.display = ""; }
function hide(el: HTMLElement) { el.style.display = "none"; }

function setStatus(msg: string, kind: "ok" | "error" | "fetching") {
  elStatus.textContent = msg;
  elStatus.className = `status ${kind}`;
}

function renderHeader(d: ApiData) {
  elServerTime.textContent = d.server_time;

  if (d.obs_err) {
    elObsErr.textContent = `API error: ${d.obs_err}`;
    show(elObsErr);
  } else {
    hide(elObsErr);
  }

  const cur = d.current;
  if (cur) {
    elObsTime.textContent = cur.dt.slice(11, 16);
    elTemp.textContent    = `${cur.t.toFixed(1)} °C`;
    elDewpt.textContent   = cur.dp !== null ? `${cur.dp.toFixed(0)} °C` : "—";
    elWind.textContent    = `${(cur.ws * 3.6).toFixed(0)} km/h`;
  } else {
    elTemp.textContent  = "—";
    elDewpt.textContent = "—";
    elWind.textContent  = "—";
    elObsTime.textContent = "--:--";
  }

  if (d.trend) {
    elTrend.textContent      = d.trend.label;
    elTrendDelta.textContent = `${d.trend.delta > 0 ? "+" : ""}${d.trend.delta.toFixed(1)} °C`;
    elTrend.className        = `trend-label ${d.trend.delta > 0 ? "rising" : d.trend.delta < 0 ? "falling" : "steady"}`;
  }
}

function renderLstm(lstm: LstmResult | null) {
  if (!lstm) {
    hide(elLstmMu.parentElement!);
    show(elLstmWait);
    return;
  }
  show(elLstmMu.parentElement!);
  hide(elLstmWait);
  elLstmMu.textContent    = `${lstm.mu.toFixed(1)} °C`;
  elLstmPi.textContent    = `${lstm.ci_low.toFixed(1)} – ${lstm.ci_high.toFixed(1)} °C`;
  elLstmSigma.textContent = `σ = ${lstm.sigma.toFixed(3)} °C`;
  elLstmInfo.textContent  = `${lstm.n_obs} obs  ·  last at ${lstm.last_dt}`;
}

function renderMos(mos: MosResult | null) {
  if (!mos || "error" in mos) {
    hide(elMosMu.parentElement!);
    elMosWait.textContent = mos?.error ?? "Waiting for 12:00 station reading + NWP data…";
    show(elMosWait);
    return;
  }
  show(elMosMu.parentElement!);
  hide(elMosWait);
  elMosMu.textContent   = `${mos.mu} °C`;
  elMosPi.textContent   = `${mos.ci_low_95} – ${mos.ci_high_95} °C`;
  const confClass = { HIGH: "conf-high", MEDIUM: "conf-med", LOW: "conf-low" }[mos.confidence];
  elMosConf.textContent = mos.confidence;
  elMosConf.className   = `conf-badge ${confClass}`;
  elMosBias.textContent = `NWP bias ${mos.nwp_bias_at_12 >= 0 ? "+" : ""}${mos.nwp_bias_at_12.toFixed(1)} °C`;
  elMosReasons.textContent = mos.reasons.join("  ·  ");
}

// ---- Chart ----------------------------------------------------------------

interface PredLine { mu: number; ciLow: number; ciHigh: number; color: string; label: string }

function drawChart(recs: ObsRec[], lstm: LstmResult | null, mos: MosResult | null) {
  const W = canvas.width  = canvas.offsetWidth  * devicePixelRatio;
  const H = canvas.height = canvas.offsetHeight * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  const w = canvas.offsetWidth;
  const h = canvas.offsetHeight;

  const PAD = { top: 30, right: 90, bottom: 40, left: 48 };
  const pw = w - PAD.left - PAD.right;
  const ph = h - PAD.top  - PAD.bottom;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0e1117";
  ctx.fillRect(0, 0, w, h);

  if (!recs.length) {
    ctx.fillStyle = "#888";
    ctx.font = "14px system-ui";
    ctx.textAlign = "center";
    ctx.fillText("No observations yet today", w / 2, h / 2);
    return;
  }

  // time domain: midnight → now (or last obs)
  const dayStart = new Date(recs[0].dt.slice(0, 10) + "T00:00:00").getTime();
  const dayEnd   = dayStart + 24 * 3600_000;
  const times    = recs.map(r => new Date(r.dt).getTime());
  const temps    = recs.map(r => r.t);

  // y domain
  const predLines: PredLine[] = [];
  if (lstm) predLines.push({ mu: lstm.mu, ciLow: lstm.ci_low,    ciHigh: lstm.ci_high,    color: "#00C864", label: "LSTM" });
  if (mos && "mu" in mos) predLines.push({ mu: +mos.mu, ciLow: +mos.ci_low_95, ciHigh: +mos.ci_high_95, color: "#FF7020", label: "MOS" });

  const allY = [
    ...temps,
    ...predLines.flatMap(p => [p.ciLow, p.ciHigh]),
  ];
  const yMin = Math.floor(Math.min(...allY) - 1);
  const yMax = Math.ceil(Math.max(...allY) + 1);

  const tx = (t: number) => PAD.left + ((t - dayStart) / (dayEnd - dayStart)) * pw;
  const ty = (v: number) => PAD.top  + (1 - (v - yMin) / (yMax - yMin)) * ph;

  // grid
  ctx.strokeStyle = "#1e2230";
  ctx.lineWidth   = 1;
  const ySteps = 6;
  for (let i = 0; i <= ySteps; i++) {
    const v = yMin + (i / ySteps) * (yMax - yMin);
    const y = ty(v);
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + pw, y); ctx.stroke();
    ctx.fillStyle  = "#555"; ctx.font = "11px system-ui"; ctx.textAlign = "right";
    ctx.fillText(`${v.toFixed(0)}°`, PAD.left - 6, y + 4);
  }
  // x-axis hour ticks
  for (let h = 0; h <= 24; h += 3) {
    const x = tx(dayStart + h * 3600_000);
    ctx.beginPath(); ctx.moveTo(x, PAD.top); ctx.lineTo(x, PAD.top + ph); ctx.stroke();
    ctx.fillStyle = "#555"; ctx.font = "11px system-ui"; ctx.textAlign = "center";
    ctx.fillText(`${String(h).padStart(2, "0")}:00`, x, PAD.top + ph + 18);
  }

  // CI bands
  for (const p of predLines) {
    ctx.save();
    ctx.globalAlpha = 0.12;
    ctx.fillStyle = p.color;
    ctx.fillRect(PAD.left, ty(p.ciHigh), pw, ty(p.ciLow) - ty(p.ciHigh));
    ctx.restore();
  }

  // prediction horizontal lines
  let labelOffset = 0;
  for (const p of predLines) {
    const y = ty(p.mu);
    ctx.save();
    ctx.strokeStyle = p.color;
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + pw, y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle  = p.color;
    ctx.font       = "bold 12px system-ui";
    ctx.textAlign  = "left";
    ctx.fillText(`${p.label} ${p.mu.toFixed ? p.mu.toFixed(1) : p.mu}°`, PAD.left + pw + 6, y + 4 + labelOffset);
    ctx.restore();
    labelOffset += 18;
  }

  // observed temp line
  ctx.beginPath();
  ctx.strokeStyle = "#4A90D9";
  ctx.lineWidth   = 2.5;
  for (let i = 0; i < recs.length; i++) {
    const x = tx(times[i]), y = ty(temps[i]);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // dots
  ctx.fillStyle = "#4A90D9";
  for (let i = 0; i < recs.length; i++) {
    ctx.beginPath();
    ctx.arc(tx(times[i]), ty(temps[i]), 3, 0, Math.PI * 2);
    ctx.fill();
  }

  // title
  ctx.fillStyle  = "#ccc";
  ctx.font       = "13px system-ui";
  ctx.textAlign  = "left";
  ctx.fillText("Today's Temperature at LLBG", PAD.left, 18);
}

// ---- Poll loop ------------------------------------------------------------

let lastData: ApiData | null = null;

async function update() {
  setStatus("Fetching…", "fetching");
  const d = await fetchData();
  if (!d) return;
  lastData = d;

  renderHeader(d);
  renderLstm(d.lstm);
  renderMos(d.mos);
  drawChart(d.nws_recs, d.lstm, d.mos);
  setStatus(`Updated at ${d.server_time}`, "ok");
}

// redraw chart on resize without refetching
window.addEventListener("resize", () => {
  if (lastData) drawChart(lastData.nws_recs, lastData.lstm, lastData.mos);
});

// kick off
update();
setInterval(update, POLL_INTERVAL);
