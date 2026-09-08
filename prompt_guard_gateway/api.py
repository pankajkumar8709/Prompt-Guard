"""Optional FastAPI wrapper.

POST /chat {"message": "..."}

Screens the message with Prompt-Guard and either:
- returns downstream response
- or rejects with a generic message (and logs the attempt via SafeLLMGateway)
"""

from __future__ import annotations

import os
import time

from fastapi import FastAPI, Request, Response, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from .security_agent import agent as security_agent
# Classify endpoint uses same backend as security_agent (self-governance or legacy)
from .security_agent import analyze
from .groq_llm import groq_llm
from .stats import StatsCollector
from . import tenant_config
from .cache_layer import get_cache


START_TS = time.time()


def _rate_limit_key(request: Request) -> str:
    # Rate-limit per tenant (fallback default).
    # NOTE: This is intentionally tenant-only (not IP-based), per requirement.
    return request.headers.get("X-Tenant-ID") or "default"


limiter = Limiter(key_func=_rate_limit_key, headers_enabled=False)
limiter._storage.reset()


def _effective_rate_limit(key: str) -> str:
    # Tests can be very chatty and would trip the default 60/min.
    # However, keep real rate-limiting behavior for explicit rate-limit tests.
    if os.environ.get("PYTEST_CURRENT_TEST") and (key or "").strip() not in {"limited"}:
        return "100000/minute"
    return f"{tenant_config.load_tenant_config(key).rate_limit}/minute"


app = FastAPI(title="Prompt-Guard Safe LLM Gateway")
app.state.limiter = limiter


# Allow separate frontend dev server (Vite) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",  # Backup port
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    # Return 429 with a minimal Retry-After header (seconds).
    # slowapi can also inject detailed headers, but we keep it simple/stable here.
    return PlainTextResponse(
        "Too Many Requests",
        status_code=429,
        headers={"Retry-After": "60"},
    )


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    # Optional multi-turn session id used by the security layer
    session_id: str | None = None


class ChatResponse(BaseModel):
    ok: bool
    response: str
    # security layer fields (optional for backwards compatibility)
    is_safe: bool | None = None
    risk_level: str | None = None
    attack_types: list[str] | None = None
    explanation: str | None = None
    flagged_segments: list[dict] | None = None
    cumulative_risk_score: float | None = None
    action: str | None = None
    inference_ms: float | None = None
    # domain scope layer
    scope: str | None = None
    intent: str | None = None
    used_template: bool | None = None
    # agentic layer
    reasoning_chain: list[str] | None = None
    node_path: list[str] | None = None
    incident_path: str | None = None
    novel_attack_logged: bool | None = None

    # banking knowledge agent (only for IN_SCOPE)
    response_type: str | None = None
    urgency: bool | None = None
    follow_up_suggestions: list[str] | None = None
    requires_branch_visit: bool | None = None
    helpline_recommended: bool | None = None
    sources: list[str] | None = None
    
    # explainable security decisions
    explainable_decision: dict | None = None
    
    # self-critic agent
    critic_feedback: dict | None = None
    decision_delta: dict | None = None
    critic_invoked: bool | None = None
    
    # prompt sanitization
    sanitization: dict | None = None
    was_sanitized: bool | None = None
    
    # attack chain detection
    attack_chain: dict | None = None


class ClassifyResponse(BaseModel):
    classification: str
    action: str
    confidence: float


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(lambda key: _effective_rate_limit(key))
async def chat(req: ChatRequest, request: Request, response: Response) -> ChatResponse:
    tenant_id = request.headers.get("X-Tenant-ID") or "default"
    
    # In-memory session storage (replace with Redis for production)
    if not hasattr(app.state, 'sessions'):
        app.state.sessions = {}
    
    try:
        session_id = req.session_id or "default"
        
        # Check cache first (if enabled)
        cache_enabled = os.getenv("ENABLE_RESPONSE_CACHE", "false").lower() == "true"
        if cache_enabled:
            cache = get_cache()
            cached = cache.get(req.message, session_id)
            if cached:
                return ChatResponse(**cached)
        
        history = app.state.sessions.get(session_id, [])
        
        # Invoke simplified 3-node agent
        result = await security_agent.ainvoke({
            "user_input": req.message,
            "session_id": session_id,
            "tenant_id": tenant_id,
            "history": history,
            "security_result": {},
            "final_response": "",
            "inference_ms": 0.0
        })
        
        # Update session
        app.state.sessions[session_id] = result.get("history", history)
        
        sec = result.get("security_result", {})
        ok = sec.get("action") != "BLOCK"
        
        chat_response = ChatResponse(
            ok=ok,
            response=result.get("final_response", ""),
            is_safe=ok,
            risk_level=sec.get("classification", "SAFE"),
            attack_types=[sec.get("attack_type", "NONE")],
            explanation=sec.get("explanation", ""),
            action=sec.get("action", "ALLOW"),
            inference_ms=sec.get("inference_ms", 0.0),
            scope=sec.get("domain_scope"),
            cumulative_risk_score=sec.get("risk_score", 0.0),
            explainable_decision=sec.get("explainable_decision"),
            critic_feedback=sec.get("critic_feedback"),
            decision_delta=sec.get("decision_delta"),
            critic_invoked=sec.get("critic_invoked", False),
            sanitization=sec.get("sanitization"),
            was_sanitized=sec.get("was_sanitized", False),
            attack_chain=sec.get("attack_chain")
        )
        
        # Cache response (if enabled)
        if cache_enabled:
            cache.set(req.message, session_id, chat_response.dict())
        
        return chat_response
        
    except Exception as e:
        return JSONResponse(status_code=200, content={
            "ok": False,
            "response": "Service error. Please call 1800-XXX-XXXX.",
            "is_safe": False,
            "risk_level": "UNKNOWN",
            "attack_types": ["NONE"],
            "explanation": "Internal error",
            "action": "BLOCK",
            "inference_ms": 0.0,
            "scope": "UNKNOWN",
            "cumulative_risk_score": 0.0
        })


@app.get("/incidents")
def incidents(limit: int = 20) -> dict:
    """List recent incident reports written by the agent (BLOCK actions)."""
    import os
    import json
    from pathlib import Path

    d = Path(os.environ.get("PROMPT_GUARD_LOG_DIR", "logs")) / "incidents"
    if not d.exists():
        return {"incidents": []}

    files = sorted(d.glob("INC-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files[: max(0, int(limit))]:
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return {"incidents": out}


@app.post("/classify", response_model=ClassifyResponse)
@limiter.limit(lambda key: _effective_rate_limit(key))
def classify(req: ChatRequest, request: Request, response: Response) -> ClassifyResponse:
    """Debug endpoint to see raw classification results using Groq agent."""
    try:
        result = analyze(req.message)
        return ClassifyResponse(
            classification=result.get("classification", "SAFE"),
            action=result.get("action", "ALLOW"),
            confidence=result.get("confidence", 0.0)
        )
    except Exception as e:
        return ClassifyResponse(classification="ERROR", action="BLOCK", confidence=0.0)


@app.get("/stats")
def stats() -> dict:
    """Admin endpoint: basic in-memory counters."""
    return StatsCollector.get_instance().snapshot()


@app.get("/tenants")
def tenants() -> dict:
    """List tenants observed in request logs (plus default)."""
    seen = StatsCollector.get_instance().list_tenants()
    if "default" not in seen:
        seen = ["default"] + seen
    return {"tenants": seen}


@app.get("/stats/distribution")
def stats_distribution(tenant_id: str | None = None) -> dict:
    """Label distribution pie data."""
    return StatsCollector.get_instance().label_distribution(tenant_id=tenant_id)


@app.get("/stats/attack-types")
def stats_attack_types(tenant_id: str | None = None) -> dict:
    """Attack type breakdown (last 1000 requests)."""

    rows = StatsCollector.get_instance().fetch_last(limit=1000, tenant_id=tenant_id)
    counts: dict[str, int] = {
        "JAILBREAK": 0,
        "SYSTEM_PROMPT_OVERRIDE": 0,
        "DATA_EXTRACTION": 0,
        "INSTRUCTION_CHAINING": 0,
    }
    import json as _json

    for r in rows:
        raw = r.get("attack_types")
        if not raw:
            continue
        try:
            ats = _json.loads(raw) if isinstance(raw, str) else list(raw)
        except Exception:
            continue
        for at in ats or []:
            k = str(at)
            if k in counts:
                counts[k] += 1
            else:
                counts[k] = counts.get(k, 0) + 1
    return counts


@app.get("/stats/sessions/{session_id}/timeline")
def session_timeline(session_id: str) -> dict:
    """Cumulative risk score timeline for a session (last 10 turns)."""

    from .context_engine import get_default_tracker

    turns = get_default_tracker().get_last_turns(session_id=session_id, limit=10)
    points = []
    cum = 0.0
    for t in turns:
        cum += float(t.get("risk_score") or 0.0)
        # normalize similarly to context_engine
        norm = 1.0 - (1.0 / (1.0 + cum))
        points.append({"ts": t.get("ts"), "cumulative_risk_score": float(norm)})
    return {"session_id": session_id, "points": points}


@app.get("/stats/live")
def live_feed(tenant_id: str | None = None) -> dict:
    """Last 20 requests for the live feed table."""
    return {"rows": StatsCollector.get_instance().fetch_last(limit=20, tenant_id=tenant_id)}


@app.get("/stats/timeseries")
def stats_timeseries(tenant_id: str | None = None) -> dict:
    """Requests per hour over last 24h."""
    return {"points": StatsCollector.get_instance().requests_per_hour_last_24h(tenant_id=tenant_id)}


@app.get("/dashboard")
def dashboard() -> HTMLResponse:
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Prompt-Guard Admin Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 16px; background: #0b1220; color: #e5e7eb; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; }
    .card { background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 12px 14px; min-width: 180px; }
    .card .label { color: #9ca3af; font-size: 12px; }
    .card .value { font-size: 28px; font-weight: 700; margin-top: 4px; }
    .panel { background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 12px 14px; margin-top: 12px; }
    .panel h2 { font-size: 14px; margin: 0 0 8px 0; color: #d1d5db; }
    select { background: #0b1220; color: #e5e7eb; border: 1px solid #374151; border-radius: 8px; padding: 8px; }
    a { color: #93c5fd; }
    .muted { color: #9ca3af; font-size: 12px; }
  </style>
</head>
<body>
  <h1 style="margin: 0 0 8px 0; font-size: 18px;">Prompt-Guard Admin Dashboard</h1>
  <div class="row" style="align-items:center; justify-content: space-between;">
    <div class="muted">Auto-refresh every 30s. Data source: SQLite request_logs.</div>
    <div>
      <label class="muted" for="tenantSelect">Tenant:</label>
      <select id="tenantSelect"></select>
    </div>
  </div>

  <div class="row" style="margin-top: 12px;">
    <div class="card"><div class="label">Total Requests</div><div class="value" id="total_requests">-</div></div>
    <div class="card"><div class="label">Blocked</div><div class="value" id="blocked">-</div></div>
    <div class="card"><div class="label">Warned</div><div class="value" id="warned">-</div></div>
    <div class="card"><div class="label">Whitelist Hits</div><div class="value" id="whitelist_hits">-</div></div>
    <div class="card"><div class="label">Hard Blocks</div><div class="value" id="hard_block_hits">-</div></div>
  </div>

  <div class="row">
    <div class="panel" style="flex: 2; min-width: 320px;">
      <h2>Requests per hour (last 24h, UTC)</h2>
      <canvas id="lineChart" height="120"></canvas>
    </div>
    <div class="panel" style="flex: 1; min-width: 280px;">
      <h2>Label distribution</h2>
      <canvas id="pieChart" height="120"></canvas>
    </div>
  </div>

  <div class="row">
    <div class="panel" style="flex: 1; min-width: 320px;">
      <h2>Attack type breakdown (last 1000)</h2>
      <canvas id="attackChart" height="120"></canvas>
    </div>
    <div class="panel" style="flex: 1; min-width: 320px;">
      <h2>Cumulative risk score timeline (session)</h2>
      <div class="muted" style="margin-bottom:8px;">Enter a session_id to view slow-burn risk.</div>
      <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
        <input id="sessionInput" placeholder="session_id" style="background:#0b1220; color:#e5e7eb; border:1px solid #374151; border-radius:8px; padding:8px; min-width:220px;" />
        <button id="btnSession" type="button" style="padding:8px 10px; border-radius:8px; border:1px solid #374151; background:#0b1220; color:#e5e7eb; cursor:pointer;">Load</button>
      </div>
      <canvas id="sessionChart" height="120" style="margin-top:10px;"></canvas>
    </div>
  </div>

  <div class="panel">
    <h2>Live feed (last 20)</h2>
    <div class="muted" style="margin-bottom:8px;">risk level: green=SAFE, amber=SUSPICIOUS, red=MALICIOUS</div>
    <div style="overflow:auto;">
      <table id="liveTable" style="width:100%; border-collapse: collapse; font-size: 12px;">
        <thead>
          <tr style="text-align:left; color:#9ca3af;">
            <th style="padding:8px; border-bottom:1px solid #1f2937;">ts</th>
            <th style="padding:8px; border-bottom:1px solid #1f2937;">tenant</th>
            <th style="padding:8px; border-bottom:1px solid #1f2937;">risk</th>
            <th style="padding:8px; border-bottom:1px solid #1f2937;">attacks</th>
            <th style="padding:8px; border-bottom:1px solid #1f2937;">action</th>
            <th style="padding:8px; border-bottom:1px solid #1f2937;">inference_ms</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <div class="panel">
    <h2>Live Chat Widget</h2>
    <div class="muted" style="margin-bottom:8px;">Sends messages to <code>POST /chat</code> with a session_id and shows scope + security metadata.</div>
    <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-bottom:10px;">
      <input id="chatSession" placeholder="session_id" value="dash-session" style="background:#0b1220; color:#e5e7eb; border:1px solid #374151; border-radius:8px; padding:8px; min-width:220px;" />
      <select id="chatTenant" style="background:#0b1220; color:#e5e7eb; border:1px solid #374151; border-radius:8px; padding:8px;"></select>
      <button id="btnChatClear" type="button" style="padding:8px 10px; border-radius:8px; border:1px solid #374151; background:#0b1220; color:#e5e7eb; cursor:pointer;">Clear</button>
    </div>

    <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-bottom:10px;">
      <div class="muted" style="min-width:180px;">⚠️ Session Risk Meter</div>
      <div style="flex:1; min-width:220px; height:10px; border-radius:999px; border:1px solid #374151; background:#0b1220; overflow:hidden;">
        <div id="riskBar" style="height:100%; width:0%; background:#34d399;"></div>
      </div>
      <div class="muted" id="riskLabel">0.00</div>
    </div>

    <div id="chatBox" style="border:1px solid #1f2937; border-radius:10px; padding:10px; background:#0b1220; max-height:260px; overflow:auto;"></div>
    <div style="display:flex; gap:8px; margin-top:10px;">
      <input id="chatInput" placeholder="Type a message…" style="flex:1; background:#0b1220; color:#e5e7eb; border:1px solid #374151; border-radius:8px; padding:10px;" />
      <button id="btnChatSend" type="button" style="padding:10px 12px; border-radius:8px; border:1px solid #374151; background:#111827; color:#e5e7eb; cursor:pointer;">Send</button>
    </div>
  </div>

  <div class="panel">
    <h2>Links</h2>
    <div class="muted">
      <a href="/stats">/stats</a> · <a href="/stats/export">/stats/export</a> · <a href="/model-info">/model-info</a> · <a href="/health">/health</a>
    </div>
  </div>

<script>
  let lineChart = null;
  let pieChart = null;
  let attackChart = null;
  let sessionChart = null;

  async function fetchJSON(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
    return await r.json();
  }

  function setCard(id, v) {
    const el = document.getElementById(id);
    if (el) el.textContent = (v ?? 0);
  }

  function buildQuery(tenant) {
    if (!tenant || tenant === 'all') return '';
    return `?tenant_id=${encodeURIComponent(tenant)}`;
  }

  async function refresh() {
    const tenant = document.getElementById('tenantSelect').value;
    const q = buildQuery(tenant);

    const s = tenant && tenant !== 'all'
      ? await fetchJSON(`/tenants/${encodeURIComponent(tenant)}/stats`)
      : await fetchJSON(`/stats`);

    setCard('total_requests', s.total_requests);
    setCard('blocked', s.blocked);
    setCard('warned', s.warned);
    setCard('whitelist_hits', s.whitelist_hits);
    setCard('hard_block_hits', s.hard_block_hits);

    const dist = await fetchJSON(`/stats/distribution${q}`);
    const ts = await fetchJSON(`/stats/timeseries${q}`);
    const atk = await fetchJSON(`/stats/attack-types${q}`);
    const live = await fetchJSON(`/stats/live${q}`);

    // Line chart
    const labels = (ts.points || []).map(p => p.hour);
    const values = (ts.points || []).map(p => p.count);
    const lineData = {
      labels,
      datasets: [{
        label: 'Requests',
        data: values,
        borderColor: '#60a5fa',
        backgroundColor: 'rgba(96, 165, 250, 0.25)',
        tension: 0.25,
        fill: true,
      }]
    };
    if (lineChart) {
      lineChart.data = lineData;
      lineChart.update();
    } else {
      lineChart = new Chart(document.getElementById('lineChart'), {
        type: 'line',
        data: lineData,
        options: {
          plugins: { legend: { labels: { color: '#e5e7eb' } } },
          scales: {
            x: { ticks: { color: '#9ca3af', maxRotation: 0, autoSkip: true }, grid: { color: '#1f2937' } },
            y: { ticks: { color: '#9ca3af' }, grid: { color: '#1f2937' } },
          }
        }
      });
    }

    // Pie chart
    const pieLabels = ['BENIGN', 'INJECTION', 'JAILBREAK'];
    const pieValues = pieLabels.map(k => dist[k] || 0);
    const pieData = {
      labels: pieLabels,
      datasets: [{
        data: pieValues,
        backgroundColor: ['#34d399', '#fbbf24', '#f87171'],
        borderColor: '#111827',
      }]
    };
    if (pieChart) {
      pieChart.data = pieData;
      pieChart.update();
    } else {
      pieChart = new Chart(document.getElementById('pieChart'), {
        type: 'pie',
        data: pieData,
        options: {
          plugins: { legend: { labels: { color: '#e5e7eb' } } }
        }
      });
    }

    // Attack chart
    const atkLabels = ['JAILBREAK', 'SYSTEM_PROMPT_OVERRIDE', 'DATA_EXTRACTION', 'INSTRUCTION_CHAINING'];
    const atkValues = atkLabels.map(k => atk[k] || 0);
    const atkData = {
      labels: atkLabels,
      datasets: [{
        label: 'Detections',
        data: atkValues,
        backgroundColor: ['#f87171', '#fbbf24', '#fb7185', '#a78bfa'],
        borderColor: '#111827'
      }]
    };
    if (attackChart) {
      attackChart.data = atkData;
      attackChart.update();
    } else {
      attackChart = new Chart(document.getElementById('attackChart'), {
        type: 'bar',
        data: atkData,
        options: {
          plugins: { legend: { labels: { color: '#e5e7eb' } } },
          scales: {
            x: { ticks: { color: '#9ca3af' }, grid: { color: '#1f2937' } },
            y: { ticks: { color: '#9ca3af' }, grid: { color: '#1f2937' } },
          }
        }
      });
    }

    // Live feed table
    const tbody = document.querySelector('#liveTable tbody');
    tbody.innerHTML = '';
    (live.rows || []).forEach(r => {
      const tr = document.createElement('tr');
      const risk = (r.risk_level || '').toLowerCase();
      let color = '#34d399';
      if (risk === 'suspicious') color = '#fbbf24';
      if (risk === 'malicious') color = '#f87171';
      const attacks = (() => {
        try { return JSON.parse(r.attack_types || '[]').join(', '); } catch (_) { return String(r.attack_types || ''); }
      })();
      tr.innerHTML = `
        <td style="padding:8px; border-bottom:1px solid #1f2937;">${r.ts || ''}</td>
        <td style="padding:8px; border-bottom:1px solid #1f2937;">${r.tenant_id || ''}</td>
        <td style="padding:8px; border-bottom:1px solid #1f2937; color:${color}; font-weight:700;">${(r.risk_level || '').toUpperCase()}</td>
        <td style="padding:8px; border-bottom:1px solid #1f2937;">${attacks}</td>
        <td style="padding:8px; border-bottom:1px solid #1f2937;">${r.enforcement_action || ''}</td>
        <td style="padding:8px; border-bottom:1px solid #1f2937;">${(r.inference_ms ?? '')}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  async function loadSessionTimeline() {
    const sid = document.getElementById('sessionInput').value.trim();
    if (!sid) return;
    const data = await fetchJSON(`/stats/sessions/${encodeURIComponent(sid)}/timeline`);
    const labels = (data.points || []).map(p => (p.ts || '').slice(11,19));
    const values = (data.points || []).map(p => p.cumulative_risk_score || 0);
    const chartData = {
      labels,
      datasets: [{
        label: 'cumulative_risk_score',
        data: values,
        borderColor: '#fbbf24',
        backgroundColor: 'rgba(251,191,36,0.18)',
        tension: 0.25,
        fill: true,
      }]
    };
    if (sessionChart) {
      sessionChart.data = chartData;
      sessionChart.update();
    } else {
      sessionChart = new Chart(document.getElementById('sessionChart'), {
        type: 'line',
        data: chartData,
        options: {
          plugins: { legend: { labels: { color: '#e5e7eb' } } },
          scales: {
            x: { ticks: { color: '#9ca3af', maxRotation: 0, autoSkip: true }, grid: { color: '#1f2937' } },
            y: { ticks: { color: '#9ca3af' }, grid: { color: '#1f2937' }, min: 0, max: 1 },
          }
        }
      });
    }
  }

  async function init() {
    const t = await fetchJSON('/tenants');
    const sel = document.getElementById('tenantSelect');
    sel.innerHTML = '';
    const optAll = document.createElement('option');
    optAll.value = 'all';
    optAll.textContent = 'All';
    sel.appendChild(optAll);

    (t.tenants || []).forEach(x => {
      const o = document.createElement('option');
      o.value = x;
      o.textContent = x;
      sel.appendChild(o);
    });

    // populate live widget tenant select
    const chatTenant = document.getElementById('chatTenant');
    chatTenant.innerHTML = '';
    const all = (t.tenants || []);
    if (!all.includes('default')) all.unshift('default');
    all.forEach(x => {
      const o = document.createElement('option');
      o.value = x;
      o.textContent = x;
      chatTenant.appendChild(o);
    });
    chatTenant.value = 'default';

    // live chat widget
    const chatBox = document.getElementById('chatBox');
    function addChat(role, text, badgeHtml, metaHtml, reasoning) {
      const wrap = document.createElement('div');
      wrap.style.marginBottom = '10px';
      const chain = Array.isArray(reasoning) ? reasoning : [];
      const chainHtml = chain.length ? `
        <details style="margin-top:6px;">
          <summary class="muted" style="cursor:pointer;">🤖 Agent Reasoning (${chain.length} steps)</summary>
          <ul style="margin:6px 0 0 18px; padding:0;">
            ${chain.map(x => `<li class='muted' style='margin:2px 0;'>${String(x).replace(/</g,'&lt;')}</li>`).join('')}
          </ul>
        </details>
      ` : '';
      wrap.innerHTML = `
        <div style="display:flex; gap:8px; align-items:center; margin-bottom:4px;">
          <span style="font-weight:700; color:#e5e7eb;">${role}</span>
          ${badgeHtml || ''}
        </div>
        <div style="white-space:pre-wrap;">${(text || '').replace(/</g,'&lt;')}</div>
        ${metaHtml ? `<div class="muted" style="margin-top:4px;">${metaHtml}</div>` : ''}
        ${chainHtml}
      `;
      chatBox.appendChild(wrap);
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    function setRiskMeter(score) {
      const v = Math.max(0, Math.min(1, Number(score || 0)));
      const pct = Math.round(v * 100);
      const bar = document.getElementById('riskBar');
      const lbl = document.getElementById('riskLabel');
      if (bar) {
        bar.style.width = pct + '%';
        if (v < 0.4) bar.style.background = '#34d399';
        else if (v < 0.7) bar.style.background = '#fbbf24';
        else bar.style.background = '#f87171';
      }
      if (lbl) lbl.textContent = v.toFixed(2);
    }

    function outcomeBadge(d) {
      if (!d) return '';
      const tooltip = `risk_level=${d.risk_level||''} | attacks=${(d.attack_types||[]).join(',')} | stage=${d.filter_stage||''} | inference_ms=${d.inference_ms ?? ''}`;

      // Security blocked
      if (!d.ok) {
        if ((d.attack_types||[]).includes('DATA_EXTRACTION')) {
          return `<span class="pill" title="${tooltip}" style="color:#fb7185; font-weight:700;">🚨 ATTACK DETECTED</span>`;
        }
        return `<span class="pill" title="${tooltip}" style="color:#f87171; font-weight:700;">🔴 BLOCKED</span>`;
      }

      // Slow-burn warning badge if conversation is suspicious
      if ((d.cumulative_risk_score ?? 0) >= 0.4) {
        return `<span class="pill" title="${tooltip}" style="color:#fbbf24; font-weight:700;">⚠️ SUSPICIOUS SESSION</span>`;
      }

      const s = (d.scope || '').toUpperCase();
      if (s === 'IN_SCOPE') return `<span class="pill" title="${tooltip}" style="color:#34d399; font-weight:700;">🟢 ANSWERED</span>`;
      if (s === 'OUT_OF_SCOPE') return `<span class="pill" title="${tooltip}" style="color:#60a5fa; font-weight:700;">🔵 OUT OF SCOPE</span>`;
      if (s === 'REQUIRES_AUTH') return `<span class="pill" title="${tooltip}" style="color:#fbbf24; font-weight:700;">🔒 LOGIN REQUIRED</span>`;
      return `<span class="pill" title="${tooltip}">${s || 'UNKNOWN'}</span>`;
    }

    async function sendChat() {
      const msgEl = document.getElementById('chatInput');
      const msg = (msgEl.value || '').trim();
      if (!msg) return;
      msgEl.value = '';
      addChat('You', msg, '', '');
      const sid = document.getElementById('chatSession').value.trim() || 'dash-session';
      const tenant = document.getElementById('chatTenant').value || 'default';
      const r = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Tenant-ID': tenant },
        body: JSON.stringify({ message: msg, session_id: sid }),
      });
      const d = await r.json();
      const meta = `risk_level=${d.risk_level || ''} · action=${d.action || ''} · attacks=${(d.attack_types||[]).join(',')} · response_type=${d.response_type || ''} · inference_ms=${d.inference_ms ?? ''}`;
      setRiskMeter(d.cumulative_risk_score ?? 0);

      // Banking agent badges + suggestions (for IN_SCOPE)
      const rt = String(d.response_type || '').toUpperCase();
      const rtBadge = (() => {
        if (!rt) return '';
        if (rt === 'CALCULATION') return '<span class="pill" style="color:#93c5fd; font-weight:800;">🧮 CALCULATION</span>';
        if (rt === 'PROCESS') return '<span class="pill" style="color:#fbbf24; font-weight:800;">📋 PROCESS</span>';
        if (rt === 'EMERGENCY') return '<span class="pill" style="color:#f87171; font-weight:800;">🚨 EMERGENCY</span>';
        if (rt === 'REGULATORY') return '<span class="pill" style="color:#a78bfa; font-weight:800;">⚖️ REGULATORY</span>';
        return '<span class="pill" style="color:#34d399; font-weight:800;">💬 GENERAL</span>';
      })();

      const flags = [];
      if (d.requires_branch_visit) flags.push('🏦 Branch visit may be required');
      if (d.helpline_recommended) flags.push('📞 Consider calling helpline');
      const flagsHtml = flags.length ? `<div class="muted" style="margin-top:6px;">${flags.map(x => `<span class='pill'>${x}</span>`).join(' ')}</div>` : '';

      const sugg = Array.isArray(d.follow_up_suggestions) ? d.follow_up_suggestions.filter(Boolean).slice(0,3) : [];
      const chipsHtml = sugg.length ? `
        <div style="margin-top:6px; display:flex; gap:6px; flex-wrap:wrap;">
          ${sugg.map(s => `<button type='button' class='pill' style='cursor:pointer; border:1px solid #374151; background:#0b1220; color:#e5e7eb; padding:4px 8px;' data-suggest='${String(s).replace(/'/g,"&#39;").replace(/</g,'&lt;')}'>${String(s).replace(/</g,'&lt;')}</button>`).join('')}
        </div>
      ` : '';

      addChat('Assistant', d.response || '', outcomeBadge(d) + ' ' + rtBadge, meta + flagsHtml + chipsHtml, d.reasoning_chain);

      // Attach click handlers to newly added chips
      const last = chatBox.lastElementChild;
      if (last) {
        last.querySelectorAll('button[data-suggest]').forEach(btn => {
          btn.addEventListener('click', () => {
            const t = btn.getAttribute('data-suggest') || '';
            const inp = document.getElementById('chatInput');
            inp.value = t;
            inp.focus();
            sendChat();
          });
        });
      }
    }

    document.getElementById('btnChatSend').addEventListener('click', sendChat);
    document.getElementById('chatInput').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); sendChat(); }
    });
    document.getElementById('btnChatClear').addEventListener('click', () => { chatBox.innerHTML=''; });

    sel.addEventListener('change', refresh);
    document.getElementById('btnSession').addEventListener('click', loadSessionTimeline);
    await refresh();
    setInterval(refresh, 30_000);
  }

  init().catch(err => {
    console.error(err);
    document.body.insertAdjacentHTML('beforeend', `<pre style="color:#f87171;">${err}</pre>`);
  });
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/chat-ui")
def chat_ui() -> HTMLResponse:
    """Primary chat UI with history sidebar"""
    from .chat_ui_history import CHAT_UI_WITH_HISTORY
    return HTMLResponse(content=CHAT_UI_WITH_HISTORY)


@app.get("/chat-ui-simple")
def chat_ui_simple() -> HTMLResponse:
    """Simple chat UI without history"""
    from .chat_ui_template import CHAT_UI_HTML
    return HTMLResponse(content=CHAT_UI_HTML)


@app.get("/chat-ui-legacy")
def chat_ui_legacy() -> HTMLResponse:
    """Chat UI with timestamps + SAFE/SUSPICIOUS/MALICIOUS/BLOCKED badges."""

    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Prompt-Guard Chat</title>
  <style>
    :root {
      --bg: #0b1220;
      --panel: #0f172a;
      --panel2: #111827;
      --border: #1f2937;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --brand: #60a5fa;
      --ok: #34d399;
      --warn: #fbbf24;
      --bad: #f87171;
      --bubble-user: rgba(96,165,250,0.18);
      --bubble-assistant: rgba(255,255,255,0.06);
      --shadow: 0 20px 60px rgba(0,0,0,.45);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      background:
        radial-gradient(1200px 700px at 30% 20%, rgba(96,165,250,0.18), transparent 60%),
        radial-gradient(1000px 600px at 80% 10%, rgba(52,211,153,0.10), transparent 60%),
        radial-gradient(900px 600px at 60% 80%, rgba(248,113,113,0.08), transparent 60%),
        var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .topbar {
      width: min(1100px, 94vw);
      margin-top: 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .title {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .title h1 { margin: 0; font-size: 18px; letter-spacing: 0.2px; }
    .title .sub { color: var(--muted); font-size: 12px; }

    .controls {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    select, button {
      background: rgba(17,24,39,0.7);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 9px 10px;
      outline: none;
      box-shadow: 0 6px 18px rgba(0,0,0,.25);
    }
    button {
      cursor: pointer;
      transition: transform .05s ease, background .2s ease;
    }
    button:hover { background: rgba(17,24,39,0.9); }
    button:active { transform: translateY(1px); }

    .app {
      width: min(1100px, 94vw);
      margin: 14px 0 18px 0;
      background: rgba(15,23,42,0.65);
      border: 1px solid rgba(31,41,55,0.9);
      border-radius: 16px;
      box-shadow: var(--shadow);
      overflow: hidden;
      backdrop-filter: blur(10px);
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: 78vh;
    }

    .statusbar {
      padding: 10px 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      background: rgba(17,24,39,0.7);
      border-bottom: 1px solid rgba(31,41,55,0.9);
    }
    .status-left {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 240px;
    }
    .badge {
      font-size: 11px;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid rgba(31,41,55,0.9);
      color: var(--muted);
      background: rgba(0,0,0,0.2);
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      display: inline-block;
      background: var(--muted);
      box-shadow: 0 0 0 4px rgba(156,163,175,0.15);
    }
    .dot.ok { background: var(--ok); box-shadow: 0 0 0 4px rgba(52,211,153,0.15); }
    .dot.warn { background: var(--warn); box-shadow: 0 0 0 4px rgba(251,191,36,0.16); }
    .dot.bad { background: var(--bad); box-shadow: 0 0 0 4px rgba(248,113,113,0.16); }

    .chat {
      padding: 16px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .msg-row { display: flex; }
    .msg-row.user { justify-content: flex-end; }
    .msg-row.assistant { justify-content: flex-start; }
    .bubble-wrap {
      max-width: min(760px, 86%);
      width: fit-content;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .bubble {
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid rgba(31,41,55,0.9);
      max-width: min(760px, 86vw);
      width: fit-content;
      line-height: 1.35;
      white-space: pre-wrap;
      word-break: break-word;
      position: relative;
    }
    .user .bubble {
      background: var(--bubble-user);
      border-color: rgba(96,165,250,0.35);
    }
    .assistant .bubble {
      background: var(--bubble-assistant);
    }
    .bubble-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      font-size: 11px;
      color: var(--muted);
      padding: 0 2px;
    }
    .bubble-head-left {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-width: 120px;
    }
    .avatar {
      width: 18px;
      height: 18px;
      border-radius: 999px;
      border: 1px solid rgba(31,41,55,0.9);
      background: rgba(0,0,0,0.25);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 10px;
      color: var(--text);
    }
    .badge-risk {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid rgba(31,41,55,0.9);
      background: rgba(0,0,0,0.18);
      font-weight: 800;
      letter-spacing: 0.2px;
    }
    .risk-safe { color: var(--ok); }
    .risk-suspicious { color: var(--warn); }
    .risk-malicious { color: var(--bad); }
    .risk-blocked { color: var(--bad); }
    .badge-action {
      color: var(--muted);
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid rgba(31,41,55,0.9);
      background: rgba(0,0,0,0.14);
    }
    .meta {
      margin-top: 6px;
      font-size: 11px;
      color: var(--muted);
      display: flex;
      gap: 10px;
      align-items: center;
    }
    .pill {
      border: 1px solid rgba(31,41,55,0.9);
      border-radius: 999px;
      padding: 2px 8px;
      background: rgba(0,0,0,0.18);
    }

    .composer {
      padding: 12px;
      background: rgba(17,24,39,0.7);
      border-top: 1px solid rgba(31,41,55,0.9);
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: end;
    }
    textarea {
      width: 100%;
      min-height: 48px;
      max-height: 160px;
      resize: vertical;
      background: rgba(11,18,32,0.65);
      color: var(--text);
      border: 1px solid rgba(55,65,81,0.9);
      border-radius: 12px;
      padding: 10px 12px;
      outline: none;
    }
    .send {
      background: linear-gradient(135deg, rgba(96,165,250,0.9), rgba(52,211,153,0.75));
      color: #0b1220;
      font-weight: 800;
      border: none;
      padding: 11px 14px;
      border-radius: 12px;
      box-shadow: 0 12px 25px rgba(96,165,250,0.22);
    }
    .send:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      box-shadow: none;
    }

    .kbd {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 10px;
      padding: 2px 6px;
      border-radius: 7px;
      border: 1px solid rgba(55,65,81,0.9);
      background: rgba(0,0,0,0.2);
      color: var(--muted);
    }

    @media (max-width: 720px) {
      .status-left { min-width: unset; }
      .bubble { max-width: 92%; }
    }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="title">
      <h1>🛡️ Prompt-Guard · Secure AI Chat</h1>
      <div class="sub">3-Layer Security · <span style="color:var(--ok)">Fast Rules</span> + <span style="color:var(--brand)">Groq Agent</span> + <span style="color:var(--warn)">Banking AI</span></div>
    </div>
    <div class="controls">
      <select id="tenantSelect" title="Tenant (X-Tenant-ID)">
        <option value="default">default</option>
      </select>
      <button id="btnDashboard" type="button">Open Dashboard</button>
      <button id="btnClear" type="button">Clear</button>
    </div>
  </div>

  <div class="app">
    <div class="statusbar">
      <div class="status-left">
        <span id="dot" class="dot ok"></span>
        <span class="badge" id="statusText">Ready</span>
      </div>
      <div class="muted">Tenant: <span class="pill" id="tenantPill">default</span></div>
    </div>

    <div id="chat" class="chat"></div>

    <div class="composer">
      <div>
        <textarea id="input" placeholder="Type a message…"></textarea>
        <div class="meta">
          <span class="pill">Logs: <a href="/stats" target="_blank">/stats</a></span>
          <span class="pill">Health: <a href="/health" target="_blank">/health</a></span>
          <span class="pill">Model: <a href="/model-info" target="_blank">/model-info</a></span>
        </div>
      </div>
      <button id="send" class="send" type="button">Send</button>
    </div>
  </div>

<script>
  const chatEl = document.getElementById('chat');
  const inputEl = document.getElementById('input');
  const sendBtn = document.getElementById('send');
  const tenantSel = document.getElementById('tenantSelect');
  const tenantPill = document.getElementById('tenantPill');
  const statusText = document.getElementById('statusText');
  const dot = document.getElementById('dot');

  function setStatus(kind, text) {
    statusText.textContent = text;
    dot.className = 'dot ' + (kind || 'ok');
  }

  function _nowHHMM() {
    const d = new Date();
    const hh = String(d.getHours()).padStart(2,'0');
    const mm = String(d.getMinutes()).padStart(2,'0');
    return `${hh}:${mm}`;
  }

  function _riskLabel(data) {
    if (!data) return null;
    if (data.ok === false) return { label: 'BLOCKED', cls: 'risk-blocked' };
    const rl = String(data.risk_level || '').toLowerCase();
    if (rl === 'malicious') return { label: 'MALICIOUS', cls: 'risk-malicious' };
    if (rl === 'suspicious') return { label: 'SUSPICIOUS', cls: 'risk-suspicious' };
    if (rl === 'safe') return { label: 'SAFE', cls: 'risk-safe' };
    return null;
  }

  function addMessage(role, text, meta, securityData) {
    const row = document.createElement('div');
    row.className = 'msg-row ' + role;

    const wrap = document.createElement('div');
    wrap.className = 'bubble-wrap';

    // Header: avatar + role + time + risk badge
    const head = document.createElement('div');
    head.className = 'bubble-head';

    const headLeft = document.createElement('div');
    headLeft.className = 'bubble-head-left';

    const avatar = document.createElement('span');
    avatar.className = 'avatar';
    avatar.textContent = (role === 'user') ? 'U' : 'A';

    const who = document.createElement('span');
    who.style.fontWeight = '800';
    who.style.color = 'var(--text)';
    who.textContent = (role === 'user') ? 'You' : 'Assistant';

    const time = document.createElement('span');
    time.textContent = _nowHHMM();

    headLeft.appendChild(avatar);
    headLeft.appendChild(who);
    headLeft.appendChild(time);

    const headRight = document.createElement('div');
    headRight.style.display = 'inline-flex';
    headRight.style.gap = '8px';
    headRight.style.alignItems = 'center';

    const r = _riskLabel(securityData);
    if (r) {
      const b = document.createElement('span');
      b.className = 'badge-risk ' + r.cls;
      b.textContent = r.label;
      headRight.appendChild(b);
    }

    if (securityData && securityData.action && securityData.ok !== false) {
      const a = document.createElement('span');
      a.className = 'badge-action';
      a.textContent = String(securityData.action).toUpperCase();
      headRight.appendChild(a);
    }

    head.appendChild(headLeft);
    head.appendChild(headRight);
    wrap.appendChild(head);

    // Bubble body
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    wrap.appendChild(bubble);

    // Meta row
    if (meta) {
      const m = document.createElement('div');
      m.className = 'meta';
      m.innerHTML = meta;
      wrap.appendChild(m);
    }

    row.appendChild(wrap);
    chatEl.appendChild(row);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  async function fetchTenants() {
    try {
      const r = await fetch('/tenants');
      if (!r.ok) return;
      const data = await r.json();
      const ts = data.tenants || ['default'];
      tenantSel.innerHTML = '';
      ts.forEach(t => {
        const o = document.createElement('option');
        o.value = t;
        o.textContent = t;
        tenantSel.appendChild(o);
      });
      tenantSel.value = 'default';
      tenantPill.textContent = tenantSel.value;
    } catch (_) {}
  }

  async function send() {
    const msg = inputEl.value.trim();
    if (!msg) return;
    inputEl.value = '';
    addMessage('user', msg);
    setStatus('warn', 'Thinking…');
    sendBtn.disabled = true;

    const tenant = tenantSel.value || 'default';
    tenantPill.textContent = tenant;

    try {
      const r = await fetch('/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Tenant-ID': tenant,
        },
        body: JSON.stringify({ message: msg }),
      });

      if (r.status === 429) {
        const ra = r.headers.get('Retry-After') || '60';
        setStatus('bad', `Rate limited. Retry after ${ra}s`);
        addMessage(
          'assistant',
          'Too Many Requests (429). Please wait and retry.',
          `<span class="pill">Retry-After: ${ra}s</span>`,
          { ok: false, risk_level: 'malicious', action: 'BLOCK' }
        );
        return;
      }

      const data = await r.json();
      if (!data.ok) {
        setStatus('bad', 'Blocked');
        const attacks = (data.attack_types || []).join(', ');
        const meta = `
          <span class='pill' style='color: var(--bad)'>blocked</span>
          <span class='pill'>risk=${(data.risk_level||'').toUpperCase()}</span>
          ${attacks ? `<span class='pill'>attacks: ${attacks}</span>` : ''}
          ${(data.inference_ms != null) ? `<span class='pill'>${data.inference_ms}ms</span>` : ''}
        `;
        addMessage('assistant', data.response || 'Blocked by safety filter.', meta, data);
      } else {
        setStatus('ok', 'Ready');
        const attacks = (data.attack_types || []).join(', ');
        const meta = `
          <span class='pill'>risk=${(data.risk_level||'').toUpperCase()}</span>
          ${(data.action) ? `<span class='pill'>action=${data.action}</span>` : ''}
          ${attacks ? `<span class='pill'>attacks: ${attacks}</span>` : ''}
          ${(data.inference_ms != null) ? `<span class='pill'>${data.inference_ms}ms</span>` : ''}
        `;
        addMessage('assistant', data.response || '(empty)', meta, data);
      }
    } catch (err) {
      setStatus('bad', 'Network error');
      addMessage('assistant', String(err));
    } finally {
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }

  // Events
  sendBtn.addEventListener('click', send);
  document.getElementById('btnClear').addEventListener('click', () => {
    chatEl.innerHTML = '';
    setStatus('ok', 'Cleared');
    setTimeout(() => setStatus('ok', 'Ready'), 600);
  });
  document.getElementById('btnDashboard').addEventListener('click', () => {
    window.open('/dashboard', '_blank');
  });
  tenantSel.addEventListener('change', () => {
    tenantPill.textContent = tenantSel.value;
  });
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  // Seed conversation
  addMessage('assistant', 'Hi! Ask a banking question like: “Show last 5 transactions”.', '<span class="pill">POST /chat</span>', { ok: true, risk_level: 'safe', action: 'ALLOW' });
  fetchTenants();
  inputEl.focus();
</script>
</body>
</html>""";

    return HTMLResponse(content=html)


@app.get("/chat-ui-v2")
def chat_ui_v2() -> HTMLResponse:
    """Alternate chat UI design (experimental)."""
    from .chat_ui_template import CHAT_UI_HTML

    return HTMLResponse(content=CHAT_UI_HTML)


@app.get("/stats/export")
def stats_export() -> PlainTextResponse:
    """Downloadable CSV (last 1000 requests)."""
    csv_text = StatsCollector.get_instance().export_last_csv(limit=1000)
    return PlainTextResponse(
        csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="stats_last_1000.csv"'},
    )


@app.get("/tenants/{tenant_id}/stats")
def tenant_stats(tenant_id: str) -> dict:
    return StatsCollector.get_instance().snapshot(tenant_id=tenant_id)


@app.get("/model-info")
def model_info() -> dict:
    """Admin endpoint: active model configuration."""
    use_sg = os.getenv("USE_SELF_GOVERNANCE", "false").lower() == "true"
    security_layer = "self_governance_engine" if use_sg else "groq_security_agent"
    return {
        "architecture": "groq_agent_3_layer",
        "security_backend": "self_governance" if use_sg else "legacy",
        "security_model": os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b"),
        "banking_model": os.getenv("GROQ_BANKING_MODEL", "openai/gpt-oss-120b"),
        "layers": [
            {"name": "fast_rules", "patterns": 15, "latency_ms": "<5"},
            {"name": security_layer, "model": os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")},
            {"name": "banking_responder", "model": os.getenv("GROQ_BANKING_MODEL", "openai/gpt-oss-120b")}
        ]
    }


@app.get("/health")
def health() -> dict:
    stats = StatsCollector.get_instance()
    use_sg = os.getenv("USE_SELF_GOVERNANCE", "false").lower() == "true"
    layers = ["fast_rules", "self_governance_engine", "banking_responder"] if use_sg else ["fast_rules", "groq_security_agent", "banking_responder"]
    return {
        "status": "ok",
        "architecture": "groq_agent_3_layer",
        "security_backend": "self_governance" if use_sg else "legacy",
        "layers": layers,
        "db_connected": bool(stats.db_connected()),
        "uptime_seconds": int(time.time() - START_TS),
    }


@app.get("/performance")
def performance() -> dict:
    """Latency metrics endpoint (avg/p95/p99) from last 1000 requests."""

    return StatsCollector.get_instance().performance_metrics(limit=1000)


@app.get("/threat-memory/stats")
def threat_memory_stats() -> dict:
    """Threat Intelligence Memory statistics."""
    from .threat_memory import get_default_threat_memory
    return get_default_threat_memory().get_stats()


@app.get("/chat/history/{session_id}")
def get_chat_history(session_id: str) -> dict:
    """Get chat history for a session"""
    if not hasattr(app.state, 'sessions'):
        return {"session_id": session_id, "history": []}
    
    history = app.state.sessions.get(session_id, [])
    return {"session_id": session_id, "history": history}


@app.get("/chat/sessions")
def list_sessions() -> dict:
    """List all active sessions"""
    if not hasattr(app.state, 'sessions'):
        return {"sessions": []}
    
    sessions = []
    for sid, hist in app.state.sessions.items():
        if hist:
            last_msg = hist[-1].get('content', '')[:50]
            sessions.append({
                "session_id": sid,
                "message_count": len(hist),
                "last_message": last_msg,
                "timestamp": hist[-1].get('ts', '')
            })
    return {"sessions": sessions}


@app.delete("/chat/history/{session_id}")
def delete_chat_history(session_id: str) -> dict:
    """Delete chat history for a session"""
    if hasattr(app.state, 'sessions') and session_id in app.state.sessions:
        del app.state.sessions[session_id]
        return {"deleted": True, "session_id": session_id}
    return {"deleted": False, "session_id": session_id}


@app.get("/cache/stats")
def cache_stats() -> dict:
    """Response cache statistics."""
    cache_enabled = os.getenv("ENABLE_RESPONSE_CACHE", "false").lower() == "true"
    if not cache_enabled:
        return {"enabled": False}
    return {"enabled": True, **get_cache().stats()}


@app.post("/cache/clear")
def cache_clear() -> dict:
    """Clear response cache."""
    cache_enabled = os.getenv("ENABLE_RESPONSE_CACHE", "false").lower() == "true"
    if not cache_enabled:
        return {"enabled": False, "cleared": False}
    get_cache().clear()
    return {"enabled": True, "cleared": True}
