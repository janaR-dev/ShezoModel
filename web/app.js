/* Shezo web app — vanilla JS talking to the Python JSON API on the same origin.
 *
 * Three role-scoped dashboards (patient, caretaker, therapist) over one shared
 * patient record. The browser only ever summarises camera/microphone input into
 * numbers and detector labels; no video or audio leaves the device, which is the
 * contract the model's perception layer expects (`shezo/ai/perception.py`).
 */
"use strict";

const TOKEN_KEY = "shezo.token";

const state = {
  token: null,
  user: null,
  home: null,
  session: null,
  cameraStream: null,
  audioStream: null,
  audioCtx: null,
  analyser: null,
  prevLuma: null,
  previewTimer: null,
  charts: {},
  selected: { caretaker: null, therapist: null },
  faceDetectorPromise: null,
  faceDetectorOff: false,
  faceResult: [],
  faceResultAt: 0,
  faceBusy: false,
  lastFaces: 0,
  lastFaceScore: 0,
  adherencePct: null,
};

const VERDICT_STYLE = {
  corroborated: {
    label: "Sensors matched it",
    cls: "bg-mint-400/15 text-mint-200 ring-mint-400/30",
  },
  no_sensor_evidence: {
    label: "No sensor evidence",
    cls: "bg-amber-400/15 text-amber-200 ring-amber-400/30",
  },
  inconclusive: {
    label: "Reading unusable",
    cls: "bg-slate-400/15 text-slate-200 ring-slate-400/30",
  },
};

const RISK_STYLE = {
  low: { label: "Low risk", cls: "bg-white/5 text-slate-300 ring-white/15" },
  elevated: { label: "Elevated", cls: "bg-amber-400/15 text-amber-200 ring-amber-400/30" },
  high: { label: "High — care team alerted", cls: "bg-orange-500/15 text-orange-200 ring-orange-400/30" },
  critical: { label: "Critical — crisis protocol", cls: "bg-rose-500/20 text-rose-100 ring-rose-400/40" },
};

const TREND_STYLE = {
  improving: { label: "Improving", cls: "bg-mint-400/15 text-mint-200 ring-mint-400/30" },
  stable: { label: "Stable", cls: "bg-slate-400/15 text-slate-200 ring-slate-400/30" },
  worsening: { label: "Needs attention", cls: "bg-rose-500/15 text-rose-200 ring-rose-400/30" },
  baseline: { label: "Baseline", cls: "bg-indigo-400/15 text-indigo-200 ring-indigo-400/30" },
  no_data: { label: "No data yet", cls: "bg-white/5 text-slate-300 ring-white/15" },
};

const DOSE_STYLE = {
  pending: { label: "Due", cls: "bg-amber-400/15 text-amber-200 ring-amber-400/30" },
  taken: { label: "Taken", cls: "bg-mint-400/15 text-mint-200 ring-mint-400/30" },
  skipped: { label: "Skipped", cls: "bg-slate-400/15 text-slate-200 ring-slate-400/30" },
  missed: { label: "Missed", cls: "bg-rose-500/15 text-rose-200 ring-rose-400/30" },
};

// ─────────────────────────────────────────────────────────────── utilities
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
  });
}

function when(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return String(iso);
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function clock(iso) {
  return iso ? `${String(iso).slice(11, 16)} UTC` : "—";
}

function pct(value, fallback = "—") {
  return value === null || value === undefined ? fallback : `${Math.round(Number(value))}%`;
}

function num(value, digits = 1, fallback = "—") {
  return value === null || value === undefined ? fallback : Number(value).toFixed(digits);
}

function initials(name) {
  return String(name || "?")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0].toUpperCase())
    .join("");
}

function toast(message, tone = "info") {
  const host = $("toast-host");
  if (!host) return;
  const tones = {
    info: "bg-ink-800/95 text-slate-100 ring-white/15",
    success: "bg-mint-400/15 text-mint-100 ring-mint-400/35",
    warn: "bg-amber-400/15 text-amber-100 ring-amber-400/35",
    error: "bg-rose-500/20 text-rose-100 ring-rose-400/40",
  };
  const node = document.createElement("div");
  node.className = `pointer-events-auto max-w-lg rounded-xl px-4 py-2.5 text-sm shadow-panel ring-1 ring-inset animate-rise ${tones[tone] || tones.info}`;
  node.setAttribute("role", tone === "error" ? "alert" : "status");
  node.textContent = message;
  host.appendChild(node);
  setTimeout(() => {
    node.style.opacity = "0";
    node.style.transition = "opacity 240ms ease";
    setTimeout(() => node.remove(), 260);
  }, 4200);
}

/** A permission prompt nobody answers must not leave the UI stuck on "Opening…". */
function withTimeout(promise, ms, message) {
  let timer;
  const timeout = new Promise((resolve, reject) => {
    timer = setTimeout(() => reject(new Error(message)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function api(path, options = {}) {
  const headers = { Accept: "application/json" };
  if (options.body !== undefined) headers["Content-Type"] = "application/json";
  if (state.token) headers.Authorization = `Bearer ${state.token}`;

  let response;
  try {
    response = await fetch(path, {
      method: options.method || "GET",
      headers,
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
    });
  } catch (error) {
    throw new ApiError("Cannot reach the Shezo API. Is `py -m shezo.api` running?", 0);
  }

  let data = {};
  try {
    data = await response.json();
  } catch (error) {
    data = {};
  }
  if (response.status === 401 && path !== "/api/auth/login") {
    clearSession();
    showLogin("Your session expired. Sign in again.");
  }
  if (!response.ok || data.ok === false) {
    throw new ApiError(data.error || `Request failed (HTTP ${response.status})`, response.status);
  }
  return data;
}

// ───────────────────────────────────────────────────────── auth and routing
function clearSession() {
  state.token = null;
  state.user = null;
  state.session = null;
  sessionStorage.removeItem(TOKEN_KEY);
  stopMedia();
  Object.keys(state.charts).forEach(destroyChart);
  $("app-shell").classList.add("hidden");
  $("login-layer").classList.remove("hidden");
}

function showLogin(message) {
  $("login-layer").classList.remove("hidden");
  $("app-shell").classList.add("hidden");
  const error = $("login-error");
  if (message) {
    error.textContent = message;
    error.classList.remove("hidden");
  } else {
    error.classList.add("hidden");
  }
}

function showView(role) {
  ["patient", "caretaker", "therapist"].forEach((name) => {
    const section = $(`view-${name}`);
    if (section) section.classList.toggle("hidden", name !== role);
  });
  const nav = $("app-nav");
  const titles = { patient: "Patient workspace", caretaker: "Caretaker workspace", therapist: "Therapist workspace" };
  if (nav) {
    nav.innerHTML = `<span class="chip bg-white/5 text-slate-300 ring-white/15">${esc(titles[role] || role)}</span>`;
  }
}

async function signIn(email, password) {
  const submit = $("login-submit");
  submit.disabled = true;
  submit.textContent = "Signing in…";
  try {
    const data = await api("/api/auth/login", { method: "POST", body: { email, password } });
    state.token = data.token;
    sessionStorage.setItem(TOKEN_KEY, data.token);
    $("login-error").classList.add("hidden");
    await enterApp(data.user);
  } catch (error) {
    showLogin(error.message);
  } finally {
    submit.disabled = false;
    submit.textContent = "Sign in";
  }
}

async function enterApp(user) {
  state.user = user;
  $("login-layer").classList.add("hidden");
  $("app-shell").classList.remove("hidden");
  $("current-user").textContent = user.display_name;
  $("current-role").textContent = user.role;
  showView(user.role);

  try {
    if (user.role === "patient") await loadPatient();
    else if (user.role === "caretaker") await loadCaretaker();
    else if (user.role === "therapist") await loadTherapist();
  } catch (error) {
    if (error.status !== 401) toast(error.message, "error");
  }
}

async function signOut() {
  try {
    await api("/api/auth/logout", { method: "POST" });
  } catch (error) {
    /* the local token is dropped regardless */
  }
  clearSession();
  showLogin("");
}

// ─────────────────────────────────────────────────────────────────── charts
function hexAlpha(hex, alpha) {
  const clean = String(hex).replace("#", "");
  const value = parseInt(clean.length === 3 ? clean.split("").map((c) => c + c).join("") : clean, 16);
  const r = (value >> 16) & 255;
  const g = (value >> 8) & 255;
  const b = value & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function destroyChart(id) {
  const existing = state.charts[id];
  if (existing) {
    try {
      existing.destroy();
    } catch (error) {
      /* canvas already gone */
    }
    delete state.charts[id];
  }
}

function setupCharts() {
  if (!window.Chart) return;
  Chart.defaults.color = "#94a3b8";
  Chart.defaults.font.family = 'Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif';
  Chart.defaults.font.size = 11;
  Chart.defaults.borderColor = "rgba(148, 163, 184, 0.12)";

  // Plugin 1: badge on the latest value of every line dataset.
  Chart.register({
    id: "shezoLastPoint",
    afterDatasetsDraw(chart) {
      if (chart.config.type !== "line") return;
      const { ctx } = chart;
      chart.data.datasets.forEach((dataset, index) => {
        const meta = chart.getDatasetMeta(index);
        if (meta.hidden) return;
        const points = meta.data.filter((point) => point && !Number.isNaN(point.y));
        const last = points[points.length - 1];
        if (!last || dataset.data[dataset.data.length - 1] === null) return;
        const value = dataset.data[dataset.data.length - 1];
        const color = dataset.borderColor;
        ctx.save();
        ctx.beginPath();
        ctx.arc(last.x, last.y, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#0b1220";
        ctx.stroke();

        const text = dataset.suffix === "%" ? `${Math.round(value)}%` : Number(value).toFixed(1);
        ctx.font = "600 10px Inter, system-ui, sans-serif";
        const width = ctx.measureText(text).width + 12;
        const x = Math.min(last.x + 8, chart.chartArea.right - width - 2);
        const y = Math.max(last.y - 20, chart.chartArea.top + 2);
        ctx.fillStyle = "rgba(11, 18, 32, 0.92)";
        ctx.strokeStyle = hexAlpha(color, 0.45);
        ctx.lineWidth = 1;
        if (ctx.roundRect) {
          ctx.beginPath();
          ctx.roundRect(x, y, width, 16, 8);
          ctx.fill();
          ctx.stroke();
        } else {
          ctx.fillRect(x, y, width, 16);
        }
        ctx.fillStyle = color;
        ctx.textBaseline = "middle";
        ctx.fillText(text, x + 6, y + 8.5);
        ctx.restore();
      });
    },
  });

  // Plugin 2: centre label for doughnut charts (used on all three dashboards).
  Chart.register({
    id: "shezoCenter",
    afterDraw(chart, args, options) {
      if (chart.config.type !== "doughnut" || !options || !options.text) return;
      const { ctx, chartArea } = chart;
      const x = (chartArea.left + chartArea.right) / 2;
      const y = (chartArea.top + chartArea.bottom) / 2;
      ctx.save();
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "#e2e8f0";
      ctx.font = "600 18px Inter, system-ui, sans-serif";
      ctx.fillText(options.text, x, y - 6);
      if (options.subtext) {
        ctx.fillStyle = "#94a3b8";
        ctx.font = "500 10px Inter, system-ui, sans-serif";
        ctx.fillText(options.subtext, x, y + 12);
      }
      ctx.restore();
    },
  });
}

function lineChart(canvasId, config) {
  const canvas = $(canvasId);
  if (!canvas || !window.Chart) return;
  destroyChart(canvasId);
  const datasets = config.series.map((line) => ({
    label: line.label,
    data: line.data,
    borderColor: line.color,
    backgroundColor: hexAlpha(line.color, 0.14),
    fill: line.fill !== false,
    borderWidth: 2,
    tension: 0.35,
    spanGaps: true,
    pointRadius: 0,
    pointHoverRadius: 4,
    suffix: line.suffix,
  }));
  state.charts[canvasId] = new Chart(canvas, {
    type: "line",
    data: { labels: config.labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      layout: { padding: { top: 18, right: 4 } },
      plugins: {
        legend: {
          display: datasets.length > 1,
          position: "bottom",
          labels: { boxWidth: 8, boxHeight: 8, usePointStyle: true, pointStyle: "circle" },
        },
        tooltip: {
          backgroundColor: "rgba(11, 18, 32, 0.95)",
          borderColor: "rgba(148, 163, 184, 0.2)",
          borderWidth: 1,
          padding: 10,
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y === null ? "no data" : ctx.parsed.y}${ctx.dataset.suffix || ""}`,
          },
        },
      },
      scales: {
        x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkipPadding: 12 } },
        y: {
          beginAtZero: config.minZero !== false,
          suggestedMax: config.max,
          grid: { color: "rgba(148, 163, 184, 0.08)" },
          ticks: { maxTicksLimit: 5, callback: (value) => (config.suffix ? `${value}${config.suffix}` : value) },
        },
      },
    },
  });
}

function doughnutChart(canvasId, config) {
  const canvas = $(canvasId);
  if (!canvas || !window.Chart) return;
  destroyChart(canvasId);
  state.charts[canvasId] = new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: config.labels,
      datasets: [
        {
          data: config.values,
          backgroundColor: config.colors,
          borderColor: "#0b1220",
          borderWidth: 2,
          hoverOffset: 3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "68%",
      plugins: {
        legend: {
          position: "bottom",
          labels: { boxWidth: 8, boxHeight: 8, usePointStyle: true, pointStyle: "circle", padding: 12 },
        },
        shezoCenter: { text: config.center || "", subtext: config.subtext || "" },
      },
    },
  });
}

// ────────────────────────────────────────────────────────── patient screen
async function loadPatient() {
  const [home, progress, doses, medications] = await Promise.all([
    api("/api/patient/home"),
    api("/api/patient/progress"),
    api("/api/patient/doses"),
    api("/api/patient/medications"),
  ]);
  state.home = home;
  $("patient-name").textContent = home.patient.display_name;
  // Same window as the doughnut below it, so the chip and the chart agree.
  renderMedications(medications.medications, progress.adherence);
  renderReminders(home.reminders);
  renderRecentChecks(home.recent_checks);
  renderTrend(home.trend, progress);
  renderCareTeam(home.care_team, home.patient.emergency_contact);
  renderAdherenceChart(progress);
  await adoptOpenSession(home.open_session);
}

/** Pick the session the server still holds open, so a reload continues it. */
async function adoptOpenSession(open) {
  const previousId = state.session ? state.session.id : null;
  state.session = open || null;
  if (!open) {
    updateSessionUi();
    return;
  }
  $("consent-camera").checked = Boolean(open.camera_consent);
  $("consent-mic").checked = Boolean(open.mic_consent);
  updateSessionUi();
  if (previousId === open.id) return;

  toast(`Your ${clock(open.started_at)} session is still open — checks continue on the same record.`, "info");
  await restoreSensors(open);
}

/** Consent for this session was already given; try the devices again quietly. */
async function restoreSensors(open) {
  if (open.camera_consent && !state.cameraStream) {
    try {
      await openCamera();
    } catch (error) {
      toast("Camera stayed off after the reload. Checks still work with your words and the distress rating.", "warn");
    }
  }
  if (open.mic_consent && !state.audioStream) {
    try {
      await openMicrophone();
    } catch (error) {
      toast("Microphone stayed off after the reload. Checks still work without it.", "warn");
    }
  }
}

function renderReminders(reminders) {
  $("reminder-message").textContent = reminders.message || "Nothing due right now.";
  // One chip carries both facts: what is due now, and the settled-dose rate.
  const due = reminders.due_now.length
    ? `${reminders.due_now.length} due now`
    : reminders.missed.length
      ? `${reminders.missed.length} unconfirmed`
      : "On track";
  const adherence = state.adherencePct;
  const chip = $("adherence-chip");
  chip.textContent = adherence === null ? due : `${due} · ${adherence}% adherence`;
  chip.className = `chip ${
    adherence === null
      ? "bg-white/5 text-slate-300 ring-white/15"
      : adherence >= 80
        ? "bg-mint-400/10 text-mint-200 ring-mint-400/30"
        : adherence >= 50
          ? "bg-amber-400/15 text-amber-200 ring-amber-400/30"
          : "bg-rose-500/15 text-rose-200 ring-rose-400/30"
  }`;

  const rows = [];
  const push = (dose, tone) => {
    const style = DOSE_STYLE[dose.status] || DOSE_STYLE.pending;
    const actionable = dose.status === "pending" || dose.status === "missed";
    rows.push(`
      <div class="flex flex-wrap items-center gap-2 rounded-xl border border-white/10 bg-ink-950/50 px-3 py-2">
        <div class="min-w-0 flex-1">
          <p class="truncate text-sm font-medium text-slate-100">${esc(dose.medication)} <span class="text-slate-400">${esc(dose.dose)}</span></p>
          <p class="text-[11px] text-slate-500">${dose.status === "pending" ? "Due" : "Was due"} ${esc(clock(dose.due_at))}${tone ? ` · ${esc(tone)}` : ""}</p>
        </div>
        <span class="chip ${style.cls}">${style.label}</span>
        ${
          actionable
            ? `<div class="flex gap-1">
                 <button class="btn-quiet bg-mint-400/15 text-mint-100 ring-1 ring-inset ring-mint-400/30" data-dose="${esc(dose.id)}" data-action="taken">Taken</button>
                 <button class="btn-quiet bg-white/5 ring-1 ring-inset ring-white/10" data-dose="${esc(dose.id)}" data-action="skipped">Skip</button>
               </div>`
            : ""
        }
      </div>`);
  };

  reminders.missed.forEach((dose) => push(dose, "not confirmed"));
  reminders.due_now.forEach((dose) => push(dose, "due now"));
  reminders.upcoming.slice(0, 3).forEach((dose) => push(dose, "upcoming"));
  reminders.recently_taken.slice(-2).forEach((dose) => push(dose, null));

  if (!rows.length) {
    rows.push('<p class="rounded-xl border border-white/10 bg-ink-950/50 px-3 py-4 text-center text-sm text-slate-400">No doses scheduled for today.</p>');
  }
  $("dose-list").innerHTML = rows.join("");
  $("dose-list")
    .querySelectorAll("[data-dose]")
    .forEach((button) => button.addEventListener("click", () => actOnDose(button.dataset.dose, button.dataset.action)));
}

async function actOnDose(doseId, action) {
  try {
    await api(`/api/patient/doses/${doseId}`, { method: "POST", body: { action } });
    toast(action === "taken" ? "Dose recorded as taken." : "Dose marked skipped — your care team can see it.", "success");
    await loadPatient();
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderMedications(medications, adherence) {
  state.adherencePct =
    adherence && adherence.adherence_pct !== null && adherence.adherence_pct !== undefined
      ? Math.round(adherence.adherence_pct)
      : null;
  $("med-list").innerHTML = medications.length
    ? medications
        .map(
          (med) => `
      <div class="rounded-xl border border-white/10 bg-ink-950/50 px-3 py-2.5">
        <div class="flex items-baseline justify-between gap-2">
          <p class="text-sm font-medium text-slate-100">${esc(med.name)}</p>
          <span class="font-mono text-xs text-slate-400">${esc(med.dose)}</span>
        </div>
        <p class="mt-0.5 text-[11px] text-slate-500">${esc(med.times.join(" · "))}${med.instructions ? ` · ${esc(med.instructions)}` : ""}</p>
      </div>`,
        )
        .join("")
    : '<p class="text-sm text-slate-400">No medication on your plan yet. Your therapist can add it.</p>';
}

function renderTrend(trend, progress) {
  const style = TREND_STYLE[trend.direction] || TREND_STYLE.no_data;
  const chip = $("patient-trend-chip");
  chip.className = `chip ${style.cls}`;
  chip.textContent = style.label;
  $("patient-trend-copy").textContent = trend.explanation || "No activity in this window.";

  const sessions = progress.sessions;
  const checks = progress.reality_checks;
  $("patient-week-stats").innerHTML = `
    <div class="rounded-xl bg-white/5 px-2 py-3"><dt class="text-[11px] text-slate-400">Sessions</dt><dd class="text-xl font-semibold text-white">${sessions.count}</dd></div>
    <div class="rounded-xl bg-white/5 px-2 py-3"><dt class="text-[11px] text-slate-400">Checks</dt><dd class="text-xl font-semibold text-white">${checks.count}</dd></div>
    <div class="rounded-xl bg-white/5 px-2 py-3"><dt class="text-[11px] text-slate-400">Avg relief</dt><dd class="text-xl font-semibold text-white">${num(sessions.avg_relief, 1, "—")}</dd></div>`;
}

function renderAdherenceChart(progress) {
  doughnutChart("chart-patient-adherence", {
    labels: ["Taken", "Skipped", "Missed"],
    values: [progress.adherence.taken, progress.adherence.skipped, progress.adherence.missed],
    colors: ["#3ee0b4", "#64748b", "#f43f5e"],
    center: pct(progress.adherence.adherence_pct, "—"),
    subtext: "settled doses",
  });
}

function renderCareTeam(team, emergency) {
  $("care-team").innerHTML = team.length
    ? team
        .map(
          (member) => `
      <li class="flex items-center gap-3">
        <span class="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-white/10 text-xs font-semibold text-slate-200">${esc(initials(member.display_name))}</span>
        <div class="min-w-0">
          <p class="truncate text-sm font-medium text-slate-100">${esc(member.display_name || member.user_id)}</p>
          <p class="text-[11px] capitalize text-slate-500">${esc(member.role)}${member.relationship ? ` · ${esc(member.relationship)}` : ""}</p>
        </div>
      </li>`,
        )
        .join("")
    : '<li class="text-sm text-slate-400">No care team assigned yet.</li>';
  $("emergency-contact").textContent = emergency
    ? `Emergency contact: ${emergency}`
    : "No emergency contact recorded — ask your therapist to add one.";
}

function renderRecentChecks(checks) {
  if (!checks.length) {
    $("recent-checks").innerHTML =
      '<p class="text-sm text-slate-400">No checks yet. Start a session and run your first reality check.</p>';
    return;
  }
  $("recent-checks").innerHTML = checks
    .slice()
    .reverse()
    .map((check) => {
      const verdict = VERDICT_STYLE[check.verdict] || VERDICT_STYLE.inconclusive;
      const risk = RISK_STYLE[check.risk_level] || RISK_STYLE.low;
      return `
      <details class="group rounded-xl border border-white/10 bg-ink-950/50 px-3 py-2.5">
        <summary class="flex cursor-pointer flex-wrap items-center gap-2 list-none">
          <span class="min-w-0 flex-1 truncate text-sm text-slate-200">${esc(check.claim)}</span>
          <span class="chip ${verdict.cls}">${verdict.label}</span>
          <span class="chip ${risk.cls}">${esc(check.risk_level)}</span>
          <span class="text-[11px] text-slate-500">${esc(when(check.created_at))}</span>
        </summary>
        <div class="mt-3 space-y-2 border-t border-white/10 pt-3 text-xs leading-relaxed text-slate-400">
          <p class="text-sm font-medium text-slate-200">${esc(check.headline)}</p>
          <p>${esc(check.explanation)}</p>
          <p class="rounded-lg bg-white/5 px-3 py-2">${esc(check.sensor_summary)}</p>
        </div>
      </details>`;
    })
    .join("");
}

function renderResult(data) {
  const check = data.reality_check;
  const verdict = VERDICT_STYLE[check.verdict] || VERDICT_STYLE.inconclusive;
  const risk = RISK_STYLE[check.risk_level] || RISK_STYLE.low;

  $("result-panel").classList.remove("hidden");
  // "start" + the panel's scroll-margin-top parks the headline just below the
  // sticky header; "nearest" left a tall panel showing only its last rows.
  $("result-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  $("result-headline").textContent = check.headline;
  $("result-verdict").className = `chip ${verdict.cls}`;
  $("result-verdict").textContent = verdict.label;
  $("result-risk").className = `chip ${risk.cls}`;
  $("result-risk").textContent = risk.label;
  $("result-confidence").textContent = `${Math.round(check.confidence * 100)}%`;
  $("result-confidence-bar").style.width = `${Math.round(check.confidence * 100)}%`;
  $("result-explanation").textContent = check.explanation;
  $("result-sensor").textContent = check.sensor_summary;
  $("result-disclosure").textContent = data.safety && data.safety.crisis_protocol
    ? `${data.safety.note || ""} ${check.disclosure}`.trim()
    : check.disclosure;

  $("result-steps").innerHTML = (check.grounding_steps || [])
    .map(
      (step, index) => `
      <li class="rounded-xl border border-white/10 bg-ink-950/50 p-3">
        <div class="flex items-baseline gap-2">
          <span class="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-mint-400/15 text-[11px] font-semibold text-mint-200">${index + 1}</span>
          <p class="text-sm font-medium text-slate-100">${esc(step.title)}</p>
          <span class="ml-auto text-[11px] text-slate-500">${esc(step.seconds)}s</span>
        </div>
        <ul class="mt-2 space-y-1 pl-7 text-xs leading-relaxed text-slate-400">
          ${(step.instructions || []).map((line) => `<li>${esc(line)}</li>`).join("")}
        </ul>
        ${step.why ? `<p class="mt-2 pl-7 text-[11px] italic text-slate-500">${esc(step.why)}</p>` : ""}
      </li>`,
    )
    .join("");
}

function updateSessionUi() {
  const open = Boolean(state.session && state.session.status === "open");
  const chip = $("session-chip");
  chip.textContent = open
    ? `Open since ${clock(state.session.started_at)}`
    : "Session closed";
  chip.className = open
    ? "chip bg-mint-400/15 text-mint-200 ring-mint-400/30"
    : "chip bg-white/5 text-slate-300 ring-white/15";
  $("start-session-btn").disabled = open;
  $("end-session-btn").disabled = !open;
  $("check-btn").disabled = !open;
  $("consent-camera").disabled = open;
  $("consent-mic").disabled = open;
}

// ─────────────────────────────────────────────────────── camera / microphone
async function openCamera() {
  if (state.cameraStream) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("This browser cannot read the camera.");
  }
  const stream = await withTimeout(
    navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    }),
    15000,
    "The camera did not respond. Allow it in the browser's permission prompt, then tick Camera again.",
  );
  state.cameraStream = stream;
  const video = $("camera-preview");
  video.srcObject = stream;
  await video.play().catch(() => {});
  $("camera-placeholder").classList.add("hidden");
  if (state.audioStream && !state.analyser) attachAnalyser();
  startPreviewLoop();
}

async function openMicrophone() {
  if (state.audioStream) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("This browser cannot read the microphone.");
  }
  state.audioStream = await withTimeout(
    navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } }),
    15000,
    "The microphone did not respond. Allow it in the browser's permission prompt, then tick Microphone again.",
  );
  attachAnalyser();
}

function attachAnalyser() {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx || !state.audioStream) return;
  state.audioCtx = state.audioCtx || new AudioCtx();
  if (state.audioCtx.state === "suspended") state.audioCtx.resume().catch(() => {});
  const source = state.audioCtx.createMediaStreamSource(state.audioStream);
  const analyser = state.audioCtx.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.5;
  source.connect(analyser);
  state.analyser = analyser;
}

function stopMedia() {
  if (state.previewTimer) {
    clearInterval(state.previewTimer);
    state.previewTimer = null;
  }
  [state.cameraStream, state.audioStream].forEach((stream) => {
    if (stream) stream.getTracks().forEach((track) => track.stop());
  });
  state.cameraStream = null;
  state.audioStream = null;
  state.analyser = null;
  state.prevLuma = null;
  const video = $("camera-preview");
  if (video) video.srcObject = null;
  const placeholder = $("camera-placeholder");
  if (placeholder) placeholder.classList.remove("hidden");
}

function startPreviewLoop() {
  if (state.previewTimer) return;
  state.previewTimer = setInterval(async () => {
    try {
      const sample = measureFrame();
      if (!sample) return;
      // BlazeFace costs a couple of milliseconds per frame, so every tick is
      // affordable.
      if (state.cameraStream) {
        const { count, score } = await detectFaces();
        state.lastFaces = count;
        state.lastFaceScore = score;
      }
      paintMetrics({ ...sample, faces: state.lastFaces, faceScore: state.lastFaceScore });
    } catch (error) {
      /* a dropped frame must never break the page */
    }
  }, 800);
}

const frameCanvas = document.createElement("canvas");
frameCanvas.width = 160;
frameCanvas.height = 120;
const frameCtx = frameCanvas.getContext("2d", { willReadFrequently: true });

function measureFrame() {
  const video = $("camera-preview");
  if (!video || !video.videoWidth || video.readyState < 2) return null;
  frameCtx.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
  const { data } = frameCtx.getImageData(0, 0, frameCanvas.width, frameCanvas.height);
  const width = frameCanvas.width;
  const height = frameCanvas.height;
  const luma = new Float32Array(width * height);
  let sum = 0;
  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    const value = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
    luma[p] = value;
    sum += value;
  }
  const brightness = sum / (luma.length * 255);

  // Blur: mean absolute 4-neighbour Laplacian, divided by the frame's own mean
  // luma so a dim room is not reported as blurry. A fixed divisor made every
  // real webcam frame score ~0.8-0.95 and look "too blurry" to the model.
  // Calibrated on this metric: crisp synthetic 0.04, 5px-blurred synthetic 0.97,
  // dim-but-sharp camera frame ~0.6.
  let lap = 0;
  let count = 0;
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const p = y * width + x;
      const value = 4 * luma[p] - luma[p - 1] - luma[p + 1] - luma[p - width] - luma[p + width];
      lap += Math.abs(value);
      count += 1;
    }
  }
  const meanLuma = luma.length ? sum / luma.length : 0;
  const meanLap = count ? lap / count : 0;
  const edgeRatio = meanLuma > 1 ? meanLap / meanLuma : 0;
  const blur = Math.max(0, Math.min(1, 1 - edgeRatio / 0.25));

  let motion = 0;
  if (state.prevLuma && state.prevLuma.length === luma.length) {
    let diff = 0;
    for (let i = 0; i < luma.length; i += 1) diff += Math.abs(luma[i] - state.prevLuma[i]);
    motion = Math.max(0, Math.min(1, diff / luma.length / 40));
  }
  state.prevLuma = luma;

  return {
    brightness: Number(brightness.toFixed(2)),
    blur: Number(blur.toFixed(2)),
    motion: Number(motion.toFixed(2)),
    faces: state.lastFaces || 0,
  };
}

/**
 * MediaPipe BlazeFace (short range), vendored in web/vendor/face and run on this
 * device. Shipping Chrome has no FaceDetector, which is why every frame used to
 * report zero faces.
 */
function ensureFaceDetector() {
  if (state.faceDetectorPromise) return state.faceDetectorPromise;
  state.faceDetectorPromise = (async () => {
    if (typeof window.FaceDetection !== "function") return null;
    const detector = new window.FaceDetection({
      locateFile: (file) => `/vendor/face/${file}`,
    });
    // 0.3 rather than the 0.5 default: a three-quarter turn or a face lit from
    // one side scores 0.35-0.5 and would otherwise read as "no face".
    detector.setOptions({ model: "short", minDetectionConfidence: 0.3 });
    detector.onResults((results) => {
      state.faceResult = results.detections || [];
      state.faceResultAt = performance.now();
    });
    await detector.initialize();
    return detector;
  })().catch(() => {
    state.faceDetectorOff = true;
    return null;
  });
  return state.faceDetectorPromise;
}

/**
 * The detector's confidence for one detection. The minified release carries it
 * as ``V[0].ga`` rather than the documented ``score``, so read both and fall
 * back to 0 rather than inventing a number.
 */
function faceScore(detection) {
  const value = detection.score ?? (detection.V && detection.V[0] && detection.V[0].ga);
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

/** Face count and best score for the current video frame. */
async function detectFaces() {
  const video = $("camera-preview");
  if (state.faceDetectorOff) return { count: 0, score: 0 };
  if (!video || !video.videoWidth || video.readyState < 2) return { count: 0, score: 0 };
  // One frame at a time: the preview loop and a grounding check can otherwise
  // call send() together, and the overlapping call comes back empty.
  if (state.faceBusy) {
    return { count: state.lastFaces || 0, score: state.lastFaceScore || 0 };
  }
  const detector = await ensureFaceDetector();
  if (!detector) return { count: 0, score: 0 };
  state.faceBusy = true;
  try {
    const before = state.faceResultAt || 0;
    // Full video resolution, not the 160x120 measuring canvas: tiny faces were
    // exactly what the old path could not see.
    await detector.send({ image: video });
    // Trust only a result produced by this frame, never last frame's array.
    if ((state.faceResultAt || 0) <= before) return { count: 0, score: 0 };
    const faces = state.faceResult;
    const score = faces.reduce((best, face) => Math.max(best, faceScore(face)), 0);
    return { count: faces.length, score };
  } catch (error) {
    return { count: 0, score: 0 };
  } finally {
    state.faceBusy = false;
  }
}

function paintMetrics(sample) {
  $("m-brightness").textContent = sample.brightness.toFixed(2);
  $("m-blur").textContent = sample.blur.toFixed(2);
  $("m-motion").textContent = sample.motion.toFixed(2);
  $("m-faces").textContent = String(sample.faces);
  $("m-labels").textContent = sample.faces > 0
    ? `person ${Number(sample.faceScore || 0).toFixed(2)}`
    : "none";
  if (state.analyser) {
    const buf = new Float32Array(state.analyser.fftSize);
    state.analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i += 1) sum += buf[i] * buf[i];
    const rms = Math.sqrt(sum / buf.length);
    $("m-audio").textContent = `${Math.round(94 + 20 * Math.log10(rms + 1e-9))} dB`;
  } else {
    $("m-audio").textContent = "mic off";
  }
}

async function grabFrameSample() {
  const sample = measureFrame();
  if (!sample) return null;
  const { count, score } = await detectFaces();
  state.lastFaces = count;
  state.lastFaceScore = score;
  const labels = {};
  // Only labels this device actually produced, at the detector's own score.
  if (count > 0) labels.person = Number(score.toFixed(2));
  return {
    labels,
    faces: count,
    brightness: sample.brightness,
    blur: sample.blur,
    motion: sample.motion,
    seconds_since_frame: 0.1,
  };
}

async function sampleAudio(durationMs = 1400) {
  if (!state.analyser) return null;
  const analyser = state.analyser;
  const buffer = new Float32Array(analyser.fftSize);
  const levels = [];
  const started = performance.now();
  while (performance.now() - started < durationMs) {
    analyser.getFloatTimeDomainData(buffer);
    let sum = 0;
    for (let i = 0; i < buffer.length; i += 1) sum += buffer[i] * buffer[i];
    const rms = Math.sqrt(sum / buffer.length);
    levels.push(94 + 20 * Math.log10(rms + 1e-9));
    await sleep(70);
  }
  if (!levels.length) return null;
  const sorted = [...levels].sort((a, b) => a - b);
  const floor = sorted[Math.floor(sorted.length * 0.2)];
  const loud = sorted[Math.floor(sorted.length * 0.85)];
  const speechFrames = levels.filter((level) => level > floor + 8).length;
  // Energy alone is not speech: a fan, hum or steady music crosses any threshold.
  // Speech is bursty, so its level keeps crossing its own mean — require that too.
  // This is still only a guess about sound, never a transcription, which is why
  // the model may not treat it as evidence of a voice.
  const mean = levels.reduce((total, level) => total + level, 0) / levels.length;
  let transitions = 0;
  let above = levels[0] > mean;
  levels.forEach((level) => {
    const nowAbove = level > mean;
    if (nowAbove !== above) transitions += 1;
    above = nowAbove;
  });
  return {
    transcript: "",
    duration_s: Number((durationMs / 1000).toFixed(1)),
    ambient_db: Number(floor.toFixed(0)),
    speech_detected: speechFrames / levels.length > 0.25 && transitions >= 3,
    other_speaker: false, // the browser cannot diarise; we do not pretend it can
    snr_db: Number(Math.max(3, Math.min(25, loud - floor)).toFixed(1)),
  };
}

async function startSession() {
  const wantCamera = $("consent-camera").checked;
  const wantMic = $("consent-mic").checked;
  const button = $("start-session-btn");
  button.disabled = true;
  button.textContent = "Opening…";
  try {
    if (wantCamera) {
      try {
        await openCamera();
      } catch (error) {
        $("consent-camera").checked = false;
        toast(`Camera not available: ${error.message}. Continuing without it.`, "warn");
      }
    }
    if (wantMic) {
      try {
        await openMicrophone();
      } catch (error) {
        $("consent-mic").checked = false;
        toast(`Microphone not available: ${error.message}. Continuing without it.`, "warn");
      }
    }
    if (!$("consent-camera").checked) stopCameraOnly();
    if (!$("consent-mic").checked) stopMicOnly();

    const data = await api("/api/patient/sessions", {
      method: "POST",
      body: {
        distress_start: Number($("distress-range").value),
        camera_consent: $("consent-camera").checked,
        mic_consent: $("consent-mic").checked,
      },
    });
    state.session = data.session;
    updateSessionUi();
    toast(data.notice || "Session opened.", "success");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.textContent = "Start session";
    updateSessionUi();
  }
}

function stopCameraOnly() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach((track) => track.stop());
    state.cameraStream = null;
  }
  // A frame from the previous session would otherwise read as motion and keep
  // reporting the face count it had before the camera was switched off.
  state.prevLuma = null;
  state.lastFaces = 0;
  state.lastFaceScore = 0;
  const video = $("camera-preview");
  if (video) video.srcObject = null;
  $("camera-placeholder").classList.remove("hidden");
}

function stopMicOnly() {
  if (state.audioStream) {
    state.audioStream.getTracks().forEach((track) => track.stop());
    state.audioStream = null;
  }
  state.analyser = null;
}

async function endSession() {
  if (!state.session) return;
  const button = $("end-session-btn");
  button.disabled = true;
  try {
    const data = await api(`/api/patient/sessions/${state.session.id}/end`, {
      method: "POST",
      body: { distress_end: Number($("distress-range").value) },
    });
    state.session = null;
    stopMedia();
    toast("Session closed. A summary report was sent to your care team.", "success");
    await loadPatient();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    updateSessionUi();
  }
}

async function runCheck() {
  const claim = $("claim-input").value.trim();
  if (!state.session) {
    toast("Start a session first, so the camera/microphone consent is recorded.", "warn");
    return;
  }
  if (!claim) {
    toast("Write what you are noticing, or tap one of the examples.", "warn");
    $("claim-input").focus();
    return;
  }
  const button = $("check-btn");
  button.disabled = true;
  const original = button.textContent;
  button.textContent = "Checking…";
  try {
    const body = { claim, distress: Number($("distress-range").value) };
    if (state.session.camera_consent && state.cameraStream) {
      const frame = await grabFrameSample();
      if (frame) body.frame = frame;
    }
    if (state.session.mic_consent && state.analyser) {
      const audio = await sampleAudio();
      if (audio) body.audio = audio;
    }
    const data = await api(`/api/patient/sessions/${state.session.id}/checks`, { method: "POST", body });
    renderResult(data);
    await loadPatient();
    if (data.safety && data.safety.crisis_protocol) {
      toast("Crisis protocol triggered — your care team was alerted.", "error");
    } else {
      toast("Check recorded and sent to your care team.", "success");
    }
  } catch (error) {
    toast(error.message, "error");
    if (error.status === 400 && /closed|timed out/i.test(error.message)) {
      // Another tab (or the idle timeout) closed it: resync instead of retrying.
      state.session = null;
      stopMedia();
      await loadPatient().catch(() => {});
    }
  } finally {
    button.textContent = original;
    updateSessionUi();
  }
}

// ─────────────────────────────────────────────────────────────── care views
function statCard(label, value, hint, accent) {
  return `
    <div class="panel p-4">
      <p class="stat-label">${esc(label)}</p>
      <p class="stat-value ${accent || ""}">${esc(value)}</p>
      <p class="mt-1 text-xs text-slate-500">${esc(hint || "")}</p>
    </div>`;
}

function renderPatientPicker(containerId, rows, role) {
  const container = $(containerId);
  if (!rows.length) {
    container.innerHTML = '<p class="text-sm text-slate-400">No patients are assigned to you yet.</p>';
    return;
  }
  container.innerHTML = rows
    .map((row) => {
      const trend = TREND_STYLE[row.trend.direction] || TREND_STYLE.no_data;
      const active = state.selected[role] === row.patient.id;
      return `
      <button class="w-full rounded-xl border px-3 py-3 text-left transition ${
        active ? "border-mint-400/40 bg-mint-400/10" : "border-white/10 bg-ink-950/50 hover:bg-white/5"
      }" data-patient="${esc(row.patient.id)}" data-role="${role}">
        <div class="flex items-center gap-3">
          <span class="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-white/10 text-xs font-semibold text-slate-200">${esc(initials(row.patient.display_name))}</span>
          <div class="min-w-0 flex-1">
            <p class="truncate text-sm font-medium text-slate-100">${esc(row.patient.display_name)}</p>
            <p class="text-[11px] text-slate-500">${row.reality_checks} checks · ${pct(row.adherence_pct)} adherence</p>
          </div>
          <span class="chip ${trend.cls}">${trend.label}</span>
        </div>
        ${row.escalations ? `<p class="mt-2 text-[11px] font-medium text-rose-200">${row.escalations} escalation(s) in this window</p>` : ""}
      </button>`;
    })
    .join("");
  container.querySelectorAll("[data-patient]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selected[button.dataset.role] = button.dataset.patient;
      if (button.dataset.role === "caretaker") loadCaretakerPatient(button.dataset.patient);
      else loadTherapistPatient(button.dataset.patient);
    });
  });
}

function progressDetailHtml(progress, prefix) {
  const sessions = progress.sessions;
  const checks = progress.reality_checks;
  const trend = TREND_STYLE[progress.trend.direction] || TREND_STYLE.no_data;
  const topClaims = checks.top_claims.length
    ? checks.top_claims.map((item) => `<li class="flex justify-between gap-2"><span>${esc(item.entity)}</span><span class="font-mono text-slate-300">${item.count}</span></li>`).join("")
    : '<li class="text-slate-500">No recognisable entities reported.</li>';

  return `
    <div class="flex flex-wrap items-center gap-3">
      <span class="chip ${trend.cls}">${trend.label}</span>
      <p class="text-xs leading-relaxed text-slate-400">${esc(progress.trend.explanation || "")}</p>
    </div>
    <div class="grid gap-3 sm:grid-cols-4">
      <div class="rounded-xl bg-white/5 px-3 py-2"><p class="text-[11px] text-slate-400">Sessions</p><p class="text-lg font-semibold text-white">${sessions.count}</p></div>
      <div class="rounded-xl bg-white/5 px-3 py-2"><p class="text-[11px] text-slate-400">Avg relief</p><p class="text-lg font-semibold text-white">${num(sessions.avg_relief, 1)}</p></div>
      <div class="rounded-xl bg-white/5 px-3 py-2"><p class="text-[11px] text-slate-400">Checks</p><p class="text-lg font-semibold text-white">${checks.count}</p></div>
      <div class="rounded-xl bg-white/5 px-3 py-2"><p class="text-[11px] text-slate-400">Escalations</p><p class="text-lg font-semibold ${checks.escalated ? "text-rose-200" : "text-white"}">${checks.escalated}</p></div>
    </div>
    <div class="grid gap-4 lg:grid-cols-2">
      <div>
        <p class="label">Adherence per week</p>
        <div class="chart-box"><canvas id="${prefix}-adherence"></canvas></div>
      </div>
      <div>
        <p class="label">Distress in session (0–10)</p>
        <div class="chart-box"><canvas id="${prefix}-distress"></canvas></div>
      </div>
    </div>
    <div class="grid gap-4 lg:grid-cols-2">
      <div>
        <p class="label">What the sensors reported</p>
        <div class="chart-box h-52"><canvas id="${prefix}-verdicts"></canvas></div>
      </div>
      <div>
        <p class="label">Most-reported entities</p>
        <ul class="space-y-1.5 text-sm text-slate-300">${topClaims}</ul>
        <p class="mt-3 text-[11px] leading-relaxed text-slate-500">
          ${progress.adherence.taken} taken · ${progress.adherence.missed} missed · ${progress.adherence.skipped} skipped in this window.
        </p>
      </div>
    </div>`;
}

function drawProgressCharts(progress, prefix) {
  const weeks = progress.weekly || [];
  const labels = weeks.map((week) => new Date(week.week_start).toLocaleDateString(undefined, { month: "short", day: "numeric" }));
  lineChart(`${prefix}-adherence`, {
    labels,
    series: [{ label: "Adherence", data: weeks.map((week) => week.adherence_pct), color: "#fbbf24", suffix: "%" }],
    max: 100,
    suffix: "%",
  });
  lineChart(`${prefix}-distress`, {
    labels,
    series: [
      { label: "Start", data: weeks.map((week) => week.avg_distress_start), color: "#818cf8" },
      { label: "End", data: weeks.map((week) => week.avg_distress_end), color: "#3ee0b4" },
    ],
    max: 10,
    minZero: false,
  });
  const verdicts = progress.reality_checks.by_verdict;
  doughnutChart(`${prefix}-verdicts`, {
    labels: ["Matched", "No evidence", "Unusable"],
    values: [verdicts.corroborated, verdicts.no_sensor_evidence, verdicts.inconclusive],
    colors: ["#3ee0b4", "#fbbf24", "#64748b"],
    center: String(progress.reality_checks.count),
    subtext: "checks",
  });
}

function reportCard(report, base) {
  const acknowledged = (report.recipients || []).some((recipient) => recipient.acknowledged_at);
  const urgent = report.urgent
    ? '<span class="chip bg-rose-500/20 text-rose-100 ring-rose-400/40">Urgent</span>'
    : "";
  return `
    <div class="rounded-xl border border-white/10 bg-ink-950/50 p-3">
      <div class="flex flex-wrap items-center gap-2">
        <p class="min-w-0 flex-1 text-sm font-medium text-slate-100">${esc(report.title)}</p>
        ${urgent}
        <span class="text-[11px] text-slate-500">${esc(when(report.created_at))}</span>
      </div>
      <p class="mt-1 line-clamp-3 whitespace-pre-line text-xs leading-relaxed text-slate-400">${esc(report.summary)}</p>
      <div class="mt-2 flex flex-wrap items-center gap-2">
        <span class="chip bg-white/5 text-slate-400 ring-white/10">${esc(report.kind.replace(/_/g, " "))}</span>
        ${
          acknowledged
            ? '<span class="chip bg-mint-400/10 text-mint-200 ring-mint-400/25">Acknowledged</span>'
            : `<button class="btn-quiet bg-white/5 ring-1 ring-inset ring-white/10" data-ack="${esc(report.id)}" data-base="${esc(base)}">Acknowledge</button>`
        }
      </div>
    </div>`;
}

function renderInbox(containerId, countId, reports, base) {
  const container = $(containerId);
  const count = $(countId);
  const open = reports.filter((report) => !(report.recipients || []).some((r) => r.acknowledged_at)).length;
  count.textContent = `${reports.length} report(s) · ${open} open`;
  if (!reports.length) {
    container.innerHTML = '<p class="text-sm text-slate-400">No reports yet.</p>';
    return;
  }
  container.innerHTML = reports.slice(0, 12).map((report) => reportCard(report, base)).join("");
  container.querySelectorAll("[data-ack]").forEach((button) => {
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        await api(`${button.dataset.base}/${button.dataset.ack}/ack`, { method: "POST" });
        toast("Report acknowledged.", "success");
        if (state.user.role === "caretaker") await loadCaretaker();
        else await loadTherapist();
      } catch (error) {
        button.disabled = false;
        toast(error.message, "error");
      }
    });
  });
}

// ──────────────────────────────────────────────────────────── caretaker UI
async function loadCaretaker() {
  const dash = await api("/api/caretaker/dashboard");
  const openReports = dash.inbox.filter((report) => !(report.recipients || []).some((r) => r.acknowledged_at)).length;
  const summary = dash.summary || {};

  $("caretaker-stats").innerHTML = [
    statCard("Patients", String(summary.patients ?? dash.patients.length), "On your care team"),
    statCard("Average adherence", pct(summary.avg_adherence_pct), "Settled doses, last 14 days"),
    statCard("Escalations", String(summary.escalations ?? 0), "High or critical checks", summary.escalations ? "text-rose-200" : ""),
    statCard("Open reports", String(openReports), "Waiting for acknowledgement", openReports ? "text-amber-200" : ""),
  ].join("");

  if (!dash.patients.length) {
    renderPatientPicker("caretaker-patients", dash.patients, "caretaker");
    $("caretaker-detail").innerHTML = '<p class="text-sm text-slate-400">No patient is assigned to you yet.</p>';
    renderInbox("caretaker-inbox", "caretaker-inbox-count", dash.inbox, "/api/caretaker/reports");
    return;
  }
  const target =
    dash.patients.find((row) => row.patient.id === state.selected.caretaker) || dash.patients[0];
  state.selected.caretaker = target.patient.id;
  renderPatientPicker("caretaker-patients", dash.patients, "caretaker");
  renderInbox("caretaker-inbox", "caretaker-inbox-count", dash.inbox, "/api/caretaker/reports");
  await loadCaretakerPatient(target.patient.id);
}

async function loadCaretakerPatient(patientId) {
  state.selected.caretaker = patientId;
  const [progress] = await Promise.all([api(`/api/caretaker/patients/${patientId}/progress`)]);
  const container = $("caretaker-detail");
  $("caretaker-detail-range").textContent = `Last ${progress.window_days} days · ${when(progress.generated_at)}`;
  container.innerHTML = progressDetailHtml(progress, "caretaker");
  drawProgressCharts(progress, "caretaker");
  highlightPicker("caretaker-patients", patientId);
}

function highlightPicker(containerId, patientId) {
  document.querySelectorAll(`#${containerId} [data-patient]`).forEach((button) => {
    const active = button.dataset.patient === patientId;
    button.classList.toggle("border-mint-400/40", active);
    button.classList.toggle("bg-mint-400/10", active);
  });
}

// ──────────────────────────────────────────────────────────── therapist UI
async function loadTherapist() {
  const dash = await api("/api/therapist/dashboard");
  const openReports = dash.inbox.filter((report) => !(report.recipients || []).some((r) => r.acknowledged_at)).length;

  $("therapist-stats").innerHTML = [
    statCard("Caseload", String(dash.patients.length), "Assigned patients"),
    statCard("Escalations", String(dash.unassigned_alerts ?? 0), "High or critical checks", dash.unassigned_alerts ? "text-rose-200" : ""),
    statCard("Reports", String(dash.inbox.length), `${openReports} unacknowledged`),
    statCard("Retention", "14 d", "Tracking window"),
  ].join("");

  if (!dash.patients.length) {
    renderPatientPicker("therapist-patients", dash.patients, "therapist");
    renderInbox("therapist-inbox", "therapist-inbox-count", dash.inbox, "/api/therapist/reports");
    $("therapist-detail").innerHTML = '<p class="text-sm text-slate-400">No patient is assigned to you yet.</p>';
    return;
  }
  const target = dash.patients.find((row) => row.patient.id === state.selected.therapist) || dash.patients[0];
  state.selected.therapist = target.patient.id;
  renderPatientPicker("therapist-patients", dash.patients, "therapist");
  renderInbox("therapist-inbox", "therapist-inbox-count", dash.inbox, "/api/therapist/reports");
  await loadTherapistPatient(target.patient.id);
}

async function loadTherapistPatient(patientId) {
  const [progress, notes, reports, meds] = await Promise.all([
    api(`/api/therapist/patients/${patientId}/progress`),
    api(`/api/therapist/patients/${patientId}/notes`),
    api(`/api/therapist/patients/${patientId}/reports`),
    api(`/api/therapist/patients/${patientId}/medications`),
  ]);
  const container = $("therapist-detail");
  // Switching patients in the picker must relabel the panel, not keep the old name.
  $("therapist-detail-title").textContent = `${(progress.patient && progress.patient.display_name) || "Patient"} · progress`;
  $("therapist-detail-range").textContent = `Last ${progress.window_days} days · ${when(progress.generated_at)}`;
  container.innerHTML = progressDetailHtml(progress, "therapist");
  drawProgressCharts(progress, "therapist");
  highlightPicker("therapist-patients", patientId);

  $("therapist-notes").innerHTML = notes.notes.length
    ? notes.notes
        .map(
          (note) => `
      <div class="rounded-xl border border-white/10 bg-ink-950/50 p-3">
        <div class="flex items-center gap-2 text-[11px] text-slate-500">
          ${note.pinned ? '<span class="chip bg-indigo-400/15 text-indigo-200 ring-indigo-400/30">Pinned</span>' : ""}
          <span>${esc(note.author_name || "Therapist")}</span>
          <span>· ${esc(when(note.created_at))}</span>
        </div>
        <p class="mt-1 whitespace-pre-line text-sm leading-relaxed text-slate-200">${esc(note.body)}</p>
      </div>`,
        )
        .join("")
    : '<p class="text-sm text-slate-400">No notes yet.</p>';

  $("therapist-meds").innerHTML = meds.medications.length
    ? meds.medications
        .map(
          (med) => `
      <div class="rounded-lg border border-white/10 bg-ink-950/50 px-3 py-2 text-xs">
        <p class="text-sm text-slate-100">${esc(med.name)} <span class="text-slate-400">${esc(med.dose)}</span></p>
        <p class="text-slate-500">${esc(med.times.join(" · "))}${med.instructions ? ` · ${esc(med.instructions)}` : ""}</p>
      </div>`,
        )
        .join("")
    : '<p class="text-xs text-slate-400">No medication on file.</p>';

  $("therapist-reports").innerHTML = reports.reports.length
    ? reports.reports
        .slice(0, 8)
        .map(
          (report) => `
      <details class="rounded-lg border border-white/10 bg-ink-950/50 px-3 py-2">
        <summary class="flex cursor-pointer list-none items-center gap-2">
          <span class="min-w-0 flex-1 truncate text-xs text-slate-200">${esc(report.title)}</span>
          ${report.urgent ? '<span class="chip bg-rose-500/20 text-rose-100 ring-rose-400/40">Urgent</span>' : ""}
          <span class="text-[10px] text-slate-500">${esc(when(report.created_at))}</span>
        </summary>
        <p class="mt-2 whitespace-pre-line text-[11px] leading-relaxed text-slate-400">${esc(report.summary)}</p>
      </details>`,
        )
        .join("")
    : '<p class="text-xs text-slate-400">No reports for this patient.</p>';

  $("note-form").dataset.patient = patientId;
  $("med-form").dataset.patient = patientId;
}

async function submitNote(event) {
  event.preventDefault();
  const patientId = $("note-form").dataset.patient;
  const body = $("note-body").value.trim();
  if (!patientId) return toast("Select a patient first.", "warn");
  if (body.length < 3) return toast("Write at least a few words.", "warn");
  try {
    await api(`/api/therapist/patients/${patientId}/notes`, {
      method: "POST",
      body: { body, pinned: $("note-pinned").checked },
    });
    $("note-body").value = "";
    $("note-pinned").checked = false;
    toast("Note saved and shared with the care team.", "success");
    await loadTherapistPatient(patientId);
  } catch (error) {
    toast(error.message, "error");
  }
}

async function submitMedication(event) {
  event.preventDefault();
  const patientId = $("med-form").dataset.patient;
  if (!patientId) return toast("Select a patient first.", "warn");
  const times = $("med-times")
    .value.split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  try {
    await api(`/api/therapist/patients/${patientId}/medications`, {
      method: "POST",
      body: {
        name: $("med-name").value.trim(),
        dose: $("med-dose").value.trim(),
        times,
        instructions: $("med-instructions").value.trim(),
      },
    });
    $("med-form").reset();
    toast("Medication added to the plan.", "success");
    await loadTherapistPatient(patientId);
  } catch (error) {
    toast(error.message, "error");
  }
}

// ─────────────────────────────────────────────────────────────────── wiring
function bind() {
  $("login-form").addEventListener("submit", (event) => {
    event.preventDefault();
    signIn($("login-email").value.trim(), $("login-password").value);
  });

  document.querySelectorAll("[data-demo]").forEach((button) => {
    button.addEventListener("click", () => {
      const [email, password] = button.dataset.demo.split("|");
      $("login-email").value = email;
      $("login-password").value = password;
      signIn(email, password);
    });
  });

  $("logout-btn").addEventListener("click", signOut);
  $("start-session-btn").addEventListener("click", startSession);
  $("end-session-btn").addEventListener("click", endSession);
  $("check-btn").addEventListener("click", runCheck);

  $("distress-range").addEventListener("input", (event) => {
    $("distress-value").textContent = `${event.target.value} / 10`;
  });

  document.querySelectorAll("[data-claim]").forEach((button) => {
    button.addEventListener("click", () => {
      $("claim-input").value = button.dataset.claim;
      $("claim-input").focus();
    });
  });

  $("consent-camera").addEventListener("change", async (event) => {
    if (event.target.checked && !state.session) {
      try {
        await openCamera();
      } catch (error) {
        event.target.checked = false;
        toast(`Camera not available: ${error.message}`, "warn");
      }
    } else if (!event.target.checked) {
      stopCameraOnly();
    }
  });

  $("consent-mic").addEventListener("change", async (event) => {
    if (event.target.checked && !state.session) {
      try {
        await openMicrophone();
      } catch (error) {
        event.target.checked = false;
        toast(`Microphone not available: ${error.message}`, "warn");
      }
    } else if (!event.target.checked) {
      stopMicOnly();
    }
  });

  $("note-form").addEventListener("submit", submitNote);
  $("med-form").addEventListener("submit", submitMedication);

  window.addEventListener("beforeunload", stopMedia);
}

async function boot() {
  setupCharts();
  bind();
  const token = sessionStorage.getItem(TOKEN_KEY);
  if (!token) return;
  state.token = token;
  try {
    const me = await api("/api/auth/me");
    await enterApp(me.user);
  } catch (error) {
    clearSession();
    if (error.status !== 401) showLogin(error.message);
  }
}

document.addEventListener("DOMContentLoaded", boot);
