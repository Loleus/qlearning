'use strict';

// ============================================================
// KONSTANTY I KONFIGURACJA
// ============================================================
const W = 720;
const H = 520;
const POP_SIZE = 100;
const STEP_LIMIT = 2600;
const REWARD_GOAL = 35.0;
const PENALTY_STEP = -0.018;
const PENALTY_WALL = -10.0;
const PENALTY_TIMEOUT = -4.0;
const SHAPING_FACTOR = 0.12;

// Akcje: 0=Góra, 1=Dół, 2=Lewo, 3=Prawo
const ACTIONS = [
  { dx: 0, dy: -1 },
  { dx: 0, dy: 1 },
  { dx: -1, dy: 0 },
  { dx: 1, dy: 0 }
];

const goal = { x: 50, y: H - 50, r: 18 };
const start = { x: W - 70, y: 60 };

const walls = [
  { x: 0, y: 0, w: W, h: 20 },
  { x: 0, y: H - 20, w: W, h: 20 },
  { x: 0, y: 0, w: 20, h: H },
  { x: W - 20, y: 0, w: 20, h: H },
  { x: 120, y: 80, w: 480, h: 20 },
  { x: 120, y: 80, w: 20, h: 300 },
  { x: 120, y: 360, w: 400, h: 20 },
  { x: 500, y: 140, w: 20, h: 240 },
  { x: 260, y: 140, w: 260, h: 20 },
  { x: 260, y: 200, w: 20, h: 160 },
  { x: 320, y: 260, w: 180, h: 20 }
];

const config = {
  popSize: POP_SIZE,
  cellSize: 20,
  alpha: 0.22,
  gamma: 0.965,
  speed: 3.2,
  strategy: 'epsilon',
  tracesEnabled: false,
  traceLambda: 0.70,
  doubleQ: false,
  // ε-greedy
  epsDecayType: 'exponential',
  epsStart: 0.35,
  epsMin: 0.02,
  epsDecay: 0.992,
  epsBoost: 0.08,
  epsFixed: 0.10,
  // Softmax
  tempStart: 1.25,
  tempMin: 0.18,
  tempDecay: 0.993,
  // Optimistic
  optimisticInit: 3.0,
  // UCB
  ucbC: 1.40,
  // NoisyNet
  noisySigma: 0.55,
  noisySigmaMin: 0.05,
  noisySigmaDecay: 0.996,
  // Hybrid
  hybridUcbC: 1.50,
  hybridEps: 0.10
};

// ============================================================
// RUNTIME STATE
// ============================================================
const runtime = {
  paused: false,
  episode: 0,
  bestSteps: null,
  lastSuccessRate: 0,
  lastAvgReward: 0,
  epsilonCurrent: config.epsStart,
  temperatureCurrent: config.tempStart,
  noiseSigmaCurrent: config.noisySigma,
  history: [],
  frameCounter: 0,
  lastFrameTime: 0,
  softmaxCache: new Map()
};

// Double Q-Learning: dwie tablice Q
// Używamy Map dla oszczędności pamięci (sparse states)
const Q_A = new Map();       // Q_A (główna)
const Q_B = new Map();     // Q_B (drugi estymator)
const visitCounts = new Map();
let agents = [];

// ============================================================
// FUNKCJE POMOCNICZE (MATEMATYKA I LOGIKA)
// ============================================================
const $ = (id) => document.getElementById(id);

function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }
function randInt(max) { return Math.floor(Math.random() * max); }

// Generowanie liczby z rozkładu normalnego (Box-Muller transform)
function gaussianRandom() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function maxArray(arr) {
  let m = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > m) m = arr[i];
  }
  return m;
}

// Klasa do zarządzania śladami eligibility dla Double Q-Learning
class Traces {
  constructor() {
    this.A = new Map(); // ślady dla Q_A
    this.B = new Map(); // ślady dla Q_B
  }
  clear() {
    this.A.clear();
    this.B.clear();
  }
}

// Argmax z losowym rozstrzyganiem remisów
function argmaxRandomTie(arr) {
  let best = -Infinity;
  const ids = [];
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (v > best + 1e-12) {
      best = v;
      ids.length = 0;
      ids.push(i);
    } else if (Math.abs(v - best) <= 1e-12) {
      ids.push(i);
    }
  }
  return ids[randInt(ids.length)];
}

function isGreedyAction(values, action) {
  const best = maxArray(values);
  return values[action] >= best - 1e-9;
}

// Softmax numerycznie stabilny (odejmujemy max) z cache
function softmax(values, temperature) {
  const t = Math.max(0.05, temperature);
  // Cache key: skrót wartości i temperatury
  const cacheKey = values.join(',') + '|' + t.toFixed(4);
  if (runtime.softmaxCache.has(cacheKey)) {
    return runtime.softmaxCache.get(cacheKey);
  }

  const maxV = maxArray(values);
  const probs = new Float32Array(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    const e = Math.exp((values[i] - maxV) / t);
    probs[i] = e;
    sum += e;
  }
  if (sum <= 0) {
    const fallback = 1 / values.length;
    for (let i = 0; i < probs.length; i++) probs[i] = fallback;
    runtime.softmaxCache.set(cacheKey, probs);
    if (runtime.softmaxCache.size > 2000) {
      const firstKey = runtime.softmaxCache.keys().next().value;
      runtime.softmaxCache.delete(firstKey);
    }
    return probs;
  }
  for (let i = 0; i < probs.length; i++) probs[i] /= sum;
  runtime.softmaxCache.set(cacheKey, probs);
  if (runtime.softmaxCache.size > 2000) {
    const firstKey = runtime.softmaxCache.keys().next().value;
    runtime.softmaxCache.delete(firstKey);
  }
  return probs;
}

function sampleDiscrete(probs) {
  const r = Math.random();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r <= acc) return i;
  }
  return probs.length - 1;
}

function strategyLabel(strategy) {
  switch (strategy) {
    case 'epsilon_dynamic': return 'ε-greedy dyn.';
    case 'epsilon': return 'ε-greedy kl.';
    case 'softmax': return 'Softmax';
    case 'optimistic': return 'Optimistic';
    case 'ucb': return 'UCB';
    case 'noisy': return 'NoisyNet';
    case 'hybrid': return 'Hybrydowa (UCB + ε)';
    default: return strategy;
  }
}

// ============================================================
// OBSŁUGA TABLIC Q (SPEEDUP przez Float32Array)
// ============================================================

function resetExplorationSchedulers() {
  runtime.epsilonCurrent = config.epsStart;
  runtime.temperatureCurrent = config.tempStart;
  runtime.noiseSigmaCurrent = config.noisySigma;
}

function getInitialQValue() {
  if (config.strategy === 'optimistic') {
    return config.optimisticInit;
  }
  return 0;
}

// Pobiera lub tworzy wiersz Q dla danego stanu
function ensureQ(qMap, key) {
  if (!qMap.has(key)) {
    qMap.set(key, new Float32Array(4).fill(getInitialQValue()));
  }
  return qMap.get(key);
}

// Średnia Q_A i Q_B (do wyboru akcji w Double Q)
function getAveragedQ(key) {
  if (!config.doubleQ) {
    return ensureQ(Q_A, key);
  }
  const qa = ensureQ(Q_A, key);
  const qb = ensureQ(Q_B, key);
  const avg = new Float32Array(4);
  for (let i = 0; i < 4; i++) {
    avg[i] = (qa[i] + qb[i]) * 0.5;
  }
  return avg;
}

function maxQ(qMap, key) {
  return maxArray(ensureQ(qMap, key));
}

function ensureCounts(key) {
  if (!visitCounts.has(key)) visitCounts.set(key, new Uint32Array(4));
  return visitCounts.get(key);
}

function ensureTrace(traces, key) {
  if (!traces.has(key)) traces.set(key, new Float32Array(4));
  return traces.get(key);
}

// ============================================================
// AKTUALIZACJA Q (CORE ALGORITHM)
// ============================================================

// Oblicz TD Target dla danego estymatora
function computeTDTarget(selectMap, evalMap, sKey, nextKey, reward, terminal) {
  if (terminal) return reward;
  const nextSelect = ensureQ(selectMap, nextKey);
  const bestAction = argmaxRandomTie(nextSelect);
  return reward + config.gamma * ensureQ(evalMap, nextKey)[bestAction];
}

// Główna funkcja aktualizacji Q
function qUpdate(sKey, action, reward, nextKey, terminal) {
  if (config.doubleQ) {
    const updateA = Math.random() < 0.5;
    const target = updateA
      ? computeTDTarget(Q_A, Q_B, sKey, nextKey, reward, terminal)
      : computeTDTarget(Q_B, Q_A, sKey, nextKey, reward, terminal);

    const q = ensureQ(updateA ? Q_A : Q_B, sKey);
    q[action] += config.alpha * (target - q[action]);
  } else {
    const q = ensureQ(Q_A, sKey);
    const target = terminal ? reward : reward + config.gamma * maxQ(Q_A, nextKey);
    q[action] += config.alpha * (target - q[action]);
  }
}

// ============================================================
// ELIGIBILITY TRACES (Watkins's Q(lambda))
// ============================================================

function applyTraces(qMap, traceMap, delta) {
  for (const [key, trace] of traceMap) {
    const q = ensureQ(qMap, key);
    for (let a = 0; a < 4; a++) {
      if (trace[a] > 1e-6) {
        q[a] += config.alpha * delta * trace[a];
        trace[a] *= config.gamma * config.traceLambda;
      }
    }
  }
}

function decayTraces(traceMap) {
  for (const trace of traceMap.values()) {
    for (let i = 0; i < 4; i++) {
      if (trace[i] > 1e-6) trace[i] *= config.gamma * config.traceLambda;
    }
  }
}

function tracesUpdate(agent, sKey, action, reward, nextKey, terminal, wasGreedy) {
  if (!config.tracesEnabled) {
    qUpdate(sKey, action, reward, nextKey, terminal);
    return;
  }

  if (!config.doubleQ) {
    // Standard Q(λ)
    const delta = reward + config.gamma * (terminal ? 0 : maxQ(Q_A, nextKey)) - ensureQ(Q_A, sKey)[action];
    const trace = agent.traces.A;
    if (!trace.has(sKey)) trace.set(sKey, new Float32Array(4));
    trace.get(sKey)[action] = 1.0; // replacing trace

    applyTraces(Q_A, trace, delta);
    if (terminal || !wasGreedy) agent.traces.clear();
  } else {
    // Double Q + Traces
    const updateA = Math.random() < 0.5;
    const delta = updateA
      ? computeTDTarget(Q_A, Q_B, sKey, nextKey, reward, terminal) - ensureQ(Q_A, sKey)[action]
      : computeTDTarget(Q_B, Q_A, sKey, nextKey, reward, terminal) - ensureQ(Q_B, sKey)[action];

    const traceMap = updateA ? agent.traces.A : agent.traces.B;
    if (!traceMap.has(sKey)) traceMap.set(sKey, new Float32Array(4));
    traceMap.get(sKey)[action] = 1.0;

    if (updateA) {
      applyTraces(Q_A, traceMap, delta);
      decayTraces(agent.traces.B);
    } else {
      applyTraces(Q_B, traceMap, delta);
      decayTraces(agent.traces.A);
    }

    if (terminal || !wasGreedy) agent.traces.clear();
  }
}

// ============================================================
// WYBÓR AKCJI I STRATEGIE
// ============================================================

function getActionValues(rep) {
  return getAveragedQ(rep.key);
}

function pickAction(rep) {
  const baseValues = getActionValues(rep);

  if (config.strategy === 'epsilon_dynamic') {
    if (Math.random() < runtime.epsilonCurrent) {
      return { action: randInt(4), baseValues };
    }
    return { action: argmaxRandomTie(baseValues), baseValues };
  }

  if (config.strategy === 'epsilon') {
    if (Math.random() < config.epsFixed) {
      return { action: randInt(4), baseValues };
    }
    return { action: argmaxRandomTie(baseValues), baseValues };
  }

  if (config.strategy === 'softmax') {
    const probs = softmax(baseValues, runtime.temperatureCurrent);
    return { action: sampleDiscrete(probs), baseValues };
  }

  if (config.strategy === 'optimistic') {
    return { action: argmaxRandomTie(baseValues), baseValues };
  }

  if (config.strategy === 'ucb') {
    const counts = ensureCounts(rep.countKey);
    let total = 0;
    for (let i = 0; i < 4; i++) total += counts[i];
    const scores = new Float32Array(4);
    const bonusBase = config.ucbC * Math.sqrt(Math.log(total + 2));
    for (let i = 0; i < 4; i++) {
      scores[i] = baseValues[i] + bonusBase / Math.sqrt(counts[i] + 1);
    }
    return { action: argmaxRandomTie(scores), baseValues };
  }

  if (config.strategy === 'hybrid') {
    const counts = ensureCounts(rep.countKey);
    let total = 0;
    for (let i = 0; i < 4; i++) total += counts[i];
    const scores = new Float32Array(4);
    const bonusBase = config.hybridUcbC * Math.sqrt(Math.log(total + 2));
    for (let i = 0; i < 4; i++) {
      scores[i] = baseValues[i] + bonusBase / Math.sqrt(counts[i] + 1);
    }
    if (Math.random() < config.hybridEps) {
      return { action: randInt(4), baseValues };
    }
    return { action: argmaxRandomTie(scores), baseValues };
  }

  // NoisyNet-style
  const noisyScores = new Float32Array(4);
  for (let i = 0; i < 4; i++) {
    noisyScores[i] = baseValues[i] + runtime.noiseSigmaCurrent * gaussianRandom();
  }
  return { action: argmaxRandomTie(noisyScores), baseValues };
}

function getExplorationMeta() {
  switch (config.strategy) {
    case 'epsilon_dynamic':
      return { text: 'ε = ' + runtime.epsilonCurrent.toFixed(3), norm: clamp(runtime.epsilonCurrent, 0, 1) };
    case 'epsilon':
      return { text: 'ε = ' + config.epsFixed.toFixed(3), norm: clamp(config.epsFixed, 0, 1) };
    case 'softmax':
      return { text: 'τ = ' + runtime.temperatureCurrent.toFixed(3), norm: clamp(runtime.temperatureCurrent / 3, 0, 1) };
    case 'optimistic':
      return { text: 'Q₀ = ' + config.optimisticInit.toFixed(2), norm: clamp(config.optimisticInit / 8, 0, 1) };
    case 'ucb':
      return { text: 'c = ' + config.ucbC.toFixed(2), norm: clamp(config.ucbC / 4, 0, 1) };
    case 'hybrid':
      return { text: 'c=' + config.hybridUcbC.toFixed(2) + ' ε=' + config.hybridEps.toFixed(3), norm: clamp(config.hybridEps, 0, 1) };
    case 'noisy':
      return { text: 'σ = ' + runtime.noiseSigmaCurrent.toFixed(3), norm: clamp(runtime.noiseSigmaCurrent / 1.5, 0, 1) };
    default:
      return { text: '—', norm: 0 };
  }
}

function advanceExplorationSchedule(successRate) {
  if (config.strategy === 'epsilon_dynamic') {
    let t = runtime.episode;
    let decayFactor;
    switch (config.epsDecayType) {
      case 'linear':
        decayFactor = Math.max(0, 1 - t * 0.001 * (1 - config.epsDecay));
        break;
      case 'exponential':
        decayFactor = Math.pow(config.epsDecay, t);
        break;
      case 'inverse':
        decayFactor = 1 / (1 + t * 0.001 * (1 - config.epsDecay));
        break;
      case 'cosine':
        decayFactor = 0.5 * (1 + Math.cos(Math.PI * t / 1000));
        break;
      default:
        decayFactor = Math.pow(config.epsDecay, t);
    }
    let nextEps = config.epsMin + (config.epsStart - config.epsMin) * decayFactor;
    nextEps += Math.max(0, 0.18 - successRate) * config.epsBoost;
    runtime.epsilonCurrent = clamp(nextEps, config.epsMin, 0.98);
  } else if (config.strategy === 'softmax') {
    runtime.temperatureCurrent = Math.max(config.tempMin, runtime.temperatureCurrent * config.tempDecay);
  } else if (config.strategy === 'noisy') {
    runtime.noiseSigmaCurrent = Math.max(config.noisySigmaMin, runtime.noiseSigmaCurrent * config.noisySigmaDecay);
  }
}

// ============================================================
// AGENT
// ============================================================

class Agent {
  constructor() {
    this.reset();
  }

  reset() {
    this.x = start.x;
    this.y = start.y;
    this.r = 4;
    this.dead = false;
    this.reached = false;
    this.steps = 0;
    this.totalReward = 0;
    this.traces = new Traces();
  }

  draw(ctx) {
    ctx.beginPath();
    ctx.fillStyle = this.reached ? '#22c55e' : (this.dead ? '#ef4444' : '#f8fafc');
    ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ============================================================
// ŚRODOWISKO
// ============================================================

const mazeCanvas = $('mazeCanvas');
const historyCanvas = $('historyCanvas');
const ctx = mazeCanvas.getContext('2d');
const histCtx = historyCanvas.getContext('2d');

function createAgents() {
  agents = Array.from({ length: config.popSize }, () => new Agent());
}

function clearKnowledge() {
  Q_A.clear();
  Q_B.clear();
  visitCounts.clear();
}

function resetEpisodesOnly() {
  runtime.episode = 0;
  runtime.bestSteps = null;
  runtime.lastSuccessRate = 0;
  runtime.lastAvgReward = 0;
  runtime.history.length = 0;
  resetExplorationSchedulers();
  createAgents();
  updateStatus();
  drawHistoryChart();
}

function resetKnowledgeAndEpisodes() {
  clearKnowledge();
  resetEpisodesOnly();
}

function collides(x, y, r) {
  for (const w of walls) {
    const nx = Math.max(w.x, Math.min(x, w.x + w.w));
    const ny = Math.max(w.y, Math.min(y, w.y + w.h));
    const dx = x - nx;
    const dy = y - ny;
    if (dx * dx + dy * dy <= r * r) return true;
  }
  if (x < 24 || x > W - 24 || y < 24 || y > H - 24) return true;
  return false;
}

function atGoal(x, y) {
  const dx = x - goal.x;
  const dy = y - goal.y;
  return dx * dx + dy * dy <= goal.r * goal.r;
}

function toCell(x, y) {
  const gx = Math.max(20, Math.min(x, W - 21));
  const gy = Math.max(20, Math.min(y, H - 21));
  const cx = Math.floor((gx - 20) / config.cellSize);
  const cy = Math.floor((gy - 20) / config.cellSize);
  return { cx, cy, key: cx + ',' + cy };
}

function getStateRep(x, y) {
  const cell = toCell(x, y);
  return { key: cell.key, countKey: cell.key };
}

function endEpisode() {
  const reached = agents.filter(a => a.reached).length;
  const avgReward = agents.reduce((sum, a) => sum + a.totalReward, 0) / agents.length;
  const successRate = reached / agents.length;
  const meta = getExplorationMeta();

  runtime.lastSuccessRate = successRate;
  runtime.lastAvgReward = avgReward;
  runtime.history.push({
    success: successRate,
    reward: avgReward,
    explorationNorm: meta.norm
  });
  if (runtime.history.length > 160) runtime.history.shift();

  runtime.episode += 1;
  advanceExplorationSchedule(successRate);
  for (const agent of agents) agent.reset();
  updateStatus();
  drawHistoryChart();
}

// ============================================================
// RYSOWANIE
// ============================================================

function drawMaze() {
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, '#0c172b');
  grad.addColorStop(1, '#08121f');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);

  // Siatka
  ctx.strokeStyle = 'rgba(95, 146, 255, 0.12)';
  ctx.lineWidth = 1;
  for (let x = 20; x <= W - 20; x += config.cellSize) {
    ctx.beginPath();
    ctx.moveTo(x, 20);
    ctx.lineTo(x, H - 20);
    ctx.stroke();
  }
  for (let y = 20; y <= H - 20; y += config.cellSize) {
    ctx.beginPath();
    ctx.moveTo(20, y);
    ctx.lineTo(W - 20, y);
    ctx.stroke();
  }

  // Ściany
  ctx.fillStyle = 'rgb(105, 110, 119)';
  for (const w of walls) {
    ctx.fillRect(w.x, w.y, w.w, w.h);
  }

  // Cel
  ctx.beginPath();
  ctx.fillStyle = '#22c55e';
  ctx.arc(goal.x, goal.y, goal.r, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(34,197,94,0.55)';
  ctx.lineWidth = 2;
  ctx.arc(goal.x, goal.y, goal.r + 9, 0, Math.PI * 2);
  ctx.stroke();

  // Start
  ctx.beginPath();
  ctx.fillStyle = '#38bdf8';
  ctx.arc(start.x, start.y, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(56,189,248,0.5)';
  ctx.arc(start.x, start.y, 15, 0, Math.PI * 2);
  ctx.stroke();

  drawQHeatmap();
  drawBrightGridLines();
  function drawQHeatmap() {
    const cellW = config.cellSize;
    const cellH = cellW;

    // Limit do N stanów dla optymalizacji — wybieramy top-N według najwyższego Q (maxVal)
    const maxStates = 330;

    // Zbierz unikalne klucze z obu map Q_A i Q_B (żeby nie pomijać stanów istniejących tylko w Q_B)
    const keys = new Set();
    for (const k of Q_A.keys()) keys.add(k);
    for (const k of Q_B.keys()) keys.add(k);

    // Zmapuj klucze na ich maksymalną wartość (uśrednione jeśli doubleQ)
    const keyed = [];
    for (const key of keys) {
      const vals = config.doubleQ ? getAveragedQ(key) : (Q_A.get(key) || null);
      if (!vals) continue;
      const maxVal = maxArray(vals);
      // pomiń praktycznie zerowe wartości
      if (maxVal <= 0.01) continue;
      keyed.push({ key, maxVal });
    }

    // sortuj malejąco i weź top N
    keyed.sort((a, b) => b.maxVal - a.maxVal);
    const top = keyed.slice(0, maxStates);

    for (const item of top) {
      const key = item.key;
      const maxVal = item.maxVal;
      const [cx, cy] = key.split(',').map(Number);
      if (isNaN(cx) || isNaN(cy)) continue;

      const px = 20 + cx * cellW + cellW / 2;
      const py = 20 + cy * cellH + cellH / 2;

      const intensity = Math.min(1, maxVal / 15);

      // Cyfry zawsze rysuj (co klatkę) jeśli wystarczająco duże
      if (config.cellSize >= 16 && intensity > 0.05) {
        ctx.fillStyle = `rgba(128, 128, 255, ${0.1 + intensity * 0.5})`;
        ctx.font = `${Math.max(9, config.cellSize * 0.38)}px monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(maxVal.toFixed(1), px, py);
      }
    }
  }
}




function drawBrightGridLines() {
  const cellW = config.cellSize;
  const cellH = cellW;

  // Znajdź maksymalną liczbę wizyt do normalizacji
  let maxVisits = 1;
  for (const counts of visitCounts.values()) {
    const total = counts[0] + counts[1] + counts[2] + counts[3];
    if (total > maxVisits) maxVisits = total;
  }

  // Rysuj jaśniejsze ramki dla każdej odwiedzonej komórki
  for (const [key, counts] of visitCounts.entries()) {
    const [cx, cy] = key.split(',').map(Number);
    if (isNaN(cx) || isNaN(cy)) continue;

    const totalVisits = counts[0] + counts[1] + counts[2] + counts[3];
    // Pomiń tylko jeśli wcale nie odwiedzono
    if (totalVisits < 1) continue;

    // Normalizuj liczbę wizyt do zakresu 0-1
    const rawNorm = Math.min(totalVisits / maxVisits, 1);

    // Delikatne podświetlenie już przy pierwszej wizycie — bez pełnej bieli
    const minNormVisible = 0.18;
    const norm = Math.max(rawNorm, minNormVisible);

    // Interpolacja od oryginalnego koloru siatki (95,146,255,0.12)
    // do pełnego niebieskiego (0,0,255,0.95)
    const start = { r: 35, g: 36, b: 75, a: 0.2 };
    const end = { r: 50, g: 50, b: 190, a: 0.90 };

    const r = Math.round(start.r + (end.r - start.r) * norm);
    const g = Math.round(start.g + (end.g - start.g) * norm);
    const b = Math.round(start.b + (end.b - start.b) * norm);
    const alpha = start.a + (end.a - start.a) * norm;

    ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`;
    ctx.lineWidth = 1 + norm * 1.2; // lekko grubsze linie dla widoczności

    // Rysuj ramkę komórki (lekki offset dla bardziej wyraźnych linii)
    const x = 20 + cx * cellW;
    const y = 20 + cy * cellH;
    ctx.beginPath();
    ctx.rect(x + 0.5, y + 0.5, cellW - 1, cellH - 1);
    ctx.stroke();
  }
}

function drawQValues() {
  const cellW = config.cellSize;
  const cellH = cellW;
  const maxStates = 330;

  // Zbierz unikalne klucze z obu map Q
  const keys = new Set();
  for (const k of Q_A.keys()) keys.add(k);
  for (const k of Q_B.keys()) keys.add(k);

  // Zmapuj klucze na ich maksymalną wartość Q
  const keyed = [];
  for (const key of keys) {
    const vals = config.doubleQ ? getAveragedQ(key) : ensureQ(Q_A, key);
    const maxVal = maxArray(vals);
    if (maxVal <= 0.01) continue;
    keyed.push({ key, maxVal });
  }

  // Sortuj malejąco po Q, weź top N
  keyed.sort((a, b) => b.maxVal - a.maxVal);
  const top = keyed.slice(0, maxStates);

  for (const item of top) {
    const [cx, cy] = item.key.split(',').map(Number);
    if (isNaN(cx) || isNaN(cy)) continue;

    const x = 20 + cx * cellW + cellW / 2;
    const y = 20 + cy * cellH + cellH / 2;
    const intensity = Math.min(1, item.maxVal / 15);

    if (config.cellSize >= 16 && intensity > 0.05) {
      ctx.fillStyle = `rgba(128, 128, 255, ${0.1 + intensity * 0.5})`;
      ctx.font = `${Math.max(9, config.cellSize * 0.38)}px monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(item.maxVal.toFixed(1), x, y);
    }
  }
}

function drawHistoryChart() {
  const w = historyCanvas.width;
  const h = historyCanvas.height;
  histCtx.clearRect(0, 0, w, h);
  histCtx.fillStyle = '#0a1425';
  histCtx.fillRect(0, 0, w, h);

  const pad = 26;
  histCtx.strokeStyle = 'rgba(114, 151, 229, 0.18)';
  histCtx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad + ((h - pad * 2) * i) / 4;
    histCtx.beginPath();
    histCtx.moveTo(pad, y);
    histCtx.lineTo(w - pad, y);
    histCtx.stroke();
  }

  histCtx.fillStyle = '#dbeafe';
  histCtx.font = '12px Inter, sans-serif';
  histCtx.textAlign = 'left';
  histCtx.fillText('Historia epizodów', pad, 16);

  const data = runtime.history.slice(-120);
  if (!data.length) {
    histCtx.fillStyle = 'rgba(203, 220, 248, 0.75)';
    histCtx.fillText('Brak ukończonych epizodów — uruchom symulację i obserwuj skuteczność, nagrodę oraz siłę eksploracji.', pad, h / 2);
    return;
  }

  const rewardMin = Math.min(...data.map(d => d.reward));
  const rewardMax = Math.max(...data.map(d => d.reward));
  const rewardRange = Math.max(1e-6, rewardMax - rewardMin);

  const xAt = (i) => pad + (data.length === 1 ? 0 : (i / (data.length - 1)) * (w - pad * 2));
  const yFromNorm = (norm) => h - pad - norm * (h - pad * 2);

  function drawLine(values, color, lineWidth) {
    histCtx.beginPath();
    histCtx.lineWidth = lineWidth || 2;
    histCtx.strokeStyle = color;
    values.forEach((v, i) => {
      const x = xAt(i);
      const y = yFromNorm(clamp(v, 0, 1));
      if (i === 0) histCtx.moveTo(x, y);
      else histCtx.lineTo(x, y);
    });
    histCtx.stroke();
  }

  drawLine(data.map(d => d.success), '#22c55e');
  drawLine(data.map(d => (d.reward - rewardMin) / rewardRange), '#f59e0b');
  drawLine(data.map(d => d.explorationNorm), '#38bdf8');

  histCtx.fillStyle = '#7d89a4';
  histCtx.font = '9px Inter, sans-serif';
  histCtx.textAlign = 'right';
  histCtx.fillText('0%', pad - 4, h - pad + 3);
  histCtx.fillText('50%', pad - 4, (pad + yFromNorm(0.5)) / 2 + 3);
  histCtx.fillText('100%', pad - 4, pad + 3);
}

// ============================================================
// KROK SYMULACJI
// ============================================================

function stepSimulation() {
  for (const ag of agents) {
    if (ag.dead || ag.reached) continue;

    const sRep = getStateRep(ag.x, ag.y);
    const decision = pickAction(sRep);
    const greedyChosen = isGreedyAction(decision.baseValues, decision.action);

    // Zliczaj wizyty zawsze — potrzebne do rysowania podświetlenia siatki
    const counts = ensureCounts(sRep.countKey);
    counts[decision.action] += 1;

    const move = ACTIONS[decision.action];
    const nx = ag.x + move.dx * config.speed;
    const ny = ag.y + move.dy * config.speed;

    let reward = PENALTY_STEP;
    let terminal = false;
    const distBefore = Math.hypot(ag.x - goal.x, ag.y - goal.y);

    if (collides(nx, ny, ag.r)) {
      reward = PENALTY_WALL;
      terminal = true;
      ag.dead = true;
    } else {
      ag.x = nx;
      ag.y = ny;
      const distAfter = Math.hypot(ag.x - goal.x, ag.y - goal.y);
      // Potential Based Reward Shaping
      reward += (distBefore - distAfter) * SHAPING_FACTOR;

      if (atGoal(ag.x, ag.y)) {
        reward = REWARD_GOAL;
        terminal = true;
        ag.reached = true;
      }
    }

    ag.steps += 1;
    if (ag.steps >= STEP_LIMIT && !ag.reached && !terminal) {
      reward += PENALTY_TIMEOUT;
      terminal = true;
      ag.dead = true;
    }

    const nextRep = getStateRep(ag.x, ag.y);

    if (config.tracesEnabled) {
      tracesUpdate(ag, sRep.key, decision.action, reward, nextRep.key, terminal, isGreedyAction(decision.baseValues, decision.action));
    } else {
      qUpdate(sRep.key, decision.action, reward, nextRep.key, terminal);
    }

    ag.totalReward += reward;
    if (ag.reached && (runtime.bestSteps == null || ag.steps < runtime.bestSteps)) {
      runtime.bestSteps = ag.steps;
    }
  }

  if (agents.every(a => a.dead || a.reached)) {
    endEpisode();
  }
}

function renderFrame() {
  // FPS capping na 60 fps
  const now = performance.now();
  const frameInterval = 1000 / 60; // 60 fps target
  if (now - runtime.lastFrameTime < frameInterval) {
    requestAnimationFrame(renderFrame);
    return;
  }
  runtime.lastFrameTime = now;
  runtime.frameCounter++;

  drawMaze();
  if (!runtime.paused) stepSimulation();
  for (const ag of agents) ag.draw(ctx);
  updateStatus();
  requestAnimationFrame(renderFrame);
}

// ============================================================
// UI UPDATES
// ============================================================

function updateStatus() {
  const learningMode = config.doubleQ ? 'Double Q (Qᴬ+Qᴮ)' : 'Standard Q';
  const estLabel = config.doubleQ ? '2 estymatory' : '1 estymator';

  $('episodeVal').textContent = runtime.episode;
  $('populationVal').textContent = config.popSize;
  $('stateBadge').textContent = learningMode;
  $('strategyBadge').textContent = strategyLabel(config.strategy);
  $('exploreVal').textContent = getExplorationMeta().text;
  $('bestVal').textContent = runtime.bestSteps == null ? '—' : runtime.bestSteps + ' kroków';
  $('successVal').textContent = Math.round(runtime.lastSuccessRate * 100) + '%';
  $('avgRewardVal').textContent = runtime.lastAvgReward.toFixed(2);
  $('doubleQLabel').textContent = config.doubleQ ? 'Włączone (' + estLabel + ')' : 'Wyłączone (' + estLabel + ')';
  $('knowledgeVal1').textContent = Q_A.size;
  $('knowledgeVal2').textContent = Q_B.size;
  $('knowledgeVal').textContent = (Q_A.size + Q_B.size);
  $('pauseBtn').textContent = runtime.paused ? 'Wznów' : 'Pauza';

  //        $('liveHint').textContent =
  //        'Tryb: ' + (config.doubleQ ? 'Double Q (2 est.)' : 'Standard Q (1 est.)') +
  //        ' • Strategia: ' + strategyLabel(config.strategy) +
  //          ' • ' + getExplorationMeta().text;
}

function renderPanels() {
  document.querySelectorAll('.strategy-panel').forEach(panel => {
    panel.classList.toggle('hidden-panel', panel.dataset.strategy !== config.strategy);
  });
}

function refreshControlOutputs() {
  $('cellSize').value = config.cellSize;
  $('cellSizeOut').textContent = config.cellSize + ' px';

  $('alpha').value = Math.round(config.alpha * 100);
  $('alphaOut').textContent = config.alpha.toFixed(2);

  $('gamma').value = Math.round(config.gamma * 1000);
  $('gammaOut').textContent = config.gamma.toFixed(3);

  $('speed').value = Math.round(config.speed * 10);
  $('speedOut').textContent = config.speed.toFixed(1);

  $('strategy').value = config.strategy;
  $('tracesEnabled').checked = config.tracesEnabled;
  $('doubleQ').checked = config.doubleQ;

  $('epsStart').value = Math.round(config.epsStart * 100);
  $('epsStartOut').textContent = config.epsStart.toFixed(2);
  $('epsMin').value = Math.round(config.epsMin * 100);
  $('epsMinOut').textContent = config.epsMin.toFixed(2);
  $('epsDecay').value = Math.round(config.epsDecay * 1000);
  $('epsDecayOut').textContent = config.epsDecay.toFixed(3);
  $('epsBoost').value = Math.round(config.epsBoost * 100);
  $('epsBoostOut').textContent = config.epsBoost.toFixed(2);

  $('epsFixed').value = Math.round(config.epsFixed * 100);
  $('epsFixedOut').textContent = config.epsFixed.toFixed(2);

  $('epsDecayType').value = config.epsDecayType;

  $('tempStart').value = Math.round(config.tempStart * 100);
  $('tempStartOut').textContent = config.tempStart.toFixed(2);
  $('tempMin').value = Math.round(config.tempMin * 100);
  $('tempMinOut').textContent = config.tempMin.toFixed(2);
  $('tempDecay').value = Math.round(config.tempDecay * 1000);
  $('tempDecayOut').textContent = config.tempDecay.toFixed(3);

  $('optimisticInit').value = Math.round(config.optimisticInit * 10);
  $('optimisticInitOut').textContent = config.optimisticInit.toFixed(1);

  $('ucbC').value = Math.round(config.ucbC * 100);
  $('ucbCOut').textContent = config.ucbC.toFixed(2);

  $('hybridUcbC').value = Math.round(config.hybridUcbC * 100);
  $('hybridUcbCOut').textContent = config.hybridUcbC.toFixed(2);
  $('hybridEps').value = Math.round(config.hybridEps * 100);
  $('hybridEpsOut').textContent = config.hybridEps.toFixed(2);

  $('noisySigma').value = Math.round(config.noisySigma * 100);
  $('noisySigmaOut').textContent = config.noisySigma.toFixed(2);
  $('noisySigmaMin').value = Math.round(config.noisySigmaMin * 100);
  $('noisySigmaMinOut').textContent = config.noisySigmaMin.toFixed(2);
  $('noisySigmaDecay').value = Math.round(config.noisySigmaDecay * 1000);
  $('noisySigmaDecayOut').textContent = config.noisySigmaDecay.toFixed(3);

  $('traceLambda').value = Math.round(config.traceLambda * 100);
  $('traceLambdaOut').textContent = config.traceLambda.toFixed(2);

  // Population / number of agents
  if ($('popSize')) {
    $('popSize').value = config.popSize;
    $('popSizeOut').textContent = config.popSize;
  }
}

// ============================================================
// BINDING KONTROLEK
// ============================================================

function bindControls() {
  $('doubleQ').addEventListener('change', (e) => {
    config.doubleQ = e.target.checked;
    resetKnowledgeAndEpisodes();
  });

  $('cellSize').addEventListener('input', (e) => {
    config.cellSize = +e.target.value;
    $('cellSizeOut').textContent = config.cellSize + ' px';
    resetKnowledgeAndEpisodes();
  });

  $('alpha').addEventListener('input', (e) => {
    config.alpha = +e.target.value / 100;
    $('alphaOut').textContent = config.alpha.toFixed(2);
    updateStatus();
  });

  $('gamma').addEventListener('input', (e) => {
    config.gamma = +e.target.value / 1000;
    $('gammaOut').textContent = config.gamma.toFixed(3);
    updateStatus();
  });

  $('speed').addEventListener('input', (e) => {
    config.speed = +e.target.value / 10;
    $('speedOut').textContent = config.speed.toFixed(1);
    updateStatus();
  });

  $('strategy').addEventListener('change', (e) => {
    const prev = config.strategy;
    config.strategy = e.target.value;
    renderPanels();
    if (prev === 'optimistic' || config.strategy === 'optimistic') {
      resetKnowledgeAndEpisodes();
    } else {
      resetExplorationSchedulers();
      if (config.strategy === 'ucb') visitCounts.clear();
      resetEpisodesOnly();
    }
  });

  // Epsilon controls
  $('epsStart').addEventListener('input', (e) => {
    config.epsStart = +e.target.value / 100;
    runtime.epsilonCurrent = config.epsStart;
    $('epsStartOut').textContent = config.epsStart.toFixed(2);
    updateStatus();
  });

  $('epsMin').addEventListener('input', (e) => {
    config.epsMin = +e.target.value / 100;
    $('epsMinOut').textContent = config.epsMin.toFixed(2);
  });

  $('epsDecay').addEventListener('input', (e) => {
    config.epsDecay = +e.target.value / 1000;
    $('epsDecayOut').textContent = config.epsDecay.toFixed(3);
  });

  $('epsBoost').addEventListener('input', (e) => {
    config.epsBoost = +e.target.value / 100;
    $('epsBoostOut').textContent = config.epsBoost.toFixed(2);
  });

  $('epsFixed').addEventListener('input', (e) => {
    config.epsFixed = +e.target.value / 100;
    $('epsFixedOut').textContent = config.epsFixed.toFixed(2);
    updateStatus();
  });

  $('epsDecayType').addEventListener('change', (e) => {
    config.epsDecayType = e.target.value;
    resetExplorationSchedulers();
    resetEpisodesOnly();
  });

  // Softmax
  $('tempStart').addEventListener('input', (e) => {
    config.tempStart = +e.target.value / 100;
    runtime.temperatureCurrent = config.tempStart;
    $('tempStartOut').textContent = config.tempStart.toFixed(2);
    updateStatus();
  });

  $('tempMin').addEventListener('input', (e) => {
    config.tempMin = +e.target.value / 100;
    $('tempMinOut').textContent = config.tempMin.toFixed(2);
  });

  $('tempDecay').addEventListener('input', (e) => {
    config.tempDecay = +e.target.value / 1000;
    $('tempDecayOut').textContent = config.tempDecay.toFixed(3);
  });

  // Optimistic
  $('optimisticInit').addEventListener('input', (e) => {
    config.optimisticInit = +e.target.value / 10;
    $('optimisticInitOut').textContent = config.optimisticInit.toFixed(1);
    if (config.strategy === 'optimistic') resetKnowledgeAndEpisodes();
  });

  // UCB
  $('ucbC').addEventListener('input', (e) => {
    config.ucbC = +e.target.value / 100;
    $('ucbCOut').textContent = config.ucbC.toFixed(2);
    updateStatus();
  });

  // Hybrid
  $('hybridUcbC').addEventListener('input', (e) => {
    config.hybridUcbC = +e.target.value / 100;
    $('hybridUcbCOut').textContent = config.hybridUcbC.toFixed(2);
    updateStatus();
  });

  $('hybridEps').addEventListener('input', (e) => {
    config.hybridEps = +e.target.value / 100;
    $('hybridEpsOut').textContent = config.hybridEps.toFixed(2);
    updateStatus();
  });

  // Noisy
  $('noisySigma').addEventListener('input', (e) => {
    config.noisySigma = +e.target.value / 100;
    runtime.noiseSigmaCurrent = config.noisySigma;
    $('noisySigmaOut').textContent = config.noisySigma.toFixed(2);
    updateStatus();
  });

  $('noisySigmaMin').addEventListener('input', (e) => {
    config.noisySigmaMin = +e.target.value / 100;
    $('noisySigmaMinOut').textContent = config.noisySigmaMin.toFixed(2);
  });

  $('noisySigmaDecay').addEventListener('input', (e) => {
    config.noisySigmaDecay = +e.target.value / 1000;
    $('noisySigmaDecayOut').textContent = config.noisySigmaDecay.toFixed(3);
  });

  // Traces
  $('tracesEnabled').addEventListener('change', (e) => {
    config.tracesEnabled = e.target.checked;
    resetEpisodesOnly();
  });

  // Population control: changing number of agents requires resetting knowledge/episodes
  if ($('popSize')) {
    $('popSize').addEventListener('input', (e) => {
      const val = +e.target.value;
      // sanitize to integer
      config.popSize = Math.max(1, Math.floor(val));
      $('popSizeOut').textContent = config.popSize;
      // According to UI badge, changing population requires clearing Q and episodes
      resetKnowledgeAndEpisodes();
    });
  }

  $('traceLambda').addEventListener('input', (e) => {
    config.traceLambda = +e.target.value / 100;
    $('traceLambdaOut').textContent = config.traceLambda.toFixed(2);
  });

  // Buttons
  $('restartEpisodes').addEventListener('click', () => {
    resetEpisodesOnly();
  });

  $('pauseBtn').addEventListener('click', () => {
    runtime.paused = !runtime.paused;
    updateStatus();
  });

  $('resetKnowledge').addEventListener('click', () => {
    resetKnowledgeAndEpisodes();
  });
}

// ============================================================
// INICJALIZACJA
// ============================================================

refreshControlOutputs();
renderPanels();
bindControls();
clearKnowledge();
createAgents();
drawHistoryChart();
updateStatus();
renderFrame();
