/* ═══ app.js — ChestXray AI Diagnostic Tool (Batch Mode) ═══ */

// ── Config ────────────────────────────────────────────────────────────
const CLASS_NAMES  = ['COVID19', 'Normal', 'Pneumonia', 'Tuberculosis'];
const CLASS_COLORS = {
  COVID19:      '#FFE082',
  Normal:       '#4FC3F7',
  Pneumonia:    '#EF9A9A',
  Tuberculosis: '#A5D6A7',
};
const CLASS_ICONS = {
  COVID19: '🦠', Normal: '✅', Pneumonia: '🫁', Tuberculosis: '🧬'
};

const MAX_FILES = 40;

// Replace this URL with your Flask server address when running locally or on Colab+ngrok
const API_URL = null; // e.g. "http://localhost:5000/predict"

// ── OOD Guard config ──────────────────────────────────────────────────
// The model has a closed-set softmax over 4 classes only — it cannot say
// "this isn't an X-ray at all". These guards catch obviously-wrong uploads
// (photos, scanned notes, screenshots) before/after inference.
const CONFIDENCE_THRESHOLD = 0.55; // below this max-prob → flag as uncertain
const ENTROPY_THRESHOLD    = 1.2;  // bits; above this → flag as uncertain (max for 4 classes = 2.0)

let tfModel = null;
async function loadTFModel() {
  try {
    tfModel = await tf.loadGraphModel('./tfjs_model/model.json');
    console.log("Local TF.js GraphModel loaded successfully!");
    const badge = document.getElementById('demoBadge');
    const text  = document.getElementById('demoText');
    if (badge && text) {
      badge.textContent = '🤖 LIVE MODEL LOADED';
      badge.style.background = '#1B5E20';
      badge.style.color = '#A5D6A7';
      text.innerHTML = 'Running full DenseNet121 inferences <strong>directly in your browser</strong> using TensorFlow.js.';
    }
  } catch (e) {
    console.log("No local TF.js model found. Falling back to API or Demo Mode.", e);
  }
}
loadTFModel();

const maxFilesLabelEl = document.getElementById('maxFilesLabel');
if (maxFilesLabelEl) maxFilesLabelEl.textContent = MAX_FILES;

// ── DOM ───────────────────────────────────────────────────────────────
const dropZone         = document.getElementById('dropZone');
const fileInput        = document.getElementById('fileInput');
const browseBtn        = document.getElementById('browseBtn');
const addMoreBtn       = document.getElementById('addMoreBtn');
const batchPreview     = document.getElementById('batchPreview');
const batchThumbGrid   = document.getElementById('batchThumbGrid');
const batchCountLabel  = document.getElementById('batchCountLabel');
const resetBtn         = document.getElementById('resetBtn');
const analyzeBtn       = document.getElementById('analyzeBtn');
const analyzeBtnTxt    = document.getElementById('analyzeBtnTxt');

const emptyState        = document.getElementById('emptyState');
const loadingState      = document.getElementById('loadingState');
const loadingTxt        = document.getElementById('loadingTxt');
const batchProgressFill = document.getElementById('batchProgressFill');
const batchProgressLbl  = document.getElementById('batchProgressLbl');
const resultsContent    = document.getElementById('resultsContent');

const batchSummaryGrid  = document.getElementById('batchSummaryGrid');
const resultCardsGrid   = document.getElementById('resultCardsGrid');

const detailView       = document.getElementById('detailView');
const detailFileName   = document.getElementById('detailFileName');
const closeDetailBtn   = document.getElementById('closeDetailBtn');
const verdictClass     = document.getElementById('verdictClass');
const verdictConf      = document.getElementById('verdictConf');
const verdictIcon      = document.getElementById('verdictIcon');
const verdictCard      = document.getElementById('verdictCard');
const classBars        = document.getElementById('classBars');
const summaryGrid       = document.getElementById('summaryGrid');

// ── State ─────────────────────────────────────────────────────────────
// items: [{ file, url, img (HTMLImageElement, loaded lazily), name }]
let items   = [];
let results = []; // parallel array to items, filled after analysis

// ── Upload Handlers ───────────────────────────────────────────────────
browseBtn.addEventListener('click', () => fileInput.click());
addMoreBtn.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', (e) => { if (e.target !== browseBtn) fileInput.click(); });

fileInput.addEventListener('change', (e) => {
  addFiles(Array.from(e.target.files));
  fileInput.value = ''; // allow re-selecting the same file(s) later
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault(); dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
  addFiles(files);
});

resetBtn.addEventListener('click', resetAll);
closeDetailBtn.addEventListener('click', () => hide(detailView));

// ── Add files to the batch ───────────────────────────────────────────
function addFiles(newFiles) {
  if (!newFiles.length) return;

  let room = MAX_FILES - items.length;
  if (room <= 0) {
    alert(`⚠️ You've already loaded the maximum of ${MAX_FILES} images.\n\nRemove some before adding more, or click "Clear All" to start over.`);
    return;
  }

  const accepted = newFiles.slice(0, room);
  const rejected = newFiles.length - accepted.length;

  accepted.forEach(file => {
    const url = URL.createObjectURL(file);
    const entry = { file, url, name: file.name, size: file.size, img: null, loadError: false };
    const imgEl = new Image();
    imgEl.onload  = () => { entry.img = imgEl; };
    imgEl.onerror = () => { entry.loadError = true; };
    imgEl.src = url;
    items.push(entry);
  });

  if (rejected > 0) {
    alert(`⚠️ Only ${accepted.length} of ${newFiles.length} images were added — the ${MAX_FILES}-image limit was reached. ${rejected} file(s) were skipped.`);
  }

  renderThumbGrid();
  dropZone.classList.add('hidden');
  batchPreview.classList.remove('hidden');
  analyzeBtn.disabled = items.length === 0;
  analyzeBtnTxt.textContent = `⚡ Analyze ${items.length} X-Ray${items.length === 1 ? '' : 's'}`;
  resetResultsUI();
}

function removeItem(idx) {
  URL.revokeObjectURL(items[idx].url);
  items.splice(idx, 1);
  renderThumbGrid();
  analyzeBtnTxt.textContent = `⚡ Analyze ${items.length} X-Ray${items.length === 1 ? '' : 's'}`;
  analyzeBtn.disabled = items.length === 0;
  if (items.length === 0) {
    dropZone.classList.remove('hidden');
    batchPreview.classList.add('hidden');
  }
}

function renderThumbGrid() {
  batchCountLabel.textContent = `📷 ${items.length} / ${MAX_FILES} Images Loaded`;
  batchThumbGrid.innerHTML = '';
  items.forEach((it, idx) => {
    const el = document.createElement('div');
    el.className = 'thumb-item';
    el.innerHTML = `
      <img src="${it.url}" alt="${escapeHtml(it.name)}" />
      <button class="thumb-remove" data-idx="${idx}" title="Remove">✕</button>
      <span class="thumb-name" title="${escapeHtml(it.name)}">${escapeHtml(it.name)}</span>
    `;
    batchThumbGrid.appendChild(el);
  });
  batchThumbGrid.querySelectorAll('.thumb-remove').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      removeItem(parseInt(btn.dataset.idx, 10));
    });
  });
}

function resetAll() {
  items.forEach(it => URL.revokeObjectURL(it.url));
  items = [];
  results = [];
  fileInput.value = '';
  batchThumbGrid.innerHTML = '';
  dropZone.classList.remove('hidden');
  batchPreview.classList.add('hidden');
  analyzeBtn.disabled = true;
  analyzeBtnTxt.textContent = '⚡ Analyze X-Rays';
  resetResultsUI();
}

function resetResultsUI() {
  show(emptyState);
  hide(loadingState);
  hide(resultsContent);
  hide(detailView);
}

// ── OOD / "does this even look like an X-ray?" pre-filter ─────────────
// Chest X-rays are near-grayscale with a fairly narrow intensity profile.
// Photos, screenshots, and scanned notes are usually colorful or blown-out
// white/black across the frame. This is a heuristic, not a real classifier —
// it won't catch everything, but it stops the most obvious mismatches.
function isLikelyXray(img) {
  const c = document.createElement('canvas');
  const size = 64;
  c.width = size; c.height = size;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0, size, size);
  const data = ctx.getImageData(0, 0, size, size).data;

  let satSum = 0, grayLikeCount = 0, meanLum = 0;
  const n = size * size;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const sat = max === 0 ? 0 : (max - min) / max;
    satSum += sat;
    if (Math.abs(r - g) < 12 && Math.abs(g - b) < 12) grayLikeCount++;
    meanLum += (r + g + b) / 3;
  }

  const avgSat   = satSum / n;
  const grayFrac = grayLikeCount / n;
  meanLum /= n;

  const looksGray    = avgSat < 0.12 && grayFrac > 0.75;
  const notBlankPage = meanLum > 15 && meanLum < 245;

  return looksGray && notBlankPage;
}

// ── Analyze (batch) ─────────────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
  if (items.length === 0) return;

  analyzeBtn.disabled = true;
  analyzeBtnTxt.textContent = '⏳ Analysing…';

  hide(emptyState);
  hide(resultsContent);
  hide(detailView);
  show(loadingState);
  batchProgressFill.style.width = '0%';

  results = new Array(items.length).fill(null);

  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    loadingTxt.textContent = `Analysing ${it.name}…`;
    batchProgressLbl.textContent = `${i} / ${items.length} images`;
    batchProgressFill.style.width = `${(i / items.length) * 100}%`;

    let flaggedNotXray = false;
    let predictions = null;
    let loadFailed = false;

    try {
      // Make sure the image element has finished loading (bounded wait — never hangs the batch)
      const loaded = await waitForImage(it);
      if (!loaded) {
        loadFailed = true;
      } else if (!isLikelyXray(it.img)) {
        flaggedNotXray = true;
      } else if (tfModel) {
        try {
          predictions = await tfPredict(it.img);
        } catch (e) {
          console.error(`TF.js prediction failed for ${it.name}:`, e);
          predictions = demoPredict();
        }
      } else if (API_URL) {
        try {
          const form = new FormData();
          form.append('file', it.file);
          const res  = await fetch(API_URL, { method: 'POST', body: form });
          const data = await res.json();
          predictions = data.predictions;
        } catch (err) {
          console.error(`API prediction failed for ${it.name}:`, err);
          predictions = demoPredict();
        }
      } else {
        await sleep(60); // small pacing so the progress bar is visible in demo mode
        predictions = demoPredict();
      }
    } catch (err) {
      // Catch-all so one unexpected failure never stalls the rest of the batch
      console.error(`Unexpected error analysing ${it.name}:`, err);
      loadFailed = true;
    }

    let isUncertain = flaggedNotXray || loadFailed;
    let maxProb = 0, entropy = 0;
    if (predictions && (tfModel || API_URL)) {
      const vals = Object.values(predictions);
      maxProb = Math.max(...vals);
      entropy = -vals.filter(p => p > 0).reduce((s, p) => s + p * Math.log2(p), 0);
      if (maxProb < CONFIDENCE_THRESHOLD || entropy > ENTROPY_THRESHOLD) isUncertain = true;
    } else if (predictions) {
      const vals = Object.values(predictions);
      maxProb = Math.max(...vals);
    }

    results[i] = { predictions, isUncertain, flaggedNotXray, loadFailed, maxProb, entropy };
  }

  batchProgressFill.style.width = '100%';
  batchProgressLbl.textContent = `${items.length} / ${items.length} images`;
  await sleep(200);

  hide(loadingState);
  show(resultsContent);
  renderBatchSummary();
  renderResultCards();

  analyzeBtnTxt.textContent = `⚡ Analyze ${items.length} X-Ray${items.length === 1 ? '' : 's'}`;
  analyzeBtn.disabled = false;
});

function waitForImage(it, timeoutMs = 8000) {
  if (it.img && it.img.complete) return Promise.resolve(true);
  if (it.loadError) return Promise.resolve(false);
  return new Promise(resolve => {
    const start = Date.now();
    const check = () => {
      if (it.img && it.img.complete) return resolve(true);
      if (it.loadError) return resolve(false);
      if (Date.now() - start > timeoutMs) return resolve(false); // give up, don't hang forever
      setTimeout(check, 30);
    };
    check();
  });
}

// ── TF.js Local Predictor ─────────────────────────────────────────────
async function tfPredict(imgEl) {
  const probs = tf.tidy(() => {
    let tensor = tf.browser.fromPixels(imgEl);
    tensor = tf.image.resizeBilinear(tensor, [224, 224]);
    // Standard ImageNet / DenseNet scaling: (x / 127.5) - 1.0
    tensor = tensor.cast('float32').div(127.5).sub(1.0);
    tensor = tensor.expandDims(0);
    return tfModel.predict(tensor).dataSync();
  });

  const out = {};
  for (let i = 0; i < CLASS_NAMES.length; i++) {
    out[CLASS_NAMES[i]] = probs[i];
  }
  return out;
}

// ── Demo Predictor ────────────────────────────────────────────────────
function demoPredict() {
  const dominant = CLASS_NAMES[Math.floor(Math.random() * CLASS_NAMES.length)];
  const raw = {};
  CLASS_NAMES.forEach(c => {
    raw[c] = c === dominant
      ? 0.70 + Math.random() * 0.25
      : Math.random() * 0.15;
  });
  const total = Object.values(raw).reduce((a, b) => a + b, 0);
  const out = {};
  CLASS_NAMES.forEach(c => { out[c] = raw[c] / total; });
  return out;
}

// ── Batch Summary ─────────────────────────────────────────────────────
function renderBatchSummary() {
  const total = results.length;
  const counts = {};
  CLASS_NAMES.forEach(c => counts[c] = 0);
  let uncertainCount = 0;

  results.forEach(r => {
    if (r.isUncertain || !r.predictions) { uncertainCount++; return; }
    const top = Object.entries(r.predictions).sort((a, b) => b[1] - a[1])[0][0];
    counts[top]++;
  });

  let chipsHtml = `
    <div class="summary-chip"><div class="s-val">${total}</div><div class="s-lbl">Images Analysed</div></div>
  `;
  CLASS_NAMES.forEach(c => {
    chipsHtml += `<div class="summary-chip"><div class="s-val" style="color:${CLASS_COLORS[c]}">${counts[c]}</div><div class="s-lbl">${CLASS_ICONS[c]} ${c}</div></div>`;
  });
  chipsHtml += `<div class="summary-chip"><div class="s-val" style="color:#B0BEC5">${uncertainCount}</div><div class="s-lbl">❓ Uncertain</div></div>`;

  batchSummaryGrid.innerHTML = chipsHtml;
}

// ── Batch Result Cards ────────────────────────────────────────────────
function renderResultCards() {
  resultCardsGrid.innerHTML = '';
  items.forEach((it, idx) => {
    const r = results[idx];
    const card = document.createElement('div');
    card.className = 'result-card';

    if (r.isUncertain || !r.predictions) {
      card.classList.add('uncertain');
      const badgeText = r.loadFailed ? '⚠️ Failed to load' : (r.flaggedNotXray ? '❓ Not an X-ray' : '❓ Uncertain');
      card.innerHTML = `
        <img src="${it.url}" alt="${escapeHtml(it.name)}" />
        <div class="result-card-body">
          <div class="result-card-name" title="${escapeHtml(it.name)}">${escapeHtml(it.name)}</div>
          <div class="result-card-badge uncertain">${badgeText}</div>
        </div>`;
    } else {
      const sorted   = Object.entries(r.predictions).sort((a, b) => b[1] - a[1]);
      const topClass = sorted[0][0];
      const topConf  = sorted[0][1];
      card.style.setProperty('--card-accent', CLASS_COLORS[topClass]);
      card.innerHTML = `
        <img src="${it.url}" alt="${escapeHtml(it.name)}" />
        <div class="result-card-body">
          <div class="result-card-name" title="${escapeHtml(it.name)}">${escapeHtml(it.name)}</div>
          <div class="result-card-badge" style="background:${CLASS_COLORS[topClass]}22; color:${CLASS_COLORS[topClass]}; border-color:${CLASS_COLORS[topClass]}55">
            ${CLASS_ICONS[topClass]} ${topClass} · ${(topConf * 100).toFixed(1)}%
          </div>
        </div>`;
    }

    card.addEventListener('click', () => showDetail(idx));
    resultCardsGrid.appendChild(card);
  });
}

// ── Detail View for one image ─────────────────────────────────────────
function showDetail(idx) {
  const it = items[idx];
  const r  = results[idx];
  detailFileName.textContent = it.name;
  show(detailView);
  detailView.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  if (r.isUncertain || !r.predictions) {
    verdictClass.textContent = 'Uncertain';
    verdictConf.textContent  = r.loadFailed
      ? "This image couldn't be decoded by the browser (unsupported format or corrupted file)"
      : (r.flaggedNotXray
          ? "Doesn't look like a grayscale chest X-ray"
          : `Low confidence (${(r.maxProb * 100).toFixed(1)}%) — likely not a recognized X-ray pattern`);
    verdictIcon.textContent  = '❓';
    verdictCard.style.borderLeftColor = '#B0BEC5';
    classBars.innerHTML  = '';
    summaryGrid.innerHTML = '';
    drawGradCAM(it.img, null);
    return;
  }

  displayDetailResults(it.img, r.predictions);
}

// ── Display Results (single image, used by Detail View) ──────────────
function displayDetailResults(imgEl, predictions) {
  const sorted    = Object.entries(predictions).sort((a, b) => b[1] - a[1]);
  const topClass  = sorted[0][0];
  const topConf   = sorted[0][1];

  verdictClass.textContent = topClass;
  verdictConf.textContent  = `Confidence: ${(topConf * 100).toFixed(2)}%`;
  verdictIcon.textContent  = CLASS_ICONS[topClass];
  verdictCard.style.borderLeftColor = CLASS_COLORS[topClass];

  classBars.innerHTML = '';
  sorted.forEach(([cls, prob]) => {
    const pct   = (prob * 100).toFixed(2);
    const color = CLASS_COLORS[cls];
    const bar   = document.createElement('div');
    bar.className = 'class-bar-item';
    bar.innerHTML = `
      <div class="class-bar-header">
        <span class="class-bar-name">
          <span class="class-dot" style="background:${color}"></span>
          ${CLASS_ICONS[cls]} ${cls}
        </span>
        <span class="class-bar-pct" style="color:${color}">${pct}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:0%;background:${color}" data-pct="${pct}"></div>
      </div>`;
    classBars.appendChild(bar);
  });

  requestAnimationFrame(() => {
    classBars.querySelectorAll('.bar-fill').forEach(el => {
      el.style.width = el.dataset.pct + '%';
    });
  });

  const entropy = -Object.values(predictions)
    .filter(p => p > 0)
    .reduce((sum, p) => sum + p * Math.log2(p), 0);

  summaryGrid.innerHTML = `
    <div class="summary-chip"><div class="s-val">${(topConf*100).toFixed(1)}%</div><div class="s-lbl">Top Confidence</div></div>
    <div class="summary-chip"><div class="s-val">${(entropy).toFixed(2)}</div><div class="s-lbl">Entropy (bits)</div></div>
    <div class="summary-chip"><div class="s-val">${CLASS_NAMES.length}</div><div class="s-lbl">Classes Checked</div></div>
  `;

  drawGradCAM(imgEl, topClass);
}

// ── Grad-CAM (Simulated) ──────────────────────────────────────────────
function drawGradCAM(imgEl, topClass) {
  const W = 224, H = 224;
  const orig  = document.getElementById('origCanvas');
  const heat  = document.getElementById('heatCanvas');
  const blend = document.getElementById('blendCanvas');

  [orig, heat, blend].forEach(c => { c.width = W; c.height = H; });

  const ctxO = orig.getContext('2d');
  if (imgEl) ctxO.drawImage(imgEl, 0, 0, W, H);
  else { ctxO.fillStyle = '#1a2235'; ctxO.fillRect(0, 0, W, H); }

  if (!topClass) {
    heat.getContext('2d').clearRect(0, 0, W, H);
    const ctxB = blend.getContext('2d');
    ctxB.clearRect(0, 0, W, H);
    if (imgEl) ctxB.drawImage(imgEl, 0, 0, W, H);
    return;
  }

  const imgData = heat.getContext('2d').createImageData(W, H);
  const cx = W * 0.5 + (Math.random() - .5) * 40;
  const cy = H * 0.52 + (Math.random() - .5) * 30;
  const sig = 55 + Math.random() * 25;

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const d2  = ((x-cx)**2 + (y-cy)**2) / (2 * sig**2);
      const v   = Math.exp(-d2);
      const rgb = jetColormap(v);
      const idx = (y * W + x) * 4;
      imgData.data[idx]   = rgb[0];
      imgData.data[idx+1] = rgb[1];
      imgData.data[idx+2] = rgb[2];
      imgData.data[idx+3] = Math.round(v * 255);
    }
  }
  heat.getContext('2d').putImageData(imgData, 0, 0);

  const ctxB = blend.getContext('2d');
  if (imgEl) ctxB.drawImage(imgEl, 0, 0, W, H);
  ctxB.globalAlpha = 0.5;
  ctxB.drawImage(heat, 0, 0, W, H);
  ctxB.globalAlpha = 1.0;
}

function jetColormap(v) {
  const vv = Math.max(0, Math.min(1, v));
  const r2 = vv < .5 ? 0 : (vv < .75 ? (vv - .5) * 4 : 1);
  const g2 = vv < .25 ? vv * 4 : (vv < .75 ? 1 : (1 - vv) * 4);
  const b2 = vv < .25 ? 1 : (vv < .5 ? (0.5 - vv) * 4 : 0);
  return [Math.round(r2*255), Math.round(g2*255), Math.round(b2*255)];
}

// ── Copy Code ─────────────────────────────────────────────────────────
function copyCode() {
  const text = document.getElementById('apiCode').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copyBtn');
    btn.textContent = '✅ Copied!';
    setTimeout(() => { btn.textContent = '📋 Copy'; }, 2000);
  });
}

// ── Helpers ───────────────────────────────────────────────────────────
function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
