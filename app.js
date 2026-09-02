/* ═══ app.js — ChestXray AI Diagnostic Tool ═══ */

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

// Replace this URL with your Flask server address when running locally or on Colab+ngrok
const API_URL = null; // e.g. "http://localhost:5000/predict"

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

// ── DOM ───────────────────────────────────────────────────────────────
const dropZone        = document.getElementById('dropZone');
const fileInput       = document.getElementById('fileInput');
const browseBtn       = document.getElementById('browseBtn');
const previewContainer= document.getElementById('previewContainer');
const previewImg      = document.getElementById('previewImg');
const previewMeta     = document.getElementById('previewMeta');
const scanLine        = document.getElementById('scanLine');
const resetBtn        = document.getElementById('resetBtn');
const analyzeBtn      = document.getElementById('analyzeBtn');
const analyzeBtnTxt   = document.getElementById('analyzeBtnTxt');

const emptyState      = document.getElementById('emptyState');
const loadingState    = document.getElementById('loadingState');
const resultsContent  = document.getElementById('resultsContent');

const verdictClass    = document.getElementById('verdictClass');
const verdictConf     = document.getElementById('verdictConf');
const verdictIcon     = document.getElementById('verdictIcon');
const verdictCard     = document.getElementById('verdictCard');
const classBars       = document.getElementById('classBars');
const summaryGrid     = document.getElementById('summaryGrid');

let currentFile  = null;
let currentImage = null; // HTMLImageElement for canvas drawing

// ── OOD Guard config ──────────────────────────────────────────────────
// The model has a closed-set softmax over 4 classes only — it cannot say
// "this isn't an X-ray at all". These guards catch obviously-wrong uploads
// (photos, scanned notes, screenshots) before/after inference.
const CONFIDENCE_THRESHOLD = 0.55; // below this max-prob → flag as uncertain
const ENTROPY_THRESHOLD    = 1.2;  // bits; above this → flag as uncertain (max for 4 classes = 2.0)

// ── Upload Handlers ───────────────────────────────────────────────────
browseBtn.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', (e) => { if (e.target !== browseBtn) fileInput.click(); });

fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault(); dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
});

resetBtn.addEventListener('click', resetAll);

// ── Handle File ───────────────────────────────────────────────────────
function handleFile(file) {
  currentFile = file;
  const url   = URL.createObjectURL(file);
  previewImg.src = url;

  previewImg.onload = () => {
    currentImage = previewImg;
    previewMeta.textContent =
      `${file.name}  |  ${(file.size / 1024).toFixed(1)} KB  |  ${previewImg.naturalWidth}×${previewImg.naturalHeight}px`;
  };

  dropZone.classList.add('hidden');
  previewContainer.classList.remove('hidden');
  analyzeBtn.disabled = false;
  resetState();
}

function resetAll() {
  currentFile  = null;
  currentImage = null;
  fileInput.value = '';
  previewImg.src  = '';
  dropZone.classList.remove('hidden');
  previewContainer.classList.add('hidden');
  analyzeBtn.disabled = true;
  analyzeBtnTxt.textContent = '⚡ Analyze X-Ray';
  resetState();
}

function resetState() {
  show(emptyState);
  hide(loadingState);
  hide(resultsContent);
  scanLine.classList.remove('active');
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

  const looksGray    = avgSat < 0.08 && grayFrac > 0.85;
  const notBlankPage = meanLum > 15 && meanLum < 245;

  return looksGray && notBlankPage;
}

// ── Analyze ───────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
  if (!currentFile) return;

  if (!isLikelyXray(currentImage)) {
    alert(
      "⚠️ This doesn't look like a grayscale chest X-ray.\n\n" +
      "This model is only trained to classify chest X-rays into COVID-19 / " +
      "Normal / Pneumonia / Tuberculosis, and will produce meaningless " +
      "results on other images (photos, scanned notes, screenshots, etc).\n\n" +
      "Please upload a chest X-ray image."
    );
    return;
  }

  analyzeBtn.disabled  = true;
  analyzeBtnTxt.textContent = '⏳ Analysing…';
  scanLine.classList.add('active');

  hide(emptyState);
  show(loadingState);
  hide(resultsContent);

  // Animate loading steps
  const steps = ['step1','step2','step3','step4'];
  for (let i = 0; i < steps.length; i++) {
    await sleep(400);
    document.getElementById(steps[i]).classList.add('active');
  }

  let predictions;

  if (tfModel) {
    // ── Real model via local TF.js
    try {
      predictions = await tfPredict();
    } catch (e) {
      console.error("TF.js prediction failed:", e);
      alert('❌ TF.js Model inference failed.\n' + e.message + '\n\nShowing demo predictions instead.');
      predictions = demoPredict();
    }
  } else if (API_URL) {
    // ── Real model via Flask API
    try {
      const form = new FormData();
      form.append('file', currentFile);
      const res  = await fetch(API_URL, { method: 'POST', body: form });
      const data = await res.json();
      predictions = data.predictions; // { COVID19: 0.02, Normal: 0.93, ... }
    } catch (err) {
      alert('❌ Could not connect to Flask API.\n' + err.message + '\n\nShowing demo predictions instead.');
      predictions = demoPredict();
    }
  } else {
    // ── Demo mode: simulate realistic prediction
    await sleep(800);
    predictions = demoPredict();
  }

  await sleep(300);

  // ── Confidence/entropy gate on the model's own output ────────────────
  // Only applied to real model output (TF.js/API), not the fabricated demo
  // predictions, since demo mode isn't looking at pixels in the first place.
  if (tfModel || API_URL) {
    const vals    = Object.values(predictions);
    const maxProb = Math.max(...vals);
    const entropy = -vals.filter(p => p > 0).reduce((s, p) => s + p * Math.log2(p), 0);
    if (maxProb < CONFIDENCE_THRESHOLD || entropy > ENTROPY_THRESHOLD) {
      hide(loadingState);
      show(resultsContent);
      verdictClass.textContent = 'Uncertain';
      verdictConf.textContent  = `Low confidence (${(maxProb * 100).toFixed(1)}%) — likely not a recognized X-ray pattern`;
      verdictIcon.textContent  = '❓';
      verdictCard.style.borderLeftColor = '#B0BEC5';
      classBars.innerHTML   = '';
      summaryGrid.innerHTML = '';
      analyzeBtnTxt.textContent = '⚡ Analyze X-Ray';
      analyzeBtn.disabled = false;
      scanLine.classList.remove('active');
      return;
    }
  }

  displayResults(predictions);

  analyzeBtnTxt.textContent = '⚡ Analyze X-Ray';
  analyzeBtn.disabled = false;
  scanLine.classList.remove('active');
});

// ── TF.js Local Predictor ─────────────────────────────────────────────
async function tfPredict() {
  const probs = tf.tidy(() => {
    let tensor = tf.browser.fromPixels(currentImage);
    tensor = tf.image.resizeBilinear(tensor, [224, 224]);
    // Standard ImageNet / DenseNet scaling: (x / 127.5) - 1.0
    tensor = tensor.cast('float32').div(127.5).sub(1.0);
    tensor = tensor.expandDims(0);
    return tfModel.predict(tensor).dataSync();
  });

  const out = {};
  for(let i=0; i<CLASS_NAMES.length; i++) {
    out[CLASS_NAMES[i]] = probs[i];
  }
  return out;
}

// ── Demo Predictor ────────────────────────────────────────────────────
function demoPredict() {
  // Randomly pick a "dominant" class with high probability
  const dominant = CLASS_NAMES[Math.floor(Math.random() * CLASS_NAMES.length)];
  const raw = {};
  CLASS_NAMES.forEach(c => {
    raw[c] = c === dominant
      ? 0.70 + Math.random() * 0.25      // 70–95% for dominant
      : Math.random() * 0.15;             // small noise for others
  });
  // Softmax normalise
  const total = Object.values(raw).reduce((a, b) => a + b, 0);
  const out = {};
  CLASS_NAMES.forEach(c => { out[c] = raw[c] / total; });
  return out;
}

// ── Display Results ───────────────────────────────────────────────────
function displayResults(predictions) {
  hide(loadingState);
  show(resultsContent);

  // Find top class
  const sorted    = Object.entries(predictions).sort((a, b) => b[1] - a[1]);
  const topClass  = sorted[0][0];
  const topConf   = sorted[0][1];

  // Verdict
  verdictClass.textContent = topClass;
  verdictConf.textContent  = `Confidence: ${(topConf * 100).toFixed(2)}%`;
  verdictIcon.textContent  = CLASS_ICONS[topClass];
  verdictCard.style.setProperty('--card-color', CLASS_COLORS[topClass]);
  verdictCard.querySelector('::before');
  // colour the left border
  verdictCard.style.borderLeftColor = CLASS_COLORS[topClass];

  // Class bars
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

  // Animate bars after render
  requestAnimationFrame(() => {
    document.querySelectorAll('.bar-fill').forEach(el => {
      el.style.width = el.dataset.pct + '%';
    });
  });

  // Summary chips
  const entropy = -Object.values(predictions)
    .filter(p => p > 0)
    .reduce((sum, p) => sum + p * Math.log2(p), 0);

  summaryGrid.innerHTML = `
    <div class="summary-chip"><div class="s-val">${(topConf*100).toFixed(1)}%</div><div class="s-lbl">Top Confidence</div></div>
    <div class="summary-chip"><div class="s-val">${(entropy).toFixed(2)}</div><div class="s-lbl">Entropy (bits)</div></div>
    <div class="summary-chip"><div class="s-val">${CLASS_NAMES.length}</div><div class="s-lbl">Classes Checked</div></div>
  `;

  // Draw Grad-CAM canvases (simulated heatmap)
  drawGradCAM(topClass);
}

// ── Grad-CAM (Simulated) ──────────────────────────────────────────────
function drawGradCAM(topClass) {
  const W = 224, H = 224;
  const orig  = document.getElementById('origCanvas');
  const heat  = document.getElementById('heatCanvas');
  const blend = document.getElementById('blendCanvas');

  [orig, heat, blend].forEach(c => { c.width = W; c.height = H; });

  // Draw original image
  const ctxO = orig.getContext('2d');
  if (currentImage) ctxO.drawImage(currentImage, 0, 0, W, H);
  else { ctxO.fillStyle = '#1a2235'; ctxO.fillRect(0,0,W,H); }

  // Generate simulated Gaussian heatmap focused on lung region
  const imgData = heat.getContext('2d').createImageData(W, H);
  const cx = W * 0.5 + (Math.random() - .5) * 40;
  const cy = H * 0.52 + (Math.random() - .5) * 30;
  const sig = 55 + Math.random() * 25;

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const d2  = ((x-cx)**2 + (y-cy)**2) / (2 * sig**2);
      const v   = Math.exp(-d2);   // 0-1, peak at centre
      const rgb = jetColormap(v);
      const idx = (y * W + x) * 4;
      imgData.data[idx]   = rgb[0];
      imgData.data[idx+1] = rgb[1];
      imgData.data[idx+2] = rgb[2];
      imgData.data[idx+3] = Math.round(v * 255);
    }
  }
  heat.getContext('2d').putImageData(imgData, 0, 0);

  // Blend original + heatmap (alpha = 0.45)
  const ctxB = blend.getContext('2d');
  if (currentImage) ctxB.drawImage(currentImage, 0, 0, W, H);
  ctxB.globalAlpha = 0.5;
  ctxB.drawImage(heat, 0, 0, W, H);
  ctxB.globalAlpha = 1.0;
}

// Jet colormap: value 0-1 → [R,G,B]
function jetColormap(v) {
  const t  = v * 3;
  const r  = Math.round(255 * Math.min(Math.max(Math.min(t - 1.5, 4.5 - t), 0), 1));
  const g  = Math.round(255 * Math.min(Math.max(Math.min(t - 0.5, 3.5 - t), 0), 1));
  const b  = Math.round(255 * Math.min(Math.max(Math.min(t + 0.5, 2.5 - t), 0), 1));
  // Classic jet: blue → cyan → green → yellow → red
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
