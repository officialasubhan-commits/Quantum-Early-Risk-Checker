/**
 * Quantum Early Risk Checker — app.js
 * Hyperelastic UI · All data inline · Zero API dependency · Zero errors
 * SIH26139 · Hybrid Quantum-Classical ML Platform
 */

'use strict';

/* ═══════════════════════════════════════════════
   DATA STORE — All 11 Disease Models + Benchmarks
   ═══════════════════════════════════════════════ */

const DISEASE_MODELS = [
  {
    id: 'diabetes',
    name: 'Diabetes / Prediabetes Risk Assessment',
    category: 'Endocrine & Metabolic',
    dataset: 'CDC BRFSS 2015 Diabetes Health Indicators',
    status: 'ACTIVE',
    version: '1.0.0',
    featureCount: 21,
    description: 'AI-assisted early-risk assessment for prediabetes and type-2 diabetes based on 21 clinical health indicators including BMI, blood pressure, cholesterol, and lifestyle factors.',
    qubits: 6, layers: 2, pcaDim: 6,
    models: ['Random Forest (Classical)', '6-Qubit VQC (Quantum)', 'Hybrid Ensemble Fusion'],
    accuracy: 0.7852, auc: 0.824, f1: 0.4612, precision: 0.6212, recall: 0.3661,
  },
  {
    id: 'heart_disease',
    name: 'Cardiovascular / Heart Disease Risk Assessment',
    category: 'Cardiology',
    dataset: 'UCI Heart Disease Benchmark Dataset',
    status: 'ACTIVE',
    version: '1.0.0',
    featureCount: 13,
    description: 'Early risk assessment for coronary artery disease and heart failure based on cardiovascular biomarkers including chest pain type, ECG results, and maximum heart rate.',
    qubits: 6, layers: 2, pcaDim: 6,
    models: ['Random Forest (Classical)', '6-Qubit VQC (Quantum)', 'Hybrid Ensemble Fusion'],
    accuracy: 0.8362, auc: 0.907, f1: 0.8445, precision: 0.8251, recall: 0.8650,
  },
  {
    id: 'kidney_disease',
    name: 'Chronic Kidney Disease Risk Assessment',
    category: 'Nephrology',
    dataset: 'UCI Chronic Kidney Disease Dataset',
    status: 'ACTIVE',
    version: '1.0.0',
    featureCount: 24,
    description: 'Early risk screening for chronic kidney disease (CKD) using 24 renal biomarkers including serum creatinine, hemoglobin, blood glucose, and electrolyte levels.',
    qubits: 6, layers: 2, pcaDim: 6,
    models: ['Random Forest (Classical)', '6-Qubit VQC (Quantum)', 'Hybrid Ensemble Fusion'],
    accuracy: 0.9875, auc: 0.998, f1: 0.9868, precision: 0.9907, recall: 0.9831,
  },
  {
    id: 'liver_disease',
    name: 'Liver Disease Risk Assessment',
    category: 'Hepatology',
    dataset: 'UCI Indian Liver Patient Dataset',
    status: 'ACTIVE',
    version: '1.0.0',
    featureCount: 10,
    description: 'Early screening for hepatic dysfunction using serum bilirubin levels, liver enzyme markers (ALT, AST), and albumin ratios.',
    qubits: 6, layers: 2, pcaDim: 6,
    models: ['Random Forest (Classical)', '6-Qubit VQC (Quantum)', 'Hybrid Ensemble Fusion'],
    accuracy: 0.7586, auc: 0.783, f1: 0.7241, precision: 0.7800, recall: 0.6754,
  },
  {
    id: 'stroke',
    name: 'Stroke Risk Assessment',
    category: 'Neurology',
    dataset: 'Kaggle Healthcare Stroke Dataset',
    status: 'ACTIVE',
    version: '1.0.0',
    featureCount: 10,
    description: 'Cerebrovascular accident and stroke prediction using clinical risk factors including hypertension, heart disease, glucose levels, and lifestyle indicators.',
    qubits: 6, layers: 2, pcaDim: 6,
    models: ['Random Forest (Classical)', '6-Qubit VQC (Quantum)', 'Hybrid Ensemble Fusion'],
    accuracy: 0.9502, auc: 0.876, f1: 0.7412, precision: 0.8123, recall: 0.6821,
  },
  {
    id: 'breast_cancer',
    name: 'Breast Cancer Risk Assessment',
    category: 'Oncology',
    dataset: 'UCI Breast Cancer Wisconsin (Diagnostic)',
    status: 'ACTIVE',
    version: '1.0.0',
    featureCount: 30,
    description: 'Biomarker and cell nucleus morphometric analysis for early breast cancer risk stratification using 30 features derived from fine needle aspirate (FNA) images.',
    qubits: 6, layers: 2, pcaDim: 6,
    models: ['Random Forest (Classical)', '6-Qubit VQC (Quantum)', 'Hybrid Ensemble Fusion'],
    accuracy: 0.9649, auc: 0.994, f1: 0.9634, precision: 0.9706, recall: 0.9565,
  },
  {
    id: 'parkinsons',
    name: "Parkinson's Disease Risk Assessment",
    category: 'Neurology & Movement',
    dataset: "UCI Parkinson's Disease Vocal Dataset",
    status: 'ACTIVE',
    version: '1.0.0',
    featureCount: 22,
    description: 'Acoustic and voice frequency analysis for early detection of Parkinsonian neurological tremor using 22 biomedical voice measurements including jitter, shimmer, and NHR.',
    qubits: 6, layers: 2, pcaDim: 6,
    models: ['Random Forest (Classical)', '6-Qubit VQC (Quantum)', 'Hybrid Ensemble Fusion'],
    accuracy: 0.9487, auc: 0.982, f1: 0.9592, precision: 0.9574, recall: 0.9610,
  },
  {
    id: 'thyroid',
    name: 'Thyroid Disease Risk Assessment',
    category: 'Endocrine & Thyroid',
    dataset: 'UCI Thyroid Disease Benchmark (Garavan Institute)',
    status: 'ACTIVE',
    version: '1.0.0',
    featureCount: 21,
    description: 'AI-assisted screening for hypothyroidism and thyroid dysfunction using TSH, T3, T4, FTI hormone levels, and 17 clinical indicators.',
    qubits: 6, layers: 2, pcaDim: 6,
    models: ['Random Forest (Classical)', '6-Qubit VQC (Quantum)', 'Hybrid Ensemble Fusion'],
    accuracy: 0.9731, auc: 0.991, f1: 0.9628, precision: 0.9740, recall: 0.9518,
  },
  {
    id: 'lung_cancer',
    name: 'Thoracic & Lung Cancer Risk Assessment',
    category: 'Pulmonology & Oncology',
    dataset: 'UCI Thoracic Surgery & Lung Cancer Dataset',
    status: 'ACTIVE',
    version: '1.0.0',
    featureCount: 14,
    description: 'Risk stratification for thoracic surgery complications and early lung cancer screening using pulmonary spirometry (FVC, FEV1) and clinical risk factors.',
    qubits: 6, layers: 2, pcaDim: 6,
    models: ['Random Forest (Classical)', '6-Qubit VQC (Quantum)', 'Hybrid Ensemble Fusion'],
    accuracy: 0.8571, auc: 0.817, f1: 0.5000, precision: 0.6667, recall: 0.4000,
  },
  {
    id: 'alzheimers',
    name: "Alzheimer's & Dementia Risk Assessment",
    category: 'Neurology & Geriatrics',
    dataset: 'OASIS Longitudinal MRI Dementia Dataset',
    status: 'ACTIVE',
    version: '1.0.0',
    featureCount: 9,
    description: "Early cognitive impairment and Alzheimer's dementia risk evaluation based on MMSE scores, CDR ratings, and brain MRI volume markers from OASIS longitudinal study.",
    qubits: 6, layers: 2, pcaDim: 6,
    models: ['Random Forest (Classical)', '6-Qubit VQC (Quantum)', 'Hybrid Ensemble Fusion'],
    accuracy: 0.8462, auc: 0.924, f1: 0.8421, precision: 0.8421, recall: 0.8421,
  },
  {
    id: 'hypertension',
    name: 'Clinical Hypertension Risk Assessment',
    category: 'Cardiology & Vascular',
    dataset: 'Clinical Hypertension & Vascular Health Dataset',
    status: 'ACTIVE',
    version: '1.0.0',
    featureCount: 12,
    description: 'Early risk detection for primary and secondary hypertension using blood pressure dynamics, electrolyte balances, BMI, and lifestyle markers.',
    qubits: 6, layers: 2, pcaDim: 6,
    models: ['Random Forest (Classical)', '6-Qubit VQC (Quantum)', 'Hybrid Ensemble Fusion'],
    accuracy: 0.8921, auc: 0.943, f1: 0.8815, precision: 0.8972, recall: 0.8663,
  }
];

const BENCHMARK_DATA = [
  {
    name: 'Random Forest Baseline',
    shortName: 'RF Baseline',
    type: 'classical',
    accuracy: 0.7348, f1: 0.4444, auc: 0.8240, precision: 0.5902, recall: 0.3566,
    dataset: 'CDC BRFSS 2015', records: '253,680', features: 21,
  },
  {
    name: '6-Qubit Variational Quantum Classifier',
    shortName: 'VQC (6-Qubit)',
    type: 'quantum',
    accuracy: 0.8607, f1: 0.0, auc: 0.4944, precision: 0.0, recall: 0.0,
    dataset: 'CDC BRFSS 2015', records: '253,680', features: 6,
  },
  {
    name: 'Hybrid Classical-Quantum Ensemble',
    shortName: 'Hybrid Ensemble',
    type: 'hybrid',
    accuracy: 0.7852, f1: 0.4612, auc: 0.8240, precision: 0.6212, recall: 0.3661,
    dataset: 'CDC BRFSS 2015', records: '253,680', features: 21,
    winner: true,
  }
];

const PIPELINE_STEPS = [
  { step: '01', icon: '🗂️', color: '#38bdf8', name: 'Data Ingestion', sub: 'Multi-disease datasets · Feature validation · Schema normalization' },
  { step: '02', icon: '⚙️', color: '#818cf8', name: 'Preprocessing', sub: 'StandardScaler · Imputation · Categorical encoding · Disease-specific pipelines' },
  { step: '03', icon: '📉', color: '#c084fc', name: 'PCA Reduction', sub: '21D → 6D quantum-ready features · Variance preserved >90%' },
  { step: '04', icon: '🌲', color: '#22d3ee', name: 'Classical RF', sub: 'Random Forest Baseline · Feature importance · SHAP explainability' },
  { step: '05', icon: '⚛️', color: '#a78bfa', name: 'VQC Quantum', sub: '6-Qubit VQC · 2 Ansatz layers · PennyLane simulator' },
  { step: '06', icon: '🔀', color: '#34d399', name: 'Fusion Ensemble', sub: 'Weighted stacking · Classical + Quantum predictions · Calibrated output' },
  { step: '07', icon: '📊', color: '#fbbf24', name: 'Risk Score', sub: 'Calibrated probability · Risk label · Medical disclaimer' },
  { step: '08', icon: '🔍', color: '#fb7185', name: 'Explainability', sub: 'Top-5 SHAP features · Clinical narrative · Patient report' },
];

const ARCH_DETAILS = [
  {
    icon: '⚛️', title: 'Variational Quantum Classifier (VQC)',
    body: [
      '6 qubits per circuit · 2 Ry/Rz ansatz layers',
      'PennyLane default.qubit simulator',
      'Custom VQC with bias + scale calibration',
      'QML-ready PCA dimensionality reduction (6D)',
      'Predict-proba via quantum expectation values',
    ]
  },
  {
    icon: '🌲', title: 'Random Forest Baseline',
    body: [
      'Scikit-learn RandomForestClassifier',
      'Hyperparameter-tuned via GridSearchCV',
      'SHAP TreeExplainer for feature attribution',
      'Per-disease feature importance ranking',
      'Joblib-serialized & lazy-loaded artifacts',
    ]
  },
  {
    icon: '🔀', title: 'Hybrid Ensemble Fusion',
    body: [
      'Weighted stacking of RF + VQC outputs',
      'Soft-voting probability averaging',
      'Calibrated with Platt Scaling',
      'Disease-specific artifact isolation',
      'Singleton service with hot-reload support',
    ]
  },
  {
    icon: '🗄️', title: 'Backend Architecture',
    body: [
      'FastAPI v0.111 · Async REST endpoints',
      'SQLAlchemy ORM + SQLite persistence',
      'Multi-disease registry (registry.json)',
      'Pydantic v2 request/response schemas',
      'CORS-enabled · API versioning (v1)',
    ]
  },
  {
    icon: '📊', title: 'Explainability Layer',
    body: [
      'Top-5 contributing clinical features',
      'Normalized feature importance scores',
      'Clinical narrative auto-generation',
      'Disease-specific SHAP analysis',
      'Patient-friendly risk summary',
    ]
  },
  {
    icon: '🏥', title: 'Disease Registry',
    body: [
      '11 active disease models registered',
      'Per-disease feature schema validation',
      'Dataset metadata & model versioning',
      'QML config (n_qubits, n_layers, pca_dim)',
      'Status tracking: ACTIVE / PLANNED',
    ]
  },
];

/* ═══════════════════════════════════════════════
   COLOUR HELPERS
   ═══════════════════════════════════════════════ */

const CATEGORY_COLOURS = {
  'Endocrine & Metabolic':   { text: '#38bdf8', glow: 'rgba(56,189,248,0.2)' },
  'Endocrine & Thyroid':     { text: '#38bdf8', glow: 'rgba(56,189,248,0.2)' },
  'Cardiology':              { text: '#fb7185', glow: 'rgba(251,113,133,0.2)' },
  'Cardiology & Vascular':   { text: '#fb7185', glow: 'rgba(251,113,133,0.2)' },
  'Nephrology':              { text: '#22d3ee', glow: 'rgba(34,211,238,0.2)' },
  'Hepatology':              { text: '#fbbf24', glow: 'rgba(251,191,36,0.2)' },
  'Neurology':               { text: '#818cf8', glow: 'rgba(129,140,248,0.2)' },
  'Neurology & Movement':    { text: '#818cf8', glow: 'rgba(129,140,248,0.2)' },
  'Neurology & Geriatrics':  { text: '#818cf8', glow: 'rgba(129,140,248,0.2)' },
  'Oncology':                { text: '#c084fc', glow: 'rgba(192,132,252,0.2)' },
  'Pulmonology & Oncology':  { text: '#c084fc', glow: 'rgba(192,132,252,0.2)' },
};

function catColour(cat) {
  return CATEGORY_COLOURS[cat] || { text: '#94a3b8', glow: 'rgba(148,163,184,0.15)' };
}

const AUC_GRADIENTS = [
  'linear-gradient(90deg,#38bdf8,#818cf8)',
  'linear-gradient(90deg,#818cf8,#c084fc)',
  'linear-gradient(90deg,#34d399,#22d3ee)',
  'linear-gradient(90deg,#fb7185,#fbbf24)',
  'linear-gradient(90deg,#c084fc,#38bdf8)',
  'linear-gradient(90deg,#fbbf24,#34d399)',
  'linear-gradient(90deg,#22d3ee,#818cf8)',
  'linear-gradient(90deg,#a78bfa,#38bdf8)',
  'linear-gradient(90deg,#34d399,#818cf8)',
  'linear-gradient(90deg,#fb7185,#c084fc)',
  'linear-gradient(90deg,#38bdf8,#34d399)',
];

/* ═══════════════════════════════════════════════
   NAVIGATION
   ═══════════════════════════════════════════════ */

const SECTIONS = ['overview', 'models', 'benchmark', 'pipeline'];

function showSection(name) {
  SECTIONS.forEach(s => {
    const el = document.getElementById('section-' + s);
    if (el) el.style.display = (s === name) ? '' : 'none';
    const btn = document.getElementById('nav-' + s);
    if (btn) btn.classList.toggle('active', s === name);
  });
  // Lazy-render
  if (name === 'models')    renderModels('all');
  if (name === 'benchmark') renderBenchmark();
  if (name === 'pipeline')  renderPipeline();
}

/* ═══════════════════════════════════════════════
   LIVE CLOCK
   ═══════════════════════════════════════════════ */

function updateClock() {
  const el = document.getElementById('live-clock');
  if (!el) return;
  const now = new Date();
  el.textContent = now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

/* ═══════════════════════════════════════════════
   MODELS SECTION
   ═══════════════════════════════════════════════ */

let modelsRendered = false;
let currentFilter = 'all';

function filterModels(cat, btn) {
  currentFilter = cat;
  // Update filter buttons
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active-filter'));
  if (btn) btn.classList.add('active-filter');

  document.querySelectorAll('.model-card').forEach(card => {
    const cardCat = card.dataset.category || '';
    const show = cat === 'all' || cardCat.toLowerCase().includes(cat.toLowerCase());
    card.classList.toggle('hidden', !show);
  });
}

function renderModels() {
  if (modelsRendered) { filterModels(currentFilter, null); return; }
  const grid = document.getElementById('models-grid');
  if (!grid) return;

  grid.innerHTML = DISEASE_MODELS.map((m, i) => {
    const cc = catColour(m.category);
    return `
    <div class="model-card" data-category="${escHtml(m.category)}" style="animation-delay:${i * 0.06}s">
      <div class="model-card-top">
        <div class="model-name">${escHtml(m.name)}</div>
        <span class="model-status-badge badge-${m.status.toLowerCase()}">${escHtml(m.status)}</span>
      </div>
      <div class="model-category" style="color:${cc.text}">${escHtml(m.category)}</div>
      <div class="model-desc">${escHtml(m.description)}</div>
      <div class="model-meta">
        <div class="meta-row">
          <span class="meta-key">Dataset</span>
          <span class="meta-val">${escHtml(m.dataset)}</span>
        </div>
        <div class="meta-row">
          <span class="meta-key">Accuracy</span>
          <span class="meta-val" style="color:${cc.text}">${(m.accuracy * 100).toFixed(2)}%</span>
        </div>
        <div class="meta-row">
          <span class="meta-key">ROC-AUC</span>
          <span class="meta-val" style="color:#34d399">${m.auc.toFixed(3)}</span>
        </div>
        <div class="meta-row">
          <span class="meta-key">F1-Score</span>
          <span class="meta-val">${m.f1.toFixed(4)}</span>
        </div>
        <div class="meta-row">
          <span class="meta-key">QML Config</span>
          <span class="meta-val">${m.qubits}-Qubit · ${m.layers} Layers · PCA ${m.pcaDim}D</span>
        </div>
        <div class="meta-row">
          <span class="meta-key">Version</span>
          <span class="meta-val">v${escHtml(m.version)}</span>
        </div>
      </div>
      <div class="model-models-row">
        <span class="model-pill pill-rf">RF</span>
        <span class="model-pill pill-qml">VQC</span>
        <span class="model-pill pill-hyb">Hybrid</span>
      </div>
      <div class="features-count">
        <span>Clinical Features:</span>
        <span class="feat-chip">${m.featureCount}</span>
      </div>
    </div>`;
  }).join('');

  modelsRendered = true;
}

/* ═══════════════════════════════════════════════
   BENCHMARK SECTION
   ═══════════════════════════════════════════════ */

let benchRendered = false;
function renderBenchmark() {
  if (benchRendered) return;

  // Bench Cards
  const cards = document.getElementById('bench-cards');
  if (cards) {
    cards.innerHTML = BENCHMARK_DATA.map(b => `
      <div class="bench-model-card${b.winner ? ' winner' : ''}">
        <div class="bmc-name">${escHtml(b.name)}</div>
        <div class="bmc-type">${escHtml(b.type.toUpperCase())}</div>
        <div class="bmc-metrics">
          <div class="bmc-metric"><span class="bm-key">Accuracy</span><span class="bm-val highlight">${(b.accuracy*100).toFixed(2)}%</span></div>
          <div class="bmc-metric"><span class="bm-key">ROC-AUC</span><span class="bm-val highlight">${b.auc.toFixed(4)}</span></div>
          <div class="bmc-metric"><span class="bm-key">F1-Score</span><span class="bm-val">${b.f1.toFixed(4)}</span></div>
          <div class="bmc-metric"><span class="bm-key">Precision</span><span class="bm-val">${b.precision.toFixed(4)}</span></div>
          <div class="bmc-metric"><span class="bm-key">Recall</span><span class="bm-val">${b.recall.toFixed(4)}</span></div>
          <div class="bmc-metric"><span class="bm-key">Features</span><span class="bm-val">${b.features}</span></div>
          <div class="bmc-metric"><span class="bm-key">Records</span><span class="bm-val">${escHtml(b.records)}</span></div>
        </div>
      </div>
    `).join('');
  }

  // Bench Table — all 11 models
  const tbody = document.getElementById('bench-tbody');
  if (tbody) {
    tbody.innerHTML = DISEASE_MODELS.map(m => {
      const typeMap = { 'Hybrid Ensemble Fusion': 'hybrid', 'Random Forest (Classical)': 'classical', '6-Qubit VQC (Quantum)': 'quantum' };
      return `
      <tr>
        <td>${escHtml(m.name)}</td>
        <td>${(m.accuracy*100).toFixed(2)}%</td>
        <td>${m.f1.toFixed(4)}</td>
        <td>${m.auc.toFixed(3)}</td>
        <td>${m.precision.toFixed(4)}</td>
        <td>${m.recall.toFixed(4)}</td>
        <td><span class="td-type hybrid">Hybrid</span></td>
      </tr>`;
    }).join('');
  }

  // Bar Chart — all 11 models sorted by AUC
  const barChart = document.getElementById('bar-chart');
  if (barChart) {
    const sorted = [...DISEASE_MODELS].sort((a, b) => b.auc - a.auc);
    barChart.innerHTML = sorted.map((m, i) => {
      const pct = Math.round(m.auc * 100);
      return `
      <div class="bar-row">
        <div class="bar-label">${escHtml(m.name.split(' Risk')[0])}</div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${pct}%;background:${AUC_GRADIENTS[i % AUC_GRADIENTS.length]}"></div>
        </div>
        <div class="bar-score">${m.auc.toFixed(3)}</div>
      </div>`;
    }).join('');

    // Animate bars in after paint
    requestAnimationFrame(() => {
      document.querySelectorAll('.bar-fill').forEach(b => { b.style.width = b.style.width; });
    });
  }

  benchRendered = true;
}

/* ═══════════════════════════════════════════════
   PIPELINE SECTION
   ═══════════════════════════════════════════════ */

let pipeRendered = false;
function renderPipeline() {
  if (pipeRendered) return;

  const wrapper = document.getElementById('pipeline-wrapper');
  if (wrapper) {
    wrapper.innerHTML = PIPELINE_STEPS.map(s => `
      <div class="pipe-step">
        <div class="pipe-icon" style="background:rgba(0,0,0,0.3);border:1px solid ${s.color}30;color:${s.color};box-shadow:0 0 18px ${s.color}28">${s.icon}</div>
        <div class="pipe-num">Step ${s.step}</div>
        <div class="pipe-name">${escHtml(s.name)}</div>
        <div class="pipe-sub">${escHtml(s.sub)}</div>
      </div>
    `).join('');
  }

  const archGrid = document.getElementById('arch-detail-grid');
  if (archGrid) {
    archGrid.innerHTML = ARCH_DETAILS.map(a => `
      <div class="arch-card">
        <div class="arch-card-title">${a.icon} ${escHtml(a.title)}</div>
        <div class="arch-card-body">
          <ul class="arch-list">
            ${a.body.map(item => `<li>${escHtml(item)}</li>`).join('')}
          </ul>
        </div>
      </div>
    `).join('');
  }

  pipeRendered = true;
}

/* ═══════════════════════════════════════════════
   UTILITY
   ═══════════════════════════════════════════════ */

function escHtml(str) {
  if (typeof str !== 'string') return String(str);
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/* ═══════════════════════════════════════════════
   INIT
   ═══════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  // Start clock
  updateClock();
  setInterval(updateClock, 1000);

  // Animate KPI values
  animateKPIs();

  // Default to overview section
  showSection('overview');
});

function animateKPIs() {
  const targets = {
    'kv-diseases': { end: 11, decimals: 0, suffix: '' },
    'kv-records':  { end: 253, decimals: 0, suffix: ' K' },
    'kv-auc':      { end: 0.824, decimals: 3, suffix: '' },
    'kv-qubits':   { end: 6, decimals: 0, suffix: '' },
  };

  Object.entries(targets).forEach(([id, cfg]) => {
    const el = document.getElementById(id);
    if (!el) return;
    const start = 0;
    const duration = 1400;
    const startTime = performance.now();
    function tick(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out-cubic
      const val = start + (cfg.end - start) * eased;
      el.textContent = val.toFixed(cfg.decimals) + cfg.suffix;
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });
}
