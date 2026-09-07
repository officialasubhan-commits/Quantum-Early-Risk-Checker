// SIH26139 Patient Web Application - Multi-Disease FastAPI Backend Client

const API_BASE_URL = "http://127.0.0.1:8001";

// Active parameters stored per disease for custom user tweaking
const customDiseaseParams = {};

// Disease Icons mapping
const DISEASE_ICONS = {
    diabetes: "🩺",
    heart_disease: "❤️",
    kidney_disease: "🫘",
    liver_disease: "🧪",
    stroke: "🧠",
    breast_cancer: "🎗️",
    parkinsons: "🎙️",
    thyroid: "🦋",
    lung_cancer: "🫁",
    alzheimers: "🧠",
    hypertension: "🩸"
};

// Preset Definitions: Low Risk vs High Risk profiles for all target diseases
const PRESETS = {
    low: {
        diabetes: {
            BMI: 22.4, GenHlth: 1, MentHlth: 0, PhysHlth: 0, HighBP: 0, HighChol: 0,
            CholCheck: 1, Smoker: 0, Stroke: 0, HeartDiseaseorAttack: 0, PhysActivity: 1,
            Fruits: 1, Veggies: 1, HvyAlcoholConsump: 0, AnyHealthcare: 1, NoDocbcCost: 0,
            DiffWalk: 0, Sex: 0, Age: 4, Education: 6, Income: 8
        },
        heart_disease: {
            age: 42.0, sex: 0.0, cp: 0.0, trestbps: 118.0, chol: 185.0, fbs: 0.0,
            restecg: 0.0, thalach: 168.0, exang: 0.0, oldpeak: 0.0, slope: 1.0, ca: 0.0, thal: 2.0
        },
        kidney_disease: {
            age: 38.0, bp: 70.0, sg: 1.025, al: 0.0, su: 0.0, rbc: 1.0, pc: 1.0, pcc: 0.0,
            ba: 0.0, bgr: 95.0, bu: 18.0, sc: 0.8, sod: 142.0, pot: 4.2, hemo: 16.2, pcv: 48.0,
            wbcc: 6400.0, rbcc: 5.4, htn: 0.0, dm: 0.0, cad: 0.0, appet: 0.0, pe: 0.0, ane: 0.0
        },
        liver_disease: {
            Age: 35.0, Gender: 1.0, Total_Bilirubin: 0.6, Direct_Bilirubin: 0.1,
            Alkaline_Phosphotase: 140.0, Alamine_Aminotransferase: 18.0,
            Aspartate_Aminotransferase: 20.0, Total_Protiens: 7.2, Albumin: 4.1,
            Albumin_and_Globulin_Ratio: 1.2
        },
        stroke: {
            gender: 0.0, age: 34.0, hypertension: 0.0, heart_disease: 0.0, ever_married: 1.0,
            work_type: 2.0, Residence_type: 1.0, avg_glucose_level: 84.5, bmi: 22.1, smoking_status: 0.0
        },
        breast_cancer: {
            "mean radius": 11.2, "mean texture": 14.5, "mean perimeter": 71.8, "mean area": 384.0,
            "mean smoothness": 0.092, "mean compactness": 0.054, "mean concavity": 0.021,
            "mean concave points": 0.015, "mean symmetry": 0.165, "mean fractal dimension": 0.059,
            "radius error": 0.22, "texture error": 0.85, "perimeter error": 1.45, "area error": 15.2,
            "smoothness error": 0.005, "compactness error": 0.011, "concavity error": 0.012,
            "concave points error": 0.006, "symmetry error": 0.014, "fractal dimension error": 0.002,
            "worst radius": 12.4, "worst texture": 18.2, "worst perimeter": 79.5, "worst area": 470.0,
            "worst smoothness": 0.121, "worst compactness": 0.102, "worst concavity": 0.085,
            "worst concave points": 0.052, "worst symmetry": 0.245, "worst fractal dimension": 0.071
        },
        parkinsons: {
            "MDVP:Fo(Hz)": 202.4, "MDVP:Fhi(Hz)": 224.1, "MDVP:Flo(Hz)": 188.5,
            "MDVP:Jitter(%)": 0.0021, "MDVP:Jitter(Abs)": 0.00001, "MDVP:RAP": 0.0011,
            "MDVP:PPQ": 0.0013, "Jitter:DDP": 0.0032, "MDVP:Shimmer": 0.0142,
            "MDVP:Shimmer(dB)": 0.128, "Shimmer:APQ3": 0.0078, "Shimmer:APQ5": 0.0089,
            "MDVP:APQ": 0.0105, "Shimmer:DDA": 0.0234, "NHR": 0.0045, "HNR": 28.5,
            "RPDE": 0.321, "DFA": 0.625, "spread1": -7.21, "spread2": 0.112,
            "D2": 1.85, "PPE": 0.098
        },
        thyroid: {
            age: 32, sex: 0, on_thyroxine: 0, query_on_thyroxine: 0, on_antithyroid_med: 0,
            sick: 0, pregnant: 0, thyroid_surgery: 0, I131_treatment: 0, query_hypothyroid: 0,
            query_hyperthyroid: 0, lithium: 0, goitre: 0, tumor: 0, hypopituitary: 0, psych: 0,
            TSH: 1.8, T3: 2.1, TT4: 110.0, T4U: 0.95, FTI: 115.8
        },
        lung_cancer: {
            age: 48, forced_vital_capacity: 3.8, fev1: 3.1, performance_status: 0, pain: 0,
            haemoptysis: 0, dyspnoea: 0, cough: 0, weakness: 0, tumor_size: 1, diabetes_history: 0,
            myocardial_infarction: 0, peripheral_arterial_disease: 0, smoking_history: 0
        },
        alzheimers: {
            gender: 0, age: 62, education_years: 16, socioeconomic_status: 2,
            mini_mental_state_exam: 29, clinical_dementia_rating: 0.0,
            estimated_total_intracranial_vol: 1450.0, normalize_whole_brain_vol: 0.76, atlas_scaling_factor: 1.21
        },
        hypertension: {
            age: 35, sex: 0, systolic_bp: 118.0, diastolic_bp: 76.0, heart_rate: 68, bmi: 22.5,
            fasting_glucose: 92.0, serum_sodium: 140.0, serum_potassium: 4.3, family_history: 0,
            smoking_status: 0, physical_activity_hours: 4.5
        }
    },
    high: {
        diabetes: {
            BMI: 34.5, GenHlth: 4, MentHlth: 8, PhysHlth: 14, HighBP: 1, HighChol: 1,
            CholCheck: 1, Smoker: 1, Stroke: 0, HeartDiseaseorAttack: 1, PhysActivity: 0,
            Fruits: 0, Veggies: 0, HvyAlcoholConsump: 0, AnyHealthcare: 1, NoDocbcCost: 1,
            DiffWalk: 1, Sex: 1, Age: 10, Education: 4, Income: 5
        },
        heart_disease: {
            age: 67.0, sex: 1.0, cp: 3.0, trestbps: 160.0, chol: 286.0, fbs: 1.0,
            restecg: 2.0, thalach: 108.0, exang: 1.0, oldpeak: 3.5, slope: 2.0, ca: 2.0, thal: 7.0
        },
        kidney_disease: {
            age: 65.0, bp: 90.0, sg: 1.010, al: 3.0, su: 2.0, rbc: 0.0, pc: 0.0, pcc: 1.0,
            ba: 1.0, bgr: 240.0, bu: 85.0, sc: 4.8, sod: 128.0, pot: 5.8, hemo: 8.5, pcv: 26.0,
            wbcc: 11200.0, rbcc: 3.1, htn: 1.0, dm: 1.0, cad: 1.0, appet: 1.0, pe: 1.0, ane: 1.0
        },
        liver_disease: {
            Age: 65.0, Gender: 1.0, Total_Bilirubin: 4.5, Direct_Bilirubin: 2.1,
            Alkaline_Phosphotase: 480.0, Alamine_Aminotransferase: 120.0,
            Aspartate_Aminotransferase: 160.0, Total_Protiens: 5.4, Albumin: 2.2,
            Albumin_and_Globulin_Ratio: 0.6
        },
        stroke: {
            gender: 1.0, age: 78.0, hypertension: 1.0, heart_disease: 1.0, ever_married: 1.0,
            work_type: 2.0, Residence_type: 1.0, avg_glucose_level: 228.69, bmi: 36.8, smoking_status: 1.0
        },
        breast_cancer: {
            "mean radius": 19.8, "mean texture": 24.2, "mean perimeter": 132.5, "mean area": 1240.0,
            "mean smoothness": 0.125, "mean compactness": 0.285, "mean concavity": 0.320,
            "mean concave points": 0.165, "mean symmetry": 0.255, "mean fractal dimension": 0.082,
            "radius error": 1.15, "texture error": 1.42, "perimeter error": 8.95, "area error": 158.0,
            "smoothness error": 0.008, "compactness error": 0.052, "concavity error": 0.058,
            "concave points error": 0.018, "symmetry error": 0.032, "fractal dimension error": 0.007,
            "worst radius": 26.8, "worst texture": 34.5, "worst perimeter": 188.0, "worst area": 2150.0,
            "worst smoothness": 0.175, "worst compactness": 0.680, "worst concavity": 0.740,
            "worst concave points": 0.285, "worst symmetry": 0.480, "worst fractal dimension": 0.125
        },
        parkinsons: {
            "MDVP:Fo(Hz)": 119.99, "MDVP:Fhi(Hz)": 157.30, "MDVP:Flo(Hz)": 74.99,
            "MDVP:Jitter(%)": 0.0078, "MDVP:Jitter(Abs)": 0.00007, "MDVP:RAP": 0.0037,
            "MDVP:PPQ": 0.0055, "Jitter:DDP": 0.0110, "MDVP:Shimmer": 0.0437,
            "MDVP:Shimmer(dB)": 0.426, "Shimmer:APQ3": 0.0218, "Shimmer:APQ5": 0.0313,
            "MDVP:APQ": 0.0297, "Shimmer:DDA": 0.0654, "NHR": 0.0221, "HNR": 21.0,
            "RPDE": 0.414, "DFA": 0.815, "spread1": -4.81, "spread2": 0.266,
            "D2": 2.30, "PPE": 0.284
        },
        thyroid: {
            age: 64, sex: 0, on_thyroxine: 0, query_on_thyroxine: 1, on_antithyroid_med: 0,
            sick: 0, pregnant: 0, thyroid_surgery: 0, I131_treatment: 0, query_hypothyroid: 1,
            query_hyperthyroid: 0, lithium: 0, goitre: 1, tumor: 0, hypopituitary: 0, psych: 0,
            TSH: 8.5, T3: 0.9, TT4: 55.0, T4U: 1.15, FTI: 47.8
        },
        lung_cancer: {
            age: 72, forced_vital_capacity: 2.1, fev1: 1.2, performance_status: 2, pain: 1,
            haemoptysis: 1, dyspnoea: 1, cough: 1, weakness: 1, tumor_size: 3, diabetes_history: 1,
            myocardial_infarction: 0, peripheral_arterial_disease: 1, smoking_history: 1
        },
        alzheimers: {
            gender: 1, age: 81, education_years: 12, socioeconomic_status: 4,
            mini_mental_state_exam: 21, clinical_dementia_rating: 1.0,
            estimated_total_intracranial_vol: 1520.0, normalize_whole_brain_vol: 0.68, atlas_scaling_factor: 1.15
        },
        hypertension: {
            age: 68, sex: 1, systolic_bp: 158.0, diastolic_bp: 96.0, heart_rate: 88, bmi: 33.5,
            fasting_glucose: 128.0, serum_sodium: 144.0, serum_potassium: 4.8, family_history: 1,
            smoking_status: 1, physical_activity_hours: 0.5
        }
    }
};

document.addEventListener("DOMContentLoaded", () => {
    checkBackendHealth();
    fetchAndPopulateDiseases();

    // Setup tab listeners
    const tabs = document.querySelectorAll(".tab-btn");
    tabs.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");
            switchTab(targetTab);
        });
    });

    calculateBMI();
});

async function fetchAndPopulateDiseases() {
    const select = document.getElementById("select-target-disease");
    if (!select) return;

    try {
        const resp = await fetch(`${API_BASE_URL}/api/v1/diseases`);
        if (resp.ok) {
            const data = await resp.json();
            const diseases = data.registered_diseases || {};

            select.innerHTML = '<option value="all" selected>🔍 Comprehensive Multi-Disease Screening (All Models)</option>';

            Object.entries(diseases).forEach(([dId, info]) => {
                if (info.status === "ACTIVE") {
                    const icon = DISEASE_ICONS[dId] || "🩺";
                    const opt = document.createElement("option");
                    opt.value = dId;
                    opt.textContent = `${icon} ${info.name || dId.toUpperCase()}`;
                    select.appendChild(opt);
                }
            });

            onDiseaseSelected();
        }
    } catch (err) {
        console.warn("Could not dynamically load diseases from backend, using fallback options:", err);
    }
}


function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

    const btn = document.querySelector(`[data-tab="${tabId}"]`);
    const content = document.getElementById(tabId);

    if (btn) btn.classList.add("active");
    if (content) content.classList.add("active");

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function checkBackendHealth() {
    const statusText = document.getElementById("api-status-text");
    try {
        const resp = await fetch(`${API_BASE_URL}/api/v1/health`);
        if (resp.ok) {
            const data = await resp.json();
            statusText.textContent = `Connected (${data.active_diseases_count} Active Disease Models)`;
            statusText.style.color = "var(--accent-emerald)";
        } else {
            statusText.textContent = "Backend Error";
            statusText.style.color = "var(--accent-rose)";
        }
    } catch (err) {
        statusText.textContent = "Offline / Connection Error";
        statusText.style.color = "var(--accent-amber)";
    }
}

function calculateBMI() {
    const heightCm = parseFloat(document.getElementById("inp-height").value) || 175;
    const weightKg = parseFloat(document.getElementById("inp-weight").value) || 88;

    if (heightCm > 0 && weightKg > 0) {
        const heightM = heightCm / 100.0;
        const bmi = weightKg / (heightM * heightM);
        document.getElementById("inp-bmi").value = bmi.toFixed(1);
    }
}

function onDiseaseSelected() {
    const selected = document.getElementById("select-target-disease").value;
    populateParameterTuner(selected);
}

function toggleParameterTuner() {
    const tuner = document.getElementById("disease-parameter-tuner");
    if (tuner.style.display === "none" || tuner.style.display === "") {
        tuner.style.display = "block";
    } else {
        tuner.style.display = "none";
    }
}

function populateParameterTuner(diseaseId) {
    const grid = document.getElementById("tuner-fields-grid");
    const title = document.getElementById("tuner-title");

    if (diseaseId === "all") {
        title.textContent = "⚙️ Multi-Disease Screening Mode";
        grid.innerHTML = `<p style="color: var(--text-muted); font-size: 0.85rem;">All 7 disease models will evaluate the patient profile features simultaneously.</p>`;
        return;
    }

    const icon = DISEASE_ICONS[diseaseId] || "🩺";
    title.textContent = `⚙️ ${icon} Custom Parameter Tuner: ${diseaseId.toUpperCase().replace("_", " ")}`;

    const currentParams = customDiseaseParams[diseaseId] || PRESETS.high[diseaseId] || {};
    grid.innerHTML = "";

    // Show top 8 primary fields for quick tuning
    const entries = Object.entries(currentParams).slice(0, 10);
    entries.forEach(([key, val]) => {
        const fieldGroup = document.createElement("div");
        fieldGroup.className = "form-group";

        const valType = typeof val;
        const inputStep = valType === "number" && !Number.isInteger(val) ? "0.01" : "1";

        fieldGroup.innerHTML = `
            <label style="font-size: 0.82rem; color: var(--text-secondary);">${key}</label>
            <input type="number" class="form-control tuner-input" data-disease="${diseaseId}" data-key="${key}" value="${val}" step="${inputStep}" onchange="updateCustomParam('${diseaseId}', '${key}', this.value)">
        `;
        grid.appendChild(fieldGroup);
    });
}

function updateCustomParam(diseaseId, key, value) {
    if (!customDiseaseParams[diseaseId]) {
        customDiseaseParams[diseaseId] = { ...PRESETS.high[diseaseId] };
    }
    customDiseaseParams[diseaseId][key] = parseFloat(value);
}

function loadClinicalPreset(type) {
    const diseaseId = document.getElementById("select-target-disease").value;
    const presetData = (PRESETS[type] && PRESETS[type][diseaseId]) ? PRESETS[type][diseaseId] : null;

    if (presetData) {
        customDiseaseParams[diseaseId] = { ...presetData };
        populateParameterTuner(diseaseId);
    }
    // Also update form inputs if diabetes is selected
    if (diseaseId === "diabetes" && type === "high") {
        document.getElementById("inp-bmi").value = "34.5";
        document.getElementById("inp-genhlth").value = "4";
        document.getElementById("inp-highbp").value = "1";
        document.getElementById("inp-highchol").value = "1";
        document.getElementById("inp-smoker").value = "1";
    } else if (diseaseId === "diabetes" && type === "low") {
        document.getElementById("inp-bmi").value = "22.4";
        document.getElementById("inp-genhlth").value = "1";
        document.getElementById("inp-highbp").value = "0";
        document.getElementById("inp-highchol").value = "0";
        document.getElementById("inp-smoker").value = "0";
    }

    runBackendAssessment();
}

function getFeaturePayload(diseaseId) {
    const patientId = document.getElementById("inp-patient-id").value.trim() || "PATIENT_DEMO_101";

    let features = customDiseaseParams[diseaseId];

    if (!features) {
        if (diseaseId === "diabetes") {
            features = {
                BMI: parseFloat(document.getElementById("inp-bmi").value) || 28.7,
                GenHlth: parseInt(document.getElementById("inp-genhlth").value, 10) || 3,
                MentHlth: parseFloat(document.getElementById("inp-menthlth").value) || 2.0,
                PhysHlth: parseFloat(document.getElementById("inp-physhlth").value) || 4.0,
                HighBP: parseInt(document.getElementById("inp-highbp").value, 10),
                HighChol: parseInt(document.getElementById("inp-highchol").value, 10),
                CholCheck: parseInt(document.getElementById("inp-cholcheck").value, 10),
                Smoker: parseInt(document.getElementById("inp-smoker").value, 10),
                Stroke: parseInt(document.getElementById("inp-stroke").value, 10),
                HeartDiseaseorAttack: parseInt(document.getElementById("inp-heartdisease").value, 10),
                PhysActivity: parseInt(document.getElementById("inp-physact").value, 10),
                Fruits: parseInt(document.getElementById("inp-fruits").value, 10),
                Veggies: parseInt(document.getElementById("inp-veggies").value, 10),
                HvyAlcoholConsump: parseInt(document.getElementById("inp-alcohol").value, 10),
                AnyHealthcare: parseInt(document.getElementById("inp-healthcare").value, 10),
                NoDocbcCost: parseInt(document.getElementById("inp-cost-barrier").value, 10),
                DiffWalk: parseInt(document.getElementById("inp-diffwalk").value, 10),
                Sex: parseInt(document.getElementById("inp-sex").value, 10),
                Age: parseInt(document.getElementById("inp-age").value, 10),
                Education: parseInt(document.getElementById("inp-education").value, 10),
                Income: parseInt(document.getElementById("inp-income").value, 10)
            };
        } else {
            features = PRESETS.high[diseaseId] || PRESETS.high["diabetes"];
        }
    }

    return {
        patient_id: patientId,
        disease_id: diseaseId,
        features: features
    };
}

async function runBackendAssessment() {
    switchTab("tab-results");

    const loadingBox = document.getElementById("assessment-loading");
    const singleContainer = document.getElementById("assessment-results-container");
    const multiContainer = document.getElementById("multi-screening-container");

    loadingBox.style.display = "block";
    singleContainer.style.display = "none";
    multiContainer.style.display = "none";

    const targetDisease = document.getElementById("select-target-disease").value;
    const modelType = document.getElementById("select-model-type").value; // hybrid, classical, qml

    try {
        if (targetDisease === "all") {
            const payload = getFeaturePayload("diabetes");
            const resp = await fetch(`${API_BASE_URL}/api/v1/predict/all`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            if (!resp.ok) throw new Error("Multi-disease screening request failed");
            const data = await resp.json();
            renderMultiDiseaseResults(data);

            loadingBox.style.display = "none";
            multiContainer.style.display = "block";
        } else {
            const payload = getFeaturePayload(targetDisease);

            // 1. Prediction Endpoint
            const predResp = await fetch(`${API_BASE_URL}/api/v1/predict/disease/${targetDisease}/${modelType}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            if (!predResp.ok) throw new Error("Prediction request failed");
            const predData = await predResp.json();

            // 2. Explainability Endpoint
            const expResp = await fetch(`${API_BASE_URL}/api/v1/explain/disease/${targetDisease}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            let expData = null;
            if (expResp.ok) expData = await expResp.json();

            renderSingleResults(predData, expData);

            loadingBox.style.display = "none";
            singleContainer.style.display = "block";
        }

    } catch (err) {
        loadingBox.style.display = "none";
        alert(`Assessment Error: ${err.message}. Ensure FastAPI server is running on http://127.0.0.1:8001.`);
    }
}

function renderSingleResults(predData, expData) {
    const riskLabel = document.getElementById("res-risk-label");
    const riskBadge = document.getElementById("res-risk-badge");
    const modelUsed = document.getElementById("res-model-used");
    const riskPct = document.getElementById("res-risk-pct");
    const meterFill = document.getElementById("res-meter-fill");
    const featuresList = document.getElementById("res-features-list");
    const narrativeBox = document.getElementById("res-clinical-narrative");

    const isHigh = predData.is_high_risk;
    const probPct = (predData.risk_probability * 100).toFixed(1);
    const icon = DISEASE_ICONS[predData.disease_id] || "🩺";

    riskLabel.textContent = `${icon} ${predData.disease_name}: ${predData.predicted_label}`;
    modelUsed.textContent = `Model Engine: ${predData.model_used}`;
    riskPct.textContent = `${probPct}% Probabilistic Risk`;

    meterFill.style.width = `${probPct}%`;
    if (isHigh) {
        riskBadge.textContent = "ELEVATED RISK";
        riskBadge.className = "risk-badge high";
        meterFill.className = "meter-fill high";
    } else {
        riskBadge.textContent = "LOW RISK";
        riskBadge.className = "risk-badge low";
        meterFill.className = "meter-fill low";
    }

    featuresList.innerHTML = "";
    const topFactors = expData ? expData.top_contributing_features : [];

    if (topFactors.length > 0) {
        topFactors.forEach(factor => {
            const scorePct = (factor.contribution_score * 100).toFixed(1);
            const isElevating = factor.contribution_score > 0.05;

            const itemHtml = `
                <div class="feature-item">
                    <div class="feature-info">
                        <strong>${factor.display_name}</strong>
                        <p>Recorded Clinical Value: <code style="color: var(--accent-teal);">${factor.feature_value}</code></p>
                    </div>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="width: 100px; height: 8px; background: var(--bg-card); border-radius: 4px; overflow: hidden;">
                            <div style="width: ${Math.min(100, Math.abs(scorePct * 2))}%; height: 100%; background: ${isElevating ? 'var(--accent-rose)' : 'var(--accent-emerald)'};"></div>
                        </div>
                        <div class="feature-score" style="color: ${isElevating ? 'var(--accent-rose)' : 'var(--accent-emerald)'};">${scorePct}%</div>
                    </div>
                </div>
            `;
            featuresList.insertAdjacentHTML("beforeend", itemHtml);
        });
    } else {
        featuresList.innerHTML = `<p style="color: var(--text-muted);">Attribution data unavailable for this prediction.</p>`;
    }

    narrativeBox.textContent = expData ? expData.clinical_narrative : "Clinical narrative generated successfully.";
}

function renderMultiDiseaseResults(multiData) {
    const cardsGrid = document.getElementById("multi-disease-cards-grid");
    cardsGrid.innerHTML = "";

    multiData.disease_results.forEach(res => {
        const probPct = (res.risk_probability * 100).toFixed(1);
        const icon = DISEASE_ICONS[res.disease_id] || "🩺";
        const isHigh = res.is_high_risk;

        const cardHtml = `
            <div class="multi-card">
                <div>
                    <div class="multi-card-header">
                        <div>
                            <div class="multi-card-title">${icon} ${res.disease_name}</div>
                            <div class="multi-card-model">Engine: ${res.model_used}</div>
                        </div>
                        <span class="risk-badge ${isHigh ? 'high' : 'low'}" style="font-size: 0.75rem; padding: 4px 10px;">
                            ${isHigh ? 'ELEVATED' : 'LOW RISK'}
                        </span>
                    </div>
                    
                    <div style="margin: 14px 0;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 4px;">
                            <span style="color: var(--text-secondary);">Risk Probability</span>
                            <strong style="color: ${isHigh ? 'var(--accent-rose)' : 'var(--accent-emerald)'};">${probPct}%</strong>
                        </div>
                        <div style="height: 8px; background: var(--bg-input); border-radius: 6px; overflow: hidden; border: 1px solid var(--border-color);">
                            <div style="width: ${probPct}%; height: 100%; background: ${isHigh ? 'linear-gradient(90deg, var(--accent-amber), var(--accent-rose))' : 'linear-gradient(90deg, var(--accent-teal), var(--accent-emerald))'};"></div>
                        </div>
                    </div>
                </div>

                <button class="btn-secondary" style="width: 100%; margin-top: 12px; font-size: 0.8rem; padding: 6px 12px;" onclick="inspectDiseaseExplanation('${res.disease_id}')">
                    Inspect Explanation →
                </button>
            </div>
        `;
        cardsGrid.insertAdjacentHTML("beforeend", cardHtml);
    });
}

function inspectDiseaseExplanation(diseaseId) {
    document.getElementById("select-target-disease").value = diseaseId;
    onDiseaseSelected();
    runBackendAssessment();
}

function handleFileSelected(input) {
    if (input.files && input.files[0]) {
        const file = input.files[0];
        document.getElementById("preview-filename").textContent = file.name;
        document.getElementById("preview-filesize").textContent = `(${(file.size / 1024).toFixed(1)} KB)`;
        document.getElementById("file-info-preview").style.display = "flex";
        document.getElementById("upload-status-display").style.display = "none";
    }
}

async function uploadReportFile() {
    const fileInput = document.getElementById("report-file-input");
    if (!fileInput.files || !fileInput.files[0]) return;

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    try {
        const resp = await fetch(`${API_BASE_URL}/api/v1/upload-report`, {
            method: "POST",
            body: formData
        });

        if (!resp.ok) {
            const errData = await resp.json();
            throw new Error(errData.detail || "File upload failed");
        }

        const data = await resp.json();

        document.getElementById("upload-status-title").textContent = `Status: ${data.status} (${data.filename})`;
        document.getElementById("upload-status-msg").textContent = data.message;
        document.getElementById("upload-status-display").style.display = "block";

    } catch (err) {
        console.error("File upload error:", err);
        const titleEl = document.getElementById("upload-status-title");
        const msgEl = document.getElementById("upload-status-msg");
        const displayEl = document.getElementById("upload-status-display");
        if (titleEl) titleEl.textContent = "Upload Failed";
        if (msgEl) msgEl.textContent = err.message || "An error occurred during file upload.";
        if (displayEl) displayEl.style.display = "block";
    }
}

// =========================================================================
// CONVERSATIONAL VOICE AI HEALTH ASSESSMENT ENGINE (SIH26139)
// Conducts natural spoken question-by-question medical interviews
// Real-time synchronization with manual health forms & model schemas
// =========================================================================

const VOICE_PRIMARY_URL = "http://127.0.0.1:8001";
const VOICE_SECONDARY_URL = "http://127.0.0.1:8002";
let currentVoiceSessionId = "session_" + Math.floor(Math.random() * 1000000);
let lastExtractedVoiceData = {};
let lastSynthesizedAudioBase64 = null;
let isRecordingMicrophone = false;
let speechRecognitionInstance = null;
let isConversationalAssessmentActive = false;
let currentQuestionData = null;

async function callVoiceApi(endpoint, body) {
    try {
        const r1 = await fetch(`${VOICE_PRIMARY_URL}${endpoint}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });
        if (r1.ok) return await r1.json();
    } catch (e) {
        console.warn(`Primary voice endpoint (${VOICE_PRIMARY_URL}) unreachable, trying secondary...`);
    }

    const r2 = await fetch(`${VOICE_SECONDARY_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
    });
    if (!r2.ok) {
        throw new Error(`Voice service returned HTTP ${r2.status}`);
    }
    return await r2.json();
}

function onVoiceDiseaseChange() {
    currentVoiceSessionId = "session_" + Math.floor(Math.random() * 1000000);
    isConversationalAssessmentActive = false;
    const indicator = document.getElementById("voice-state-indicator");
    const stateText = document.getElementById("voice-state-text");
    const progressText = document.getElementById("question-progress-text");
    const bubble = document.getElementById("assistant-spoken-bubble");
    const completionCard = document.getElementById("voice-completion-card");

    if (indicator) indicator.style.background = "#94a3b8";
    if (stateText) stateText.textContent = "Ready to Begin";
    if (progressText) progressText.textContent = "Question 0 of 0";
    if (completionCard) completionCard.style.display = "none";
    if (bubble) bubble.textContent = "Click 'Start Spoken Assessment' to begin.";
}

function onVoiceLanguageChange() {
    if (isConversationalAssessmentActive) {
        handleQuickCommand("repeat");
    }
}

async function startConversationalAssessment() {
    const diseaseSelect = document.getElementById("select-voice-disease");
    const langSelect = document.getElementById("select-voice-language");
    const disease = diseaseSelect ? diseaseSelect.value : "diabetes";
    const lang = langSelect ? langSelect.value : "en";

    const indicator = document.getElementById("voice-state-indicator");
    const stateText = document.getElementById("voice-state-text");
    const progressText = document.getElementById("question-progress-text");
    const bubble = document.getElementById("assistant-spoken-bubble");
    const completionCard = document.getElementById("voice-completion-card");
    const startBtnText = document.getElementById("start-btn-text");

    if (completionCard) completionCard.style.display = "none";
    if (indicator) indicator.style.background = "#06b6d4";
    if (stateText) stateText.textContent = "Connecting Voice AI...";
    if (startBtnText) startBtnText.textContent = "Restart Assessment";

    try {
        currentVoiceSessionId = "session_" + Math.floor(Math.random() * 1000000);
        const data = await callVoiceApi("/api/v1/voice/conversation/start", {
            session_id: currentVoiceSessionId,
            disease_id: disease,
            language: lang
        });

        isConversationalAssessmentActive = true;
        currentQuestionData = data.current_question;
        lastSynthesizedAudioBase64 = data.audio_base64;

        if (bubble) bubble.textContent = data.assistant_reply;
        if (progressText && data.current_question) {
            progressText.textContent = `Question ${data.current_question.index} of ${data.total_questions} (${disease.replace('_', ' ').toUpperCase()})`;
        }
        if (indicator) indicator.style.background = "#10b981";
        if (stateText) stateText.textContent = "Assistant Speaking...";

        // Play audio spoken question
        playLastSynthesizedAudio(data.assistant_reply, lang);

        // Focus input
        const inp = document.getElementById("inp-voice-transcript");
        if (inp) {
            inp.value = "";
            inp.focus();
        }

    } catch (err) {
        console.error("Failed to start voice assessment via API:", err);
        fallbackStartConversation(disease, lang);
    }
}

async function submitConversationalTurn() {
    const inp = document.getElementById("inp-voice-transcript");
    const text = inp ? inp.value.trim() : "";
    if (!text) {
        alert("Please speak your answer or type into the input field.");
        return;
    }

    const diseaseSelect = document.getElementById("select-voice-disease");
    const langSelect = document.getElementById("select-voice-language");
    const disease = diseaseSelect ? diseaseSelect.value : "diabetes";
    const lang = langSelect ? langSelect.value : "en";

    const indicator = document.getElementById("voice-state-indicator");
    const stateText = document.getElementById("voice-state-text");
    const bubble = document.getElementById("assistant-spoken-bubble");
    const progressText = document.getElementById("question-progress-text");

    if (indicator) indicator.style.background = "#eab308";
    if (stateText) stateText.textContent = "Processing Response...";

    try {
        const data = await callVoiceApi("/api/v1/voice/conversation/turn", {
            session_id: currentVoiceSessionId,
            user_input: text,
            disease_id: disease,
            language: lang
        });

        lastSynthesizedAudioBase64 = data.audio_base64;
        currentQuestionData = data.current_question;

        // 1. Update Assistant speech bubble
        if (bubble) {
            bubble.textContent = data.assistant_reply;
        }

        // 2. Play Verbal Audio
        playLastSynthesizedAudio(data.assistant_reply, data.language || lang);

        // 3. Update Extracted Data & Real-time Form Sync
        if (data.extracted_data) {
            lastExtractedVoiceData = data.extracted_data;
            updateExtractedParameterBadges(lastExtractedVoiceData, data.newly_extracted);
            syncExtractedVoiceToForm(false);
        }

        // 4. Update Progress Header
        if (progressText) {
            if (data.is_completed) {
                progressText.textContent = `Completed (${data.total_questions} Questions Assessed)`;
            } else if (data.current_question) {
                progressText.textContent = `Question ${data.current_question.index} of ${data.total_questions} (${disease.replace('_', ' ').toUpperCase()})`;
            }
        }

        // 5. Check Completion
        if (data.is_completed) {
            isConversationalAssessmentActive = false;
            if (indicator) indicator.style.background = "#10b981";
            if (stateText) stateText.textContent = "Assessment Complete - Awaiting Review";
            const completionCard = document.getElementById("voice-completion-card");
            const completionText = document.getElementById("completion-summary-text");
            if (completionCard) completionCard.style.display = "block";
            if (completionText) completionText.textContent = data.assistant_reply;
        } else {
            if (indicator) indicator.style.background = "#10b981";
            if (stateText) stateText.textContent = "Listening for Answer...";
        }

        // 6. Clear input box
        if (inp) {
            inp.value = "";
            inp.focus();
        }

    } catch (err) {
        console.error("Turn processing API error:", err);
        fallbackConversationalTurn(text, disease, lang);
    }
}

function handleQuickCommand(cmd) {
    const inp = document.getElementById("inp-voice-transcript");
    let cmdText = "skip";
    if (cmd === "skip") cmdText = "skip";
    else if (cmd === "dont_know") cmdText = "I don't know";
    else if (cmd === "back") cmdText = "go back";
    else if (cmd === "repeat") cmdText = "repeat";
    else if (cmd === "stop") cmdText = "stop";

    if (inp) inp.value = cmdText;
    submitConversationalTurn();
}

function injectScenarioAnswer(answerText) {
    const inp = document.getElementById("inp-voice-transcript");
    if (inp) inp.value = answerText;
    submitConversationalTurn();
}

function updateExtractedParameterBadges(params, newlyExtracted = {}) {
    const ageEl = document.getElementById("badge-param-age");
    const sexEl = document.getElementById("badge-param-sex");
    const bmiEl = document.getElementById("badge-param-bmi");
    const bpEl = document.getElementById("badge-param-highbp");
    const bpreadingEl = document.getElementById("badge-param-bpreading");
    const glucoseEl = document.getElementById("badge-param-glucose");
    const hrEl = document.getElementById("badge-param-heartrate");
    const smokeEl = document.getElementById("badge-param-smoker");
    const alcEl = document.getElementById("badge-param-alcohol");
    const actEl = document.getElementById("badge-param-activity");

    if (ageEl) ageEl.textContent = params.age ? `${params.age} yrs (Cat ${params.Age || '-'})` : "-";
    if (sexEl) sexEl.textContent = (params.Sex !== undefined) ? (params.Sex === 1 ? "Male" : "Female") : "-";

    const h = params.height ? `${params.height}cm` : "";
    const w = params.weight ? `${params.weight}kg` : "";
    const b = params.BMI ? `BMI ${params.BMI}` : "";
    const hwStr = [h, w, b].filter(Boolean).join(" / ");
    if (bmiEl) bmiEl.textContent = hwStr || "-";

    if (bpEl) bpEl.textContent = (params.HighBP !== undefined) ? (params.HighBP === 1 ? "Yes (High)" : "No (Normal)") : "-";

    if (bpreadingEl) {
        if (params.systolic_bp && params.diastolic_bp) {
            bpreadingEl.textContent = `${params.systolic_bp}/${params.diastolic_bp} mmHg`;
        } else if (params.trestbps) {
            bpreadingEl.textContent = `${params.trestbps} mmHg`;
        } else {
            bpreadingEl.textContent = "-";
        }
    }

    if (glucoseEl) {
        const g = params.avg_glucose_level || params.fasting_glucose || params.bgr;
        glucoseEl.textContent = g ? `${g} mg/dL` : "-";
    }

    if (hrEl) {
        const hr = params.heart_rate || params.thalach;
        hrEl.textContent = hr ? `${hr} bpm` : "-";
    }

    if (smokeEl) smokeEl.textContent = (params.Smoker !== undefined) ? (params.Smoker === 1 ? "Yes (Smoker)" : "No") : "-";
    if (alcEl) alcEl.textContent = (params.HvyAlcoholConsump !== undefined) ? (params.HvyAlcoholConsump === 1 ? "Heavy / Regular" : "None / Moderate") : "-";
    if (actEl) actEl.textContent = (params.PhysActivity !== undefined) ? (params.PhysActivity === 1 ? "Active" : "Sedentary") : "-";

    // Highlight newly updated cards with a brief glow
    if (newlyExtracted) {
        const mapCards = {
            "Age": "card-param-age", "age": "card-param-age",
            "Sex": "card-param-sex", "sex": "card-param-sex",
            "height": "card-param-bmi", "weight": "card-param-bmi", "BMI": "card-param-bmi",
            "HighBP": "card-param-highbp",
            "systolic_bp": "card-param-bp", "diastolic_bp": "card-param-bp", "trestbps": "card-param-bp",
            "fasting_glucose": "card-param-glucose", "avg_glucose_level": "card-param-glucose",
            "heart_rate": "card-param-heartrate", "thalach": "card-param-heartrate",
            "Smoker": "card-param-smoker",
            "HvyAlcoholConsump": "card-param-alcohol",
            "PhysActivity": "card-param-activity"
        };
        for (const key of Object.keys(newlyExtracted)) {
            const cardId = mapCards[key];
            if (cardId) {
                const card = document.getElementById(cardId);
                if (card) {
                    card.style.borderColor = "var(--accent-teal)";
                    card.style.boxShadow = "0 0 10px rgba(6, 182, 212, 0.4)";
                    setTimeout(() => {
                        card.style.borderColor = "var(--border-color)";
                        card.style.boxShadow = "none";
                    }, 2500);
                }
            }
        }
    }
}

function syncExtractedVoiceToForm(showToast = true) {
    const params = lastExtractedVoiceData;
    if (!params || Object.keys(params).length === 0) return;

    // Age Mapping
    if (params.Age !== undefined) {
        const inpAge = document.getElementById("inp-age");
        if (inpAge) inpAge.value = String(params.Age);
    } else if (params.age !== undefined) {
        const a = parseFloat(params.age);
        let cat = 1;
        if (a < 25) cat = 1; else if (a < 30) cat = 2; else if (a < 35) cat = 3;
        else if (a < 40) cat = 4; else if (a < 45) cat = 5; else if (a < 50) cat = 6;
        else if (a < 55) cat = 7; else if (a < 60) cat = 8; else if (a < 65) cat = 9;
        else if (a < 70) cat = 10; else if (a < 75) cat = 11; else if (a < 80) cat = 12;
        else cat = 13;
        const inpAge = document.getElementById("inp-age");
        if (inpAge) inpAge.value = String(cat);
    }

    // Sex Mapping
    if (params.Sex !== undefined) {
        const inpSex = document.getElementById("inp-sex");
        if (inpSex) inpSex.value = String(params.Sex);
    }

    // BMI Mapping
    if (params.BMI !== undefined) {
        const inpBmi = document.getElementById("inp-bmi");
        if (inpBmi) inpBmi.value = parseFloat(params.BMI).toFixed(1);
    }

    // High Blood Pressure Mapping
    if (params.HighBP !== undefined) {
        const inpHbp = document.getElementById("inp-highbp");
        if (inpHbp) inpHbp.value = String(params.HighBP);
    }

    // High Cholesterol Mapping
    if (params.HighChol !== undefined) {
        const inpHc = document.getElementById("inp-highchol");
        if (inpHc) inpHc.value = String(params.HighChol);
        const inpCc = document.getElementById("inp-cholcheck");
        if (inpCc) inpCc.value = "1";
    }

    // Smoking Mapping
    if (params.Smoker !== undefined) {
        const inpSmk = document.getElementById("inp-smoker");
        if (inpSmk) inpSmk.value = String(params.Smoker);
    }

    // Heart Disease Mapping
    if (params.HeartDiseaseorAttack !== undefined) {
        const inpHd = document.getElementById("inp-heartdisease");
        if (inpHd) inpHd.value = String(params.HeartDiseaseorAttack);
    }

    // Stroke Mapping
    if (params.Stroke !== undefined) {
        const inpStk = document.getElementById("inp-stroke");
        if (inpStk) inpStk.value = String(params.Stroke);
    }

    // Physical Activity Mapping
    if (params.PhysActivity !== undefined) {
        const inpPa = document.getElementById("inp-physactivity");
        if (inpPa) inpPa.value = String(params.PhysActivity);
    }

    // Fruits & Veggies Mapping
    if (params.Fruits !== undefined) {
        const inpFr = document.getElementById("inp-fruits");
        if (inpFr) inpFr.value = String(params.Fruits);
    }
    if (params.Veggies !== undefined) {
        const inpVg = document.getElementById("inp-veggies");
        if (inpVg) inpVg.value = String(params.Veggies);
    }

    // Heavy Alcohol Consumption
    if (params.HvyAlcoholConsump !== undefined) {
        const inpAlc = document.getElementById("inp-alcohol");
        if (inpAlc) inpAlc.value = String(params.HvyAlcoholConsump);
    }

    // Also store into customDiseaseParams for active target disease
    const activeDis = document.getElementById("select-target-disease") ? document.getElementById("select-target-disease").value : "diabetes";
    if (!customDiseaseParams[activeDis]) customDiseaseParams[activeDis] = {};
    Object.assign(customDiseaseParams[activeDis], params);

    // Update status badge
    const syncBadge = document.getElementById("sync-status-badge");
    if (syncBadge) {
        syncBadge.textContent = "✅ Real-Time Sync Active";
        syncBadge.style.background = "rgba(16, 185, 129, 0.15)";
        syncBadge.style.color = "var(--accent-emerald)";
    }
}

function playLastSynthesizedAudio(textFallback = null, lang = "en") {
    if (lastSynthesizedAudioBase64) {
        try {
            const snd = new Audio("data:audio/wav;base64," + lastSynthesizedAudioBase64);
            snd.play().catch(e => {
                console.warn("Audio autoplay blocked by browser policy, falling back to Web Speech:", e);
                if (textFallback) speakBrowserTTS(textFallback, lang);
            });
            return;
        } catch (e) {
            console.warn("Could not play audio base64:", e);
        }
    }

    if (textFallback) {
        speakBrowserTTS(textFallback, lang);
    }
}

function speakBrowserTTS(text, lang = "en") {
    if (!("speechSynthesis" in window)) return;
    try {
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(text);
        if (lang === "hi") utter.lang = "hi-IN";
        else if (lang === "bn") utter.lang = "bn-IN";
        else utter.lang = "en-US";
        utter.rate = 1.0;
        window.speechSynthesis.speak(utter);
    } catch (e) {
        console.warn("Browser speech synthesis error:", e);
    }
}

function toggleMicrophoneCapture() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const btnText = document.getElementById("mic-btn-text");
    const micIcon = document.getElementById("mic-icon");

    if (!SpeechRecognition) {
        alert("Web Speech API is not supported in this browser. You can type your answer into the input field or click quick test scenarios.");
        return;
    }

    if (isRecordingMicrophone) {
        if (speechRecognitionInstance) speechRecognitionInstance.stop();
        isRecordingMicrophone = false;
        if (btnText) btnText.textContent = "Speak Answer";
        if (micIcon) micIcon.textContent = "🎙️";
        return;
    }

    try {
        speechRecognitionInstance = new SpeechRecognition();
        speechRecognitionInstance.continuous = false;
        speechRecognitionInstance.interimResults = false;

        const langSelect = document.getElementById("select-voice-language");
        const langVal = langSelect ? langSelect.value : "en";
        if (langVal === "hi") speechRecognitionInstance.lang = "hi-IN";
        else if (langVal === "bn") speechRecognitionInstance.lang = "bn-IN";
        else speechRecognitionInstance.lang = "en-US";

        speechRecognitionInstance.onstart = function() {
            isRecordingMicrophone = true;
            if (btnText) btnText.textContent = "Listening... (Click to Send)";
            if (micIcon) micIcon.textContent = "🔴";
            const stateText = document.getElementById("voice-state-text");
            if (stateText) stateText.textContent = "Listening for Voice...";
        };

        speechRecognitionInstance.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            const input = document.getElementById("inp-voice-transcript");
            if (input) input.value = transcript;
            submitConversationalTurn();
        };

        speechRecognitionInstance.onerror = function(event) {
            console.warn("Speech recognition notice:", event.error);
            isRecordingMicrophone = false;
            if (btnText) btnText.textContent = "Speak Answer";
            if (micIcon) micIcon.textContent = "🎙️";
        };

        speechRecognitionInstance.onend = function() {
            isRecordingMicrophone = false;
            if (btnText) btnText.textContent = "Speak Answer";
            if (micIcon) micIcon.textContent = "🎙️";
        };

        speechRecognitionInstance.start();
    } catch (e) {
        console.warn("Microphone capture error:", e);
        isRecordingMicrophone = false;
        if (btnText) btnText.textContent = "Speak Answer";
        if (micIcon) micIcon.textContent = "🎙️";
    }
}

// Client-side simulation fallback when standalone servers are starting up
let fallbackQuestionIndex = 0;
const FALLBACK_QUESTIONS = [
    { key: "age", text: "What is your age?" },
    { key: "sex", text: "What is your biological sex?" },
    { key: "height", text: "What is your height?" },
    { key: "weight", text: "What is your weight?" },
    { key: "highbp", text: "Do you have high blood pressure?" },
    { key: "smoker", text: "Do you smoke or use tobacco?" }
];

function fallbackStartConversation(disease, lang) {
    fallbackQuestionIndex = 0;
    isConversationalAssessmentActive = true;
    const bubble = document.getElementById("assistant-spoken-bubble");
    const progressText = document.getElementById("question-progress-text");
    const stateText = document.getElementById("voice-state-text");

    const q = FALLBACK_QUESTIONS[0];
    if (bubble) bubble.textContent = q.text;
    if (progressText) progressText.textContent = `Question 1 of ${FALLBACK_QUESTIONS.length} (${disease.toUpperCase()})`;
    if (stateText) stateText.textContent = "Assistant Speaking...";
    speakBrowserTTS(q.text, lang);
}

function fallbackConversationalTurn(text, disease, lang) {
    const extracted = {};
    const lower = text.toLowerCase();

    const ageM = lower.match(/\b(\d{1,3})\b/);
    if (ageM && fallbackQuestionIndex === 0) extracted.age = parseFloat(ageM[1]);
    if (lower.includes("male") || lower.includes("man")) extracted.Sex = 1;
    if (lower.includes("female") || lower.includes("woman")) extracted.Sex = 0;
    if (lower.includes("high bp") || lower.includes("yes")) extracted.HighBP = 1;
    if (lower.includes("smoke")) extracted.Smoker = 1;

    Object.assign(lastExtractedVoiceData, extracted);
    updateExtractedParameterBadges(lastExtractedVoiceData, extracted);
    syncExtractedVoiceToForm(false);

    fallbackQuestionIndex++;
    const bubble = document.getElementById("assistant-spoken-bubble");
    const progressText = document.getElementById("question-progress-text");

    if (fallbackQuestionIndex < FALLBACK_QUESTIONS.length) {
        const nextQ = FALLBACK_QUESTIONS[fallbackQuestionIndex];
        const reply = `Got it, recorded. ${nextQ.text}`;
        if (bubble) bubble.textContent = reply;
        if (progressText) progressText.textContent = `Question ${fallbackQuestionIndex + 1} of ${FALLBACK_QUESTIONS.length}`;
        speakBrowserTTS(reply, lang);
    } else {
        const comp = "I've collected the information needed for your assessment. I'll show you what I understood so you can review and correct anything before we continue.";
        if (bubble) bubble.textContent = comp;
        const compCard = document.getElementById("voice-completion-card");
        if (compCard) compCard.style.display = "block";
        speakBrowserTTS(comp, lang);
    }
}

