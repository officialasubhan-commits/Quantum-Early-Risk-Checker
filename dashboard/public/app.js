document.addEventListener("DOMContentLoaded", () => {
    fetchDashboardData();
    // Auto refresh every 10 seconds
    setInterval(fetchDashboardData, 10000);
});

async function fetchDashboardData() {
    try {
        const response = await fetch("/api/data");
        if (!response.ok) {
            throw new Error(`HTTP Error ${response.status}`);
        }
        const data = await response.json();
        renderDashboard(data);
    } catch (err) {
        console.error("Failed to fetch dashboard data:", err);
    }
}

function renderDashboard(data) {
    // 1. Render Header & Current Task Info
    if (data.project_info) {
        document.getElementById("header-status").textContent = data.project_info.status || "Development";
        document.getElementById("header-env").textContent = data.project_info.environment || "Local Dev Server";
    }

    if (data.current_task) {
        const ct = data.current_task;
        document.getElementById("val-current-phase").textContent = ct.current_phase || "N/A";
        document.getElementById("val-current-task").textContent = ct.current_task || "N/A";
        document.getElementById("val-last-completed").textContent = ct.last_completed_task || "N/A";
        document.getElementById("val-next-task").textContent = ct.next_task || "N/A";

        const pct = ct.progress_percentage || 0;
        document.getElementById("progress-pct-badge").textContent = `${pct}% Complete`;
        document.getElementById("main-progress-fill").style.width = `${pct}%`;
    }

    // 2. Render ML Status & Metrics
    if (data.ml_status) {
        const ml = data.ml_status;
        document.getElementById("ml-records").textContent = ml.dataset_records || "Not available";
        document.getElementById("ml-features").textContent = ml.num_features || "Not available";
        document.getElementById("ml-best-model").textContent = ml.best_classical_model || "Not available";
        document.getElementById("ml-accuracy").textContent = ml.accuracy || "Not available";
        document.getElementById("ml-precision").textContent = ml.precision || "Not available";
        document.getElementById("ml-recall").textContent = ml.recall || "Not available";
        document.getElementById("ml-f1").textContent = ml.f1 || "Not available";
        document.getElementById("ml-roc-auc").textContent = ml.roc_auc || "Not available";
    }

    // 3. Render Project Progress Phases
    if (data.phases) {
        renderPhases(data.phases);
    }

    // 4. Render Architecture Flow
    if (data.architecture_flow) {
        renderArchitectureFlow(data.architecture_flow);
    }

    // 5. Render Task Checklist
    if (data.checklist) {
        renderChecklist(data.checklist);
    }
}

function renderPhases(phases) {
    const container = document.getElementById("phases-grid-container");
    container.innerHTML = "";

    phases.forEach(phase => {
        const card = document.createElement("div");
        card.className = "phase-card";

        const statusClean = phase.status.toLowerCase().replace(" ", "_");
        const statusBadgeClass = `badge-${statusClean}`;

        card.innerHTML = `
            <div class="phase-name">${escapeHtml(phase.name)}</div>
            <div class="phase-desc">${escapeHtml(phase.description || '')}</div>
            <span class="badge ${statusBadgeClass}">${escapeHtml(phase.status)}</span>
        `;
        container.appendChild(card);
    });
}

function renderArchitectureFlow(flowSteps) {
    const container = document.getElementById("architecture-flow-container");
    container.innerHTML = "";

    flowSteps.forEach((step, idx) => {
        const stepEl = document.createElement("div");
        stepEl.className = "flow-step";

        stepEl.innerHTML = `
            <div class="step-num">${step.step}</div>
            <div class="step-info">
                <span class="step-title">${escapeHtml(step.title)}</span>
                <span class="step-desc">${escapeHtml(step.desc)}</span>
            </div>
        `;
        container.appendChild(stepEl);
    });
}

function renderChecklist(checklist) {
    const container = document.getElementById("checklist-container");
    container.innerHTML = "";

    checklist.forEach(item => {
        const itemEl = document.createElement("div");
        itemEl.className = "checklist-item";

        const statusOptions = ["NOT STARTED", "IN PROGRESS", "COMPLETED", "BLOCKED"];
        const optionsHtml = statusOptions.map(opt => {
            const selected = item.status === opt ? "selected" : "";
            return `<option value="${opt}" ${selected}>${opt}</option>`;
        }).join("");

        itemEl.innerHTML = `
            <div class="checklist-left">
                <span class="task-id">#${item.id}</span>
                <div>
                    <span class="task-text">${escapeHtml(item.task)}</span>
                    <span class="task-phase-tag">${escapeHtml(item.phase)}</span>
                </div>
            </div>
            <select class="status-select" onchange="updateTaskStatus(${item.id}, this.value)">
                ${optionsHtml}
            </select>
        `;
        container.appendChild(itemEl);
    });
}

async function updateTaskStatus(taskId, newStatus) {
    try {
        const response = await fetch("/api/tasks", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ id: taskId, status: newStatus })
        });
        if (response.ok) {
            fetchDashboardData();
        } else {
            console.error("Failed to update task status");
        }
    } catch (err) {
        console.error("Error updating task status:", err);
    }
}

function escapeHtml(str) {
    if (typeof str !== "string") return str;
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, '&quot;');
}
