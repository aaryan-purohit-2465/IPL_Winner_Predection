const state = {
  dashboard: null,
};

const elements = {
  datasetName: document.getElementById("dataset-name"),
  datasetStats: document.getElementById("dataset-stats"),
  metricCards: document.getElementById("metric-cards"),
  insightsList: document.getElementById("insights-list"),
  winsChart: document.getElementById("wins-chart"),
  tossChart: document.getElementById("toss-chart"),
  healthStatus: document.getElementById("health-status"),
  uploadForm: document.getElementById("upload-form"),
  predictionForm: document.getElementById("prediction-form"),
  predictionResult: document.getElementById("prediction-result"),
};

async function apiFetch(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Something went wrong.");
  }
  return data;
}

function renderStatCards(container, items) {
  container.innerHTML = items
    .map(
      (item) => `
        <article class="stat-card">
          <span class="stat-label">${item.label}</span>
          <span class="stat-value">${item.value}</span>
        </article>
      `
    )
    .join("");
}

function renderInsights(report) {
  const winningLeader = Object.entries(report.top_winning_teams || {})[0];
  const tossLeader = Object.entries(report.top_toss_winners || {})[0];
  const insights = [
    winningLeader
      ? `${winningLeader[0]} leads this dataset with ${winningLeader[1]} recorded match wins.`
      : "Upload a dataset to inspect team win trends.",
    tossLeader
      ? `${tossLeader[0]} has the strongest toss record in the current data with ${tossLeader[1]} toss wins.`
      : "Toss performance insights will appear here once the data loads.",
    `The dashboard currently spans ${report.total_seasons || 0} IPL seasons and ${report.total_matches || 0} matches.`,
  ];

  elements.insightsList.innerHTML = insights
    .map((text) => `<article class="insight-card">${text}</article>`)
    .join("");
}

function populateSelect(id, options, preferredValue = null) {
  const select = document.getElementById(id);
  const markup = options.map((value) => `<option value="${value}">${value}</option>`).join("");
  select.innerHTML = markup;
  if (preferredValue && options.includes(preferredValue)) {
    select.value = preferredValue;
  }
}

function hydratePredictionForm(dataset) {
  const latestSeason = dataset.seasons[dataset.seasons.length - 1] || new Date().getFullYear();
  document.getElementById("season").value = latestSeason;
  document.getElementById("match_year").value = latestSeason;

  populateSelect("team1", dataset.teams);
  populateSelect("team2", dataset.teams, dataset.teams[1] || dataset.teams[0]);
  populateSelect("toss_winner", dataset.teams);
  populateSelect("venue", dataset.venues);
  populateSelect("city", dataset.cities.length ? dataset.cities : ["Unknown"]);
}

function renderDashboard(data) {
  state.dashboard = data;
  elements.datasetName.textContent = data.dataset.name;

  renderStatCards(elements.datasetStats, [
    { label: "Matches", value: data.dataset.dataset_rows },
    { label: "Teams", value: data.dataset.teams.length },
    { label: "Venues", value: data.dataset.venues.length },
    { label: "Seasons", value: data.dataset.seasons.length },
  ]);

  renderStatCards(elements.metricCards, [
    { label: "Train Rows", value: data.metrics.train_rows },
    { label: "Test Rows", value: data.metrics.test_rows },
    { label: "Accuracy", value: `${(data.metrics.accuracy * 100).toFixed(1)}%` },
    { label: "Weighted F1", value: `${(data.metrics.weighted_f1 * 100).toFixed(1)}%` },
  ]);

  elements.winsChart.src = data.charts.top_winning_teams || "";
  elements.tossChart.src = data.charts.top_toss_winners || "";

  renderInsights(data.report);
  hydratePredictionForm(data.dataset);
}

function showToast(message) {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 2600);
}

async function loadHealth() {
  try {
    const response = await apiFetch("/api/health");
    elements.healthStatus.textContent = response.status === "ok" ? "Backend online" : "Backend unavailable";
  } catch (error) {
    elements.healthStatus.textContent = "Backend unavailable";
  }
}

async function loadDashboard() {
  const data = await apiFetch("/api/dashboard-data");
  renderDashboard(data);
}

elements.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(elements.uploadForm);
  try {
    const data = await apiFetch("/api/upload", {
      method: "POST",
      body: formData,
    });
    renderDashboard(data);
    showToast("Dataset uploaded and model retrained.");
  } catch (error) {
    showToast(error.message);
  }
});

elements.predictionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(elements.predictionForm);
  const payload = Object.fromEntries(form.entries());
  payload.season = Number(payload.season);
  payload.match_year = Number(payload.match_year);
  payload.dl_applied = Number(payload.dl_applied || 0);

  if (payload.team1 === payload.team2) {
    showToast("Choose two different teams for prediction.");
    return;
  }

  try {
    const prediction = await apiFetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const rows = prediction.top_probabilities
      .map(
        (entry) => `
          <div class="probability-row">
            <span>${entry.team}</span>
            <span>${(entry.probability * 100).toFixed(1)}%</span>
          </div>
        `
      )
      .join("");

    elements.predictionResult.classList.remove("empty");
    elements.predictionResult.innerHTML = `
      <div>Predicted winner</div>
      <strong>${prediction.predicted_winner}</strong>
      <div>Confidence: ${(prediction.confidence * 100).toFixed(1)}%</div>
      <div class="probability-list">${rows}</div>
    `;
  } catch (error) {
    showToast(error.message);
  }
});

Promise.all([loadHealth(), loadDashboard()]).catch((error) => {
  showToast(error.message);
});
