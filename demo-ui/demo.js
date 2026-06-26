const sampleTrainingReport = {
  generated_at: "2026-06-26T14:09:43+00:00",
  model_path: "models/ctr_model.pt",
  report_path: "models/training_report.json",
  training: {
    epochs: 8,
    batch_size: 4096,
    learning_rate: 0.001,
    device: "cpu",
    torch_threads: 4,
    lazy_dataset: false,
  },
  dataset: {
    ctr_dataset_path: "data/precompute/ctr_dataset.csv",
    total_rows: 1135225,
    used_rows: 1135225,
    train_rows: 964941,
    validation_rows: 170284,
    validation_size: 0.15,
    max_rows: null,
  },
  metrics: [
    { epoch: 1, train_loss: 0.4967, validation_auc: 0.6977 },
    { epoch: 2, train_loss: 0.4647, validation_auc: 0.7219 },
    { epoch: 3, train_loss: 0.4565, validation_auc: 0.7304 },
    { epoch: 4, train_loss: 0.4517, validation_auc: 0.7355 },
    { epoch: 5, train_loss: 0.4476, validation_auc: 0.7393 },
    { epoch: 6, train_loss: 0.4447, validation_auc: 0.7409 },
    { epoch: 7, train_loss: 0.4417, validation_auc: 0.7426 },
    { epoch: 8, train_loss: 0.4391, validation_auc: 0.7435 },
  ],
  final_metrics: { epoch: 8, train_loss: 0.4391, validation_auc: 0.7435 },
};

const sampleRecommendation = {
  ranking_source: "ctr",
  cold_start: false,
  fallback_reason: null,
  recommendations: [
    {
      news_id: "N37079",
      score: 0.48103243112564087,
      title: "Detroit To Have Residential Streets Plowed Within 24 Hours",
      text: "Contractors are clearing nearly 2,000 miles of residential streets under updated snow response guidelines.",
      category: "weather",
    },
    {
      news_id: "N52782",
      score: 0.3493347764015198,
      title: "40 Best Belly-Shrinking Foods",
      text: "A health article covering food choices associated with weight management and daily nutrition habits.",
      category: "health",
    },
    {
      news_id: "N24717",
      score: 0.3607912063598633,
      title: "New owners lay out plans for the historic Green Lantern bar",
      text: "The former watering hole on Old Troy Pike will reopen with pizza and a refreshed neighborhood concept.",
      category: "travel",
    },
    {
      news_id: "N41835",
      score: 0.35555291175842285,
      title: "Today in History: November 2",
      text: "A news digest revisiting notable events and milestones from November 2.",
      category: "news",
    },
    {
      news_id: "N52119",
      score: 0.4425748586654663,
      title: "Man, 29, in critical condition after falling 20 feet on North Side",
      text: "Pittsburgh police responded after a man fell from a steep area near Federal and South Commons streets.",
      category: "news",
    },
  ],
};

const elements = {
  reportInput: document.querySelector("[data-report-input]"),
  reportError: document.querySelector("[data-report-error]"),
  finalAuc: document.querySelector("[data-final-auc]"),
  finalLoss: document.querySelector("[data-final-loss]"),
  usedRows: document.querySelector("[data-used-rows]"),
  splitLabel: document.querySelector("[data-split-label]"),
  epochs: document.querySelector("[data-epochs]"),
  trainConfig: document.querySelector("[data-train-config]"),
  reportCaption: document.querySelector("[data-report-caption]"),
  chart: document.querySelector("[data-training-chart]"),
  epochTable: document.querySelector("[data-epoch-table]"),
  recommendationInput: document.querySelector("[data-json-input]"),
  recommendationError: document.querySelector("[data-recommendation-error]"),
  results: document.querySelector("[data-results]"),
  template: document.querySelector("#result-template"),
  source: document.querySelector("[data-summary-source]"),
  cold: document.querySelector("[data-summary-cold]"),
  count: document.querySelector("[data-summary-count]"),
  caption: document.querySelector("[data-result-caption]"),
};

const numberFormat = new Intl.NumberFormat("en-US");
const compactFormat = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function formatRows(value) {
  return compactFormat.format(Number(value) || 0);
}

function setError(element, message) {
  element.hidden = !message;
  element.textContent = message || "";
}

function normalizeTrainingReport(report) {
  if (!report || !Array.isArray(report.metrics) || report.metrics.length === 0) {
    throw new Error("Training report must include a non-empty metrics array.");
  }

  const metrics = report.metrics.map((item, index) => ({
    epoch: Number.isFinite(Number(item.epoch)) ? Number(item.epoch) : index + 1,
    train_loss: Number.isFinite(Number(item.train_loss)) ? Number(item.train_loss) : 0,
    validation_auc: Number.isFinite(Number(item.validation_auc)) ? Number(item.validation_auc) : 0,
  }));

  return {
    generated_at: report.generated_at || "",
    model_path: report.model_path || "models/ctr_model.pt",
    report_path: report.report_path || "models/training_report.json",
    training: report.training || {},
    dataset: report.dataset || {},
    metrics,
    final_metrics: report.final_metrics || metrics[metrics.length - 1],
  };
}

function renderTrainingReport(rawReport) {
  const report = normalizeTrainingReport(rawReport);
  const final = report.final_metrics || report.metrics[report.metrics.length - 1];
  const dataset = report.dataset;
  const training = report.training;

  elements.finalAuc.textContent = Number(final.validation_auc).toFixed(4);
  elements.finalLoss.textContent = Number(final.train_loss).toFixed(4);
  elements.usedRows.textContent = formatRows(dataset.used_rows || dataset.total_rows);
  elements.splitLabel.textContent = `${numberFormat.format(dataset.train_rows || 0)} train / ${numberFormat.format(dataset.validation_rows || 0)} validation`;
  elements.epochs.textContent = String(training.epochs || report.metrics.length);
  elements.trainConfig.textContent = `Batch ${numberFormat.format(training.batch_size || 0)}, ${String(training.device || "cpu").toUpperCase()}`;
  elements.reportCaption.textContent = `${report.model_path} with ${report.metrics.length} recorded epochs.`;

  renderChart(report.metrics);
  renderEpochTable(report.metrics);
}

function renderChart(metrics) {
  elements.chart.innerHTML = "";
  const maxLoss = Math.max(...metrics.map((item) => item.train_loss), 0.01);

  metrics.forEach((item) => {
    const group = document.createElement("div");
    group.className = "epoch-group";
    group.setAttribute(
      "aria-label",
      `Epoch ${item.epoch}, AUC ${item.validation_auc.toFixed(4)}, loss ${item.train_loss.toFixed(4)}`,
    );

    const bars = document.createElement("div");
    bars.className = "bar-pair";

    const auc = document.createElement("span");
    auc.className = "bar bar-auc";
    auc.style.height = `${Math.max(6, Math.min(100, item.validation_auc * 100))}%`;

    const loss = document.createElement("span");
    loss.className = "bar bar-loss";
    loss.style.height = `${Math.max(6, Math.min(100, (item.train_loss / maxLoss) * 100))}%`;

    const label = document.createElement("strong");
    label.textContent = String(item.epoch);

    bars.append(auc, loss);
    group.append(bars, label);
    elements.chart.appendChild(group);
  });
}

function renderEpochTable(metrics) {
  elements.epochTable.innerHTML = "";

  metrics.forEach((item, index) => {
    const previous = metrics[index - 1];
    const delta = previous ? item.validation_auc - previous.validation_auc : 0;
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${item.epoch}</td>
      <td>${item.train_loss.toFixed(4)}</td>
      <td>${item.validation_auc.toFixed(4)}</td>
      <td>${index === 0 ? "Baseline" : `${delta >= 0 ? "+" : ""}${delta.toFixed(4)}`}</td>
    `;
    elements.epochTable.appendChild(row);
  });
}

function normalizeRecommendation(result) {
  if (!result || !Array.isArray(result.recommendations)) {
    throw new Error("Recommendation JSON must include a recommendations array.");
  }

  return {
    ranking_source: result.ranking_source || "similarity",
    cold_start: Boolean(result.cold_start),
    fallback_reason: result.fallback_reason || null,
    recommendations: result.recommendations.map((item, index) => ({
      news_id: String(item.news_id || `N${index + 1}`),
      score: Number.isFinite(Number(item.score)) ? Number(item.score) : 0,
      title: item.title || item.news_id || `Article ${index + 1}`,
      text: item.text || "No article text was included in the CLI output.",
      category: item.category || "news",
    })),
  };
}

function renderRecommendation(rawResult) {
  const result = normalizeRecommendation(rawResult);
  elements.results.innerHTML = "";
  elements.source.textContent = result.ranking_source.toUpperCase();
  elements.cold.textContent = result.cold_start ? "Yes" : "No";
  elements.count.textContent = String(result.recommendations.length);
  elements.caption.textContent = result.fallback_reason
    ? `Fallback active: ${result.fallback_reason}`
    : `Ranked by ${result.ranking_source.toUpperCase()}, diversified by MMR.`;

  result.recommendations.forEach((item, index) => {
    const node = elements.template.content.cloneNode(true);
    node.querySelector("[data-rank]").textContent = `#${index + 1}`;
    node.querySelector("[data-category]").textContent = item.category;
    node.querySelector("[data-news-id]").textContent = item.news_id;
    node.querySelector("[data-title]").textContent = item.title;
    node.querySelector("[data-text]").textContent = item.text;
    node.querySelector("[data-score-meter]").value = Math.max(0, Math.min(1, item.score));
    node.querySelector("[data-score]").textContent = item.score.toFixed(3);
    elements.results.appendChild(node);
  });
}

function loadTrainingSample() {
  const report = clone(sampleTrainingReport);
  elements.reportInput.value = JSON.stringify(report, null, 2);
  renderTrainingReport(report);
  setError(elements.reportError, "");
}

function loadRecommendationSample() {
  const result = clone(sampleRecommendation);
  elements.recommendationInput.value = JSON.stringify(result, null, 2);
  renderRecommendation(result);
  setError(elements.recommendationError, "");
}

function renderReportFromInput() {
  try {
    const parsed = JSON.parse(elements.reportInput.value);
    renderTrainingReport(parsed);
    setError(elements.reportError, "");
  } catch (error) {
    setError(elements.reportError, error.message);
  }
}

function renderRecommendationFromInput() {
  try {
    const parsed = JSON.parse(elements.recommendationInput.value);
    renderRecommendation(parsed);
    setError(elements.recommendationError, "");
  } catch (error) {
    setError(elements.recommendationError, error.message);
  }
}

async function copyText(button) {
  const text = button.dataset.copy;
  try {
    await navigator.clipboard?.writeText(text);
    button.textContent = "Copied";
  } catch (error) {
    button.textContent = "Copy failed";
  }
  setTimeout(() => {
    button.textContent = "Copy";
  }, 1400);
}

document.querySelectorAll("[data-load-report]").forEach((button) => {
  button.addEventListener("click", loadTrainingSample);
});

document.querySelectorAll("[data-load-recommendation]").forEach((button) => {
  button.addEventListener("click", loadRecommendationSample);
});

document.querySelector("[data-render-report]").addEventListener("click", renderReportFromInput);
document.querySelector("[data-render-json]").addEventListener("click", renderRecommendationFromInput);

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", () => copyText(button));
});

loadTrainingSample();
loadRecommendationSample();
