const sampleResult = {
  ranking_source: "ctr",
  cold_start: false,
  fallback_reason: null,
  recommendations: [
    {
      news_id: "N64222",
      score: 0.91,
      title: "Markets digest latest policy signals as technology shares rise",
      text: "A concise market brief summarizing the signals that moved major technology and consumer stocks during the session.",
      category: "finance",
    },
    {
      news_id: "N11890",
      score: 0.84,
      title: "Health researchers publish new findings on sleep consistency",
      text: "The report links stable sleep schedules with better long-term health indicators across a multi-year population sample.",
      category: "health",
    },
    {
      news_id: "N40118",
      score: 0.79,
      title: "Local teams prepare for a packed weekend sports schedule",
      text: "Coaches and analysts preview key matchups, roster changes, and the conditions likely to shape the weekend results.",
      category: "sports",
    },
    {
      news_id: "N22509",
      score: 0.73,
      title: "New consumer devices put battery life ahead of thin design",
      text: "Manufacturers are adjusting product priorities as buyers ask for practical durability over smaller year-to-year changes.",
      category: "technology",
    },
    {
      news_id: "N90310",
      score: 0.68,
      title: "Travel planners see renewed interest in regional rail routes",
      text: "Seasonal travel data points to stronger demand for short-haul rail itineraries and flexible weekend schedules.",
      category: "travel",
    },
  ],
};

const elements = {
  input: document.querySelector("[data-json-input]"),
  error: document.querySelector("[data-error]"),
  results: document.querySelector("[data-results]"),
  template: document.querySelector("#result-template"),
  loadSample: document.querySelector("[data-load-sample]"),
  renderJson: document.querySelector("[data-render-json]"),
  copyCommand: document.querySelector("[data-copy-command]"),
  source: document.querySelector("[data-summary-source]"),
  cold: document.querySelector("[data-summary-cold]"),
  count: document.querySelector("[data-summary-count]"),
  caption: document.querySelector("[data-result-caption]"),
  modeButtons: document.querySelectorAll("[data-mode]"),
};

function normalizeResult(result) {
  if (!result || !Array.isArray(result.recommendations)) {
    throw new Error("JSON must include a recommendations array.");
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

function setError(message) {
  elements.error.hidden = !message;
  elements.error.textContent = message || "";
}

function renderResult(rawResult) {
  const result = normalizeResult(rawResult);
  elements.results.innerHTML = "";
  elements.source.textContent = result.ranking_source.toUpperCase();
  elements.cold.textContent = result.cold_start ? "Yes" : "No";
  elements.count.textContent = String(result.recommendations.length);
  elements.caption.textContent = result.fallback_reason
    ? `Fallback active: ${result.fallback_reason}`
    : `Ranked by ${result.ranking_source.toUpperCase()}, diversified by MMR`;

  result.recommendations.forEach((item, index) => {
    const node = elements.template.content.cloneNode(true);
    node.querySelector("[data-rank]").textContent = `#${index + 1}`;
    node.querySelector("[data-category]").textContent = item.category;
    node.querySelector("[data-title]").textContent = item.title;
    node.querySelector("[data-text]").textContent = item.text;
    node.querySelector("[data-news-id]").textContent = item.news_id;
    node.querySelector("[data-score-meter]").value = Math.max(0, Math.min(1, item.score));
    node.querySelector("[data-score]").textContent = item.score.toFixed(3);
    elements.results.appendChild(node);
  });
}

function renderFromInput() {
  try {
    const parsed = JSON.parse(elements.input.value);
    renderResult(parsed);
    setError("");
  } catch (error) {
    setError(error.message);
  }
}

function loadSample(mode = "ctr") {
  const next = structuredClone(sampleResult);
  if (mode === "similarity") {
    next.ranking_source = "similarity";
    next.fallback_reason = "CTR model artifact was not available.";
    next.recommendations = next.recommendations.map((item, index) => ({
      ...item,
      score: Math.max(0.4, item.score - index * 0.035),
    }));
  }
  elements.input.value = JSON.stringify(next, null, 2);
  renderResult(next);
  setError("");
}

elements.loadSample.addEventListener("click", () => loadSample(currentMode()));
elements.renderJson.addEventListener("click", renderFromInput);
elements.copyCommand.addEventListener("click", async () => {
  const command = "python -m src.run_recommend_cli --user U8125 --topk 10 --json";
  await navigator.clipboard?.writeText(command);
  elements.copyCommand.textContent = "Copied";
  setTimeout(() => {
    elements.copyCommand.textContent = "Copy";
  }, 1200);
});

function currentMode() {
  return document.querySelector("[data-mode].active")?.dataset.mode || "ctr";
}

elements.modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    elements.modeButtons.forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    loadSample(button.dataset.mode);
  });
});

loadSample("ctr");
