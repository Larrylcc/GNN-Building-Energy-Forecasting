function formatNumber(value) {
  return Number(value).toLocaleString("zh-CN");
}

function chartLabel(item) {
  return item.label || item.primary_use || item.column || item.meter || "";
}

function chartValue(item) {
  return Number(item.count ?? item.missing_count ?? 0);
}

function chartSuffix(item) {
  if (item.percent !== undefined) {
    return `${formatNumber(chartValue(item))} (${item.percent}%)`;
  }
  if (item.missing_percent !== undefined) {
    return `${formatNumber(chartValue(item))} (${item.missing_percent}%)`;
  }
  return formatNumber(chartValue(item));
}

function renderBarChart(container) {
  const values = JSON.parse(container.dataset.values || "[]");
  const maxValue = Math.max(...values.map(chartValue), 1);
  container.innerHTML = "";

  values.forEach((item) => {
    const row = document.createElement("div");
    row.className = "bar-row";

    const label = document.createElement("div");
    label.className = "bar-label";
    label.textContent = chartLabel(item);
    label.title = chartLabel(item);

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max((chartValue(item) / maxValue) * 100, 2)}%`;
    track.appendChild(fill);

    const value = document.createElement("div");
    value.className = "bar-value";
    value.textContent = chartSuffix(item);

    row.append(label, track, value);
    container.appendChild(row);
  });
}

function setupSampleTabs() {
  const buttons = document.querySelectorAll("[data-sample-target]");
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      buttons.forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".sample-panel").forEach((panel) => panel.classList.remove("active"));
      button.classList.add("active");
      document.querySelector(`#sample-${button.dataset.sampleTarget}`).classList.add("active");
    });
  });
}

function setupWindCompass() {
  document.querySelectorAll(".wind-compass").forEach((container) => {
    function setDetail(item, button) {
      container.querySelectorAll(".wind-sector").forEach((sector) => sector.classList.remove("active"));
      button.classList.add("active");
      const detail = container.querySelector("[data-wind-detail]");
      detail.innerHTML = `
        <strong>编码 ${item.code} · ${item.label}</strong>
        <span>${item.formula}</span>
        <span>计算式：int(wind_direction / 22.5) % 16</span>
      `;
    }

    container.querySelectorAll(".wind-sector").forEach((button) => {
      const item = {
        code: button.dataset.code,
        label: button.dataset.label,
        formula: button.dataset.formula,
      };
      button.addEventListener("mouseenter", () => setDetail(item, button));
      button.addEventListener("focus", () => setDetail(item, button));
      button.addEventListener("click", () => setDetail(item, button));
    });
  });
}

document.querySelectorAll(".bar-chart").forEach(renderBarChart);
setupSampleTabs();
setupWindCompass();
