/* ── State ──────────────────────────────────────────────── */
const state = {
  slideId: null,
  polling: null,
  lastPollTime: null,
  nucleiSummary: null,
  nucleiInstances: [],
  nucleiInstancesUrl: null,
  selectedNucleiRank: null,
  fillContours: true,
  metricComparison: null,
  patchMetrics: [],
  llmReport: null,
  milResult: null,
  // heatmap canvas cache
  heatmapGeoJson: null,
  heatmapGeoJsonUrl: null,
  heatmapThumbnailUrl: null,
  heatmapBgImage: null,       // HTMLImageElement cache
  heatmapShowBg: true,
  heatmapFillPatch: true,
  traceExpanded: true,
  // mode
  uiMode: "manual",
  // agent full-run
  agentLivePolling: null,
  _lastAgentStage: null,
};

/* ── DOM refs ────────────────────────────────────────────── */
const healthPill          = document.querySelector("#health-pill");
const slideFileInput      = document.querySelector("#slide-file");
const selectedSlide       = document.querySelector("#selected-slide");
const hfTokenInput        = document.querySelector("#hf-token");
const uploadButton        = document.querySelector("#upload-slide");
const uploadProgressBar   = document.querySelector("#upload-progress-bar");
const uploadProgressFill  = document.querySelector("#upload-progress-fill");
const uploadProgressLabel = document.querySelector("#upload-progress-label");
const manualControls      = document.querySelector("#manual-controls");
const agentControls       = document.querySelector("#agent-controls");
const modeManualBtn       = document.querySelector("#mode-manual");
const modeAgentBtn        = document.querySelector("#mode-agent");
const preprocessButton    = document.querySelector("#run-preprocess");
const reasonPreprocess    = document.querySelector("#reason-preprocess");
const inferenceButton     = document.querySelector("#run-inference");
const reasonInference     = document.querySelector("#reason-inference");
const nucleiButton        = document.querySelector("#run-nuclei");
const reasonNuclei        = document.querySelector("#reason-nuclei");
const runAgentFullBtn     = document.querySelector("#run-agent-full");
const agentRunStatus      = document.querySelector("#agent-run-status");
const agentLivePanel      = document.querySelector("#agent-live-panel");
const agentLiveState      = document.querySelector("#agent-live-state");
const agentLiveLog        = document.querySelector("#agent-live-log");
const psPreprocess        = document.querySelector("#ps-preprocess");
const psInference         = document.querySelector("#ps-inference");
const psNuclei            = document.querySelector("#ps-nuclei");
const psAnalysis          = document.querySelector("#ps-analysis");
const generateReportButton= document.querySelector("#generate-report");
const statusSlideId       = document.querySelector("#status-slide-id");
const statusCopy          = document.querySelector("#status-copy");
const statusPreprocess    = document.querySelector("#status-preprocess");
const statusInference     = document.querySelector("#status-inference");
const statusNuclei        = document.querySelector("#status-nuclei");
const statusReport        = document.querySelector("#status-report");
const timePreprocess      = document.querySelector("#time-preprocess");
const timeInference       = document.querySelector("#time-inference");
const timeNuclei          = document.querySelector("#time-nuclei");
const timeReport          = document.querySelector("#time-report");
const statusOutput        = document.querySelector("#status-output");
const logitOutput         = document.querySelector("#logit-output");
const confidencePanel     = document.querySelector("#confidence-panel");
const pollIndicator       = document.querySelector("#poll-indicator");
const heatmapFrame        = document.querySelector("#heatmap-frame");
const heatmapCanvas       = document.querySelector("#heatmap-canvas");
const heatmapImage        = document.querySelector("#heatmap-image");
const emptyHeatmap        = document.querySelector("#empty-heatmap");
const topkGrid            = document.querySelector("#topk-grid");
const nucleiTotal         = document.querySelector("#nuclei-total");
const nucleiPatches       = document.querySelector("#nuclei-patches");
const nucleiModel         = document.querySelector("#nuclei-model");
const nucleiTypeCounts    = document.querySelector("#nuclei-type-counts");
const nucleiOverlayGrid   = document.querySelector("#nuclei-overlay-grid");
const nucleiDetailTitle   = document.querySelector("#nuclei-detail-title");
const nucleiDetailFrame   = document.querySelector("#nuclei-detail-frame");
const nucleiDetailError   = document.querySelector("#nuclei-detail-error");
const contourFillToggle   = document.querySelector("#contour-fill-toggle");
const metricsReference    = document.querySelector("#metrics-reference");
const metricsPatchCount   = document.querySelector("#metrics-patch-count");
const metricComparisonTable = document.querySelector("#metric-comparison-table");
const patchMetricTable    = document.querySelector("#patch-metric-table");
const reportState         = document.querySelector("#report-state");
const diagnosticReport    = document.querySelector("#diagnostic-report");
const agentTraceSection   = document.querySelector("#agent-trace-section");
const agentTrace          = document.querySelector("#agent-trace");
const agentTraceToggle    = document.querySelector("#agent-trace-toggle");
const agentLogConsole     = document.querySelector("#agent-log-console");
const agentLogStateBadge  = document.querySelector("#agent-log-state-badge");
const agentLogIter        = document.querySelector("#agent-log-iter");
const agentOnlyTab        = document.querySelector(".agent-only-tab");
const geojsonDownload     = document.querySelector("#geojson-download");
const topkDownload        = document.querySelector("#topk-download");
const nucleiGeojsonDownload   = document.querySelector("#nuclei-geojson-download");
const nucleiCountsDownload    = document.querySelector("#nuclei-counts-download");
const patchMetricsDownload    = document.querySelector("#patch-metrics-download");
const metricComparisonDownload= document.querySelector("#metric-comparison-download");
const reportDownload      = document.querySelector("#report-download");
const tabButtons          = document.querySelectorAll(".tab-button");
const tabPanels           = document.querySelectorAll(".tab-panel");

/* ── 세포 색상 상수 ─────────────────────────────────────── */
const typeNames = {
  0: "Background", 1: "Neoplastic", 2: "Inflammatory",
  3: "Connective",  4: "Dead",       5: "Epithelial",
  "0": "Background", "1": "Neoplastic", "2": "Inflammatory",
  "3": "Connective",  "4": "Dead",       "5": "Epithelial",
  Background: "Background", Neoplastic: "Neoplastic", Inflammatory: "Inflammatory",
  Connective: "Connective",  Dead: "Dead",             Epithelial: "Epithelial",
};

const typeColors = {
  Background:  "#777777",
  Neoplastic:  "#ff0000",
  Inflammatory:"#22dd4d",
  Connective:  "#235cec",
  Dead:        "#e07b2a",   // 개선: 기존 #feff00(황색) → 주황색으로 가독성 향상
  Epithelial:  "#ff9f44",
  Unknown:     "#b4b4b4",
};

/* ── 유틸 함수 ──────────────────────────────────────────── */
function setBusy(button, busyText, busy, spinner = false) {
  if (busy) {
    button.dataset.originalText = button.textContent;
    button.innerHTML = spinner
      ? `<span class="btn-spinner"></span>${busyText}`
      : busyText;
    button.disabled = true;
  } else {
    button.textContent = button.dataset.originalText || button.textContent;
    button.disabled = false;
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function fmt(value, digits = 3) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  if (Math.abs(n) > 0 && Math.abs(n) < 0.001) return n.toExponential(2);
  return n.toFixed(digits);
}

function slideIdFromFilename(filename) {
  const base = filename.split(/[\\/]/).pop() || filename;
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(0, dot) : base;
}

function normalizeType(value) {
  return typeNames[value] || typeNames[String(value)] || String(value || "Unknown");
}

function rgbaFromHex(hex, alpha) {
  const c = hex.replace("#", "");
  return `rgba(${parseInt(c.slice(0,2),16)},${parseInt(c.slice(2,4),16)},${parseInt(c.slice(4,6),16)},${alpha})`;
}

function polygonPoints(points) {
  return (points || []).map(p => `${Number(p[0]).toFixed(2)},${Number(p[1]).toFixed(2)}`).join(" ");
}

function instancesForRank(rank) {
  return state.nucleiInstances.filter(i => Number(i.patch_rank) === Number(rank));
}

/* ── 상태 뱃지 / 다운로드 ────────────────────────────────── */
function setStateBadge(el, value) {
  el.textContent = value;
  el.className = `state ${value.replace("_", "-")}`;
}

function setStageTime(el, seconds) {
  const v = Number(seconds);
  el.textContent = Number.isFinite(v) ? `${v.toFixed(1)}s` : "";
}

function setDownload(anchor, url) {
  if (url) {
    anchor.href = url;
    anchor.classList.remove("disabled");
  } else {
    anchor.href = "#";
    anchor.classList.add("disabled");
  }
}

/* ── 버튼 비활성 이유 표시 ───────────────────────────────── */
function updateButtonReasons(data) {
  const copyDone = data?.copy_status === "completed";
  const prepDone = data?.preprocess_status === "completed";
  const infDone  = data?.inference_status === "completed";

  reasonPreprocess.textContent = copyDone ? "" : "Slide 준비 후 활성화됩니다";
  reasonInference.textContent  = prepDone ? "" : "Preprocess 완료 후 활성화됩니다";
  reasonNuclei.textContent     = infDone  ? "" : "Model Inference 완료 후 활성화됩니다";

  const nucleiDone = data?.nuclei_status === "completed";
  const generateReasonEl = document.querySelector("#generate-report + .disabled-reason");
  if (generateReasonEl) {
    generateReasonEl.textContent = nucleiDone ? "" : "Nuclei Analysis 완료 후 활성화됩니다";
  }
}

/* ── 탭 전환 (단계 잠금 포함) ────────────────────────────── */
function setActiveTab(tabId, force = false) {
  if (!force) {
    const btn = document.querySelector(`[data-tab-target="${tabId}"]`);
    if (btn?.disabled) return; // 비활성 탭은 이동 불가
  }
  tabButtons.forEach(b => b.classList.toggle("active", b.dataset.tabTarget === tabId));
  tabPanels.forEach(p => { p.hidden = p.id !== tabId; });
}

function updateTabLocks(data) {
  // Agent 모드에서는 파이프라인이 agent가 관리하므로 탭 잠금 해제
  if (state.uiMode === "agent") {
    tabButtons.forEach(btn => { btn.disabled = false; });
    return;
  }
  tabButtons.forEach(btn => {
    const req = btn.dataset.requires;
    if (!req) { btn.disabled = false; return; }
    const status = req === "inference" ? data?.inference_status
                 : req === "nuclei"    ? data?.nuclei_status
                 : "completed";
    btn.disabled = status !== "completed";
  });
}

/* ── 폴링 타임스탬프 ─────────────────────────────────────── */
function updatePollIndicator() {
  if (!state.lastPollTime) { pollIndicator.textContent = ""; return; }
  const sec = Math.round((Date.now() - state.lastPollTime) / 1000);
  pollIndicator.textContent = sec < 5 ? "방금 업데이트" : `${sec}초 전 업데이트`;
}

/* ── 헬스 체크 ───────────────────────────────────────────── */
async function refreshHealth() {
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    healthPill.textContent = d.status === "ok" ? "backend ok" : "backend issue";
    healthPill.classList.toggle("ok", d.status === "ok");
  } catch {
    healthPill.textContent = "offline";
    healthPill.classList.remove("ok");
  }
}

/* ── 히트맵 Canvas 렌더러 (GeoJSON → Canvas) ─────────────── */
async function loadBgImage(url) {
  if (!url) return null;
  if (state.heatmapThumbnailUrl === url && state.heatmapBgImage) return state.heatmapBgImage;
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      state.heatmapBgImage = img;
      state.heatmapThumbnailUrl = url;
      resolve(img);
    };
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

async function renderHeatmapCanvas(geojsonUrl, thumbnailUrl) {
  if (!geojsonUrl) { showFallbackHeatmap(); return; }

  // 썸네일 URL 항상 저장 (토글 재렌더링 시 재사용)
  if (thumbnailUrl) state.heatmapThumbnailUrl = thumbnailUrl;

  // GeoJSON 캐시
  if (state.heatmapGeoJsonUrl !== geojsonUrl) {
    try {
      const resp = await fetch(geojsonUrl);
      if (!resp.ok) { showFallbackHeatmap(); return; }
      state.heatmapGeoJson = (await resp.json()).features || [];
      state.heatmapGeoJsonUrl = geojsonUrl;
    } catch { showFallbackHeatmap(); return; }
  }

  const features = state.heatmapGeoJson;
  if (!features.length) { showFallbackHeatmap(); return; }

  // WSI 배경 이미지 로드 (optional)
  const bgImg = state.heatmapShowBg ? await loadBgImage(thumbnailUrl) : null;

  // 패치 바운딩박스 계산
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const f of features) {
    for (const [x, y] of f.geometry.coordinates[0]) {
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
  }

  const mil = state.milResult;
  const slideW = mil?.slide_width  || (maxX + (maxX - minX) / features.length);
  const slideH = mil?.slide_height || (maxY + (maxY - minY) / features.length);

  const dpr = window.devicePixelRatio || 1;
  const displayW = heatmapFrame.clientWidth  || 800;
  const displayH = Math.max(500, heatmapFrame.clientHeight || 500);

  heatmapCanvas.width  = displayW * dpr;
  heatmapCanvas.height = displayH * dpr;
  heatmapCanvas.style.width  = displayW + "px";
  heatmapCanvas.style.height = displayH + "px";

  const ctx = heatmapCanvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const PAD = 16;
  const scale = Math.min((displayW - PAD * 2) / slideW, (displayH - PAD * 2) / slideH);
  const offX  = (displayW - slideW * scale) / 2;
  const offY  = (displayH - slideH * scale) / 2;
  const tileW = slideW * scale;
  const tileH = slideH * scale;

  // 외곽 배경
  ctx.fillStyle = getComputedStyle(document.documentElement)
    .getPropertyValue("--heatmap-bg").trim() || "#f5eeeb";
  ctx.fillRect(0, 0, displayW, displayH);

  // WSI 배경: 이미지 있으면 썸네일, 없으면 단색
  if (bgImg) {
    ctx.drawImage(bgImg, offX, offY, tileW, tileH);
  } else {
    ctx.fillStyle = getComputedStyle(document.documentElement)
      .getPropertyValue("--tissue-bg").trim() || "#e8ddd8";
    ctx.fillRect(offX, offY, tileW, tileH);
  }

  // 어텐션 패치 렌더링 (Heatmap 토글이 켜진 경우에만)
  if (state.heatmapFillPatch) {
    for (const f of features) {
      const [r, g, b] = f.properties.color_rgb;
      const score  = f.properties.attention_norm;
      const coords = f.geometry.coordinates[0];
      const x0 = offX + coords[0][0] * scale;
      const y0 = offY + coords[0][1] * scale;
      const pw = Math.max(1, (coords[1][0] - coords[0][0]) * scale);
      const ph = Math.max(1, (coords[3][1] - coords[0][1]) * scale);
      const alpha = (80 + 155 * score) / 255;
      ctx.fillStyle = `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
      ctx.fillRect(x0, y0, pw, ph);
    }
  }

  heatmapCanvas.style.display = "block";
  heatmapImage.style.display  = "none";
  emptyHeatmap.style.display  = "none";
}

function showFallbackHeatmap() {
  heatmapCanvas.style.display = "none";
  // heatmapImage는 renderStatus에서 처리
}

/* ── 신뢰도 패널 ─────────────────────────────────────────── */
function renderConfidence(result) {
  const rows = Object.entries(result?.softmax || {});
  if (!rows.length) {
    confidencePanel.innerHTML = '<div class="muted">Run model inference first.</div>';
    logitOutput.textContent = "{}";
    return;
  }

  const prediction = result?.prediction;
  const topScore   = result?.confidence_score ?? 0;
  let interpText = "", interpClass = "";
  if (topScore >= 0.85) {
    interpText  = `높은 신뢰도 (${(topScore*100).toFixed(1)}%) — ${prediction}`;
    interpClass = "high";
  } else if (topScore < 0.60) {
    interpText  = `낮은 신뢰도 (${(topScore*100).toFixed(1)}%) — 추가 검토 권장`;
    interpClass = "low";
  }

  const uncertain = topScore < 0.60;
  confidencePanel.innerHTML =
    (interpText ? `<div class="confidence-interpretation ${interpClass}">${interpText}</div>` : "") +
    rows.map(([label, value]) => {
      const pct = Math.round(value * 1000) / 10;
      return `
        <div class="confidence-row">
          <strong>${escapeHtml(label)}</strong>
          <span class="confidence-track">
            <span class="confidence-fill ${uncertain ? "uncertain" : ""}" style="width:${pct}%"></span>
          </span>
          <span>${pct}%</span>
        </div>`;
    }).join("");

  logitOutput.textContent = JSON.stringify({
    prediction:               result.prediction,
    logits:                   result.logits,
    softmax:                  result.softmax,
    toxic_softmax_confidence: result.abnormal_confidence_score,
    confidence_score:         result.confidence_score,
    num_patches:              result.num_patches,
  }, null, 2);
}

/* ── Top-K 패치 ─────────────────────────────────────────── */
function renderTopK(patches) {
  if (!patches?.length) {
    topkGrid.innerHTML = '<div class="muted">Top patches appear after model inference.</div>';
    return;
  }
  topkGrid.innerHTML = patches.slice(0, 25).map(patch => {
    const score = Number(patch.attention_norm || 0).toFixed(3);
    return `
      <figure class="patch-tile">
        <img src="${patch.image_url}" alt="rank ${patch.rank}" loading="lazy" />
        <figcaption>#${patch.rank} score ${score}</figcaption>
      </figure>`;
  }).join("");
}

/* ── NuLite ─────────────────────────────────────────────── */
function renderNuclei(summary) {
  if (!summary) {
    state.nucleiSummary = null;
    state.selectedNucleiRank = null;
    nucleiTotal.textContent   = "-";
    nucleiPatches.textContent = "-";
    nucleiModel.textContent   = "-";
    nucleiTypeCounts.innerHTML  = '<div class="muted">Run nuclei level analysis after model inference.</div>';
    nucleiOverlayGrid.innerHTML = '<div class="muted">NuLite overlays appear here after analysis.</div>';
    renderSelectedNucleiPatch();
    return;
  }

  state.nucleiSummary = summary;
  if (!state.selectedNucleiRank && summary.overlays?.length) {
    state.selectedNucleiRank = Number(summary.overlays[0].rank);
  }

  nucleiTotal.textContent   = String(summary.total_nuclei ?? 0);
  nucleiPatches.textContent = String(summary.num_patches ?? 0);
  nucleiModel.textContent   = summary.model || "NuLite-H";

  const counts = Object.entries(summary.type_counts || {}).sort((a, b) => b[1] - a[1]);
  nucleiTypeCounts.innerHTML = counts.length
    ? counts.map(([type, count]) => {
        const label = normalizeType(type);
        const color = typeColors[label] || typeColors.Unknown;
        return `
          <div class="type-row">
            <span class="type-swatch" style="background:${color}"></span>
            <span>${escapeHtml(label)}</span>
            <span>${count}</span>
          </div>`;
      }).join("")
    : '<div class="muted">No nuclei detected in selected top-k patches.</div>';

  const overlays = summary.overlays || [];
  nucleiOverlayGrid.innerHTML = overlays.length
    ? overlays.map(overlay => {
        const countsText = Object.entries(overlay.type_counts || {})
          .map(([type, count]) => `${normalizeType(type)}: ${count}`).join(" · ");
        const selected = Number(overlay.rank) === Number(state.selectedNucleiRank) ? " selected" : "";
        return `
          <figure class="patch-tile clickable${selected}" data-nuclei-rank="${overlay.rank}">
            <img src="${overlay.image_url || overlay.overlay_url}"
              alt="NuLite patch rank ${overlay.rank}" loading="lazy" />
            <figcaption>#${overlay.rank} cells ${overlay.cell_count}
              ${countsText ? `<br>${escapeHtml(countsText)}` : ""}
            </figcaption>
          </figure>`;
      }).join("")
    : '<div class="muted">No NuLite overlay images available.</div>';

  nucleiOverlayGrid.querySelectorAll("[data-nuclei-rank]").forEach(tile => {
    tile.addEventListener("click", () => {
      state.selectedNucleiRank = Number(tile.dataset.nucleiRank);
      renderNuclei(state.nucleiSummary);
    });
  });
  renderSelectedNucleiPatch();
}

function renderSelectedNucleiPatch() {
  const summary = state.nucleiSummary;
  const rank    = state.selectedNucleiRank;
  const overlay = summary?.overlays?.find(o => Number(o.rank) === Number(rank));
  if (!summary || !overlay) {
    nucleiDetailTitle.textContent = "Selected Patch";
    nucleiDetailFrame.innerHTML = '<span id="nuclei-detail-empty">Click a NuLite patch to inspect contours.</span>';
    return;
  }

  const instances = instancesForRank(rank);
  const polygons  = instances
    .filter(i => (i.contour_local || []).length >= 3)
    .map(i => {
      const type  = normalizeType(i.type);
      const color = typeColors[type] || typeColors.Unknown;
      const fill  = state.fillContours ? rgbaFromHex(color, 0.24) : "transparent";
      return `
        <polygon class="nuclei-contour"
          points="${polygonPoints(i.contour_local)}"
          fill="${fill}" stroke="${color}">
          <title>${escapeHtml(type)} ${Number(i.type_prob || 0).toFixed(3)}</title>
        </polygon>`;
    }).join("");

  const countsText = Object.entries(overlay.type_counts || {})
    .map(([type, count]) => `${normalizeType(type)} ${count}`).join(" · ");
  nucleiDetailTitle.textContent = `Rank ${overlay.rank} Patch · ${overlay.cell_count} nuclei`;
  nucleiDetailFrame.innerHTML = `
    <div class="nuclei-patch-canvas">
      <img src="${overlay.image_url || overlay.overlay_url}" alt="Rank ${overlay.rank} patch" />
      <svg viewBox="0 0 256 256" preserveAspectRatio="xMidYMid meet"
        aria-label="${escapeHtml(countsText)}">
        ${polygons}
      </svg>
    </div>`;
}

/* ── Metrics 테이블 ─────────────────────────────────────── */
function renderMetrics(comparison, patchMetrics) {
  state.metricComparison = comparison;
  state.patchMetrics = patchMetrics || [];
  if (!comparison) {
    metricsReference.textContent = "-";
    metricsPatchCount.textContent = "-";
    metricComparisonTable.innerHTML = '<div class="muted">Run nuclei level analysis to compute patch-wise metrics.</div>';
    patchMetricTable.innerHTML      = '<div class="muted">Patch-wise metrics appear here after NuLite analysis.</div>';
    return;
  }

  const reference = comparison.reference || {};
  metricsReference.textContent  = `case ${reference.case_n ?? "-"} / control ${reference.control_n ?? "-"}`;
  metricsPatchCount.textContent = String(patchMetrics?.length || 0);

  const comparisonRows = comparison.metrics || [];
  metricComparisonTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th>Top-k mean</th>
          <th>Top-k median</th>
          <th>Control mean</th>
          <th>Case mean</th>
          <th title="case 값이 control 분포에서 몇 표준편차 벗어났는지. |Z| > 2 = 유의미한 차이">Z vs control</th>
          <th>Control pct</th>
          <th>Closer</th>
        </tr>
      </thead>
      <tbody>
        ${comparisonRows.map(row => {
          const z = Number(row.z_vs_control);
          const zClass = Number.isFinite(z) ? (z > 2 ? " z-high" : z < -2 ? " z-low" : "") : "";
          return `
            <tr>
              <td>${escapeHtml(row.metric)}</td>
              <td>${fmt(row.topk_mean)}</td>
              <td>${fmt(row.topk_median)}</td>
              <td>${fmt(row.control?.mean)}</td>
              <td>${fmt(row.case?.mean)}</td>
              <td class="${zClass}">${fmt(row.z_vs_control)}</td>
              <td>${fmt(row.percentile_vs_control, 1)}</td>
              <td>${escapeHtml(row.closer_to || "-")}</td>
            </tr>`;
        }).join("")}
      </tbody>
    </table>`;

  const metricColumns = [
    "patch_rank","Hep_Area_Mean","Hep_Area_Median","Hep_Area_P90",
    "Hep_Solidity_Mean","Hep_Circularity_Mean","Hep_Convexity_Mean","Hep_AspectRatio_Mean",
    "NPC_Area_Mean","NPC_Circularity_Mean","Imm_Area_Mean","Imm_Circularity_Mean",
  ];
  patchMetricTable.innerHTML = `
    <table>
      <thead>
        <tr>${metricColumns.map(c => `<th>${escapeHtml(c)}</th>`).join("")}</tr>
      </thead>
      <tbody>
        ${(patchMetrics || []).map(row => `
          <tr>
            ${metricColumns.map(c => `<td>${c === "patch_rank" ? escapeHtml(row[c]) : fmt(row[c])}</td>`).join("")}
          </tr>`).join("")}
      </tbody>
    </table>`;
}

/* ── LLM 보고서 ─────────────────────────────────────────── */
function renderReport(report, reportStatus) {
  state.llmReport = report;
  const hasMetrics = Boolean(state.metricComparison);
  generateReportButton.disabled = !state.slideId || !hasMetrics || reportStatus === "running";

  if (!report) {
    reportState.textContent = hasMetrics
      ? "Ready to generate."
      : "Run nuclei level analysis first to prepare report inputs.";
    diagnosticReport.innerHTML = '<div class="muted">Saved report will appear here, or press generate after nuclei analysis.</div>';
    return;
  }

  const agentModel = report.agent?.model || report.agent?.agent_model || "";
  const iterations = report.agent?.iterations != null ? ` · ${report.agent.iterations} iters` : "";
  const forced     = report.agent?.forced_finish ? " ⚠ 강제 종료" : "";
  const src        = report.source === "agent_full_run" ? " · Agent" : " · Manual";
  reportState.textContent = `Saved ${report.generated_at || ""}${src}${iterations}${forced}`;

  diagnosticReport.innerHTML = `<div class="report-markdown">${markdownToHtml(report.report_text || "")}</div>`;
}

/* ── 마크다운 렌더러 ─────────────────────────────────────── */
function parseMarkdownCells(line) {
  return line.trim().replace(/^\|/,"").replace(/\|$/,"").split("|").map(c => c.trim());
}

function inlineMarkdown(value) {
  let t = escapeHtml(value);
  t = t.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  t = t.replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" rel="noreferrer">$1</a>');
  t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
  return t;
}

function markdownToHtml(markdown) {
  const lines = String(markdown || "").replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let paragraph = [];

  function flushParagraph() {
    if (!paragraph.length) return;
    html.push(`<p>${inlineMarkdown(paragraph.join(" "))}</p>`);
    paragraph = [];
  }

  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i];
    const line = rawLine.trim();
    if (!line) { flushParagraph(); continue; }

    const nextLine = lines[i+1]?.trim() || "";
    const isTableHeader = line.includes("|") && /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(nextLine);
    if (isTableHeader) {
      flushParagraph();
      const headers = parseMarkdownCells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
        rows.push(parseMarkdownCells(lines[i])); i++;
      }
      i--;
      html.push(`
        <div class="markdown-table"><table>
          <thead><tr>${headers.map(c => `<th>${inlineMarkdown(c)}</th>`).join("")}</tr></thead>
          <tbody>${rows.map(r => `<tr>${r.map(c => `<td>${inlineMarkdown(c)}</td>`).join("")}</tr>`).join("")}</tbody>
        </table></div>`);
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      const level = Math.min(heading[1].length + 2, 5);
      html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const unordered = line.match(/^[-*]\s+(.+)$/);
    if (unordered) {
      flushParagraph();
      const items = [];
      while (i < lines.length) {
        const item = lines[i].trim().match(/^[-*]\s+(.+)$/);
        if (!item) break;
        items.push(`<li>${inlineMarkdown(item[1])}</li>`); i++;
      }
      i--;
      html.push(`<ul>${items.join("")}</ul>`);
      continue;
    }

    const ordered = line.match(/^\d+\.\s+(.+)$/);
    if (ordered) {
      flushParagraph();
      const items = [];
      while (i < lines.length) {
        const item = lines[i].trim().match(/^\d+\.\s+(.+)$/);
        if (!item) break;
        items.push(`<li>${inlineMarkdown(item[1])}</li>`); i++;
      }
      i--;
      html.push(`<ol>${items.join("")}</ol>`);
      continue;
    }

    paragraph.push(line);
  }
  flushParagraph();
  return html.join("");
}

/* ── Nuclei instances 로드 ──────────────────────────────── */
async function loadNucleiInstances(url, summary) {
  if (!url || state.nucleiInstancesUrl === url) return;
  state.nucleiInstancesUrl = url;
  nucleiDetailError.style.display = "none";
  try {
    const resp = await fetch(url);
    if (!resp.ok) {
      state.nucleiInstances = [];
      nucleiDetailError.textContent = `Nuclei instance 로드 실패 (HTTP ${resp.status}). 핵 분석을 다시 실행해보세요.`;
      nucleiDetailError.style.display = "";
      return;
    }
    const instances = await resp.json();
    state.nucleiInstances = Array.isArray(instances)
      ? instances.map(i => ({ ...i, type: normalizeType(i.type) }))
      : [];
    if (summary) renderNuclei(summary);
  } catch (err) {
    state.nucleiInstances = [];
    nucleiDetailError.textContent = `Nuclei instance 로드 오류: ${err.message}`;
    nucleiDetailError.style.display = "";
  }
}

/* ── 전체 상태 렌더링 ────────────────────────────────────── */
function renderStatus(data) {
  state.slideId   = data.slide_id;
  state.milResult = data.mil_result;
  statusSlideId.textContent = data.slide_id || "-";

  setStateBadge(statusCopy,       data.copy_status       || "not_started");
  setStateBadge(statusPreprocess, data.preprocess_status || "not_started");
  setStateBadge(statusInference,  data.inference_status  || "not_started");
  setStateBadge(statusNuclei,     data.nuclei_status     || "not_started");
  setStateBadge(statusReport,     data.report_status     || "not_started");

  // Agent 파이프라인 step 상태 캐시 (live.pipeline이 "unknown"일 때 fallback용)
  state._agentPrepStatus   = data.preprocess_status || "unknown";
  state._agentInferStatus  = data.inference_status  || "unknown";
  state._agentNucleiStatus = data.nuclei_status     || "unknown";

  setStageTime(timePreprocess, data.durations?.preprocess_seconds);
  setStageTime(timeInference,  data.durations?.inference_seconds);
  setStageTime(timeNuclei,     data.durations?.nuclei_seconds);
  setStageTime(timeReport,     data.durations?.report_seconds);

  statusOutput.textContent = JSON.stringify({
    run_dir:           data.run_dir,
    slide_path:        data.slide_path,
    feature_h5:        data.paths?.feature_h5,
    preprocess_job_id: data.preprocess_job_id,
    inference_job_id:  data.inference_job_id,
    nuclei_job_id:     data.nuclei_job_id,
    nuclei_summary:    data.paths?.nuclei_summary,
    report_json:       data.paths?.report_json,
  }, null, 2);

  // 버튼 활성/비활성 및 이유 표시
  preprocessButton.disabled = !data.slide_id || data.preprocess_status === "running";
  inferenceButton.disabled  = data.preprocess_status !== "completed" || data.inference_status === "running";
  nucleiButton.disabled     = data.inference_status  !== "completed" || data.nuclei_status    === "running";
  // Agent full-run button: active once slide is ready
  if (runAgentFullBtn) {
    const agentRunning = state.agentLivePolling !== null;
    runAgentFullBtn.disabled = !data.slide_id || agentRunning;
  }
  updateButtonReasons(data);
  updateTabLocks(data);

  renderConfidence(data.mil_result);
  renderTopK(data.topk);
  renderNuclei(data.nuclei_summary);
  renderMetrics(data.metric_comparison, data.patch_metrics);
  renderReport(data.llm_report, data.report_status);
  loadNucleiInstances(data.urls?.nuclei_instances_json, data.nuclei_summary);

  // 다운로드 링크
  setDownload(geojsonDownload,          data.urls?.attention_geojson);
  setDownload(topkDownload,             data.urls?.topk_manifest);
  setDownload(nucleiGeojsonDownload,    data.urls?.nuclei_geojson);
  setDownload(nucleiCountsDownload,     data.urls?.nuclei_counts_csv);
  setDownload(patchMetricsDownload,     data.urls?.patch_metrics_csv);
  setDownload(metricComparisonDownload, data.urls?.metric_comparison);
  setDownload(reportDownload,           data.urls?.report_markdown);

  // 히트맵: GeoJSON 있으면 Canvas 렌더, 없으면 PNG 폴백
  if (data.urls?.attention_geojson) {
    // wsi_thumbnail: 히트맵 없는 깨끗한 슬라이드 썸네일. 없으면 API endpoint fallback
    const wsiThumbUrl = data.urls?.wsi_thumbnail
      || (data.slide_id ? `/api/workflow/runs/${data.slide_id}/wsi-thumbnail` : null);
    renderHeatmapCanvas(data.urls.attention_geojson, wsiThumbUrl);
  } else if (data.urls?.attention_thumbnail) {
    heatmapCanvas.style.display = "none";
    heatmapImage.src  = data.urls.attention_thumbnail;
    heatmapImage.style.display = "block";
    emptyHeatmap.style.display = "none";
  } else {
    heatmapCanvas.style.display = "none";
    heatmapImage.removeAttribute("src");
    heatmapImage.style.display = "none";
    emptyHeatmap.style.display = "";
  }
}

/* ── 상태 폴링 ──────────────────────────────────────────── */
async function fetchStatus() {
  if (!state.slideId) return;
  try {
    const resp = await fetch(`/api/workflow/runs/${state.slideId}`);
    if (resp.ok) {
      renderStatus(await resp.json());
      state.lastPollTime = Date.now();
      updatePollIndicator();
    }
  } catch { /* silent */ }
}

function startPolling() {
  if (state.polling) clearInterval(state.polling);
  state.polling = setInterval(fetchStatus, 3000);
  // 폴 인디케이터 표시 타이머
  setInterval(updatePollIndicator, 5000);
}

/* ── 뷰 리셋 ────────────────────────────────────────────── */
function resetRunView() {
  state.slideId = null; state.nucleiSummary = null;
  state.nucleiInstances = []; state.nucleiInstancesUrl = null;
  state.selectedNucleiRank = null; state.metricComparison = null;
  state.patchMetrics = []; state.llmReport = null;
  state.milResult = null; state.heatmapGeoJson = null; state.heatmapGeoJsonUrl = null;
  state.lastPollTime = null; state._lastAgentStage = null; state._stageStartMs = {}; pollIndicator.textContent = "";

  statusSlideId.textContent = "-";
  setStateBadge(statusCopy,       "not_started");
  setStateBadge(statusPreprocess, "not_started");
  setStateBadge(statusInference,  "not_started");
  setStateBadge(statusNuclei,     "not_started");
  setStateBadge(statusReport,     "not_started");
  setStageTime(timePreprocess, null);
  setStageTime(timeInference,  null);
  setStageTime(timeNuclei,     null);
  setStageTime(timeReport,     null);

  preprocessButton.disabled = true; inferenceButton.disabled = true;
  nucleiButton.disabled     = true; generateReportButton.disabled = true;
  if (runAgentFullBtn) runAgentFullBtn.disabled = true;
  if (agentRunStatus)  agentRunStatus.textContent = "";
  agentLivePanel.style.display = "none";
  stopAgentLivePolling();
  reasonPreprocess.textContent = "Slide 준비 후 활성화됩니다";
  reasonInference.textContent  = "Preprocess 완료 후 활성화됩니다";
  reasonNuclei.textContent     = "Model Inference 완료 후 활성화됩니다";
  statusOutput.textContent = "{}";
  renderConfidence(null);
  renderTopK([]);
  renderNuclei(null);
  renderMetrics(null, []);
  renderReport(null, "not_started");

  setDownload(geojsonDownload,          null);
  setDownload(topkDownload,             null);
  setDownload(nucleiGeojsonDownload,    null);
  setDownload(nucleiCountsDownload,     null);
  setDownload(patchMetricsDownload,     null);
  setDownload(metricComparisonDownload, null);
  setDownload(reportDownload,           null);

  heatmapCanvas.style.display = "none";
  heatmapImage.removeAttribute("src"); heatmapImage.style.display = "none";
  emptyHeatmap.style.display = "";
  renderSelectedNucleiPatch();
  nucleiDetailError.style.display = "none";
  updateTabLocks(null);
}

async function loadExistingRun(slideId) {
  const resp = await fetch(`/api/workflow/runs/${encodeURIComponent(slideId)}`);
  if (!resp.ok) return false;
  const data = await resp.json();
  renderStatus(data);
  if (data.llm_report)        setActiveTab("llm-tab", true);
  else if (data.nuclei_summary) setActiveTab("nuclei-tab", true);
  else                          setActiveTab("attention-tab", true);
  startPolling();
  return true;
}

/* ── 모드 전환 ───────────────────────────────────────────── */
function setUiMode(mode) {
  state.uiMode = mode;
  modeManualBtn.classList.toggle("active", mode === "manual");
  modeAgentBtn.classList.toggle("active",  mode === "agent");
  manualControls.style.display = mode === "manual" ? "" : "none";
  agentControls.style.display  = mode === "agent"  ? "" : "none";
  // Agent Log 탭은 agent 모드에서만 보임
  if (agentOnlyTab) agentOnlyTab.style.display = mode === "agent" ? "" : "none";
}

modeManualBtn.addEventListener("click", () => setUiMode("manual"));
modeAgentBtn.addEventListener("click",  () => setUiMode("agent"));

/* ── Agent Full-Run ─────────────────────────────────────── */
function updatePipelineStepUI(stepEl, status) {
  const iconMap = { completed: "✅", already_completed: "✅", running: "⏳", failed: "❌", unknown: "⬜", not_started: "⬜" };
  const classMap = { completed: "done", already_completed: "done", running: "running", failed: "failed" };
  const label = stepEl.dataset.label || stepEl.textContent.replace(/^[^\s]+\s/, "");
  const icon  = iconMap[status] || "⬜";
  stepEl.textContent = `${icon} ${label}`;
  stepEl.className = "pipeline-step " + (classMap[status] || "");
}

const STAGE_LABELS = {
  "initializing":             "초기화 중...",
  "thinking":                 "LLM 추론 중 (다음 tool 결정)...",
  "checking_pipeline_status": "파이프라인 상태 확인 중...",
  "running_preprocess":       "TRIDENT 전처리 실행 중...",
  "running_inference":        "ABMIL 추론 실행 중...",
  "running_nuclei_topk":      "NuLite-H 핵 분할 실행 중...",
  "report_done":              "보고서 생성 완료",
  "report_done_forced":       "보고서 생성 완료 (최대 반복 도달)",
};

function stageToKorean(stage) {
  if (!stage) return "";
  if (STAGE_LABELS[stage]) return STAGE_LABELS[stage];
  if (stage.startsWith("tool:")) {
    const tool = stage.replace("tool:", "");
    const toolLabels = {
      "get_mil_summary":          "ABMIL 예측 결과 조회 중...",
      "get_attention_heatmap":    "어텐션 히트맵 조회 중...",
      "get_topk_patches":         "Top-K 패치 이미지 조회 중...",
      "get_nulite_overlays":      "NuLite 오버레이 조회 중...",
      "get_nuclei_summary":       "핵 분석 요약 조회 중...",
      "get_patch_metrics":        "패치 메트릭 조회 중...",
      "get_metric_comparison":    "Case-Control 비교 통계 조회 중...",
      "get_all_patch_attention":  "전체 패치 어텐션 점수 조회 중...",
      "extract_patch_image":      "패치 이미지 추출 중 (openslide)...",
      "run_nulite_on_patches":    "NuLite-H 핵 분할 실행 중 (2-5분)...",
      "compute_metrics_for_patches": "패치 메트릭 계산 중...",
    };
    return toolLabels[tool] || `${tool} 실행 중...`;
  }
  return stage;
}

/* ── Agent Log Console 렌더러 ─────────────────────────────── */
const TOOL_LABELS_KO = {
  get_pipeline_status:         "파이프라인 상태 확인",
  run_preprocess_pipeline:     "TRIDENT 전처리 실행",
  run_inference_pipeline:      "ABMIL 추론 실행",
  run_nulite_topk_pipeline:    "NuLite-H 핵 분할 실행",
  get_mil_summary:             "ABMIL 예측 결과 조회",
  get_attention_heatmap:       "어텐션 분포 데이터 조회",
  get_topk_patches:            "Top-K 패치 이미지 조회",
  get_nulite_overlays:         "NuLite 오버레이 이미지 조회",
  get_nuclei_summary:          "핵 분석 요약 조회",
  get_patch_metrics:           "패치 형태 메트릭 조회",
  get_metric_comparison:       "Case-Control 통계 비교 조회",
  get_all_patch_attention:     "전체 패치 어텐션 점수 조회",
  extract_patch_image:         "임의 패치 이미지 직접 추출",
  run_nulite_on_patches:       "On-demand NuLite 핵 분할 실행",
  compute_metrics_for_patches: "선택 패치 형태 메트릭 계산",
  run_tta_inference:           "TTA 반복 추론 실행",
};

function renderAgentLogConsole(live) {
  if (!agentLogConsole) return;

  const stateLabel = { running: "실행 중", completed: "완료", failed: "실패" }[live.state];
  if (agentLogStateBadge) {
    agentLogStateBadge.textContent = stateLabel || "";
    agentLogStateBadge.className = stateLabel ? "agent-state-pill " + (live.state || "") : "";
  }
  if (agentLogIter) agentLogIter.textContent = "";

  const parts = [];
  const pl = live.pipeline || {};
  const ICONS = { completed:"✅", already_completed:"✅", running:"⏳", failed:"❌", unknown:"⬜", not_started:"⬜" };

  // ── Pipeline 상태 헤더 ─────────────────────────────────
  parts.push(`<span class="log-sep">── Pipeline ──────────────────────────────────</span>`);
  for (const [key, label] of [["preprocess","Preprocess"],["inference","Inference"],["nuclei_topk","Nuclei"]]) {
    const st = pl[key] || "unknown";
    parts.push(`<span class="log-info">  ${ICONS[st]||"⬜"} ${label}: <em>${st}</em></span>`);
  }
  parts.push(`<span class="log-sep">── Log ───────────────────────────────────────</span>`);

  // ── log entries 렌더링 ─────────────────────────────────
  const entries = live.log || [];
  for (const e of entries) {
    // 구버전 string 포맷 fallback
    if (typeof e === "string") {
      parts.push(`<span class="log-info">${escapeHtml(e)}</span>`);
      continue;
    }
    const ts = e.ts ? `<span class="log-ts">[${e.ts}]</span> ` : "";

    switch (e.type) {
      case "info":
        parts.push(`${ts}<span class="log-info">${escapeHtml(e.content || "")}</span>`);
        break;

      case "thinking": {
        const short = (e.content || "").slice(0, 80).replace(/\n/g, " ");
        const full  = escapeHtml(e.content || "");
        parts.push(
          `${ts}<span class="log-thinking">💭 모델 판단: ` +
          `<span class="log-think-short" onclick="this.parentElement.querySelector('.log-think-full').style.display='';this.style.display='none'" style="cursor:pointer;text-decoration:underline">${escapeHtml(short)}${(e.content || "").length > 80 ? "…" : ""}</span>` +
          `<span class="log-think-full" style="display:none">${full}</span></span>`
        );
        break;
      }

      case "tool_call": {
        const label = TOOL_LABELS_KO[e.tool] || e.tool;
        const argsStr = e.args && Object.keys(e.args).length
          ? ` <span class="log-args">(${escapeHtml(JSON.stringify(e.args))})</span>` : "";
        parts.push(
          `${ts}<span class="log-tool-call">🔧 → <strong>${escapeHtml(e.tool)}</strong>` +
          `${argsStr}<br><span class="log-tool-label">   ${escapeHtml(label)}</span></span>`
        );
        break;
      }

      case "tool_result": {
        const dur = e.duration_ms != null ? ` (${(e.duration_ms/1000).toFixed(1)}s)` : "";
        const mm  = e.multimodal ? ` [이미지 ${e.image_count}장]` : "";
        const preview = (e.result_preview || "").slice(0, 200);
        parts.push(
          `${ts}<span class="log-tool-result">  ↩ <strong>${escapeHtml(e.tool)}</strong>${dur}${mm}<br>` +
          `<span class="log-result-preview">     ${escapeHtml(preview)}${preview.length >= 200 ? "…" : ""}</span></span>`
        );
        break;
      }

      case "warn":
        parts.push(`${ts}<span class="log-warn">⚠ ${escapeHtml(e.content || "")}</span>`);
        break;

      case "complete":
        parts.push(`${ts}<span class="log-tool-result">✅ ${escapeHtml(e.content || "완료")}</span>`);
        break;

      case "error":
        parts.push(`${ts}<span class="log-error">❌ ${escapeHtml(e.content || "")}</span>`);
        break;

      default:
        parts.push(`${ts}<span class="log-info">${escapeHtml(JSON.stringify(e))}</span>`);
    }
  }

  // ── 현재 실행 중인 stage + 경과 시간 ─────────────────
  if (live.state === "running" && live.stage && live.stage !== "thinking") {
    // 해당 stage의 tool_call을 프론트에서 처음 본 시각(Date.now())을 기록해 경과 계산
    // (UTC 서버 타임스탬프와 로컬 시간 혼용 문제 회피)
    const stageKey = live.stage;
    if (!state._stageStartMs) state._stageStartMs = {};
    if (!state._stageStartMs[stageKey]) state._stageStartMs[stageKey] = Date.now();
    const elapsedSec = Math.round((Date.now() - state._stageStartMs[stageKey]) / 1000);
    const elapsedStr = ` <span class="log-elapsed">(${elapsedSec}s 경과)</span>`;
    parts.push(`<span class="log-sep">──────────────────────────────────────────────</span>`);
    parts.push(`<span class="log-stage">▶ ${escapeHtml(stageToKorean(live.stage))}${elapsedStr}</span>`);
  }
  if (live.state === "failed") {
    parts.push(`<span class="log-sep">──────────────────────────────────────────────</span>`);
    parts.push(`<span class="log-error">❌ 실패: ${escapeHtml(live.error || "")}</span>`);
  }

  const wasAtBottom = agentLogConsole.scrollHeight - agentLogConsole.scrollTop
    <= agentLogConsole.clientHeight + 60;
  agentLogConsole.innerHTML = parts.join("\n");
  if (wasAtBottom) agentLogConsole.scrollTop = agentLogConsole.scrollHeight;
}

function renderAgentLiveStatus(live) {
  if (!live || live.state === "not_started") {
    agentLivePanel.style.display = "none";
    return;
  }
  agentLivePanel.style.display = "";

  // State pill
  const stateLabel = { running: "실행 중", completed: "완료", failed: "실패" }[live.state] || live.state;
  agentLiveState.textContent = stateLabel;
  agentLiveState.className = "agent-state-pill " + (live.state || "");

  // 현재 상태 한국어 표시
  const currentStageText = stageToKorean(live.stage);
  const iterText = live.iteration > 0 ? ` [반복 ${live.iteration}]` : "";
  agentRunStatus.textContent = currentStageText + iterText;

  // Pipeline steps — live.pipeline에서 unknown이면 fetchStatus 결과로 fallback
  const pl = live.pipeline || {};
  const inferSt  = state._agentInferStatus  || "unknown";
  const nucleiSt = state._agentNucleiStatus || "unknown";
  const prepSt   = state._agentPrepStatus   || "unknown";
  updatePipelineStepUI(psPreprocess, pl.preprocess   !== "unknown" ? pl.preprocess   : prepSt);
  updatePipelineStepUI(psInference,  pl.inference    !== "unknown" ? pl.inference    : inferSt);
  updatePipelineStepUI(psNuclei,     pl.nuclei_topk  !== "unknown" ? pl.nuclei_topk  : nucleiSt);

  // Analysis step
  const resolvedPrep  = pl.preprocess  !== "unknown" ? pl.preprocess  : prepSt;
  const resolvedInfer = pl.inference   !== "unknown" ? pl.inference   : inferSt;
  const resolvedNuc   = pl.nuclei_topk !== "unknown" ? pl.nuclei_topk : nucleiSt;
  const pipelineDone = ["completed","already_completed"].includes(resolvedPrep)
    && ["completed","already_completed"].includes(resolvedInfer)
    && ["completed","already_completed"].includes(resolvedNuc);
  const analysisDone = live.report !== null && live.report !== "";
  const analysisStatus = analysisDone ? "completed" : (pipelineDone && live.state === "running") ? "running" : (pipelineDone ? "completed" : "unknown");
  updatePipelineStepUI(psAnalysis, analysisStatus);

  // ── Agent Log 탭 업데이트 ──────────────────────────────
  renderAgentLogConsole(live);

  // 좌측 패널 tools called 미니 요약
  const tools = live.tools_called || [];
  if (tools.length) {
    const lines = tools.map(t => {
      const args = t.args && Object.keys(t.args).length
        ? ` (${JSON.stringify(t.args)})` : "";
      return `[${t.iteration}] → ${t.name}${args}`;
    });
    agentLiveLog.textContent = lines.join("\n");
    agentLiveLog.scrollTop = agentLiveLog.scrollHeight;
  }

  if (live.state === "completed") {
    stopAgentLivePolling();
    runAgentFullBtn.disabled = false;
    setBusy(runAgentFullBtn, "Run Agent", false);
    if (live.report) {
      agentRunStatus.textContent = "완료! 보고서가 생성되었습니다.";
      fetchStatus().then(() => {
        // Agent Log 탭을 먼저 보여주고, LLM 탭에 데이터가 준비됨을 알림
        setActiveTab("agent-log-tab", true);
      });
    } else {
      agentRunStatus.textContent = "완료 (보고서 없음)";
      fetchStatus().then(() => setActiveTab("agent-log-tab", true));
    }
  } else if (live.state === "failed") {
    agentRunStatus.textContent = `실패: ${live.error || "알 수 없는 오류"}`;
    stopAgentLivePolling();
    runAgentFullBtn.disabled = false;
    setBusy(runAgentFullBtn, "Run Agent", false);
  }
}

function stopAgentLivePolling() {
  if (state.agentLivePolling) {
    clearInterval(state.agentLivePolling);
    state.agentLivePolling = null;
  }
}

async function pollAgentLiveStatus() {
  if (!state.slideId) return;
  try {
    const resp = await fetch(`/api/workflow/runs/${state.slideId}/agent-live-status`);
    if (!resp.ok) return;
    const live = await resp.json();
    const prevStage = state._lastAgentStage;
    // 파이프라인 단계가 바뀌면 fetchStatus도 같이 트리거해 탭 업데이트
    if (live.stage !== prevStage) {
      state._lastAgentStage = live.stage;
      if (["running_preprocess","running_inference","running_nuclei_topk","report_done","report_done_forced",
           "checking_pipeline_status"].includes(live.stage) || live.state !== "running") {
        fetchStatus();
      }
    }
    renderAgentLiveStatus(live);
  } catch { /* silent */ }
}

function startAgentLivePolling() {
  stopAgentLivePolling();
  state.agentLivePolling = setInterval(pollAgentLiveStatus, 2000);
  pollAgentLiveStatus();
}

runAgentFullBtn.addEventListener("click", async () => {
  if (!state.slideId) return;
  const instructions = document.querySelector("#agent-instructions")?.value?.trim() || "";

  setBusy(runAgentFullBtn, "에이전트 실행 중...", true, true);
  agentRunStatus.textContent = "파이프라인 시작 중...";

  const resp = await fetch(`/api/workflow/runs/${state.slideId}/agent-full-run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_instructions: instructions }),
  });
  const data = await resp.json();

  if (!resp.ok) {
    setBusy(runAgentFullBtn, "Run Agent", false);
    runAgentFullBtn.disabled = false;
    agentRunStatus.textContent = `오류: ${data.detail || "실행 시작 실패"}`;
    return;
  }

  if (data.status === "already_running") {
    agentRunStatus.textContent = "이미 실행 중입니다.";
    setBusy(runAgentFullBtn, "Run Agent", false);
    runAgentFullBtn.disabled = true;
  } else {
    agentRunStatus.textContent = "실행 시작됨 — 파이프라인 진행 중...";
  }

  startAgentLivePolling();
  startPolling();
});

/* ── 이벤트: 파일 선택 ──────────────────────────────────── */
slideFileInput.addEventListener("change", async () => {
  const file = slideFileInput.files?.[0];
  selectedSlide.textContent = file
    ? `${file.name} (${(file.size / 1024 ** 3).toFixed(2)} GB)`
    : "No slide selected";
  resetRunView();
  if (!file) return;

  const slideId = slideIdFromFilename(file.name);
  statusSlideId.textContent = slideId;
  setStateBadge(statusCopy, "checking");
  const found = await loadExistingRun(slideId);
  if (!found) {
    statusSlideId.textContent = slideId;
    setStateBadge(statusCopy, "not_started");
    uploadButton.disabled = false;
  }
});

/* ── 이벤트: 슬라이드 업로드 (XHR + 프로그레스) ──────────── */
uploadButton.addEventListener("click", () => {
  const file = slideFileInput.files?.[0];
  if (!file) { window.alert("Select a WSI slide first."); return; }

  setBusy(uploadButton, "Copying...", true);
  setStateBadge(statusCopy, "running");
  uploadProgressBar.style.display = "flex";
  uploadProgressFill.style.width  = "0%";
  uploadProgressLabel.textContent = "0%";

  const form = new FormData();
  form.append("slide", file);

  const xhr = new XMLHttpRequest();

  xhr.upload.addEventListener("progress", e => {
    if (!e.lengthComputable) return;
    const pct = Math.round(e.loaded / e.total * 100);
    uploadProgressFill.style.width  = pct + "%";
    uploadProgressLabel.textContent = pct + "%";
    const loadedGB = (e.loaded / 1024 ** 3).toFixed(2);
    const totalGB  = (e.total  / 1024 ** 3).toFixed(2);
    selectedSlide.textContent = `${file.name} — ${loadedGB} / ${totalGB} GB`;
  });

  xhr.addEventListener("load", () => {
    setBusy(uploadButton, "Copying...", false);
    uploadProgressBar.style.display = "none";
    if (xhr.status >= 400) {
      let msg = "Slide copy failed.";
      try { msg = JSON.parse(xhr.responseText).detail || msg; } catch {}
      window.alert(msg); return;
    }
    try {
      const data = JSON.parse(xhr.responseText);
      renderStatus(data);
      startPolling();
    } catch { window.alert("Server response parse error."); }
  });

  xhr.addEventListener("error", () => {
    setBusy(uploadButton, "Copying...", false);
    uploadProgressBar.style.display = "none";
    window.alert("Network error during slide upload.");
  });

  xhr.open("POST", "/api/workflow/slides/upload");
  xhr.send(form);
});

/* ── 이벤트: Preprocess ─────────────────────────────────── */
preprocessButton.addEventListener("click", async () => {
  if (!state.slideId) return;
  setBusy(preprocessButton, "Starting...", true);
  const form = new FormData();
  if (hfTokenInput.value.trim()) form.append("hf_token", hfTokenInput.value.trim());
  const resp = await fetch(`/api/workflow/runs/${state.slideId}/preprocess`, { method: "POST", body: form });
  const data = await resp.json();
  setBusy(preprocessButton, "Starting...", false);
  if (!resp.ok) { window.alert(data.detail || "Preprocess failed to start."); return; }
  renderStatus(data);
  startPolling();
});

/* ── 이벤트: Inference ──────────────────────────────────── */
inferenceButton.addEventListener("click", async () => {
  if (!state.slideId) return;
  setBusy(inferenceButton, "Starting...", true);
  const resp = await fetch(`/api/workflow/runs/${state.slideId}/inference`, { method: "POST" });
  const data = await resp.json();
  setBusy(inferenceButton, "Starting...", false);
  if (!resp.ok) { window.alert(data.detail || "Inference failed to start."); return; }
  renderStatus(data);
  startPolling();
});

/* ── 이벤트: Nuclei ─────────────────────────────────────── */
nucleiButton.addEventListener("click", async () => {
  if (!state.slideId) return;
  setBusy(nucleiButton, "Starting...", true);
  const resp = await fetch(`/api/workflow/runs/${state.slideId}/nuclei`, { method: "POST" });
  const data = await resp.json();
  setBusy(nucleiButton, "Starting...", false);
  if (!resp.ok) { window.alert(data.detail || "Nuclei level analysis failed to start."); return; }
  renderStatus(data);
  setActiveTab("nuclei-tab", true);
  startPolling();
});

/* ── 이벤트: Report 생성 ─────────────────────────────────── */
generateReportButton.addEventListener("click", async () => {
  if (!state.slideId) return;
  setBusy(generateReportButton, "보고서 생성 중...", true, true);
  setStateBadge(statusReport, "running");
  const resp = await fetch(`/api/workflow/runs/${state.slideId}/report`, { method: "POST" });
  const data = await resp.json();
  setBusy(generateReportButton, "보고서 생성 중...", false);
  if (!resp.ok) {
    setStateBadge(statusReport, "not_started");
    window.alert(data.detail || "Diagnostic report generation failed.");
    return;
  }
  renderStatus(data);
  setActiveTab("llm-tab", true);
});

/* ── 이벤트: 탭 클릭 ─────────────────────────────────────── */
tabButtons.forEach(btn => {
  btn.addEventListener("click", () => setActiveTab(btn.dataset.tabTarget));
});

/* ── 이벤트: Heatmap 표시 옵션 토글 ──────────────────────── */
document.querySelector("#heatmap-bg-toggle").addEventListener("change", e => {
  state.heatmapShowBg = e.target.checked;
  if (state.heatmapGeoJsonUrl) {
    renderHeatmapCanvas(state.heatmapGeoJsonUrl, state.heatmapThumbnailUrl);
  }
});

document.querySelector("#heatmap-fill-toggle").addEventListener("change", e => {
  state.heatmapFillPatch = e.target.checked;
  if (state.heatmapGeoJsonUrl) {
    renderHeatmapCanvas(state.heatmapGeoJsonUrl, state.heatmapThumbnailUrl);
  }
});

/* ── 이벤트: Contour fill toggle ────────────────────────── */
contourFillToggle.addEventListener("change", () => {
  state.fillContours = contourFillToggle.checked;
  renderSelectedNucleiPatch();
});

/* ── 이벤트: Agent trace 접기/펴기 ─────────────────────── */
// agentTraceToggle removed — agent trace moved to Agent Log tab

/* ── 초기화 ─────────────────────────────────────────────── */
setUiMode("manual");
renderConfidence(null);
renderTopK([]);
renderNuclei(null);
refreshHealth();
setInterval(refreshHealth, 60000);
