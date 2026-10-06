/**
 * script.js — Customer Churn Analytics Platform
 * Handles sidebar, Chart.js charts, prediction form submission.
 */

'use strict';

// ─────────────────────────────────────────────
// CHART DEFAULTS
// ─────────────────────────────────────────────

const ACCENT = '#2563EB';
const COLORS = {
  blue:   '#2563EB',
  red:    '#EF4444',
  green:  '#22C55E',
  amber:  '#F59E0B',
  slate:  '#94A3B8',
  purple: '#7C3AED',
  teal:   '#0D9488',
};

const CHART_PALETTE = [
  '#2563EB','#EF4444','#22C55E','#F59E0B','#7C3AED',
  '#0D9488','#F97316','#EC4899','#06B6D4','#84CC16'
];

Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.color = '#475569';
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.padding = 16;


// ─────────────────────────────────────────────
// SIDEBAR (mobile)
// ─────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  const sidebar  = document.getElementById('sidebar');
  const overlay  = document.getElementById('sidebarOverlay');
  const toggle   = document.getElementById('sidebarToggle');

  if (toggle) {
    toggle.addEventListener('click', () => {
      sidebar.classList.toggle('open');
      overlay.classList.toggle('open');
    });
  }

  if (overlay) {
    overlay.addEventListener('click', () => {
      sidebar.classList.remove('open');
      overlay.classList.remove('open');
    });
  }

  // Check model status
  fetch('/api/model-info')
    .then(r => r.ok ? r.json() : null)
    .then(info => {
      const statusEl = document.getElementById('modelStatus');
      if (!statusEl) return;
      if (info && info.model_name) {
        statusEl.innerHTML = `<i class="bi bi-circle-fill text-success"></i><span>${info.model_name} Active</span>`;
      } else {
        statusEl.innerHTML = `<i class="bi bi-circle-fill text-warning"></i><span>Model Not Trained</span>`;
      }
    })
    .catch(() => {});
});


// ─────────────────────────────────────────────
// CHART HELPERS
// ─────────────────────────────────────────────

function makeBarChart(canvasId, labels, values, label = 'Churn Rate (%)', colors = null) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const barColors = colors || labels.map((_, i) => CHART_PALETTE[i % CHART_PALETTE.length]);

  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label,
        data: values,
        backgroundColor: barColors,
        borderRadius: 5,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.raw.toFixed(1)}%`
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: { callback: v => v + '%' },
          grid: { color: '#F1F5F9' },
        },
        x: { grid: { display: false } }
      }
    }
  });
}

function makeDonutChart(canvasId, labels, values, colors) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  return new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderWidth: 2,
        borderColor: '#fff',
        hoverOffset: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '60%',
      plugins: {
        legend: { position: 'bottom' },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.label}: ${ctx.raw.toLocaleString()}`
          }
        }
      }
    }
  });
}

function makeHorizontalBarChart(canvasId, labels, values, label = 'Importance') {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label,
        data: values,
        backgroundColor: ACCENT,
        borderRadius: 4,
        borderSkipped: false,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ctx.raw.toFixed(4)
          }
        }
      },
      scales: {
        x: {
          beginAtZero: true,
          grid: { color: '#F1F5F9' },
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 11 } }
        }
      }
    }
  });
}


// ─────────────────────────────────────────────
// DASHBOARD CHARTS
// ─────────────────────────────────────────────

function renderDashboardCharts(stats) {
  if (!stats || Object.keys(stats).length === 0) return;

  // 1. Churn Donut
  makeDonutChart(
    'churnDonutChart',
    ['Active', 'Churned'],
    [stats.active_customers || 0, stats.churned_customers || 0],
    [COLORS.green, COLORS.red]
  );

  // 2. Contract Type Churn
  if (stats.contract_churn) {
    const labels = Object.keys(stats.contract_churn);
    const values = labels.map(k => +(stats.contract_churn[k] * 100).toFixed(1));
    makeBarChart('contractChurnChart', labels, values, 'Churn Rate (%)', CHART_PALETTE);
  }

  // 3. Plan Churn
  if (stats.plan_churn) {
    const labels = Object.keys(stats.plan_churn);
    const values = labels.map(k => +(stats.plan_churn[k] * 100).toFixed(1));
    makeBarChart('planChurnChart', labels, values, 'Churn Rate (%)', CHART_PALETTE);
  }

  // 4. Auto-Renew Churn
  if (stats.autorenew_churn) {
    const colorMap = { Yes: COLORS.green, No: COLORS.red, Unknown: COLORS.slate };
    const labels = Object.keys(stats.autorenew_churn);
    const values = labels.map(k => +(stats.autorenew_churn[k] * 100).toFixed(1));
    const colors = labels.map(k => colorMap[k] || COLORS.blue);
    makeBarChart('autoRenewChart', labels, values, 'Churn Rate (%)', colors);
  }

  generateBusinessInsights(stats);
}

function generateBusinessInsights(stats) {
  const container = document.getElementById('businessInsightsContainer');
  if (!container) return;
  
  const insights = [];

  // Insight 1: Contract Type
  if (stats.contract_churn) {
    const sortedContracts = Object.entries(stats.contract_churn).sort((a,b) => b[1] - a[1]);
    const highestContract = sortedContracts[0];
    const lowestContract = sortedContracts[sortedContracts.length - 1];
    insights.push(`<strong>${highestContract[0]} contracts</strong> exhibit the highest churn risk at <strong>${(highestContract[1] * 100).toFixed(1)}%</strong>, whereas <strong>${lowestContract[0]} contracts</strong> provide the best retention (${(lowestContract[1] * 100).toFixed(1)}%). Suggest pushing ${lowestContract[0]} subscriptions.`);
  }

  // Insight 2: Auto-renewal
  if (stats.autorenew_churn && stats.autorenew_churn['No']) {
    insights.push(`Customers with <strong>Auto-Renewal disabled</strong> churn at a rate of <strong>${(stats.autorenew_churn['No'] * 100).toFixed(1)}%</strong>. Consider targeted incentives to enable auto-renewal.`);
  }

  // Insight 3: Financial
  if (stats.avg_monthly_spend) {
    insights.push(`The average monthly spend is <strong>₹${formatNumber(Math.round(stats.avg_monthly_spend))}</strong>. Focus retention efforts on high-value customers above this threshold.`);
  }

  container.innerHTML = `
    <ul class="mb-0 text-dark" style="line-height: 1.8;">
      ${insights.map(i => `<li><i class="bi bi-check-circle-fill text-success me-2"></i>${i}</li>`).join('')}
    </ul>
  `;
}


// ─────────────────────────────────────────────
// ANALYTICS PAGE CHARTS
// ─────────────────────────────────────────────

function renderAnalyticsCharts(stats, featureImportance) {
  if (!stats || Object.keys(stats).length === 0) return;

  // Region churn
  if (stats.region_churn) {
    const labels = Object.keys(stats.region_churn);
    const values = labels.map(k => +(stats.region_churn[k] * 100).toFixed(1));
    makeBarChart('regionChurnChart', labels, values, 'Churn Rate (%)', CHART_PALETTE);
  }

  // Tenure group churn
  if (stats.tenure_group_churn) {
    const order = ['New', 'Growing', 'Established', 'Loyal'];
    const labels = order.filter(k => stats.tenure_group_churn[k] !== undefined);
    const values = labels.map(k => +(stats.tenure_group_churn[k] * 100).toFixed(1));
    const gradColors = ['#EF4444', '#F97316', '#F59E0B', '#22C55E'];
    makeBarChart('tenureGroupChart', labels, values, 'Churn Rate (%)', gradColors);
  }

  // Contract churn (analytics)
  if (stats.contract_churn) {
    const labels = Object.keys(stats.contract_churn);
    const values = labels.map(k => +(stats.contract_churn[k] * 100).toFixed(1));
    makeBarChart('analyticsContractChart', labels, values, 'Churn Rate (%)', CHART_PALETTE);
  }

  // Plan churn (analytics)
  if (stats.plan_churn) {
    const labels = Object.keys(stats.plan_churn);
    const values = labels.map(k => +(stats.plan_churn[k] * 100).toFixed(1));
    makeBarChart('analyticsPlanChart', labels, values, 'Churn Rate (%)', CHART_PALETTE);
  }

  // Auto-Renew (analytics)
  if (stats.autorenew_churn) {
    const colorMap = { Yes: COLORS.green, No: COLORS.red, Unknown: COLORS.slate };
    const labels = Object.keys(stats.autorenew_churn);
    const values = labels.map(k => +(stats.autorenew_churn[k] * 100).toFixed(1));
    const colors = labels.map(k => colorMap[k] || COLORS.blue);
    makeBarChart('analyticsAutoRenewChart', labels, values, 'Churn Rate (%)', colors);
  }

  // Feature importance (horizontal bar)
  if (featureImportance && featureImportance.length > 0) {
    const labels = featureImportance.map(f => f.Feature).reverse();
    const values = featureImportance.map(f => +f.Importance.toFixed(4)).reverse();
    makeHorizontalBarChart('featureImportanceChart', labels, values, 'Importance Score');
  }
}


// ─────────────────────────────────────────────
// PREDICTION FORM
// ─────────────────────────────────────────────

function initPredictionForm() {
  const form = document.getElementById('predictionForm');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Bootstrap validation
    if (!form.checkValidity()) {
      form.classList.add('was-validated');
      return;
    }

    const submitBtn = document.getElementById('submitBtn');
    const btnText   = submitBtn.querySelector('.btn-text');
    const btnSpinner= submitBtn.querySelector('.btn-spinner');

    // Show loading state
    submitBtn.disabled = true;
    btnText.classList.add('d-none');
    btnSpinner.classList.remove('d-none');

    // Collect form data
    const formData = new FormData(form);
    const payload = {};
    for (const [key, value] of formData.entries()) {
      // Convert numeric strings to numbers
      const num = parseFloat(value);
      payload[key] = isNaN(num) || value === '' ? value : num;
    }

    // String fields must stay as strings
    ['Plan', 'ContractType', 'AutoRenew', 'Region', 'SignupDate'].forEach(k => {
      if (formData.get(k)) payload[k] = formData.get(k);
    });

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const result = await response.json();

      if (!response.ok) {
        showError(result.error || 'Prediction failed. Please check your inputs.');
        return;
      }

      displayPredictionResult(result);
    } catch (err) {
      showError('Network error. Please ensure the Flask server is running.');
    } finally {
      submitBtn.disabled = false;
      btnText.classList.remove('d-none');
      btnSpinner.classList.add('d-none');
    }
  });
}

function displayPredictionResult(result) {
  const banner = document.getElementById('predictionResult');
  const badge  = document.getElementById('resultBadge');
  const icon   = document.getElementById('resultIcon');
  const pred   = document.getElementById('resultPrediction');
  const prob   = document.getElementById('resultProb');
  const risk   = document.getElementById('resultRisk');
  const expl   = document.getElementById('resultExplanation');

  if (!banner) return;

  const riskLower = result.risk_level.toLowerCase();

  // Update banner class
  banner.className = `prediction-result-banner mb-4 churn-${riskLower}`;

  // Update badge
  badge.className = `result-badge ${riskLower}`;
  if (result.prediction === 'Churn') {
    icon.className = 'bi bi-person-x-fill';
  } else {
    icon.className = 'bi bi-person-check-fill';
  }

  // Update text
  pred.textContent = result.prediction === 'Churn' ? '⚠ High Churn Risk Detected' : '✓ Customer Likely to Stay';
  prob.textContent = `${result.probability_pct}%`;
  risk.textContent = result.risk_level;

  // Explanation text
  const explanations = {
    High:   'Based on the provided customer information, the model estimates a high likelihood of churn. We recommend proactive retention outreach.',
    Medium: 'The model estimates a moderate churn probability. Consider a targeted retention offer to reduce this risk.',
    Low:    'The model estimates this customer has a low probability of churning. Continue monitoring engagement metrics.',
  };
  
  let explanationHtml = explanations[result.risk_level] || '';
  
  // Add feature explanations if provided by API
  if (result.top_factors && result.top_factors.length > 0) {
    explanationHtml += `<hr><h6 class="mt-3"><strong>Key Contributing Factors:</strong></h6><ul class="mb-2">`;
    result.top_factors.forEach(f => {
      explanationHtml += `<li><code>${f.feature}</code> (Score: ${f.importance.toFixed(3)})</li>`;
    });
    explanationHtml += `</ul>`;
  }
  
  if (result.retention_actions && result.retention_actions.length > 0) {
    explanationHtml += `<h6 class="mt-3 text-success"><strong>Suggested Retention Actions:</strong></h6><ul class="mb-0 text-success">`;
    result.retention_actions.forEach(a => {
      explanationHtml += `<li><i class="bi bi-arrow-right-short"></i> ${a}</li>`;
    });
    explanationHtml += `</ul>`;
  }

  expl.innerHTML = explanationHtml;

  // Show banner and scroll to it
  banner.classList.remove('d-none');
  banner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(message) {
  let alertEl = document.getElementById('predictionError');
  if (!alertEl) {
    alertEl = document.createElement('div');
    alertEl.id = 'predictionError';
    alertEl.className = 'alert alert-danger d-flex align-items-center gap-2 mt-3';
    document.getElementById('predictionForm').prepend(alertEl);
  }
  alertEl.innerHTML = `<i class="bi bi-exclamation-triangle-fill"></i><div>${message}</div>`;
  alertEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function resetForm() {
  const banner = document.getElementById('predictionResult');
  if (banner) banner.classList.add('d-none');

  const form = document.getElementById('predictionForm');
  if (form) {
    form.classList.remove('was-validated');
    form.reset();
  }

  const errorEl = document.getElementById('predictionError');
  if (errorEl) errorEl.remove();
}

// ─────────────────────────────────────────────
// FORMAT HELPERS
// ─────────────────────────────────────────────

function formatNumber(n) {
  return new Intl.NumberFormat().format(n);
}
