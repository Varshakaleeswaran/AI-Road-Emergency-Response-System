/* ================================================================
   ResQLink AI — App JS
   ================================================================ */

// ── Auto-dismiss flash alerts after 5s ──────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.flash-container .alert').forEach(el => {
    setTimeout(() => {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(el);
      bsAlert.close();
    }, 5000);
  });

  // Activate Bootstrap tooltips
  document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
    new bootstrap.Tooltip(el, { trigger: 'hover' });
  });

  // Confirm on destructive buttons
  document.querySelectorAll('[data-confirm]').forEach(btn => {
    btn.addEventListener('click', e => {
      if (!confirm(btn.dataset.confirm)) e.preventDefault();
    });
  });

  // Number counter animation for stat cards
  document.querySelectorAll('.stat-value[data-count]').forEach(el => {
    const target = parseInt(el.dataset.count, 10);
    animateCount(el, target);
  });
});

function animateCount(el, target) {
  const duration = 700;
  const start = Date.now();
  const tick = () => {
    const elapsed = Date.now() - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // ease out cubic
    el.textContent = Math.round(eased * target);
    if (progress < 1) requestAnimationFrame(tick);
  };
  tick();
}

// ── Chart defaults ───────────────────────────────────────────────
if (typeof Chart !== 'undefined') {
  Chart.defaults.color = '#8b949e';
  Chart.defaults.borderColor = '#21262d';
  Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
  Chart.defaults.font.size = 12;
}

// ── Utility: build doughnut chart ───────────────────────────────
function buildDoughnut(ctx, labels, data, colors) {
  return new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: colors,
        borderColor: '#161b22',
        borderWidth: 3,
        hoverOffset: 6,
      }]
    },
    options: {
      cutout: '68%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: { padding: 14, boxWidth: 10, boxHeight: 10 }
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.parsed}`
          }
        }
      }
    }
  });
}

// ── Utility: build bar chart ────────────────────────────────────
function buildBar(ctx, labels, data, color) {
  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: color || 'rgba(249,115,22,.6)',
        borderColor: color || 'rgb(249,115,22)',
        borderWidth: 1,
        borderRadius: 4,
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 11 } } },
        y: { grid: { color: '#21262d' }, ticks: { precision: 0, font: { size: 11 } } }
      }
    }
  });
}

// ── Utility: build line chart ────────────────────────────────────
function buildLine(ctx, labels, data, color) {
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data,
        borderColor: color || 'rgb(249,115,22)',
        backgroundColor: color
          ? color.replace('rgb', 'rgba').replace(')', ', 0.12)')
          : 'rgba(249,115,22,0.12)',
        borderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6,
        fill: true,
        tension: 0.4,
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 11 } } },
        y: { grid: { color: '#21262d' }, ticks: { precision: 0, font: { size: 11 } } }
      }
    }
  });
}

// ── Live clock in topbar ─────────────────────────────────────────
const clockEl = document.getElementById('live-clock');
if (clockEl) {
  const tick = () => {
    const now = new Date();
    clockEl.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };
  tick();
  setInterval(tick, 1000);
}
