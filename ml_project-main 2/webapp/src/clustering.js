import Chart from 'chart.js/auto';

// Shared color palette from main
const COLORS = {
  purple: '#6c5ce7',
  lavender: '#a29bfe',
  blue: '#74b9ff',
  cyan: '#00cec9',
  green: '#55efc4',
  yellow: '#fdcb6e',
  orange: '#e17055',
  red: '#ff7675',
  pink: '#fd79a8',
  teal: '#81ecec',
};

function alpha(hex, a) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

let radarChartInstance = null;
let spendingChartInstance = null;

export function initClustering(clustersData) {
  const selectElement = document.getElementById('persona-select');
  const contentElement = document.getElementById('cluster-content');
  
  if (!selectElement || !clustersData) return;

  // Clear loading option
  selectElement.innerHTML = '<option value="" disabled selected>Choose a Persona...</option>';

  // Sort clusters by ID or Size (let's sort by ID to match K-Means order)
  const sortedKeys = Object.keys(clustersData).sort();

  // Populate options
  sortedKeys.forEach(key => {
    const cluster = clustersData[key];
    const option = document.createElement('option');
    option.value = key;
    option.textContent = `Persona ${cluster.id + 1}: ${cluster.name} (${cluster.percentage}%)`;
    selectElement.appendChild(option);
  });

  // Event listener for dropdown change
  selectElement.addEventListener('change', (e) => {
    const selectedKey = e.target.value;
    if (selectedKey && clustersData[selectedKey]) {
      contentElement.style.display = 'block';
      updateClusterView(clustersData[selectedKey]);
      
      // Add slight animation to cards when updated
      const cards = contentElement.querySelectorAll('.card, .feature-card');
      cards.forEach(card => {
        card.style.animation = 'none';
        card.offsetHeight; // trigger reflow
        card.style.animation = 'float 0.5s ease-out forwards';
      });
    }
  });
}

function updateClusterView(cluster) {
  // Update Metrics
  document.getElementById('cluster-size').textContent = cluster.size.toLocaleString();
  document.getElementById('cluster-percentage').textContent = `${cluster.percentage}% of total`;
  document.getElementById('cluster-age').textContent = Math.round(cluster.features.customer_age);
  document.getElementById('cluster-tenure').textContent = cluster.features.years_tenure.toFixed(1);

  // Update Radar Chart (Behavior vs Global)
  renderRadarChart(cluster);
  
  // Update Spending Chart
  renderSpendingProfileChart(cluster);
}

function renderRadarChart(cluster) {
  const ctx = document.getElementById('chart-cluster-radar');
  if (!ctx) return;

  // We want to show how this cluster deviates from the global average (percentage difference)
  // for a few key behavioral metrics
  const radarMetrics = [
    { key: 'customer_age', label: 'Age' },
    { key: 'total_children', label: 'Children' },
    { key: 'years_tenure', label: 'Tenure' },
    { key: 'distinct_stores_visited', label: 'Stores Visited' },
    { key: 'number_complaints', label: 'Complaints' },
    { key: 'percentage_of_products_bought_promotion', label: 'Promotion Seeking' }
  ];

  const labels = radarMetrics.map(m => m.label);
  
  // The data points are the percentage difference from the global average
  const data = radarMetrics.map(m => {
    const diff = cluster.global_comparison[m.key] || 0;
    // Cap extremes so the chart doesn't look completely distorted
    return Math.max(Math.min(diff, 200), -100); 
  });

  if (radarChartInstance) {
    radarChartInstance.destroy();
  }

  radarChartInstance = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: labels,
      datasets: [{
        label: '% Difference vs Global Average',
        data: data,
        backgroundColor: alpha(COLORS.purple, 0.4),
        borderColor: COLORS.purple,
        pointBackgroundColor: COLORS.lavender,
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: COLORS.purple,
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      scales: {
        r: {
          angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
          grid: { color: 'rgba(255, 255, 255, 0.1)' },
          pointLabels: { font: { size: 11 }, color: 'var(--text-secondary)' },
          ticks: { display: false } // Hide the numbers on the axis
        }
      },
      plugins: {
        legend: { position: 'bottom' },
        tooltip: {
          callbacks: {
            label: (item) => {
              const val = item.raw;
              const sign = val > 0 ? '+' : '';
              return ` ${sign}${val.toFixed(1)}% vs Average`;
            }
          }
        }
      }
    }
  });
}

function renderSpendingProfileChart(cluster) {
  const ctx = document.getElementById('chart-cluster-spending');
  if (!ctx) return;

  // Filter spending features
  const spendKeys = Object.keys(cluster.features).filter(k => k.includes('lifetime_spend'));
  
  // Sort them by highest spend for this cluster
  spendKeys.sort((a, b) => cluster.features[b] - cluster.features[a]);

  const labels = spendKeys.map(k => k.replace('lifetime_spend_', '').replace('_', ' ').title());
  const values = spendKeys.map(k => cluster.features[k]);

  // Give them a nice gradient of colors
  const barColors = [
    COLORS.blue, COLORS.cyan, COLORS.teal, COLORS.green, 
    COLORS.yellow, COLORS.orange, COLORS.pink
  ].map(c => alpha(c, 0.8));

  if (spendingChartInstance) {
    spendingChartInstance.destroy();
  }

  spendingChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Average Spend (€)',
        data: values,
        backgroundColor: barColors,
        borderWidth: 0,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      indexAxis: 'y', // Make it horizontal
      plugins: { legend: { display: false } },
      scales: {
        x: { beginAtZero: true, ticks: { font: { size: 10 } } },
        y: { ticks: { font: { size: 10 } } }
      }
    }
  });
}

// Helper for title casing
String.prototype.title = function() {
  return this.replace(/(?:^|\s)\w/g, function(match) {
    return match.toUpperCase();
  });
};
