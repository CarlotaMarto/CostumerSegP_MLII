// Simulator Logic for the "What-If" Calculator

let globalStats = null;
let clusterData = null;

const FEATURES = {
  'sim-age': 'customer_age',
  'sim-children': 'total_children',
  'sim-tenure': 'years_tenure',
  'sim-promo': 'percentage_of_products_bought_promotion',
  'sim-meat': 'lifetime_spend_meat_fish',
  'sim-tech': 'lifetime_spend_electronics_videogames',
  'sim-veg': 'lifetime_spend_vegetables',
  'sim-hyg': 'lifetime_spend_hygiene',
  'sim-stores': 'distinct_stores_visited',
  'sim-complaints': 'number_complaints'
};

const ICONS = {
  0: '👴', // Older Long-Tenure
  1: '👨‍👩‍👧‍👦', // Large Families
  2: '🌱', // Clean and Green
  3: '🎓', // Young Most Recent Customer
  4: '💻', // Tech Enthusiasts
  5: '🏷️', // Promotion Seekers
  6: '🥩', // Protein Lovers
  7: '🛎️'  // Service Sensitive
};

export function initSimulator(overview, clusters) {
  if (!overview || !overview.numeric_stats || !clusters) return;
  
  globalStats = overview.numeric_stats;
  clusterData = clusters;

  // Add event listeners to all sliders
  Object.keys(FEATURES).forEach(sliderId => {
    const slider = document.getElementById(sliderId);
    if (slider) {
      slider.addEventListener('input', (e) => {
        // Update the value display
        const valDisplay = document.getElementById(`${sliderId}-val`);
        let displayValue = e.target.value;
        if (sliderId === 'sim-promo') displayValue += '%';
        if (['sim-meat', 'sim-tech', 'sim-veg', 'sim-hyg'].includes(sliderId)) {
          displayValue = `€${displayValue}`;
        }
        valDisplay.textContent = displayValue;
        
        // Trigger prediction
        predictPersona();
      });
    }
  });

  // Initial prediction
  predictPersona();
}

function predictPersona() {
  if (!globalStats || !clusterData) return;

  // 1. Read input values
  const inputValues = {};
  Object.entries(FEATURES).forEach(([sliderId, featureName]) => {
    const slider = document.getElementById(sliderId);
    if (slider) {
      inputValues[featureName] = parseFloat(slider.value);
      // Ensure promo is scaled 0-1 if dataset uses 0-1
      if (featureName === 'percentage_of_products_bought_promotion' && globalStats[featureName].max <= 1) {
        inputValues[featureName] = inputValues[featureName] / 100;
      }
    }
  });

  // 2. Standardize inputs (Z-score = (x - mean) / std)
  const standardizedInput = {};
  Object.entries(inputValues).forEach(([feature, val]) => {
    const mean = globalStats[feature].mean;
    const std = globalStats[feature].std;
    standardizedInput[feature] = std > 0 ? (val - mean) / std : 0;
  });

  // 3. Calculate distance to each cluster
  let bestClusterId = null;
  let minDistance = Infinity;

  Object.values(clusterData).forEach(cluster => {
    let distanceSq = 0;
    
    Object.keys(cluster.features).forEach(feature => {
      // Standardize the cluster's feature mean
      const clusterMean = cluster.features[feature] || 0;
      const globalMean = globalStats[feature] ? globalStats[feature].mean : 0;
      const std = globalStats[feature] ? globalStats[feature].std : 1;
      
      const standardizedClusterVal = std > 0 ? (clusterMean - globalMean) / std : 0;
      
      // If feature is in sliders, use user's standardized input, else 0 (global mean)
      const userVal = standardizedInput[feature] !== undefined ? standardizedInput[feature] : 0;
      
      const diff = userVal - standardizedClusterVal;
      
      // Give slightly higher weight to the slider features so they feel more responsive
      const weight = standardizedInput[feature] !== undefined ? 2.0 : 1.0;
      
      distanceSq += (diff * diff) * weight;
    });

    const distance = Math.sqrt(distanceSq);
    if (distance < minDistance) {
      minDistance = distance;
      bestClusterId = cluster.id;
    }
  });

  // 4. Calculate a match score (heuristic based on max possible reasonable distance)
  // Max distance in 6D space where each dimension varies mostly between -3 and 3 is around 6-7.
  const matchScore = Math.max(0, Math.min(100, Math.round(100 - (minDistance * 12))));

  // 5. Update UI
  updateSimulatorUI(bestClusterId, matchScore);
}

function updateSimulatorUI(clusterId, score) {
  const nameEl = document.getElementById('sim-result-name');
  const iconEl = document.getElementById('sim-result-icon');
  const scoreEl = document.getElementById('sim-result-score');
  
  // Find the cluster data
  const clusterKey = `cluster_${clusterId}`;
  const cluster = clusterData[clusterKey];
  
  if (cluster) {
    nameEl.textContent = cluster.name;
    iconEl.textContent = ICONS[clusterId] || '👤';
    scoreEl.textContent = `${score}%`;
    
    // Animate
    iconEl.style.animation = 'none';
    iconEl.offsetHeight; // trigger reflow
    iconEl.style.animation = 'float 0.5s ease-out forwards';
  }
}
