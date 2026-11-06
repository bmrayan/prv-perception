async function copyBib() {
  const text = document.getElementById('bibtex-code').innerText;
  const button = document.querySelector('.copy-bib-btn');
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = 'Copied!';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.getElementById('bibtex-code'));
    selection.removeAllRanges();
    selection.addRange(range);
    button.textContent = 'Select and copy';
  }
  setTimeout(() => { button.textContent = 'Copy'; }, 2000);
}

const resultSettings = {
  muma: { name: 'MuMA-ToM', values: [44.7, 47.4, null, 58.1, 54.0] },
  mindpower: { name: 'MindPower', values: [19.1, 18.3, 24.1, 27.3, 33.0] },
  abstract: { name: 'Abstract false belief', values: [73.9, 75.7, 88.8, 98.4, 94.5] }
};
const conditionNames = ['A · Visual-only', 'Bₐ · Automatic', 'Bₚ · Perceptual', 'Bꜰ · Full scaffold', 'C · Text-only'];
const conditionColors = ['#707070', '#9b9b9b', '#5296c6', '#3273dc', '#8b71ad'];

function renderResults() {
  const setting = resultSettings[document.getElementById('benchmark-select').value];
  const gain = document.getElementById('metric-select').value === 'gain';
  const selected = [...document.querySelectorAll('.condition-controls input:checked')].map(input => Number(input.value));
  const bars = document.getElementById('interactive-bars');
  const detail = document.getElementById('chart-detail');
  bars.replaceChildren();
  document.getElementById('chart-summary').textContent = `${setting.name} · Seven-model mean · ${gain ? 'Change from A (percentage points)' : 'Reported score (%)'} · ${gain ? 'Scale: −25 to +25 pp' : 'Scale: 0–100%'}`;
  detail.textContent = selected.length ? 'Hover, focus, or select a bar for details. Values are rounded as reported in the paper.' : 'Select at least one condition to display its result.';
  for (const i of selected) {
    const score = setting.values[i];
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'result-bar-row';
    const label = document.createElement('span');
    label.className = 'result-label';
    label.textContent = conditionNames[i];
    const track = document.createElement('span');
    track.className = `result-track${gain ? ' gain-track' : ''}`;
    const fill = document.createElement('span');
    fill.className = 'result-fill';
    fill.style.background = conditionColors[i];
    const value = document.createElement('span');
    value.className = 'result-value';
    let description;
    if (score === null) {
      value.textContent = 'n/a';
      description = `${conditionNames[i]} is not defined for ${setting.name}.`;
    } else {
      const delta = score - setting.values[0];
      const signed = `${delta > 0 ? '+' : ''}${delta.toFixed(1)}`;
      value.textContent = gain ? `${signed} pp` : `${score.toFixed(1)}%`;
      fill.style.width = `${gain ? Math.abs(delta) * 2 : score}%`;
      fill.style.left = `${gain ? (delta < 0 ? 50 + delta * 2 : 50) : 0}%`;
      description = `${setting.name} — ${conditionNames[i]}: ${score.toFixed(1)}%; ${signed} percentage points compared with visual-only (${setting.values[0].toFixed(1)}%).`;
    }
    track.append(fill);
    row.append(label, track, value);
    row.setAttribute('aria-label', description);
    row.title = description;
    for (const event of ['mouseenter', 'focus', 'click']) row.addEventListener(event, () => { detail.textContent = description; });
    bars.append(row);
  }
}

if (document.getElementById('results-explorer')) {
  document.getElementById('benchmark-select').addEventListener('change', renderResults);
  document.getElementById('metric-select').addEventListener('change', renderResults);
  document.querySelectorAll('.condition-controls input').forEach(input => input.addEventListener('change', renderResults));
  renderResults();
}
