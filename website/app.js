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
const conditionNames = ['A - Visual-only', 'Ba - Automatic', 'Bp - Perceptual', 'Bf - Full scaffold', 'C - Text-only'];
const conditionColors = ['#707070', '#9b9b9b', '#5296c6', '#3273dc', '#8b71ad'];

function renderResults(dataset = 'muma') {
  const setting = resultSettings[dataset];
  const bars = document.getElementById('interactive-bars');
  bars.replaceChildren();
  document.getElementById('chart-summary').textContent = `${setting.name}: seven-model mean (%). Scale: 0-100.`;
  document.querySelectorAll('[data-dataset]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.dataset === dataset));
  });
  setting.values.forEach((score, i) => {
    const row = document.createElement('div');
    row.className = 'result-bar-row';
    const label = document.createElement('span');
    label.className = 'result-label';
    label.textContent = conditionNames[i];
    const track = document.createElement('span');
    track.className = 'result-track';
    track.setAttribute('aria-hidden', 'true');
    const fill = document.createElement('span');
    fill.className = 'result-fill';
    fill.style.background = conditionColors[i];
    fill.style.width = `${score ?? 0}%`;
    fill.style.left = '0';
    const value = document.createElement('span');
    value.className = 'result-value';
    value.textContent = score === null ? 'n/a' : `${score.toFixed(1)}%`;
    track.append(fill);
    row.append(label, track, value);
    bars.append(row);
  });
}

if (document.getElementById('results-explorer')) {
  document.querySelectorAll('[data-dataset]').forEach(button => {
    button.addEventListener('click', () => renderResults(button.dataset.dataset));
  });
  renderResults();
}
