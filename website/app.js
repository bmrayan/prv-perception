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
const conditionNames = ['Visual-only', 'Automatic', 'Perceptual', 'Full scaffold', 'Text-only'];
const conditionSymbols = ['A', 'Ba', 'Bp', 'Bf', 'C'];
const conditionColors = ['#FCC495', '#C3D9C5', '#E9DDB7', '#9ED2E0', '#D5C6E0'];

function renderResults(dataset = 'muma') {
  const setting = resultSettings[dataset];
  const bars = document.getElementById('interactive-bars');
  bars.replaceChildren();
  document.getElementById('chart-summary').textContent = 'Seven-model mean (%)';
  document.querySelectorAll('[data-dataset]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.dataset === dataset));
  });
  function svgElement(tag, attributes = {}, text = '') {
    const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    if (text) element.textContent = text;
    return element;
  }
  const chart = svgElement('svg', { viewBox: '0 0 760 330', role: 'img', 'aria-label': `${setting.name}: ${setting.values.map((score, i) => `${conditionNames[i]} ${score === null ? 'not defined' : score.toFixed(1) + '%'}`).join(', ')}` });
  const left = 48, right = 744, top = 30, bottom = 260;
  for (let tick = 0; tick <= 100; tick += 20) {
    const y = bottom - tick / 100 * (bottom - top);
    chart.append(svgElement('line', { x1: left, x2: right, y1: y, y2: y, stroke: tick === 0 ? '#555' : '#dedede', 'stroke-width': 1, 'stroke-dasharray': tick === 0 ? 'none' : '3 3' }));
    chart.append(svgElement('text', { x: left - 12, y: y + 4, 'text-anchor': 'end', class: 'plot-tick' }, String(tick)));
  }
  setting.values.forEach((score, i) => {
    const x = left + (i + 0.5) * (right - left) / 5;
    if (score !== null) {
      const height = score / 100 * (bottom - top);
      chart.append(svgElement('rect', { x: x - 36, y: bottom - height, width: 72, height, fill: conditionColors[i], stroke: '#444', 'stroke-width': 1.1 }));
      chart.append(svgElement('text', { x, y: bottom - height - 9, 'text-anchor': 'middle', class: 'plot-value' }, score.toFixed(1)));
    } else {
      chart.append(svgElement('text', { x, y: bottom - 12, 'text-anchor': 'middle', class: 'plot-tick' }, 'n/a'));
    }
    const symbol = svgElement('text', { x, y: bottom + 25, 'text-anchor': 'middle', class: 'plot-symbol' });
    symbol.append(svgElement('tspan', {}, conditionSymbols[i][0]));
    if (conditionSymbols[i].length > 1) symbol.append(svgElement('tspan', { 'baseline-shift': 'sub', 'font-size': 10 }, conditionSymbols[i][1]));
    chart.append(symbol);
    chart.append(svgElement('text', { x, y: bottom + 47, 'text-anchor': 'middle', class: 'plot-label' }, conditionNames[i]));
  });
  bars.append(chart);
}

if (document.getElementById('results-explorer')) {
  document.querySelectorAll('[data-dataset]').forEach(button => {
    button.addEventListener('click', () => renderResults(button.dataset.dataset));
  });
  renderResults();
}
