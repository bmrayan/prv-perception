'use strict';
const labels = {A:'A · Visual',B_auto:'Bₐ · Automatic',B_percept:'Bₚ · Perceptual',B_full:'Bf · Full',C:'C · Text',A_visual:'A · Visual',B_auto_gpt4o:'Bₐ · GPT parser',B_auto_claude:'Bₐ · Claude parser',C_text:'C · Text'};
const colors = {A:'--a',A_visual:'--a',B_auto:'--ba',B_auto_gpt4o:'--ba',B_auto_claude:'--ba',B_percept:'--bp',B_full:'--bf',C:'--c',C_text:'--c'};
let selectedBenchmark = 'abstract';
const modelSelect = document.getElementById('model-select');
const probeSelect = document.getElementById('probe-select');

function cell(tag, text) {
  const element = document.createElement(tag);
  element.textContent = text;
  return element;
}

function renderResults() {
  const benchmark = window.PAPER_DATA.benchmarks[selectedBenchmark];
  const row = benchmark.rows.find(item => item.model === modelSelect.value) || benchmark.rows.at(-1);
  document.getElementById('chart-title').textContent = benchmark.title;
  document.getElementById('chart-unit').textContent = benchmark.unit;
  document.getElementById('chart-note').textContent = benchmark.note;
  const chart = document.getElementById('bar-chart');
  chart.replaceChildren();
  benchmark.conditions.forEach(condition => {
    const line = document.createElement('div');
    line.className = 'bar-row';
    const label = cell('span', labels[condition]);
    label.className = 'bar-label';
    const track = document.createElement('div');
    track.className = 'bar-track';
    const fill = document.createElement('div');
    fill.className = 'bar-fill';
    fill.style.background = `var(${colors[condition]})`;
    fill.style.width = `${row[condition]}%`;
    track.append(fill);
    const value = cell('span',row[condition].toFixed(1));
    value.className = 'bar-value';
    line.append(label,track,value);
    chart.append(line);
  });
  const table = document.getElementById('results-table');
  const head = document.createElement('tr');
  ['Model',...benchmark.conditions.map(c=>labels[c])].forEach(text=>head.append(cell('th',text)));
  table.querySelector('thead').replaceChildren(head);
  const body = table.querySelector('tbody');
  body.replaceChildren();
  benchmark.rows.forEach(item=>{
    const tr = document.createElement('tr');
    tr.append(cell('td',item.model));
    benchmark.conditions.forEach(c=>tr.append(cell('td',item[c].toFixed(1))));
    body.append(tr);
  });
}

function renderProbe() {
  const probe = window.PAPER_DATA.examples.find(p=>p.id===probeSelect.value);
  const img = document.getElementById('probe-image');
  img.src = `assets/${probe.id}.png`;
  img.alt = `${probe.label}: ${probe.panel_count}-panel released stimulus`;
  document.getElementById('probe-question').textContent = probe.question;
  document.getElementById('expected-answer').textContent = probe.answer;
  document.getElementById('probe-description').textContent = probe.description;
  document.getElementById('probe-id').textContent = probe.id;
  document.getElementById('probe-answer').open = false;
}

window.PAPER_DATA.benchmarks.abstract.rows.filter(r=>r.model!=='Mean').forEach(row=>{
  const option = cell('option',row.model);
  option.value = row.model;
  modelSelect.append(option);
});
window.PAPER_DATA.examples.forEach(probe=>{
  const option = cell('option',probe.label);
  option.value = probe.id;
  probeSelect.append(option);
});
document.querySelectorAll('[data-benchmark]').forEach(button=>button.addEventListener('click',()=>{
  selectedBenchmark = button.dataset.benchmark;
  document.querySelectorAll('[data-benchmark]').forEach(other=>{
    const active = other === button;
    other.classList.toggle('active',active);
    other.setAttribute('aria-pressed',String(active));
  });
  renderResults();
}));
modelSelect.addEventListener('change',renderResults);
probeSelect.addEventListener('change',renderProbe);
document.querySelectorAll('[data-copy]').forEach(button=>button.addEventListener('click',async()=>{
  const text = document.getElementById(button.dataset.copy).textContent;
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else {
      const field = document.createElement('textarea');
      field.value = text;
      field.style.position='fixed';
      field.style.left='-10000px';
      document.body.append(field);
      field.select();
      const ok = document.execCommand('copy');
      field.remove();
      if (!ok) throw new Error('Clipboard unavailable');
    }
    const previous = button.textContent;
    button.textContent = 'Copied ✓';
    document.getElementById('copy-status').textContent = button.dataset.copy==='bibtex'?'Citation copied to clipboard.':'Command copied to clipboard.';
    setTimeout(()=>{button.textContent=previous;},2000);
  } catch {
    document.getElementById('copy-status').textContent='Select the text to copy it, or download the citation file.';
  }
}));
renderResults();
renderProbe();
