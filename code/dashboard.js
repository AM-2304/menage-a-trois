/* Support Triage Agent — Dashboard Logic */

let DATA = [];
let SORTED_COL = null;
let SORTED_ASC = true;
let FILTERS = { search: '', status: '', risk: '', product: '' };

const COLORS = {
  low: '#34d399', medium: '#fbbf24', high: '#f87171', critical: '#d946ef',
  replied: '#34d399', escalated: '#f87171',
  devplatform: '#60a5fa', claude: '#d946ef', visa: '#fbbf24', other: '#94a3b8'
};

/* ── CSV Loader ── */
async function loadData() {
  try {
    const res = await fetch('../support_tickets/output.csv');
    if (!res.ok) throw new Error('fetch failed');
    parseCSV(await res.text());
  } catch {
    // Fallback: try same-directory path
    try {
      const res2 = await fetch('output.csv');
      if (res2.ok) { parseCSV(await res2.text()); return; }
    } catch {}
    document.querySelector('.container').innerHTML =
      '<div class="glass" style="padding:40px;text-align:center;margin-top:40px">' +
      '<h3 style="margin-bottom:12px">No data loaded</h3>' +
      '<p style="color:var(--text-dim)">Serve this directory: <code>python -m http.server 8080</code> from repo root, then open <code>http://localhost:8080/code/dashboard.html</code></p></div>';
  }
}

function parseCSV(text) {
  // Full multi-line CSV parser — handles quoted fields with newlines
  const rows = [];
  let i = 0, row = [], field = '', inQ = false;
  while (i < text.length) {
    const c = text[i];
    if (inQ) {
      if (c === '"' && text[i+1] === '"') { field += '"'; i += 2; }
      else if (c === '"') { inQ = false; i++; }
      else { field += c; i++; }
    } else {
      if (c === '"') { inQ = true; i++; }
      else if (c === ',') { row.push(field); field = ''; i++; }
      else if (c === '\n' || c === '\r') {
        row.push(field); field = '';
        if (c === '\r' && text[i+1] === '\n') i++;
        i++;
        if (row.length > 1 || row[0] !== '') rows.push(row);
        row = [];
      } else { field += c; i++; }
    }
  }
  if (field || row.length) { row.push(field); rows.push(row); }

  const headers = rows[0];
  DATA = [];
  for (let r = 1; r < rows.length; r++) {
    if (rows[r].length < headers.length) continue;
    const obj = {};
    headers.forEach((h, j) => obj[h.trim()] = (rows[r][j] || '').trim());
    obj._idx = r;
    DATA.push(obj);
  }
  renderAll();
}

/* ── Render All ── */
function renderAll() {
  renderStats();
  renderRiskChart();
  renderStatusDonut();
  renderFilters();
  renderTable();
}

/* ── Stats Row ── */
function renderStats() {
  const el = document.getElementById('stats-row');
  const replied = DATA.filter(r => r.status === 'replied').length;
  const escalated = DATA.filter(r => r.status === 'escalated').length;
  const avgConf = (DATA.reduce((s, r) => s + parseFloat(r.confidence_score || 0), 0) / DATA.length).toFixed(2);
  const piiCount = DATA.filter(r => r.pii_detected === 'true').length;
  const toolCount = DATA.filter(r => { try { return JSON.parse(r.actions_taken || '[]').length > 0; } catch { return false; } }).length;

  const cards = [
    { label: 'Total Tickets', value: DATA.length, sub: 'processed', cls: 'accent' },
    { label: 'Replied', value: replied, sub: `${(replied/DATA.length*100).toFixed(0)}% of total`, cls: '' },
    { label: 'Escalated', value: escalated, sub: `${(escalated/DATA.length*100).toFixed(0)}% of total`, cls: '' },
    { label: 'Avg Confidence', value: avgConf, sub: 'calibrated 0-1', cls: 'accent' },
    { label: 'PII Flagged', value: piiCount, sub: `${toolCount} tool actions`, cls: '' },
  ];

  el.innerHTML = cards.map((c, i) =>
    `<div class="glass stat-card ${c.cls} animate-in delay-${i+1}">
      <div class="label">${c.label}</div>
      <div class="value">${c.value}</div>
      <div class="sub">${c.sub}</div>
    </div>`
  ).join('');
}

/* ── Risk Bar Chart ── */
function renderRiskChart() {
  const el = document.getElementById('risk-bars');
  const counts = { low: 0, medium: 0, high: 0, critical: 0 };
  DATA.forEach(r => { if (counts[r.risk_level] !== undefined) counts[r.risk_level]++; });
  const max = Math.max(...Object.values(counts), 1);

  el.innerHTML = Object.entries(counts).map(([level, count]) => {
    const pct = (count / max * 100).toFixed(0);
    return `<div class="bar-row">
      <div class="bar-label">${level}</div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${pct}%;background:${COLORS[level]}"><span>${count}</span></div>
      </div>
    </div>`;
  }).join('');

  // Animate bars in
  setTimeout(() => {
    el.querySelectorAll('.bar-fill').forEach(b => { b.style.width = b.style.width; });
  }, 100);
}

/* ── Status Donut (CSS-based) ── */
function renderStatusDonut() {
  const el = document.getElementById('status-donut');
  const replied = DATA.filter(r => r.status === 'replied').length;
  const escalated = DATA.filter(r => r.status === 'escalated').length;
  const total = replied + escalated;
  const rPct = (replied / total * 100).toFixed(1);
  const ePct = (escalated / total * 100).toFixed(1);

  // Product breakdown
  const products = {};
  DATA.forEach(r => {
    const p = inferDomain(r.product_area || '');
    products[p] = (products[p] || 0) + 1;
  });

  el.innerHTML = `
    <svg viewBox="0 0 120 120" width="140" height="140" style="transform:rotate(-90deg)">
      <circle cx="60" cy="60" r="50" fill="none" stroke="rgba(255,255,255,.05)" stroke-width="16"/>
      <circle cx="60" cy="60" r="50" fill="none" stroke="${COLORS.replied}" stroke-width="16"
        stroke-dasharray="${rPct * 3.14} ${314}" stroke-linecap="round" style="transition:stroke-dasharray 1s ease"/>
      <circle cx="60" cy="60" r="50" fill="none" stroke="${COLORS.escalated}" stroke-width="16"
        stroke-dasharray="${ePct * 3.14} ${314}" stroke-dashoffset="${-rPct * 3.14}" stroke-linecap="round" style="transition:all 1s ease"/>
    </svg>
    <div class="donut-legend">
      <div class="legend-item"><div class="legend-dot" style="background:${COLORS.replied}"></div>Replied: ${replied} (${rPct}%)</div>
      <div class="legend-item"><div class="legend-dot" style="background:${COLORS.escalated}"></div>Escalated: ${escalated} (${ePct}%)</div>
      <div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--glass-border)">
        ${Object.entries(products).sort((a,b)=>b[1]-a[1]).map(([p, c]) =>
          `<div class="legend-item"><div class="legend-dot" style="background:${COLORS[p] || COLORS.other}"></div>${capitalize(p)}: ${c}</div>`
        ).join('')}
      </div>
    </div>`;
}

function inferDomain(area) {
  const a = area.toLowerCase();
  if (a.includes('devplatform') || a.includes('hackerrank') || a.includes('codepair') || a.includes('assessment') || a.includes('test') || a.includes('interview') || a.includes('coding')) return 'devplatform';
  if (a.includes('claude') || a.includes('anthropic') || a.includes('conversation') || a.includes('project') || a.includes('artifact') || a.includes('api')) return 'claude';
  if (a.includes('visa') || a.includes('card') || a.includes('transaction') || a.includes('dispute') || a.includes('fraud') || a.includes('payment') || a.includes('chargeback')) return 'visa';
  return 'other';
}
function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

/* ── Filters ── */
function renderFilters() {
  const el = document.getElementById('filter-bar');
  el.innerHTML = `
    <input type="text" placeholder="Search tickets..." oninput="setFilter('search',this.value)" id="f-search">
    <select onchange="setFilter('status',this.value)">
      <option value="">All Status</option><option value="replied">Replied</option><option value="escalated">Escalated</option>
    </select>
    <select onchange="setFilter('risk',this.value)">
      <option value="">All Risk</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option><option value="critical">Critical</option>
    </select>
    <select onchange="setFilter('product',this.value)">
      <option value="">All Products</option><option value="devplatform">DevPlatform</option><option value="claude">Claude</option><option value="visa">Visa</option>
    </select>
    <button class="pill-btn" onclick="resetFilters()">Reset</button>`;
}

function setFilter(key, val) {
  FILTERS[key] = val.toLowerCase();
  renderTable();
}

function resetFilters() {
  FILTERS = { search: '', status: '', risk: '', product: '' };
  document.querySelectorAll('.filter-bar input, .filter-bar select').forEach(el => el.value = '');
  renderTable();
}

function getFiltered() {
  return DATA.filter(r => {
    if (FILTERS.search) {
      const s = FILTERS.search;
      const hay = `${r.subject} ${r.response} ${r.product_area} ${r.justification}`.toLowerCase();
      if (!hay.includes(s)) return false;
    }
    if (FILTERS.status && r.status !== FILTERS.status) return false;
    if (FILTERS.risk && r.risk_level !== FILTERS.risk) return false;
    if (FILTERS.product && inferDomain(r.product_area) !== FILTERS.product) return false;
    return true;
  });
}

/* ── Table ── */
const TABLE_COLS = [
  { key: '_idx', label: '#', w: '40px' },
  { key: 'subject', label: 'Subject', w: '220px' },
  { key: 'status', label: 'Status', w: '90px' },
  { key: 'product_area', label: 'Product', w: '160px' },
  { key: 'risk_level', label: 'Risk', w: '80px' },
  { key: 'confidence_score', label: 'Confidence', w: '110px' },
  { key: 'request_type', label: 'Type', w: '100px' },
  { key: 'pii_detected', label: 'PII', w: '50px' },
  { key: 'language', label: 'Lang', w: '50px' },
];

function renderTable() {
  const thead = document.getElementById('thead');
  const tbody = document.getElementById('tbody');

  thead.innerHTML = '<tr>' + TABLE_COLS.map(c =>
    `<th style="width:${c.w}" onclick="sortBy('${c.key}')" class="${SORTED_COL===c.key?'sorted':''}">
      ${c.label}<span class="sort-arrow">${SORTED_COL===c.key?(SORTED_ASC?'▲':'▼'):'▸'}</span>
    </th>`
  ).join('') + '</tr>';

  let rows = getFiltered();

  if (SORTED_COL) {
    rows.sort((a, b) => {
      let va = a[SORTED_COL] || '', vb = b[SORTED_COL] || '';
      if (SORTED_COL === 'confidence_score' || SORTED_COL === '_idx') {
        va = parseFloat(va) || 0; vb = parseFloat(vb) || 0;
        return SORTED_ASC ? va - vb : vb - va;
      }
      return SORTED_ASC ? va.localeCompare(vb) : vb.localeCompare(va);
    });
  }

  tbody.innerHTML = rows.map(r => `<tr onclick="openDetail(${DATA.indexOf(r)})">
    ${TABLE_COLS.map(c => {
      const v = r[c.key] || '';
      if (c.key === 'status') return `<td><span class="badge badge-${v}">${v}</span></td>`;
      if (c.key === 'risk_level') return `<td><span class="badge badge-${v}">${v}</span></td>`;
      if (c.key === 'confidence_score') {
        const pct = (parseFloat(v) * 100).toFixed(0);
        const color = parseFloat(v) >= .75 ? COLORS.low : parseFloat(v) >= .5 ? COLORS.medium : COLORS.high;
        return `<td><div style="display:flex;align-items:center;gap:8px"><span style="font-size:13px;font-weight:600;color:${color}">${v}</span><div class="conf-meter"><div class="conf-fill" style="width:${pct}%;background:${color}"></div></div></div></td>`;
      }
      if (c.key === 'pii_detected') return `<td style="color:${v==='true'?COLORS.high:COLORS.low}">${v==='true'?'Yes':'--'}</td>`;
      if (c.key === 'product_area') return `<td style="max-width:160px;overflow:hidden;text-overflow:ellipsis" title="${v}">${v}</td>`;
      if (c.key === 'subject') return `<td style="max-width:220px;overflow:hidden;text-overflow:ellipsis" title="${v}">${v || '(none)'}</td>`;
      return `<td>${v}</td>`;
    }).join('')}
  </tr>`).join('');
}

function sortBy(col) {
  if (SORTED_COL === col) SORTED_ASC = !SORTED_ASC;
  else { SORTED_COL = col; SORTED_ASC = true; }
  renderTable();
}

/* ── Detail Panel ── */
function openDetail(idx) {
  const r = DATA[idx];
  document.getElementById('detail-title').textContent = r.subject || '(no subject)';

  let actions = '[]';
  try { actions = JSON.stringify(JSON.parse(r.actions_taken || '[]'), null, 2); } catch { actions = r.actions_taken || '[]'; }

  const sources = (r.source_documents || '').split('|').filter(Boolean);

  document.getElementById('detail-body').innerHTML = `
    <div class="meta-grid" style="margin-bottom:20px">
      <div class="meta-item"><div class="mk">Status</div><div class="mv"><span class="badge badge-${r.status}">${r.status}</span></div></div>
      <div class="meta-item"><div class="mk">Risk Level</div><div class="mv"><span class="badge badge-${r.risk_level}">${r.risk_level}</span></div></div>
      <div class="meta-item"><div class="mk">Confidence</div><div class="mv" style="color:${parseFloat(r.confidence_score)>=.75?COLORS.low:parseFloat(r.confidence_score)>=.5?COLORS.medium:COLORS.high}">${r.confidence_score}</div></div>
      <div class="meta-item"><div class="mk">Request Type</div><div class="mv">${r.request_type}</div></div>
      <div class="meta-item"><div class="mk">Product Area</div><div class="mv">${r.product_area}</div></div>
      <div class="meta-item"><div class="mk">Language</div><div class="mv">${r.language}</div></div>
      <div class="meta-item"><div class="mk">PII Detected</div><div class="mv" style="color:${r.pii_detected==='true'?COLORS.high:COLORS.low}">${r.pii_detected}</div></div>
      <div class="meta-item"><div class="mk">Ticket #</div><div class="mv">${r._idx}</div></div>
    </div>
    <div class="detail-section"><h4>Agent Response</h4><pre>${escapeHtml(r.response || '')}</pre></div>
    <div class="detail-section"><h4>Justification</h4><pre>${escapeHtml(r.justification || '')}</pre></div>
    <div class="detail-section"><h4>Source Documents</h4>
      ${sources.length ? sources.map(s => `<div style="font-size:12px;color:var(--fuchsia-dim);padding:4px 0;word-break:break-all">${s}</div>`).join('') : '<p style="color:var(--text-muted)">None</p>'}
    </div>
    <div class="detail-section"><h4>Actions Taken</h4><pre style="color:var(--fuchsia-dim)">${escapeHtml(actions)}</pre></div>
    <div class="detail-section"><h4>Original Conversation</h4><pre style="max-height:400px">${escapeHtml(formatConversation(r.issue))}</pre></div>`;

  document.getElementById('detail-panel').classList.add('open');
  document.getElementById('overlay').classList.add('open');

  // Highlight row
  document.querySelectorAll('tbody tr').forEach(tr => tr.classList.remove('selected'));
}

function closeDetail() {
  document.getElementById('detail-panel').classList.remove('open');
  document.getElementById('overlay').classList.remove('open');
}

function formatConversation(issueJson) {
  try {
    const msgs = JSON.parse(issueJson);
    if (Array.isArray(msgs)) {
      return msgs.map(m => `[${(m.role || '').toUpperCase()}]\n${m.content || ''}`).join('\n\n---\n\n');
    }
  } catch {}
  return issueJson || '';
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// Keyboard shortcut
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeDetail();
});

// Init
loadData();
