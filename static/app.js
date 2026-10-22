/* Marine Debris Mapper: front end */
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const S = {
  meta: null, res: null, imgs: {}, labels: null, conf: null, gt: null,
  layer: 'pred', opacity: 0.75, zoom: 1, view: 'image',
  sampleDs: 'MARIDA', regionFilter: 'all', active: null, map: null, mapLayers: [],
};

const CLASS_ORDER = [2, 3, 4, 1, 5, 6];
const fmt = (n, d = 0) => Number(n).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
const area = (m2) => m2 >= 1e6 ? `${fmt(m2 / 1e6, 2)} km²` : m2 >= 1e4 ? `${fmt(m2 / 1e4, 2)} ha` : `${fmt(m2)} m²`;
const rgba = (c) => `rgba(${c[0]},${c[1]},${c[2]},${Math.max(c[3], 160) / 255})`;
const swatch = (k) => `<i class="sw" style="background:${rgba(S.meta.classes[k].color)}"></i>`;
const cname = (k) => S.meta.classes[k].name;
const pct = (x) => x == null ? '–' : (x * 100).toFixed(1) + '%';

/* ---------------------------------------------------------------- setup */
async function init() {
  S.meta = await (await fetch('/api/meta')).json();
  const sel = $('#model');
  sel.innerHTML = `<option value="">Auto (matches the image source)</option>` +
    Object.entries(S.meta.models).map(([k, m]) => `<option value="${k}">${m.label}</option>`).join('');
  renderGallery();
  renderLegend();
  renderAbout();
  if (!Object.keys(S.meta.models).length) showError('No trained models found. Run <code>python train_models.py</code> first.');
}

function renderGallery() {
  const items = S.meta.samples.filter((s) => s.dataset === S.sampleDs);
  $('#gallery').innerHTML = items.map((s) =>
    `<button class="thumb" data-id="${s.id}" data-ds="${s.dataset}" title="${s.id}\n${Object.entries(s.counts).map(([k, v]) => `${k}: ${v} px`).join('\n')}">
       <img loading="lazy" src="/api/thumb/${s.dataset}/${encodeURIComponent(s.id)}.png" alt="">
       <span>${s.why}</span></button>`).join('') || '<p class="muted small">No samples (dataset not trained yet).</p>';
  $$('#gallery .thumb').forEach((b) => b.onclick = () => {
    $$('#gallery .thumb').forEach((x) => x.classList.remove('active'));
    b.classList.add('active');
    runSample(b.dataset.ds, b.dataset.id);
  });
}

function renderLegend() {
  $('#legend').innerHTML = [2, 3, 4, 1, 5, 6].map((k) =>
    `<span title="${S.meta.classes[k].description}">${swatch(k)}${cname(k)}</span>`).join('');
}

function renderAbout() {
  const rows = Object.entries(S.meta.models).filter(([, m]) => m.scores).map(([k, m]) => {
    const s = m.scores, pc = s.per_class;
    return `<tr><td>${m.label}</td><td>${s.macro_f1.toFixed(3)}</td><td>${s.miou.toFixed(3)}</td>
      <td>${pc['Marine Debris'].f1.toFixed(3)}</td><td>${pc['Algae/Sargassum'].f1.toFixed(3)}</td>
      <td>${pc['Other'].f1.toFixed(3)}</td><td>${fmt(s.test_pixels)}</td></tr>`;
  }).join('');
  $('#scoreTable').innerHTML = rows ? `<table class="t"><tr><th>Model</th><th>Macro F1</th><th>mIoU</th>
    <th>Debris F1</th><th>Algae F1</th><th>Other F1</th><th>Test px</th></tr>${rows}</table>` : '<p class="muted">Not trained yet.</p>';
  $('#classTable').innerHTML = `<table class="t">${[1, 2, 3, 4, 5, 6].map((k) =>
    `<tr><td style="white-space:nowrap">${swatch(k)}${cname(k)}</td><td style="text-align:left">${S.meta.classes[k].description}</td></tr>`).join('')}</table>`;
}

/* ------------------------------------------------------------- requests */
function opts() {
  return { model: $('#model').value, threshold: $('#thr').value, scope: $('#scope').checked };
}

async function call(url, init, label) {
  hideError();
  $('#loading').classList.remove('hidden');
  $('#loadingText').textContent = label || 'Analysing…';
  try {
    const r = await fetch(url, init);
    const j = await r.json().catch(() => ({ error: `Server error (${r.status}).` }));
    if (!r.ok || j.error) throw new Error(j.error || `Server error (${r.status}).`);
    await show(j);
  } catch (e) {
    showError(e.message);
  } finally {
    $('#loading').classList.add('hidden');
  }
}

const runSample = (dataset, id) => call('/api/sample', {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ dataset, id, ...opts() }),
}, 'Reading 11 bands and classifying every pixel…');

function runUpload(files) {
  if (!files.length) return;
  const fd = new FormData();
  [...files].forEach((f) => fd.append('files', f, f.name));
  Object.entries(opts()).forEach(([k, v]) => fd.append(k, v));
  fd.append('boa_offset', $('#boa').checked);
  $$('#gallery .thumb').forEach((x) => x.classList.remove('active'));
  call('/api/upload', { method: 'POST', body: fd }, `Uploading ${files.length} file(s) and analysing…`);
}

let reTimer;
function reanalyse() {
  if (!S.res) return;
  clearTimeout(reTimer);
  reTimer = setTimeout(() => call('/api/reanalyse', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token: S.res.token, ...opts() }),
  }, 'Updating…'), 250);
}

/* ------------------------------------------------------------- results */
const b64bytes = (b) => Uint8Array.from(atob(b), (c) => c.charCodeAt(0));
const loadImg = (src) => new Promise((ok) => { const i = new Image(); i.onload = () => ok(i); i.src = src; });

async function show(res) {
  S.res = res;
  S.active = null;
  S.labels = b64bytes(res.labels);
  S.conf = b64bytes(res.conf);
  S.gt = res.gt_labels ? b64bytes(res.gt_labels) : null;
  S.imgs = {};
  for (const [k, v] of Object.entries(res.layers)) S.imgs[k] = await loadImg(v);

  $('#title').textContent = res.name;
  const src = { marida: 'MARIDA patch', mados: 'MADOS patch', sentinel2: 'Sentinel-2 image' }[res.source];
  $('#subtitle').textContent = `${src} · ${res.width}×${res.height} px · ${res.pixel_size_m[0]} m pixels · ${res.model_label} · ${res.seconds}s`;
  $('#empty').classList.add('hidden');
  $('#gtBtn').disabled = !S.gt;
  if (!S.gt && S.layer === 'gt') setLayer('pred');
  $('#mapBtn').disabled = !res.georeferenced;
  if (!res.georeferenced && S.view === 'map') setView('image');

  const notes = res.notes || [];
  $('#notes').innerHTML = notes.map((n) => `⚠ ${n}`).join('<br>');
  $('#notes').classList.toggle('hidden', !notes.length);

  fitZoom();
  setView(S.view);
  renderSummary();
  renderRegions();
  renderGT();
  $('#csvBtn').disabled = !res.regions.length;
}

function renderSummary() {
  const r = S.res, sm = Object.fromEntries(r.summary.map((s) => [s.class, s]));
  const by = (c) => r.regions.filter((x) => x.class === c);
  const deb = by(2), alg = by(3), oth = by(4);
  const where = (x) => x.lat != null ? `${Math.abs(x.lat).toFixed(4)}°${x.lat >= 0 ? 'N' : 'S'}, ${Math.abs(x.lon).toFixed(4)}°${x.lon >= 0 ? 'E' : 'W'}`
    : `pixel (row ${Math.round(x.row)}, col ${Math.round(x.col)})`;
  const parts = [];
  if (deb.length) parts.push(`<b style="color:#ff6b6b">${deb.length} marine-debris patch${deb.length > 1 ? 'es' : ''}</b> covering ${area(sm[2].area_m2)}; the largest (${area(deb[0].area_m2)}, ${(deb[0].mean_conf * 100).toFixed(0)}% confidence) is at ${where(deb[0])}`);
  else parts.push('<b>No marine debris</b> detected');
  if (alg.length) parts.push(`<b style="color:#5be37a">${alg.length} algae/Sargassum patch${alg.length > 1 ? 'es' : ''}</b> (${area(sm[3].area_m2)})`);
  if (oth.length) parts.push(`<b style="color:#ffc933">${oth.length} other floating feature${oth.length > 1 ? 's' : ''}</b> (${area(sm[4].area_m2)})`);
  $('#headline').innerHTML = parts.join('; ') + `. The rest is ${sm[1].percent.toFixed(1)}% water` +
    (sm[5].pixels ? `, ${sm[5].percent.toFixed(1)}% uncertain` : '') + (sm[6].pixels ? `, ${sm[6].percent.toFixed(1)}% cloud/land` : '') + '.';
  $('#headline').classList.remove('muted');

  $('#summary').innerHTML = CLASS_ORDER.map((k) => {
    const s = sm[k];
    if (!s || (!s.pixels && k >= 5)) return '';
    return `<div class="bar-row"><span>${swatch(k)}${cname(k)}</span><span class="muted">${fmt(s.pixels)} px · ${area(s.area_m2)} · ${s.percent.toFixed(2)}%</span>
      <div class="bar"><i style="width:${Math.max(s.percent, s.pixels ? 0.8 : 0)}%;background:${rgba(S.meta.classes[k].color)}"></i></div></div>`;
  }).join('');
}

function renderRegions() {
  const list = S.res.regions.filter((r) => S.regionFilter === 'all' || r.class === +S.regionFilter);
  if (!list.length) { $('#regions').innerHTML = '<p class="muted">No patches of this kind.</p>'; return; }
  $('#regions').innerHTML = list.map((r) => {
    const loc = r.lat != null
      ? `<a href="https://www.google.com/maps/@${r.lat},${r.lon},16z/data=!3m1!1e3" target="_blank" rel="noopener" onclick="event.stopPropagation()">${r.lat.toFixed(5)}, ${r.lon.toFixed(5)}</a>`
      : `row ${Math.round(r.row)}, col ${Math.round(r.col)}`;
    return `<div class="region ${S.active === r.id ? 'active' : ''}" data-id="${r.id}">
      <span class="num">#${r.id}</span><span>${swatch(r.class)}<b>${cname(r.class).split(' (')[0]}</b></span>
      <span class="conf">${(r.mean_conf * 100).toFixed(0)}%</span>
      <span class="meta">${fmt(r.pixels)} px · ${area(r.area_m2)} · ${loc}</span></div>`;
  }).join('');
  $$('#regions .region').forEach((el) => el.onclick = () => focusRegion(+el.dataset.id));
}

function renderGT() {
  const g = S.res.gt_metrics;
  $('#gtSection').classList.toggle('hidden', !g);
  if (!g) return;
  const rows = g.per_class.map((c) => `<tr><td>${swatch(c.class)}${cname(c.class).split(' (')[0]}</td>
    <td>${fmt(c.support)}</td><td>${pct(c.precision)}</td><td>${pct(c.recall)}</td><td>${c.f1 == null ? '–' : c.f1.toFixed(3)}</td></tr>`).join('');
  const d = g.per_class[1];
  $('#gtMetrics').innerHTML = `<div class="kpis">
      <div class="kpi"><b>${pct(g.accuracy)}</b><span>pixel accuracy (${fmt(g.annotated_pixels)} labelled px)</span></div>
      <div class="kpi"><b>${d.support ? `${d.tp}/${d.support}` : '–'}</b><span>debris pixels found · ${d.fp} false alarms</span></div></div>
    <table class="t"><tr><th>Class</th><th>Labelled</th><th>Precision</th><th>Recall</th><th>F1</th></tr>${rows}</table>
    <p class="muted small">Uses the raw top class on annotated pixels only (no uncertainty or cloud masking), as in the paper. Switch to <em>Ground truth</em> to compare visually.</p>`;
}

/* --------------------------------------------------------------- canvas */
function fitZoom() {
  const st = $('#stage');
  S.zoom = Math.max(0.25, Math.min((st.clientWidth - 20) / S.res.width, (st.clientHeight - 20) / S.res.height));
  draw();
}

function draw() {
  if (!S.res) return;
  const c = $('#canvas'), ctx = c.getContext('2d');
  const W = Math.round(S.res.width * S.zoom), H = Math.round(S.res.height * S.zoom);
  c.width = W; c.height = H;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(S.imgs.rgb, 0, 0, W, H);
  const ov = S.imgs[S.layer];
  if (ov && S.layer !== 'none') {
    ctx.globalAlpha = S.opacity;
    ctx.drawImage(ov, 0, 0, W, H);
    ctx.globalAlpha = 1;
  }
  if (S.active) {
    const r = S.res.regions.find((x) => x.id === S.active);
    if (r) {
      const [r0, c0, r1, c1] = r.bbox, pad = 3;
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#fff';
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(c0 * S.zoom - pad, r0 * S.zoom - pad, (c1 - c0) * S.zoom + 2 * pad, (r1 - r0) * S.zoom + 2 * pad);
      ctx.setLineDash([]);
    }
  }
}

function pixelAt(ev) {
  const rect = $('#canvas').getBoundingClientRect();
  const col = Math.floor((ev.clientX - rect.left) / S.zoom), row = Math.floor((ev.clientY - rect.top) / S.zoom);
  if (col < 0 || row < 0 || col >= S.res.width || row >= S.res.height) return null;
  return { row, col, i: row * S.res.width + col };
}

function latlonAt(row, col) {
  const b = S.res.bounds;               // tl, tr, bl, br
  if (!b) return null;
  const u = col / Math.max(S.res.width - 1, 1), v = row / Math.max(S.res.height - 1, 1);
  const lerp = (a, b2, t) => a + (b2 - a) * t;
  const top = [lerp(b[0][0], b[1][0], u), lerp(b[0][1], b[1][1], u)];
  const bot = [lerp(b[2][0], b[3][0], u), lerp(b[2][1], b[3][1], u)];
  return [lerp(top[0], bot[0], v), lerp(top[1], bot[1], v)];
}

function onHover(ev) {
  const p = S.res && pixelAt(ev);
  const tt = $('#tooltip');
  if (!p) { tt.classList.add('hidden'); return; }
  const k = S.labels[p.i];
  const ll = latlonAt(p.row, p.col);
  let html = k === 0 ? '<b>No data</b>' : `${swatch(k)}<b>${cname(k)}</b><br>confidence ${(S.conf[p.i] / 2.55).toFixed(0)}%`;
  if (S.gt) { const g = S.gt[p.i]; html += `<br>ground truth: ${g >= 1 && g <= 4 ? swatch(g) + cname(g).split(' (')[0] : '<span class="muted">unlabelled</span>'}`; }
  html += `<br><span class="muted">row ${p.row}, col ${p.col}${ll ? `<br>${ll[0].toFixed(5)}, ${ll[1].toFixed(5)}` : ''}</span>`;
  tt.innerHTML = html;
  const st = $('#stage').getBoundingClientRect();
  let x = ev.clientX - st.left + 14, y = ev.clientY - st.top + 14;
  if (x > st.width - 190) x -= 200;
  if (y > st.height - 110) y -= 120;
  tt.style.left = x + 'px'; tt.style.top = y + 'px';
  tt.classList.remove('hidden');
}

function onClick(ev) {
  const p = S.res && pixelAt(ev);
  if (!p) return;
  // the region containing (or nearest to) the clicked pixel
  let best = null, bd = 1e9;
  for (const r of S.res.regions) {
    const [r0, c0, r1, c1] = r.bbox;
    const d = (p.row >= r0 && p.row < r1 && p.col >= c0 && p.col < c1) ? 0 : Math.hypot(p.row - r.row, p.col - r.col);
    if (d < bd) { bd = d; best = r; }
  }
  if (best && bd < 6) focusRegion(best.id, false);
}

function focusRegion(id, scroll = true) {
  S.active = S.active === id ? null : id;
  draw();
  renderRegions();
  const r = S.res.regions.find((x) => x.id === id);
  if (S.view === 'map' && r && r.lat != null) S.map.flyTo([r.lat, r.lon], 17);
  if (scroll && r && S.view === 'image') {
    const w = $('#canvasWrap');
    w.scrollTo({ left: r.col * S.zoom - w.clientWidth / 2, top: r.row * S.zoom - w.clientHeight / 2, behavior: 'smooth' });
  }
  const el = document.querySelector(`.region[data-id="${id}"]`);
  if (el && !scroll) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

/* ------------------------------------------------------------------ map */
function composite() {
  const c = document.createElement('canvas');
  c.width = S.res.width; c.height = S.res.height;
  const ctx = c.getContext('2d');
  ctx.drawImage(S.imgs.rgb, 0, 0);
  const ov = S.imgs[S.layer];
  if (ov && S.layer !== 'none') { ctx.globalAlpha = S.opacity; ctx.drawImage(ov, 0, 0); }
  return c.toDataURL();
}

function drawMap() {
  if (!S.map) {
    S.map = L.map('map', { zoomControl: true });
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 19, attribution: 'Imagery © Esri, Maxar, Earthstar Geographics',
    }).addTo(S.map);
  }
  S.mapLayers.forEach((l) => S.map.removeLayer(l));
  S.mapLayers = [];
  const b = S.res.bounds;
  const lats = b.map((x) => x[0]), lons = b.map((x) => x[1]);
  const bounds = [[Math.min(...lats), Math.min(...lons)], [Math.max(...lats), Math.max(...lons)]];
  S.mapLayers.push(L.imageOverlay(composite(), bounds, { opacity: 1 }).addTo(S.map));
  S.mapLayers.push(L.rectangle(bounds, { color: '#2fb3e6', weight: 1, fill: false }).addTo(S.map));
  S.res.regions.filter((r) => r.lat != null && r.class !== 4).slice(0, 60).forEach((r) => {
    const m = L.circleMarker([r.lat, r.lon], {
      radius: r.class === 2 ? 6 : 4, color: '#000', weight: 1.5,
      fillColor: rgba(S.meta.classes[r.class].color), fillOpacity: 1,
    }).bindPopup(`<b>#${r.id} ${cname(r.class)}</b><br>${fmt(r.pixels)} px · ${area(r.area_m2)}<br>confidence ${(r.mean_conf * 100).toFixed(0)}%<br>${r.lat.toFixed(5)}, ${r.lon.toFixed(5)}`);
    m.on('click', () => focusRegion(r.id, false));
    S.mapLayers.push(m.addTo(S.map));
  });
  setTimeout(() => { S.map.invalidateSize(); S.map.fitBounds(bounds, { padding: [20, 20] }); }, 30);
}

/* -------------------------------------------------------------- controls */
function setLayer(l) {
  S.layer = l;
  $$('#layerBtns .seg').forEach((b) => b.classList.toggle('active', b.dataset.layer === l));
  draw();
  if (S.view === 'map') drawMap();
}
function setView(v) {
  S.view = v;
  $$('.view-switch .seg').forEach((b) => b.classList.toggle('active', b.dataset.view === v));
  $('#canvasWrap').classList.toggle('hidden', v !== 'image' || !S.res);
  $('#map').classList.toggle('hidden', v !== 'map');
  $$('#zoomIn,#zoomOut,#zoomFit').forEach((b) => b.disabled = v === 'map');
  if (v === 'map' && S.res) drawMap();
}
function showError(msg) { $('#error').innerHTML = msg; $('#error').classList.remove('hidden'); }
function hideError() { $('#error').classList.add('hidden'); }

function downloadCSV() {
  const head = 'id,class,pixels,area_m2,mean_confidence,max_confidence,lat,lon,centre_row,centre_col,bbox_row0,bbox_col0,bbox_row1,bbox_col1';
  const rows = S.res.regions.map((r) => [r.id, `"${cname(r.class)}"`, r.pixels, r.area_m2.toFixed(0), r.mean_conf.toFixed(3),
    r.max_conf.toFixed(3), r.lat ?? '', r.lon ?? '', r.row.toFixed(1), r.col.toFixed(1), ...r.bbox].join(','));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([[head, ...rows].join('\n')], { type: 'text/csv' }));
  a.download = `${S.res.name.replace(/\.[^.]+$/, '')}_patches.csv`;
  a.click();
}

$$('.tab').forEach((t) => t.onclick = () => {
  $$('.tab').forEach((x) => x.classList.toggle('active', x === t));
  $('#tab-analyse').classList.toggle('hidden', t.dataset.tab !== 'analyse');
  $('#tab-about').classList.toggle('hidden', t.dataset.tab !== 'about');
  if (t.dataset.tab === 'analyse' && S.view === 'map' && S.map) setTimeout(() => S.map.invalidateSize(), 30);
});
$$('.sample-filter .chip').forEach((c) => c.onclick = () => {
  $$('.sample-filter .chip').forEach((x) => x.classList.toggle('active', x === c));
  S.sampleDs = c.dataset.ds;
  renderGallery();
});
$$('.region-filter .chip').forEach((c) => c.onclick = () => {
  $$('.region-filter .chip').forEach((x) => x.classList.toggle('active', x === c));
  S.regionFilter = c.dataset.cls;
  if (S.res) renderRegions();
});
$$('#layerBtns .seg').forEach((b) => b.onclick = () => setLayer(b.dataset.layer));
$$('.view-switch .seg').forEach((b) => b.onclick = () => !b.disabled && setView(b.dataset.view));
$('#opacity').oninput = (e) => { S.opacity = +e.target.value; draw(); };
$('#opacity').onchange = () => { if (S.view === 'map') drawMap(); };
$('#zoomIn').onclick = () => { S.zoom *= 1.5; draw(); };
$('#zoomOut').onclick = () => { S.zoom /= 1.5; draw(); };
$('#zoomFit').onclick = () => S.res && fitZoom();
$('#thr').oninput = (e) => { $('#thrv').textContent = (+e.target.value).toFixed(2); };
$('#thr').onchange = reanalyse;
$('#scope').onchange = reanalyse;
$('#model').onchange = reanalyse;
$('#csvBtn').onclick = downloadCSV;
$('#canvas').addEventListener('mousemove', onHover);
$('#canvas').addEventListener('mouseleave', () => $('#tooltip').classList.add('hidden'));
$('#canvas').addEventListener('click', onClick);
$('#file').onchange = (e) => { runUpload(e.target.files); e.target.value = ''; };
const drop = $('#drop');
['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => runUpload(e.dataTransfer.files));
window.addEventListener('resize', () => { if (S.res && S.view === 'image') fitZoom(); });

init();
