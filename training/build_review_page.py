#!/usr/bin/env python3
"""Render review_sample.jsonl into the adjudication page (single HTML file)."""
import html
import json
import sys
from pathlib import Path

GEN_DIR = Path(__file__).resolve().parent / "data" / "generated"

checks = [json.loads(line) for line in
          (GEN_DIR / "review_sample.jsonl").read_text().splitlines() if line.strip()]
data_json = json.dumps(checks).replace("<", "\\u003c")
n_spans = sum(1 for c in checks if c["kind"] == "span")
n_decoys = len(checks) - n_spans

page = """<title>Redaction Label Bench</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root{
  --bg:#0A1128; --panel:#101A3C; --panel2:#16224C; --line:#26335E;
  --text:#E9EDFA; --dim:#8FA0C9; --accent:#5EAF73; --accent-ink:#0A1128;
  --bad:#E07A7A; --warn:#D9A75A;
  --chip-person:#7FA8FF; --chip-address:#5BC4B0; --chip-dob:#B08CE8; --chip-org:#E88CA8;
  --mark-bg:rgba(94,175,115,.16); --mark-line:rgba(94,175,115,.55);
  --shadow:0 6px 24px rgba(0,0,0,.35);
}
@media (prefers-color-scheme: light){
  :root:not([data-theme="dark"]){
    --bg:#F4F6FB; --panel:#FFFFFF; --panel2:#EDF0F8; --line:#D8DEEC;
    --text:#17213F; --dim:#5A6688; --accent:#3E8E58; --accent-ink:#FFFFFF;
    --bad:#C04848; --warn:#A8762E;
    --chip-person:#3B62C4; --chip-address:#2E8A79; --chip-dob:#7A4FC0; --chip-org:#B84868;
    --mark-bg:rgba(62,142,88,.12); --mark-line:rgba(62,142,88,.5);
    --shadow:0 4px 16px rgba(23,33,63,.12);
  }
}
:root[data-theme="light"]{
  --bg:#F4F6FB; --panel:#FFFFFF; --panel2:#EDF0F8; --line:#D8DEEC;
  --text:#17213F; --dim:#5A6688; --accent:#3E8E58; --accent-ink:#FFFFFF;
  --bad:#C04848; --warn:#A8762E;
  --chip-person:#3B62C4; --chip-address:#2E8A79; --chip-dob:#7A4FC0; --chip-org:#B84868;
  --mark-bg:rgba(62,142,88,.12); --mark-line:rgba(62,142,88,.5);
  --shadow:0 4px 16px rgba(23,33,63,.12);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font:15px/1.55 "IBM Plex Sans",system-ui,sans-serif;}
.wrap{max-width:860px;margin:0 auto;padding:0 20px 120px}
header.top{position:sticky;top:0;z-index:5;background:var(--bg);
  border-bottom:1px solid var(--line);padding:14px 0 10px;margin-bottom:18px}
.top .wrap{padding-bottom:0}
h1{font-size:1.25rem;margin:0 0 2px;letter-spacing:.01em}
.sub{color:var(--dim);font-size:.85rem;margin:0 0 10px}
.filters{display:flex;gap:8px;flex-wrap:wrap;padding-bottom:12px}
.fbtn{background:var(--panel);border:1px solid var(--line);color:var(--dim);
  border-radius:999px;padding:4px 14px;font:600 .78rem "IBM Plex Sans",sans-serif;
  cursor:pointer;letter-spacing:.03em}
.fbtn.on{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
details.crib{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:12px 16px;margin:0 0 18px;font-size:.86rem;color:var(--dim)}
details.crib summary{cursor:pointer;color:var(--text);font-weight:600}
details.crib td{padding:3px 12px 3px 0;vertical-align:top}
details.crib b{color:var(--text)}
.card{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--line);
  border-radius:10px;padding:14px 18px;margin-bottom:12px;box-shadow:var(--shadow)}
.card.v-ok{border-left-color:var(--accent)}
.card.v-bad{border-left-color:var(--bad)}
.card.v-unsure{border-left-color:var(--warn)}
.card.focused{outline:2px solid var(--accent);outline-offset:2px}
.meta{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.id{font:600 .75rem "JetBrains Mono",monospace;color:var(--dim)}
.chip{font:600 .68rem "IBM Plex Sans",sans-serif;letter-spacing:.05em;
  padding:2px 10px;border-radius:999px;text-transform:uppercase}
.chip.PERSON{background:color-mix(in srgb,var(--chip-person) 18%,transparent);color:var(--chip-person)}
.chip.ADDRESS{background:color-mix(in srgb,var(--chip-address) 18%,transparent);color:var(--chip-address)}
.chip.DATE_OF_BIRTH{background:color-mix(in srgb,var(--chip-dob) 18%,transparent);color:var(--chip-dob)}
.chip.SENSITIVE_ORGANIZATION{background:color-mix(in srgb,var(--chip-org) 18%,transparent);color:var(--chip-org)}
.chip.decoy{background:transparent;border:1px dashed var(--dim);color:var(--dim)}
.q{font-size:.8rem;color:var(--dim);margin-left:auto;text-align:right}
.ctx{font:.84rem/1.7 "JetBrains Mono",monospace;color:var(--dim);
  background:var(--panel2);border-radius:8px;padding:10px 14px;
  overflow-x:auto;white-space:pre-wrap;word-break:break-word}
.ctx mark{background:var(--mark-bg);color:var(--text);border:1px solid var(--mark-line);
  border-radius:4px;padding:0 4px;font-weight:600}
.verdict{display:flex;gap:8px;margin-top:10px;align-items:center;flex-wrap:wrap}
.vbtn{border:1px solid var(--line);background:var(--panel2);color:var(--text);
  border-radius:8px;padding:6px 16px;font:600 .8rem "IBM Plex Sans",sans-serif;cursor:pointer}
.vbtn:focus-visible,.fbtn:focus-visible,#copy:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.vbtn[data-v="ok"].on{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
.vbtn[data-v="bad"].on{background:var(--bad);border-color:var(--bad);color:#fff}
.vbtn[data-v="unsure"].on{background:var(--warn);border-color:var(--warn);color:#fff}
.note{flex:1;min-width:180px;background:var(--panel2);border:1px solid var(--line);
  color:var(--text);border-radius:8px;padding:6px 10px;font:.8rem "IBM Plex Sans",sans-serif;display:none}
.note.show{display:block}
footer.bar{position:fixed;left:0;right:0;bottom:0;background:var(--panel);
  border-top:1px solid var(--line);padding:10px 0;z-index:6}
.bar .wrap{padding:0 20px;display:flex;gap:14px;align-items:center}
#prog{font:600 .85rem "JetBrains Mono",monospace}
#pbar{flex:1;height:6px;background:var(--panel2);border-radius:3px;overflow:hidden}
#pfill{height:100%;width:0;background:var(--accent);transition:width .2s}
#copy{background:var(--accent);color:var(--accent-ink);border:none;border-radius:8px;
  padding:8px 18px;font:700 .85rem "IBM Plex Sans",sans-serif;cursor:pointer}
#out{display:none;margin:14px 0 0;width:100%;height:130px;background:var(--panel2);
  color:var(--text);border:1px solid var(--line);border-radius:8px;
  font:.8rem "JetBrains Mono",monospace;padding:10px}
.kbd{color:var(--dim);font-size:.75rem}
.kbd b{font:600 .72rem "JetBrains Mono",monospace;background:var(--panel2);
  border:1px solid var(--line);border-radius:4px;padding:0 5px;color:var(--text)}
@media (prefers-reduced-motion: reduce){#pfill{transition:none}}
</style>

<header class="top"><div class="wrap">
  <h1>Redaction Label Bench</h1>
  <p class="sub">__N__ checks from the generated training corpus &mdash; __NS__ labeled spans, __ND__ decoys.
  For a <b>span</b>: is the highlighted text correctly labeled per the guide? For a <b>decoy</b>:
  was it correctly <em>left unlabeled</em>? Mark <b>Wrong</b> if it should have been labeled.</p>
  <div class="filters" id="filters">
    <button class="fbtn on" data-f="all">All</button>
    <button class="fbtn" data-f="todo">Unreviewed</button>
    <button class="fbtn" data-f="flag">Flagged</button>
  </div>
</div></header>

<div class="wrap">
  <details class="crib"><summary>Guide crib sheet</summary>
    <table>
    <tr><td><b>PERSON</b></td><td>names in any layout; job titles and company names are not persons</td></tr>
    <tr><td><b>ADDRESS</b></td><td>full street + apt + city + zip as one span; standalone cities stay visible</td></tr>
    <tr><td><b>DATE_OF_BIRTH</b></td><td>the date value only, and only near &ldquo;born / DOB / date of birth&rdquo;</td></tr>
    <tr><td><b>SENSITIVE_ORG</b></td><td>clinics, hospitals, schools, shelters as third-party mentions; the document&rsquo;s own letterhead org stays visible (kept by a rule)</td></tr>
    <tr><td><b>decoys</b></td><td>employers, letterhead orgs, warehouse/branch cities, issue dates, batches/invoices/flights &mdash; correct = unlabeled. Serial and order numbers are redacted by rules, so they never appear as decoys.</td></tr>
    </table>
  </details>
  <div id="cards"></div>
  <p class="kbd">Keyboard: <b>j</b>/<b>k</b> next &amp; previous &middot; <b>1</b> agree &middot; <b>2</b> wrong &middot; <b>3</b> unsure</p>
  <textarea id="out" readonly aria-label="verdict text"></textarea>
</div>

<footer class="bar"><div class="wrap">
  <span id="prog"></span>
  <div id="pbar"><div id="pfill"></div></div>
  <button id="copy">Copy verdict</button>
</div></footer>

<script>
const CHECKS = __DATA__;
let state = {};
try { state = JSON.parse(localStorage.getItem('labelbench-r2') || '{}'); } catch (e) {}
const save = () => { try { localStorage.setItem('labelbench-r2', JSON.stringify(state)); } catch (e) {} };
const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

const cards = document.getElementById('cards');
cards.innerHTML = CHECKS.map(c => {
  const chip = c.kind === 'span'
    ? `<span class="chip ${c.label}">${c.label.replace('SENSITIVE_ORGANIZATION','SENSITIVE_ORG')}</span>`
    : `<span class="chip decoy">${c.label.replace('decoy_','decoy \\u00b7 ')}</span>`;
  const q = c.kind === 'span' ? 'labeled correctly?' : 'correctly left unlabeled?';
  return `<article class="card" id="${c.id}" tabindex="-1">
    <div class="meta"><span class="id">${c.id}</span>${chip}<span class="q">${q}</span></div>
    <div class="ctx">${esc(c.lead)}<mark>${esc(c.text)}</mark>${esc(c.tail)}</div>
    <div class="verdict">
      <button class="vbtn" data-v="ok">Agree</button>
      <button class="vbtn" data-v="bad">Wrong</button>
      <button class="vbtn" data-v="unsure">Unsure</button>
      <input class="note" placeholder="why? (optional, lands in the verdict)" aria-label="note for ${c.id}">
    </div></article>`;
}).join('');

function paint(id){
  const s = state[id] || {};
  const card = document.getElementById(id);
  card.classList.remove('v-ok','v-bad','v-unsure');
  if (s.v) card.classList.add('v-' + s.v);
  card.querySelectorAll('.vbtn').forEach(b =>
    b.classList.toggle('on', b.dataset.v === s.v));
  const note = card.querySelector('.note');
  note.classList.toggle('show', s.v === 'bad' || s.v === 'unsure');
  if (s.note !== undefined && note.value !== s.note) note.value = s.note;
}
function progress(){
  const done = CHECKS.filter(c => state[c.id] && state[c.id].v).length;
  document.getElementById('prog').textContent = done + ' / ' + CHECKS.length;
  document.getElementById('pfill').style.width = (100 * done / CHECKS.length) + '%';
}
CHECKS.forEach(c => paint(c.id));
progress();

cards.addEventListener('click', e => {
  const btn = e.target.closest('.vbtn');
  if (!btn) return;
  const id = btn.closest('.card').id;
  state[id] = state[id] || {};
  state[id].v = (state[id].v === btn.dataset.v) ? null : btn.dataset.v;
  paint(id); progress(); save(); applyFilter();
});
cards.addEventListener('input', e => {
  if (!e.target.classList.contains('note')) return;
  const id = e.target.closest('.card').id;
  state[id] = state[id] || {};
  state[id].note = e.target.value; save();
});

let filter = 'all';
function applyFilter(){
  CHECKS.forEach(c => {
    const s = state[c.id] || {};
    const show = filter === 'all' || (filter === 'todo' && !s.v)
      || (filter === 'flag' && (s.v === 'bad' || s.v === 'unsure'));
    document.getElementById(c.id).style.display = show ? '' : 'none';
  });
}
document.getElementById('filters').addEventListener('click', e => {
  const b = e.target.closest('.fbtn'); if (!b) return;
  filter = b.dataset.f;
  document.querySelectorAll('.fbtn').forEach(x => x.classList.toggle('on', x === b));
  applyFilter();
});

let cursor = -1;
function visible(){ return CHECKS.filter(c => document.getElementById(c.id).style.display !== 'none'); }
function focusCard(step){
  const vis = visible(); if (!vis.length) return;
  cursor = Math.min(Math.max(cursor + step, 0), vis.length - 1);
  document.querySelectorAll('.card.focused').forEach(x => x.classList.remove('focused'));
  const el = document.getElementById(vis[cursor].id);
  el.classList.add('focused');
  el.scrollIntoView({block:'center', behavior:'smooth'});
}
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'j') focusCard(1);
  else if (e.key === 'k') focusCard(-1);
  else if ('123'.includes(e.key) && cursor >= 0) {
    const vis = visible(); if (!vis.length) return;
    const id = vis[Math.min(cursor, vis.length-1)].id;
    state[id] = state[id] || {};
    state[id].v = {'1':'ok','2':'bad','3':'unsure'}[e.key];
    paint(id); progress(); save();
    focusCard(1);
  }
});

document.getElementById('copy').addEventListener('click', () => {
  const rows = kind => CHECKS
    .filter(c => state[c.id] && state[c.id].v === kind)
    .map(c => c.id + ((state[c.id].note || '').trim() ? ' (' + state[c.id].note.trim() + ')' : ''));
  const agree = CHECKS.filter(c => state[c.id] && state[c.id].v === 'ok').length;
  const done = CHECKS.filter(c => state[c.id] && state[c.id].v).length;
  const text = ['REVIEW VERDICT \\u2014 ' + done + '/' + CHECKS.length + ' reviewed',
    'AGREE: ' + agree,
    'WRONG: ' + (rows('bad').join(', ') || 'none'),
    'UNSURE: ' + (rows('unsure').join(', ') || 'none')].join('\\n');
  const out = document.getElementById('out');
  out.style.display = 'block'; out.value = text; out.focus(); out.select();
  try { navigator.clipboard.writeText(text); } catch (e) {}
});
</script>
"""
page = (page.replace("__N__", str(len(checks)))
            .replace("__NS__", str(n_spans))
            .replace("__ND__", str(n_decoys))
            .replace("__DATA__", data_json))
out = Path(sys.argv[1]) if len(sys.argv) > 1 else GEN_DIR / "review_page.html"
out.write_text(page, encoding="utf-8")
print(f"wrote {out} ({len(page)} bytes, {len(checks)} checks)")
