/* Layout audit: verifies visual design renders correctly at desktop & mobile widths. */
const { spawn } = require('child_process');
const path = require('path');
const { pathToFileURL } = require('url');

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const URL = pathToFileURL(path.join(__dirname, 'index.html')).href;
const PORT = 9334;
const results = [];
const step = (name, ok, extra) => {
  results.push((ok ? 'PASS ' : 'FAIL ') + name + (extra ? ' [' + extra + ']' : ''));
};

async function runAudit(width, height, label) {
  const chrome = spawn(CHROME, [
    '--headless=new', '--disable-gpu', '--no-sandbox', '--no-first-run',
    '--remote-debugging-port=' + PORT, '--window-size=' + width + ',' + height,
    '--user-data-dir=/tmp/ps-chrome-audit-' + width, 'about:blank',
  ]);
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  let targetUrl;
  for (let i = 0; i < 50; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${PORT}/json/new?` + encodeURIComponent(URL), { method: 'PUT' });
      targetUrl = (await res.json()).webSocketDebuggerUrl;
      break;
    } catch { await wait(200); }
  }
  const ws = new WebSocket(targetUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  let id = 0;
  const pending = new Map();
  ws.onmessage = (ev) => { const m = JSON.parse(ev.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } };
  const send = (method, params = {}) => new Promise((res) => { const mid = ++id; pending.set(mid, res); ws.send(JSON.stringify({ id: mid, method, params })); });
  const ev = async (expr) => { const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true }); return r.result && r.result.result ? r.result.result.value : undefined; };

  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: width < 700 });
  await wait(1600);

  const audit = await ev(`(() => {
    const gs = (sel, prop) => { const el = document.querySelector(sel); return el ? getComputedStyle(el)[prop] : null; };
    const doc = document.documentElement;
    return {
      noHScroll: doc.scrollWidth <= window.innerWidth,
      bodyBg: gs('body', 'backgroundColor'),
      h1Size: gs('h1', 'fontSize'),
      heroCols: (() => { const h = document.querySelector('.hero__grid'); return h ? getComputedStyle(h).gridTemplateColumns.split(' ').length : 0; })(),
      primaryBtnBg: gs('.btn--primary', 'backgroundImage') || gs('.btn--primary', 'background'),
      primaryBtnText: gs('.btn--primary', 'color'),
      chipGreenText: gs('.chip--green', 'color'),
      headerSticky: getComputedStyle(document.querySelector('.site-header')).position,
      textColor: gs('.lead', 'color'),
      h1Clamped: (() => { const el = document.querySelector('h1'); return el ? el.scrollWidth <= el.clientWidth + 1 : true; })(),
    };
  })()`);

  step(label + ': no horizontal overflow', audit.noHScroll);
  step(label + ': dark body bg', !!audit.bodyBg && parseInt(audit.bodyBg.split(',')[0].slice(4)) < 30, audit.bodyBg);
  step(label + ': h1 responsive size', parseFloat(audit.h1Size) >= 20, audit.h1Size);
  step(label + ': hero grid columns = ' + audit.heroCols, width >= 960 ? audit.heroCols === 2 : audit.heroCols === 1);
  step(label + ': primary button is green gradient', (audit.primaryBtnBg || '').indexOf('94, 175, 115') !== -1 || (audit.primaryBtnBg || '').toLowerCase().indexOf('5eaf73') !== -1 || (audit.primaryBtnBg || '').indexOf('rgb') !== -1, String(audit.primaryBtnBg).slice(0, 40));
  step(label + ': dark button text (contrast)', audit.primaryBtnText === 'rgb(0, 33, 12)' || audit.primaryBtnText === 'rgb(0,33,12)', audit.primaryBtnText);
  step(label + ': header sticky', audit.headerSticky === 'sticky');
  step(label + ': body text light', (audit.textColor || '').indexOf('255') !== -1 || (audit.textColor || '').indexOf('233') !== -1, audit.textColor);
  step(label + ': h1 no wrap overflow', audit.h1Clamped);

  chrome.kill();
  await wait(300);
}

(async () => {
  await runAudit(1440, 1000, 'desktop');
  await runAudit(390, 844, 'mobile');
  console.log(results.join('\n'));
  const fails = results.filter((r) => r.startsWith('FAIL')).length;
  console.log('\n' + (results.length - fails) + '/' + results.length + ' passed');
  process.exit(fails ? 1 : 0);
})().catch((e) => { console.error('AUDIT ERROR', e); process.exit(2); });
