/* CDP smoke test: drives the prototype through the full user flow in headless Chrome. */
const { spawn } = require('child_process');
const path = require('path');
const { pathToFileURL } = require('url');

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const URL = pathToFileURL(path.join(__dirname, 'index.html')).href;
const PORT = 9333;
const results = [];
const step = (name, ok) => { results.push((ok ? 'PASS ' : 'FAIL ') + name); };

async function main() {
  const chrome = spawn(CHROME, [
    '--headless=new', '--disable-gpu', '--no-sandbox', '--no-first-run',
    '--remote-debugging-port=' + PORT,
    '--user-data-dir=/tmp/ps-chrome-test',
    'about:blank',
  ]);
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  let ws, id = 0, pending = new Map();

  let targetUrl;
  for (let i = 0; i < 50; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${PORT}/json/new?` + encodeURIComponent(URL), { method: 'PUT' });
      const t = await res.json();
      targetUrl = t.webSocketDebuggerUrl;
      break;
    } catch { await wait(200); }
  }
  if (!targetUrl) throw new Error('no devtools target');

  const wsHandlers = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
  };

  ws = new WebSocket(targetUrl);
  ws.onmessage = wsHandlers;
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });

  const send = (method, params = {}) => new Promise((resolve) => {
    const mid = ++id;
    pending.set(mid, resolve);
    ws.send(JSON.stringify({ id: mid, method, params }));
  });
  const evalJS = async (expr) => {
    const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
    return r.result && r.result.result ? r.result.result.value : undefined;
  };

  await send('Runtime.enable');
  await wait(1500); // let page + script load

  step('welcome active on load',
    await evalJS(`document.querySelector('#view-welcome').classList.contains('is-active')`));
  step('start button present', await evalJS(`!!document.querySelector('#start-button')`));

  await evalJS(`document.querySelector('#start-button').click()`);
  step('start routes to select',
    await evalJS(`document.querySelector('#view-select').classList.contains('is-active')`));

  await evalJS(`document.querySelector('.track[data-track="csv"]').click()`);
  step('csv track revealed', await evalJS(`!document.querySelector('#track-csv').hidden`));
  step('text track hidden', await evalJS(`document.querySelector('#track-text').hidden`));
  step('track focus moved', await evalJS(`document.activeElement && document.activeElement.id === 'csv-title'`));

  await evalJS(`document.querySelector('[data-demo-upload="csv"]').click()`);
  step('csv loaded panel shown', await evalJS(`!document.querySelector('[data-csv-loaded]').hidden`));

  await evalJS(`document.querySelector('[data-scan="csv"]').click()`);
  step('csv progress shown', await evalJS(`!document.querySelector('[data-progress="csv"]').hidden`));
  await wait(1400);
  step('csv results shown', await evalJS(`!document.querySelector('[data-results="csv"]').hidden`));
  step('results heading focused', await evalJS(`document.activeElement && document.activeElement.id === 'csv-results-title'`));

  await evalJS(`document.querySelector('[data-reset="csv"]').click()`);
  step('csv reset hides results', await evalJS(`document.querySelector('[data-results="csv"]').hidden`));

  // ---- Text track ----
  await evalJS(`document.querySelector('.track[data-track="text"]').click()`);
  await evalJS(`document.querySelector('[data-demo-upload="text"]').click()`);
  step('text loaded panel shown', await evalJS(`!document.querySelector('[data-text-loaded]').hidden`));
  await evalJS(`document.querySelectorAll('input[name="text-strategy"]').forEach(b => b.checked = false)`);
  await evalJS(`document.querySelector('[data-scan="text"]').click()`);
  step('text warning when no strategy', await evalJS(`!document.querySelector('[data-text-warning]').hidden`));
  await evalJS(`document.querySelector('input[name="text-strategy"][value="mask"]').checked = true`);
  await evalJS(`document.querySelector('[data-scan="text"]').click()`);
  await wait(1400);
  step('text results shown', await evalJS(`!document.querySelector('[data-results="text"]').hidden`));

  // ---- Tabs ----
  await evalJS(`document.querySelector('#tab-pseudo').click()`);
  step('pseudo tab active',
    await evalJS(`!document.querySelector('#panel-pseudo').hidden && document.querySelector('#panel-mask').hidden`));
  await evalJS(`document.querySelector('#tab-mask').click()`);
  step('mask tab re-activated',
    await evalJS(`!document.querySelector('#panel-mask').hidden && document.querySelector('#panel-pseudo').hidden`));

  // ---- Image track ----
  await evalJS(`document.querySelector('.track[data-track="image"]').click()`);
  await evalJS(`document.querySelector('[data-demo-upload="image"]').click()`);
  step('image loaded panel shown', await evalJS(`!document.querySelector('[data-image-loaded]').hidden`));
  await evalJS(`document.querySelectorAll('[data-img-toggle]').forEach(b => b.checked = false)`);
  await evalJS(`document.querySelector('[data-scan="image"]').click()`);
  step('image warning when no method', await evalJS(`!document.querySelector('[data-image-warning]').hidden`));
  await evalJS(`document.querySelector('[data-img-toggle="faces"]').checked = true`);
  await evalJS(`document.querySelector('[data-scan="image"]').click()`);
  await wait(1400);
  step('image results shown', await evalJS(`!document.querySelector('[data-results="image"]').hidden`));

  // ---- Slider sync ----
  const sv = await evalJS(`(() => { const s = document.querySelector('#csv-confidence');
      s.value = 0.4; s.dispatchEvent(new Event('input', { bubbles: true }));
      return document.querySelector('[data-conf-out="csv"]').textContent; })()`);
  step('slider readout sync (' + sv + ')', sv === '0.4');

  // ---- Downloads feedback ----
  await evalJS(`document.querySelector('[data-results="csv"] .download').click()`);
  step('download feedback shown',
    await evalJS(`document.querySelector('[data-results="csv"] .download strong').textContent.indexOf('Saved') !== -1`));

  // ---- About ----
  await evalJS(`document.querySelector('.nav__link[data-goto="about"]').click()`);
  step('about view shows',
    await evalJS(`document.querySelector('#view-about').classList.contains('is-active')`));

  console.log(results.join('\n'));
  const fails = results.filter((r) => r.startsWith('FAIL')).length;
  console.log('\nALL: ' + (results.length - fails) + '/' + results.length + ' passed');
  chrome.kill();
  process.exit(fails ? 1 : 0);
}
main().catch((e) => { console.error('TEST ERROR', e); process.exit(2); });
