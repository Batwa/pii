/* ============================================================
   Privacy Sandbox (PII Identifier) — design interactions
   Routing, track selection, and REAL scans for all three tracks.
   Files are sent to the local Python server (server.py) which runs
   the actual detectors (Presidio + spaCy + regex + OpenCV/Tesseract).
   Nothing leaves this machine.
   ============================================================ */
(function () {
  'use strict';

  var VIEWS = ['welcome', 'select', 'about'];
  var TRACKS = ['csv', 'text', 'image'];

  /* ---------------- View routing ---------------- */
  function showView(name) {
    VIEWS.forEach(function (v) {
      var el = document.getElementById('view-' + v);
      if (el) { el.classList.toggle('is-active', v === name); }
    });

    document.querySelectorAll('.nav__link[data-goto]').forEach(function (link) {
      if (link.getAttribute('data-goto') === name) {
        link.setAttribute('aria-current', 'page');
      } else {
        link.removeAttribute('aria-current');
      }
    });

    window.scrollTo({ top: 0, behavior: 'auto' });
    if (name === 'select') { setStep(1); }
  }

  /* ---------------- Stepper ----------------
     0 welcome · 1 choose file type · 2 upload · 3 scan · 4 results */
  function setStep(index) {
    var stepper = document.querySelector('[data-stepper]');
    if (!stepper) { return; }
    Array.prototype.forEach.call(stepper.children, function (li, i) {
      li.classList.toggle('is-done', i < index);
      if (i === index) {
        li.setAttribute('aria-current', 'step');
      } else {
        li.removeAttribute('aria-current');
      }
    });
  }

  /* ---------------- Track selection (Page 2 → Page 3) ---------------- */
  function selectTrack(track) {
    document.querySelectorAll('.track').forEach(function (btn) {
      btn.setAttribute('aria-pressed', String(btn.getAttribute('data-track') === track));
    });

    TRACKS.forEach(function (t) {
      var section = document.getElementById('track-' + t);
      if (section) { section.hidden = (t !== track); }
    });

    setStep(2);

    var heading = document.getElementById(track + '-title');
    if (heading) {
      heading.focus();
      heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

    /* ---------------- Simulated scan ---------------- */
  function runScan(track) {
    // Text needs >= 1 strategy; images need >= 1 detection method.
    if (track === 'text') {
      var strategies = document.querySelectorAll('input[name="text-strategy"]:checked');
      var textWarning = document.querySelector('[data-text-warning]');
      if (strategies.length === 0) {
        if (textWarning) { textWarning.hidden = false; }
        return;
      }
      if (textWarning) { textWarning.hidden = true; }
    }

    if (track === 'image') {
      var methods = document.querySelectorAll('[data-img-toggle]:checked');
      var imgWarning = document.querySelector('[data-image-warning]');
      if (methods.length === 0) {
        if (imgWarning) { imgWarning.hidden = false; }
        return;
      }
      if (imgWarning) { imgWarning.hidden = true; }
    }

    setStep(3);

    var progress = document.querySelector('[data-progress="' + track + '"]');
    var bar = progress ? progress.querySelector('.progress__bar') : null;
    var results = document.querySelector('[data-results="' + track + '"]');

    if (progress) { progress.hidden = false; }
    if (bar) {
      bar.style.width = '0%';
      requestAnimationFrame(function () { bar.style.width = '45%'; });
      setTimeout(function () { bar.style.width = '100%'; }, 500);
    }

    // All three tracks run real Python processing on server.py:
    //   image -> /api/redact-image  (OpenCV + Tesseract + Presidio)
    //   csv   -> /api/process-csv   (PIIDetector)
    //   text  -> /api/process-text  (TextPIIDetector)
    if (track === 'image') {
      scanImage(track, results);
      return;
    }
    if (track === 'csv') {
      scanCsv(track, results);
      return;
    }
    if (track === 'text') {
      scanText(track, results);
      return;
    }

    setTimeout(function () {
      if (progress) { progress.hidden = true; }
      if (results) {
        results.hidden = false;
        setStep(4);
        var title = document.getElementById(track + '-results-title');
        if (title) {
          title.focus();
          title.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }
        }, 1100);
  }

  /* ---------------- Real image redaction (local server) ----------------
     Scans the picked image via POST /api/redact-image on server.py, which
     runs the real OpenCV + Tesseract + Presidio pipeline. Falls back to the
     SVG placeholder when opened from file:// or if the server is down. -------- */
  function toDataURL(file) {
    return new Promise(function (resolve, reject) {
      var reader = new FileReader();
      reader.onload = function () { resolve(reader.result); };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  function showError(track, message) {
    var box = document.querySelector('[data-upload-error="' + track + '"]');
    if (!box) { return; }
    var p = box.querySelector('p');
    if (p) { p.textContent = message || ''; }
    box.hidden = !message;
  }

  function showImageError(message) {
    showError('image', message);
  }

  function revealResults(track) {
    var progress = document.querySelector('[data-progress="' + track + '"]');
    if (progress) { progress.hidden = true; }
    var results = document.querySelector('[data-results="' + track + '"]');
    if (results) {
      results.hidden = false;
      setStep(4);
      var title = document.getElementById(track + '-results-title');
      if (title) {
        title.focus();
        title.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  }

  function revealImageResults(results) {
    revealResults('image');
  }

  function renderImageResult(data) {
    var panel = document.querySelector('[data-results="image"]');
    if (!panel) { return; }

    var heading = panel.querySelector('h3');
    if (heading) { heading.textContent = '\ud83d\udcf8 ' + (data.filename || 'image'); }

    var metrics = panel.querySelectorAll('.metric__value');
    if (metrics.length >= 1) { metrics[0].textContent = data.faces; }
    if (metrics.length >= 2) { metrics[1].textContent = data.textRegions; }
    if (metrics.length >= 3) { metrics[2].textContent = data.piiRegions; }

    var hint = panel.querySelector('.panel__hint');
    if (hint) {
      var parts = [];
      if (data.faces) { parts.push(data.faces + ' face(s) blurred'); }
      if (data.textRegions) { parts.push(data.textRegions + ' text region(s)'); }
      if (data.piiRegions) { parts.push(data.piiRegions + ' PII text region(s) redacted'); }
      hint.textContent = (parts.length ? parts.join(' \u00b7 ') : 'No PII detected') + ' \u00b7 processed locally';
    }

    var frames = panel.querySelectorAll('.shot__frame');
    if (frames.length >= 2 && data.redacted) {
      frames[1].innerHTML = '<img src="' + data.redacted + '" alt="Redacted image">';
    }

    if (data.redacted) {
      var base = baseName(data.filename || 'image');
      DOWNLOADS['img-redacted'] = { url: data.redacted, name: base + '_redacted.png' };
      var dlBtns = panel.querySelectorAll('.download');
      if (dlBtns[0]) { dlBtns[0].setAttribute('data-dl', 'img-redacted'); }
      if (dlBtns[1]) { dlBtns[1].setAttribute('data-dl', 'img-redacted'); }
    }
  }

  function scanImage(track, results) {
    var info = currentData.image;
    var file = info && info.file ? info.file : null;

    function finish() { revealImageResults(results); }

    if (!file) {
      showImageError('No image selected. Go back and choose an image to scan.');
      finish();
      return;
    }

    // file:// cannot reach the local API via fetch -> graceful fallback.
    if (window.location.protocol === 'file:') {
      showImageError('For real redaction, serve this page with the local server: ' +
        'run `python3 server.py` and open http://localhost:8000. ' +
        'Showing the illustrative preview instead.');
      finish();
      return;
    }

    var facesToggle = document.querySelector('[data-img-toggle="faces"]');
    var ocrToggle = document.querySelector('[data-img-toggle="text"]');
    var faceStyle = (document.getElementById('face-style') || { value: '' }).value || 'blur';
    var textStyle = (document.getElementById('text-style') || { value: '' }).value || 'black';

    toDataURL(file).then(function (url) {
      return fetch('/api/redact-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: url,
          filename: file.name,
          faces: !facesToggle || facesToggle.checked !== false,
          ocr: !ocrToggle || ocrToggle.checked !== false,
          faceStyle: faceStyle,
          textStyle: textStyle
        })
      });
    }).then(function (resp) { return resp.json(); }).then(function (data) {
      if (data && data.ok) {
        renderImageResult(data);
        showImageError(null);
      } else {
        showImageError(data && data.error ? data.error : 'Image processing failed.');
      }
      finish();
    }).catch(function (err) {
      showImageError('Could not reach the local redaction server: ' + err.message);
      finish();
    });
  }

  function scanCsv(track, results) {
    var file = (currentFiles.csv || [])[0];
    function finish() { revealResults(track); }
    if (!file) {
      showError('csv', 'No CSV selected. Go back and choose a file to scan.');
      finish();
      return;
    }
    if (window.location.protocol === 'file:') {
      showError('csv', 'For real processing, serve this page with the local server: ' +
        'run `python3 server.py` and open http://localhost:8000.');
      finish();
      return;
    }
    var method = (document.querySelector('input[name="csv-method"]:checked') || {}).value || 'smart';
    var threshold = Number((document.getElementById('csv-confidence') || { value: 0.8 }).value || 0.8);

    toDataURL(file).then(function (url) {
      return fetch('/api/process-csv', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: url, filename: file.name, method: method, threshold: threshold })
      });
    }).then(function (resp) { return resp.json(); }).then(function (data) {
      if (data && data.ok) {
        fillCsvResults(data, file);
        showError('csv', null);
      } else {
        showError('csv', data && data.error ? data.error : 'CSV processing failed.');
      }
      finish();
    }).catch(function (err) {
      showError('csv', 'Could not reach the local processing server: ' + err.message);
      finish();
    });
  }

  function fillCsvResults(data, file) {
    var m = data.metrics || {};
    setText('#csv-found', number(m.total_pii));
    setText('#csv-affected', m.affected);
    setText('#csv-clean', m.clean);
    setText('#csv-changed', number(m.changed));
    setText('#csv-aff-count', m.affected);
    var methodLabel = data.method === 'smart' ? 'Smart Redaction' :
      data.method === 'mask' ? 'Complete Masking' : 'Partial Redaction';
    setText('#csv-results-hint', file.name + ' · ' + methodLabel + ' · threshold ' + Number(data.threshold).toFixed(1));

    var fb = document.getElementById('csv-findings-body');
    if (data.findings && data.findings.length) {
      fb.innerHTML = data.findings.map(function (f) {
        return '<tr><td>' + esc(f.column) + '</td><td>' + esc(f.types.join(', ')) +
          '</td><td>' + number(f.items) + '</td><td class="redacted">' + esc(f.action) + '</td></tr>';
      }).join('');
    } else {
      fb.innerHTML = '<tr><td colspan="4">No PII detected — file was already clean</td></tr>';
    }

    var heads = '<tr>' + (data.headers || []).map(function (h) { return '<th>' + esc(h) + '</th>'; }).join('') + '</tr>';
    setInner('csv-comp-before-head', heads);
    setInner('csv-comp-after-head', heads);
    document.getElementById('csv-comp-before-body').innerHTML = (data.before || []).map(function (r) {
      return '<tr>' + r.map(function (c) { return '<td>' + esc(c) + '</td>'; }).join('') + '</tr>';
    }).join('');
    document.getElementById('csv-comp-after-body').innerHTML = (data.after || []).map(function (r) {
      return '<tr>' + r.map(function (c) { return '<td class="redacted">' + esc(c) + '</td>'; }).join('') + '</tr>';
    }).join('');

    var base = baseName(file.name);
    DOWNLOADS['csv-clean'] = { url: textDataUrl(data.clean_csv, 'text/csv'), name: 'clean_' + base + '.csv' };
    DOWNLOADS['csv-report'] = { url: textDataUrl(data.report, 'text/plain'), name: 'privacy_report_' + base + '.txt' };
    setText('#csv-clean-name', 'clean_' + base + '.csv');
    setText('#csv-report-name', 'privacy_report_' + base + '.txt');
    var dlBtns = document.querySelectorAll('[data-results="csv"] .download');
    if (dlBtns[0]) { dlBtns[0].setAttribute('data-dl', 'csv-clean'); }
    if (dlBtns[1]) { dlBtns[1].setAttribute('data-dl', 'csv-report'); }
  }

  function scanText(track, results) {
    var docs = currentData.text && currentData.text.docs;
    function finish() { revealResults(track); }
    if (!docs || !docs.length) {
      showError('text', 'No documents selected. Go back and choose a text file to scan.');
      finish();
      return;
    }
    if (window.location.protocol === 'file:') {
      showError('text', 'For real processing, serve this page with the local server: ' +
        'run `python3 server.py` and open http://localhost:8000.');
      finish();
      return;
    }
    var strategies = Array.prototype.slice.call(document.querySelectorAll('input[name="text-strategy"]:checked'))
      .map(function (b) { return b.value; });
    if (!strategies.length) { strategies = ['mask']; }
    var threshold = Number((document.getElementById('text-confidence') || { value: 0.8 }).value || 0.8);

    var jobs = docs.map(function (d) {
      return toDataURL(d.file).then(function (url) {
        return fetch('/api/process-text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: url, filename: d.file.name, strategies: strategies, threshold: threshold })
        }).then(function (resp) { return resp.json(); });
      });
    });

    Promise.all(jobs).then(function (responses) {
      var bad = responses.filter(function (r) { return !r || !r.ok; });
      if (bad.length) {
        showError('text', (bad[0] && bad[0].error) ? bad[0].error : 'Text processing failed.');
      } else {
        showError('text', null);
        renderTextBlocks(docs, responses, strategies, threshold);
      }
      finish();
    }).catch(function (err) {
      showError('text', 'Could not reach the local processing server: ' + err.message);
      finish();
    });
  }

  function renderTextBlocks(docs, analyses, strategies, threshold) {
    // Keep the real response data available for the one-click ZIP download.
    currentData.text.analyses = analyses;
    var totalPii = 0, totalSize = 0, typeCounts = {}, sourceCounts = {};
    analyses.forEach(function (a) {
      totalPii += (a.total_pii_found || 0);
      totalSize += (a.file_size || 0);
      Object.keys(a.pii_by_type || {}).forEach(function (t) { typeCounts[t] = (typeCounts[t] || 0) + a.pii_by_type[t]; });
      Object.keys(a.pii_by_source || {}).forEach(function (s) { sourceCounts[s] = (sourceCounts[s] || 0) + a.pii_by_source[s]; });
    });

    setText('#text-block-heading', (docs.length === 1 ? docs[0].file.name : '📄 ' + docs.length + ' document(s) scanned'));
    setText('#text-size', number(totalSize) + ' chars');
    setText('#text-total', number(totalPii));
    setText('#text-types', String(Object.keys(typeCounts).length));
    setText('#text-versions-count', String(strategies.length));
    setText('#text-results-hint', docs.length + ' file(s) scanned · strategies: ' + strategies.join(', ') + ' · threshold ' + Number(threshold).toFixed(1));

    setInner('text-types-chips', Object.keys(typeCounts).map(function (t) {
      return '<li class="chip">' + esc(t) + ' · ' + number(typeCounts[t]) + '</li>';
    }).join('') || '<li class="chip">No PII found</li>');
    setInner('text-sources-chips', Object.keys(sourceCounts).map(function (s) {
      return '<li class="chip chip--green">' + esc(s) + ' · ' + number(sourceCounts[s]) + '</li>';
    }).join('') || '<li class="chip">no engine matched</li>');

    setInner('text-versions', docs.map(function (d, i) { return textBlock(d, analyses[i], i); }).join(''));
    REVIEWS = {};
    docs.forEach(function (d, i) { initReview(i, analyses[i], strategies); });
    updateZipState();
  }

  async function downloadAllTextVersions(button) {
    var docs = currentData.text && currentData.text.docs;
    var analyses = currentData.text && currentData.text.analyses;
    if (!docs || !analyses || !docs.length) { return; }

    var files = [];
    docs.forEach(function (doc, index) {
      var versions = (analyses[index] && analyses[index].versions) || {};
      Object.keys(versions).forEach(function (strategy) {
        files.push({
          name: baseName(doc.file.name) + '_' + strategy + '.txt',
          content: textDataUrl(versions[strategy].content || '', 'text/plain')
        });
      });
    });
    if (!files.length) { return; }

    var label = button.querySelector('strong');
    var original = label && label.textContent;
    button.disabled = true;
    if (label) { label.textContent = 'Creating ZIP…'; }
    try {
      var response = await fetch('/api/create-zip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ files: files, filename: 'privacy-sandbox-results.zip' })
      });
      var data = await response.json();
      if (!response.ok || !data.ok) { throw new Error(data.error || 'Could not create ZIP archive.'); }
      var a = document.createElement('a');
      a.href = data.archive;
      a.download = data.filename || 'privacy-sandbox-results.zip';
      document.body.appendChild(a);
      a.click();
      a.remove();
      if (label) { label.textContent = '✓ ZIP saved to your downloads folder'; }
    } catch (err) {
      if (label) { label.textContent = err.message || 'Could not create ZIP archive'; }
    } finally {
      button.disabled = false;
      setTimeout(function () { if (label) { label.textContent = original; } }, 2200);
    }
  }

  function textBlock(doc, a, idx) {
    var types = Object.keys(a.pii_by_type || {}).map(function (t) { return { t: t, c: a.pii_by_type[t] }; })
      .sort(function (x, y) { return y.c - x.c; });
    var sources = Object.keys(a.pii_by_source || {}).map(function (s) { return { t: s, c: a.pii_by_source[s] }; })
      .sort(function (x, y) { return y.c - x.c; });
    var versions = Object.keys(a.versions || {});
    var id = 'tf' + idx;

    var byType = types.map(function (x) { return '<li class="chip">' + esc(x.t) + ' · ' + x.c + '</li>'; }).join('') || '<li class="chip">No PII found</li>';
    var bySource = sources.map(function (x) { return '<li class="chip chip--green">' + esc(x.t) + ' · ' + x.c + '</li>'; }).join('') || '<li class="chip">no engine matched</li>';

    return '<div class="file-block mt-3">' +
      '<h3>📄 ' + esc(doc.file.name) + '</h3>' +
      '<div class="metrics">' +
      '<div class="metric"><span class="metric__label">File size</span><span class="metric__value">' + number(a.file_size || 0) + ' chars</span></div>' +
      '<div class="metric metric--warn"><span class="metric__label">Total PII found</span><span class="metric__value">' + number(a.total_pii_found || 0) + '</span></div>' +
      '<div class="metric metric--accent"><span class="metric__label">PII types</span><span class="metric__value">' + types.length + '</span></div>' +
      '<div class="metric"><span class="metric__label">Redacted versions</span><span class="metric__value">' + versions.length + '</span></div>' +
      '</div>' +
      '<div class="grid grid-2 mt-2">' +
      '<div class="card card--flat"><h3>By type</h3><ul class="chips">' + byType + '</ul></div>' +
      '<div class="card card--flat"><h3>By detection source</h3><ul class="chips">' + bySource + '</ul></div>' +
      '</div>' +
      (typeof a.original === 'string' ?
        '<h3 class="mt-3">Review before you download</h3>' +
        '<p class="small muted">Highlighted text will be redacted. Click a highlight to keep that text visible. ' +
        'Select any other text to redact it. Each place is changed on its own.</p>' +
        '<pre class="preview review" data-review="' + id + '" aria-label="Original text of ' + esc(doc.file.name) + ' with the redactions marked"></pre>' +
        '<div class="review__bar">' +
        '<p class="small review__status" data-review-status="' + id + '" role="status"></p>' +
        '<button class="btn btn--ghost review__reset" type="button" data-review-action="reset" data-review-id="' + id + '">Reset to detected</button>' +
        '</div>' +
        '<div class="review__prompt" data-review-prompt="' + id + '" role="status" hidden></div>' : '') +
      '<h3 class="mt-3">Redacted versions</h3>' +
      '<div data-versions="' + id + '">' + versionPanels(doc, a, idx, versions[0]) + '</div>' +
      '</div>';
  }

  // The strategy tabs and their panels for one document. Called again each
  // time the review changes, with the tab that was open.
  function versionPanels(doc, a, idx, selected) {
    var versions = Object.keys(a.versions || {});
    var id = 'tf' + idx;
    if (versions.indexOf(selected) === -1) { selected = versions[0]; }

    var tabs = versions.map(function (s) {
      return '<button class="tab" role="tab" aria-selected="' + (s === selected ? 'true' : 'false') +
        '" data-tab="' + id + '-' + s + '" aria-controls="' + id + '-panel-' + s + '" id="' + id + '-tab-' + s + '">' + esc(s) + '</button>';
    }).join('');

    var panels = versions.map(function (s) {
      var v = a.versions[s] || {};
      var reds = v.redactions || [];
      var kv = reds.slice(0, 30).map(function (r) {
        return '<li><span class="kv__type">' + esc(String(r.entity_type).toLowerCase()) +
          '</span><span class="kv__from">' + esc(r.original) +
          '</span><span class="kv__arrow">→</span><span class="kv__to">' + esc(r.replacement) +
          '</span><span class="small muted">' + (r.source === 'manual' ? 'added by you' :
            'confidence ' + Number(r.confidence).toFixed(2) + ' · ' + esc(r.source)) + '</span></li>';
      }).join('') || '<li class="small muted">No redactions for this strategy</li>';
      if (reds.length > 30) { kv += '<li class="small muted">… and ' + (reds.length - 30) + ' more</li>'; }
      var fname = baseName(doc.file.name) + '_' + s + '.txt';
      DOWNLOADS[id + '-dl-' + s] = { url: textDataUrl(v.content, 'text/plain'), name: fname };
      return '<div role="tabpanel" id="' + id + '-panel-' + s + '" aria-labelledby="' + id + '-tab-' + s +
        '" data-tabpanel="' + id + '-' + s + '"' + (s === selected ? '' : ' hidden') + '>' +
        '<pre class="preview">' + (v.marked || esc(v.content || '')) + '</pre>' +
        '<p class="small muted mt-1">Redactions applied: ' + reds.length + '</p>' +
        '<ul class="kv">' + kv + '</ul>' +
        '<div class="downloads mt-2"><button class="download" type="button" data-dl="' + id + '-dl-' + s + '">' +
        '<span class="download__icon" aria-hidden="true">📥</span><span><strong>Download ' + esc(s) + ' version</strong><span>' + fname + '</span></span></button></div>' +
        '</div>';
    }).join('');

    return '<div class="tabs" role="tablist" aria-label="Redaction strategies for ' + esc(doc.file.name) + '">' + tabs + '</div>' + panels;
  }

  /* ---------------- Review step (text results) ----------------
     The scan returns the original text and every detection. The user removes
     wrong redactions and adds missed ones here; each change is sent to
     /api/apply-redactions, and the previews and downloads are rebuilt from
     the corrected list. Positions are UTF-16 offsets in the page and code
     points on the server, so they are converted at the boundary. */
  var REVIEWS = {};
  var reviewJustSelected = false;
  var WORD_CHAR = /[\p{L}\p{N}_]/u;

  function offsetMaps(text) {
    var cp2u = [], u2cp = new Array(text.length + 1), cp = 0;
    for (var i = 0; i < text.length; cp++) {
      var code = text.charCodeAt(i);
      var pair = code >= 0xD800 && code <= 0xDBFF && i + 1 < text.length;
      cp2u.push(i);
      u2cp[i] = cp;
      if (pair) { u2cp[i + 1] = cp; }
      i += pair ? 2 : 1;
    }
    cp2u.push(text.length);
    u2cp[text.length] = cp;
    return { cp2u: cp2u, u2cp: u2cp };
  }

  function detectedItems(state) {
    var items = [];
    (state.detections || []).forEach(function (d) {
      var start = state.maps.cp2u[d.start], end = state.maps.cp2u[d.end];
      if (start === undefined || end === undefined || end <= start) { return; }
      items.push({ start: start, end: end, entity_type: d.entity_type, score: d.score,
                   source: d.source || 'unknown', on: true, manual: false });
    });
    items.sort(function (a, b) { return a.start - b.start || b.end - a.end; });
    // Detections never overlap after the server's dedup; merge if they do.
    var merged = [];
    items.forEach(function (it) {
      var last = merged[merged.length - 1];
      if (last && it.start < last.end) { last.end = Math.max(last.end, it.end); } else { merged.push(it); }
    });
    return merged;
  }

  function initReview(idx, analysis, strategies) {
    var id = 'tf' + idx;
    if (!analysis || typeof analysis.original !== 'string') { return; }
    var state = { id: id, idx: idx, text: analysis.original, strategies: strategies.slice(),
                  detections: analysis.detection_details || [], seq: 0, timer: null,
                  pending: false, failed: false, prompt: null };
    state.maps = offsetMaps(state.text);
    state.items = detectedItems(state);
    REVIEWS[id] = state;
    renderReview(state);
  }

  function reviewEl(state, name) { return document.querySelector('[data-review' + name + '="' + state.id + '"]'); }

  function renderReview(state) {
    var pre = reviewEl(state, '');
    if (!pre) { return; }
    while (pre.firstChild) { pre.removeChild(pre.firstChild); }
    var last = 0;
    // Text nodes, not innerHTML: the document's own characters must stay
    // exactly as they are, so that selection offsets match the text.
    state.items.forEach(function (it, i) {
      if (it.start > last) { pre.appendChild(document.createTextNode(state.text.slice(last, it.start))); }
      var mark = document.createElement('mark');
      mark.className = 'rv' + (it.on ? '' : ' is-off') + (it.manual ? ' is-manual' : '');
      mark.setAttribute('role', 'button');
      mark.setAttribute('tabindex', '0');
      mark.setAttribute('aria-pressed', String(it.on));
      mark.setAttribute('data-i', String(i));
      mark.title = it.on
        ? (it.manual ? 'Added by you' : String(it.entity_type).toLowerCase().replace(/_/g, ' ')) + ' — click to keep this visible'
        : 'Kept visible — click to redact it again';
      mark.appendChild(document.createTextNode(state.text.slice(it.start, it.end)));
      pre.appendChild(mark);
      last = it.end;
    });
    if (last < state.text.length) { pre.appendChild(document.createTextNode(state.text.slice(last))); }
    renderReviewStatus(state);
    renderReviewPrompt(state);
  }

  function renderReviewStatus(state) {
    var el = reviewEl(state, '-status');
    if (!el) { return; }
    var on = 0, off = 0, added = 0;
    state.items.forEach(function (it) { if (!it.on) { off++; } else { on++; if (it.manual) { added++; } } });
    var parts = [on + ' will be redacted'];
    if (off) { parts.push(off + ' kept visible by you'); }
    if (added) { parts.push(added + ' added by you'); }
    var tail = state.failed ? 'Could not update the downloads: ' + state.failed
      : state.pending ? 'Updating the downloads…' : 'The downloads match this review.';
    el.textContent = parts.join(' · ') + '. ' + tail;
    el.classList.toggle('is-error', Boolean(state.failed));
    if (state.failed) {
      var retry = document.createElement('button');
      retry.type = 'button'; retry.className = 'review__link';
      retry.setAttribute('data-review-action', 'retry'); retry.setAttribute('data-review-id', state.id);
      retry.textContent = 'Try again';
      el.appendChild(document.createTextNode(' ')); el.appendChild(retry);
    }
  }

  function renderReviewPrompt(state) {
    var el = reviewEl(state, '-prompt');
    if (!el) { return; }
    while (el.firstChild) { el.removeChild(el.firstChild); }
    var p = state.prompt;
    el.hidden = !p;
    if (!p) { return; }
    var n = p.targets.length, places = n === 1 ? '1 other place' : n + ' other places';
    var shown = p.word.length > 40 ? p.word.slice(0, 40) + '…' : p.word;
    var message = p.kind === 'add'
      ? '“' + shown + '” also appears in ' + places + ' that will stay visible.'
      : '“' + shown + '” is still redacted in ' + places + '.';
    el.appendChild(document.createTextNode(message + ' '));
    [[p.kind === 'add' ? 'Redact ' + (n === 1 ? 'that one' : 'all ' + n) + ' too'
                       : 'Keep ' + (n === 1 ? 'that one' : 'all ' + n) + ' visible too', 'all', 'btn btn--ghost'],
     ['No, only this one', 'dismiss', 'review__link']].forEach(function (def) {
      var b = document.createElement('button');
      b.type = 'button'; b.className = def[2]; b.textContent = def[0];
      b.setAttribute('data-review-action', def[1]); b.setAttribute('data-review-id', state.id);
      el.appendChild(b);
    });
  }

  // Whole-word occurrences of a string, ignoring case: "Ann" never matches
  // inside "Annual" or "Joanna".
  function wordOccurrences(text, word) {
    var found = [], hay = text.toLowerCase(), needle = word.toLowerCase();
    if (!needle || hay.length !== text.length) { hay = text; needle = word; }
    for (var at = hay.indexOf(needle); at !== -1; at = hay.indexOf(needle, at + 1)) {
      var end = at + needle.length;
      var before = at > 0 && WORD_CHAR.test(text.charAt(at - 1)) && WORD_CHAR.test(text.charAt(at));
      var after = end < text.length && WORD_CHAR.test(text.charAt(end)) && WORD_CHAR.test(text.charAt(end - 1));
      if (!before && !after) { found.push([at, end]); }
    }
    return found;
  }

  function insertManual(state, start, end) {
    // An overlapped redaction is replaced by one that covers both.
    state.items = state.items.filter(function (it) {
      if (it.end <= start || it.start >= end) { return true; }
      if (it.on) { start = Math.min(start, it.start); end = Math.max(end, it.end); }
      return false;
    }).filter(function (it) { return it.end <= start || it.start >= end; });
    state.items.push({ start: start, end: end, entity_type: 'REDACTED', score: 1, source: 'manual', on: true, manual: true });
    state.items.sort(function (a, b) { return a.start - b.start; });
  }

  function reviewAddSelection(state, start, end) {
    var text = state.text;
    while (start < end && /\s/.test(text.charAt(start))) { start++; }
    while (end > start && /\s/.test(text.charAt(end - 1))) { end--; }
    if (end <= start) { return; }
    // Whole words only: a selection that starts or ends inside a word grows
    // to the word's edge.
    while (start > 0 && WORD_CHAR.test(text.charAt(start - 1)) && WORD_CHAR.test(text.charAt(start))) { start--; }
    while (end < text.length && WORD_CHAR.test(text.charAt(end)) && WORD_CHAR.test(text.charAt(end - 1))) { end++; }
    var covered = state.items.some(function (it) { return it.on && it.start <= start && end <= it.end; });
    state.prompt = null;
    if (!covered) {
      insertManual(state, start, end);
      var word = text.slice(start, end);
      var others = word.indexOf('\n') !== -1 ? [] : wordOccurrences(text, word).filter(function (r) {
        return !state.items.some(function (it) { return it.on && it.start < r[1] && r[0] < it.end; });
      });
      if (others.length) { state.prompt = { kind: 'add', word: word, targets: others }; }
      scheduleApply(state);
    }
    renderReview(state);
  }

  function reviewToggle(state, index) {
    var it = state.items[index];
    if (!it) { return; }
    state.prompt = null;
    var word = state.text.slice(it.start, it.end);
    if (it.on && it.manual) {
      state.items.splice(index, 1);        // what the user added is simply taken back
    } else {
      it.on = !it.on;
    }
    if (!it.on || it.manual) {
      var same = state.items.filter(function (other) {
        return other !== it && other.on && state.text.slice(other.start, other.end) === word;
      });
      if (same.length) { state.prompt = { kind: 'remove', word: word, targets: same }; }
    }
    renderReview(state);
    scheduleApply(state);
  }

  function reviewAction(button) {
    var state = REVIEWS[button.getAttribute('data-review-id')];
    if (!state) { return; }
    var action = button.getAttribute('data-review-action');
    if (action === 'retry') { scheduleApply(state); renderReviewStatus(state); return; }
    if (action === 'reset') {
      state.items = detectedItems(state);
    } else if (action === 'all' && state.prompt) {
      if (state.prompt.kind === 'add') {
        state.prompt.targets.forEach(function (r) { insertManual(state, r[0], r[1]); });
      } else {
        state.prompt.targets.forEach(function (it) {
          var at = state.items.indexOf(it);
          if (at === -1) { return; }
          if (it.manual) { state.items.splice(at, 1); } else { it.on = false; }
        });
      }
    }
    var changed = action !== 'dismiss';
    state.prompt = null;
    renderReview(state);
    if (changed) { scheduleApply(state); }
  }

  function setBlockDownloads(state, disabled) {
    var box = document.querySelector('[data-versions="' + state.id + '"]');
    if (box) {
      Array.prototype.forEach.call(box.querySelectorAll('.download'), function (b) { b.disabled = disabled; });
    }
    updateZipState();
  }

  // The ZIP holds every document, so it waits for every review.
  function updateZipState() {
    var zip = document.querySelector('[data-download-all-text]');
    if (!zip) { return; }
    zip.disabled = Object.keys(REVIEWS).some(function (id) { return REVIEWS[id].pending || REVIEWS[id].failed; });
  }

  // While a change is on its way to the server the downloads of this
  // document are switched off, so an older version cannot be saved by
  // mistake. They stay off if the update fails.
  function scheduleApply(state) {
    state.pending = true;
    state.failed = false;
    setBlockDownloads(state, true);
    renderReviewStatus(state);
    clearTimeout(state.timer);
    state.timer = setTimeout(function () { applyReview(state); }, 250);
  }

  function applyReview(state) {
    var seq = ++state.seq;
    var redactions = state.items.filter(function (it) { return it.on; }).map(function (it) {
      return { start: state.maps.u2cp[it.start], end: state.maps.u2cp[it.end],
               entity_type: it.entity_type, score: it.score, source: it.source };
    });
    fetch('/api/apply-redactions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: state.text, strategies: state.strategies, redactions: redactions })
    }).then(function (resp) { return resp.json(); }).then(function (data) {
      if (seq !== state.seq) { return; }               // a newer change is on its way
      if (!data || !data.ok) { throw new Error((data && data.error) || 'the server refused the change'); }
      var docs = currentData.text.docs, analyses = currentData.text.analyses;
      analyses[state.idx].versions = data.versions;
      var box = document.querySelector('[data-versions="' + state.id + '"]');
      if (box) {
        var open = box.querySelector('[role="tab"][aria-selected="true"]');
        var selected = open ? open.getAttribute('data-tab').slice(state.id.length + 1) : null;
        box.innerHTML = versionPanels(docs[state.idx], analyses[state.idx], state.idx, selected);
      }
      state.pending = false;
      setBlockDownloads(state, false);
      renderReviewStatus(state);
    }).catch(function (err) {
      if (seq !== state.seq) { return; }
      state.pending = false;
      state.failed = err.message || 'no answer from the local server';
      updateZipState();
      renderReviewStatus(state);
    });
  }

  // Offset of a selection boundary in the review text: the length of the
  // text between the start of the review and that boundary.
  function reviewOffset(pre, node, offset) {
    var range = document.createRange();
    range.selectNodeContents(pre);
    range.setEnd(node, offset);
    return range.toString().length;
  }

  document.addEventListener('mouseup', function () {
    var sel = window.getSelection && window.getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) { return; }
    var range = sel.getRangeAt(0);
    var anchor = range.commonAncestorContainer;
    var pre = (anchor.nodeType === 1 ? anchor : anchor.parentNode).closest('pre.review');
    if (!pre) { return; }
    var state = REVIEWS[pre.getAttribute('data-review')];
    if (!state) { return; }
    var start = reviewOffset(pre, range.startContainer, range.startOffset);
    var end = reviewOffset(pre, range.endContainer, range.endOffset);
    sel.removeAllRanges();
    reviewJustSelected = true;                      // the click that follows is not a toggle
    setTimeout(function () { reviewJustSelected = false; }, 0);
    reviewAddSelection(state, start, end);
  });

  function reviewMarkActivated(mark) {
    var pre = mark.closest('pre.review');
    var state = pre && REVIEWS[pre.getAttribute('data-review')];
    if (!state) { return; }
    var index = Number(mark.getAttribute('data-i'));
    reviewToggle(state, index);
    var again = pre.querySelector('mark.rv[data-i="' + index + '"]');
    if (again) { again.focus({ preventScroll: true }); }
  }


  function resetTrack(track) {
    var loaded = document.querySelector('[data-' + track + '-loaded]');
    var results = document.querySelector('[data-results="' + track + '"]');
    if (loaded) { loaded.hidden = true; }
    if (results) { results.hidden = true; }
    setStep(2);
    var heading = document.getElementById(track + '-title');
    if (heading) {
      heading.focus();
      heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  /* ---------------- Tabs (text results) ---------------- */
  function activateTab(tab) {
    var list = tab.closest('.tabs');
    if (!list) { return; }
    var tabs = Array.prototype.slice.call(list.querySelectorAll('[role="tab"]'));

    tabs.forEach(function (t) {
      var selected = t === tab;
      t.setAttribute('aria-selected', String(selected));
      var panel = document.querySelector('[data-tabpanel="' + t.getAttribute('data-tab') + '"]');
      if (panel) { panel.hidden = !selected; }
    });
    tab.focus();
  }

  /* ---------------- Wire it all up ---------------- */
  document.addEventListener('click', function (event) {
    var goto = event.target.closest('[data-goto]');
    if (goto) { showView(goto.getAttribute('data-goto')); return; }

    var track = event.target.closest('.track');
    if (track) { selectTrack(track.getAttribute('data-track')); return; }

        var upload = event.target.closest('[data-upload]');
    if (upload) {
      var upTrack = upload.getAttribute('data-upload');
      var upInput = document.getElementById(upTrack + '-input');
      if (upInput) { upInput.click(); }
      return;
    }

    var scan = event.target.closest('[data-scan]');
    if (scan) { runScan(scan.getAttribute('data-scan')); return; }

    var reset = event.target.closest('[data-reset]');
    if (reset) { resetTrack(reset.getAttribute('data-reset')); return; }

    var tab = event.target.closest('[role="tab"]');
    if (tab) { activateTab(tab); return; }

    var reviewMark = event.target.closest('mark.rv');
    if (reviewMark) {
      if (!reviewJustSelected) { reviewMarkActivated(reviewMark); }
      return;
    }
    var reviewButton = event.target.closest('[data-review-action]');
    if (reviewButton) { reviewAction(reviewButton); return; }

    var downloadAllText = event.target.closest('[data-download-all-text]');
    if (downloadAllText) { downloadAllTextVersions(downloadAllText); return; }

    var download = event.target.closest('.download');
    if (download) {
      var dlKey = download.getAttribute('data-dl');
      if (dlKey && DOWNLOADS[dlKey]) {
        var dl = DOWNLOADS[dlKey];
        try {
          var a = document.createElement('a');
          a.href = dl.url;
          a.download = dl.name || 'download';
          document.body.appendChild(a);
          a.click();
          a.remove();
        } catch (err) { /* ignore download errors */ }
      }
      var label = download.querySelector('strong');
      if (label && label.dataset.original === undefined) {
        label.dataset.original = label.textContent;
        label.textContent = '✓ Saved to your downloads folder';
        setTimeout(function () {
          label.textContent = label.dataset.original;
          delete label.dataset.original;
        }, 1800);
      }
    }
  });

  /* Arrow-key navigation for tabs; Enter or Space on a review highlight */
  document.addEventListener('keydown', function (event) {
    var mark = event.target.closest && event.target.closest('mark.rv');
    if (mark && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      reviewMarkActivated(mark);
      return;
    }
    var tab = event.target.closest && event.target.closest('[role="tab"]');
    if (!tab) { return; }
    if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') { return; }

    var tabs = Array.prototype.slice.call(tab.closest('.tabs').querySelectorAll('[role="tab"]'));
    var i = tabs.indexOf(tab);
    var next = event.key === 'ArrowRight' ? (i + 1) % tabs.length : (i - 1 + tabs.length) % tabs.length;
    event.preventDefault();
    activateTab(tabs[next]);
  });

  /* Confidence sliders keep every readout in sync */
  document.addEventListener('input', function (event) {
    var slider = event.target.closest('[data-conf]');
    if (slider) {
      var scope = slider.getAttribute('data-conf');
      document.querySelectorAll('[data-conf-out="' + scope + '"]').forEach(function (out) {
        out.textContent = Number(slider.value).toFixed(1);
      });
    }
  });

  /* Live validation for the two multi-select groups */
  document.addEventListener('change', function (event) {
    if (event.target.name === 'text-strategy') {
      var warning = document.querySelector('[data-text-warning]');
      if (warning) {
        warning.hidden = document.querySelectorAll('input[name="text-strategy"]:checked').length > 0;
      }
    }
    if (event.target.hasAttribute && event.target.hasAttribute('data-img-toggle')) {
      var imgWarning = document.querySelector('[data-image-warning]');
      if (imgWarning) {
        imgWarning.hidden = document.querySelectorAll('[data-img-toggle]:checked').length > 0;
      }
    }
  });

  /* Drag & drop affordance on the dropzones (visual only) */
  document.querySelectorAll('.dropzone').forEach(function (zone) {
    ['dragenter', 'dragover'].forEach(function (type) {
      zone.addEventListener(type, function (e) { e.preventDefault(); zone.classList.add('is-drag'); });
    });
        ['dragleave', 'drop'].forEach(function (type) {
      zone.addEventListener(type, function (e) {
        e.preventDefault();
        zone.classList.remove('is-drag');
        if (type === 'drop' && e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
          var dropTrack = zone.getAttribute('data-upload');
          if (dropTrack) { handleFiles(dropTrack, e.dataTransfer.files); }
        }
      });
    });
  });

    /* ----- local file handling + client-side PII scan ----- */
  var currentFiles = { csv: [], text: [], image: [] };
  var currentData  = { csv: null, text: null, image: null };
  var ACCEPT = { csv: '.csv', text: '.txt,.json,.md', image: '.jpg,.jpeg,.png,.bmp' };

  var DOWNLOADS = {};
  function baseName(name) { return String(name).replace(/\.[^.]+$/, ''); }
  function textDataUrl(str, mime) {
    return 'data:' + (mime || 'text/plain') + ';charset=utf-8,' + encodeURIComponent(String(str));
  }
  function setInner(sel, html) { var el = document.getElementById(sel); if (el) { el.innerHTML = html; } }
  function readFilesAsText(files) {
    return Promise.all(Array.prototype.map.call(files, function (f) {
      return new Promise(function (resolve, reject) {
        var r = new FileReader();
        r.onload = function () { resolve({ file: f, text: r.result }); };
        r.onerror = reject;
        r.readAsText(f);
      });
    }));
  }

  function fileAccepted(track, file) {
    var ext = '.' + (file.name.split('.').pop() || '').toLowerCase();
    return ACCEPT[track].split(',').map(function (s) { return s.trim(); }).indexOf(ext) !== -1;
  }
  function humanSize(bytes) {
    if (bytes < 1024) { return bytes + ' B'; }
    var u = ['KB', 'MB', 'GB'], i = -1, b = bytes;
    do { b /= 1024; i++; } while (b >= 1024 && i < u.length - 1);
    return b.toFixed(1) + ' ' + u[i];
  }
  function setText(sel, value) { var el = document.querySelector(sel); if (el) { el.textContent = value; } }
  function number(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ','); }
  function esc(str) {
    return String(str == null ? '' : str).replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  var PII_PATTERNS = [
    { type: 'EMAIL_ADDRESS', label: 'email', mask: '****@****.***', score: 1.0 },
    { type: 'US_SSN',        label: 'ssn',   mask: '***-**-****',   score: 1.0 },
    { type: 'PHONE_NUMBER',  label: 'phone', mask: '***-***-****',  score: 0.95 },
    { type: 'CREDIT_CARD',   label: 'card',  mask: '****-****-****-****', score: 0.9 }
  ];
  var PII_REGEX = {
        EMAIL_ADDRESS: /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi,
    US_SSN:        /\b\d{3}-\d{2}-\d{4}\b/g,
    PHONE_NUMBER:  /\+?\(?\d{1,3}\)?[ .-]?\d{3}[ .-]?\d{4}\b/g,
    CREDIT_CARD:   /\b(?:\d[ -]?){13,16}\b/g
  };

  function detectPII(text) {
    var items = [];
    PII_PATTERNS.forEach(function (p) {
      var src = PII_REGEX[p.type]; var re = new RegExp(src.source, src.flags); var m;
      while ((m = re.exec(text)) !== null) {
        items.push({ type: p.type, label: p.label, value: m[0], start: m.index, end: m.index + m[0].length, score: p.score });
        if (m[0].length === 0) { re.lastIndex++; }
      }
    });
    items.sort(function (a, b) { return a.start - b.start || b.score - a.score; });
    var out = [];
    items.forEach(function (it) { if (out.length && it.start < out[out.length - 1].end) { return; } out.push(it); });
    return out;
  }
  function wrapMarked(text, replacer) {
    var items = detectPII(text).sort(function (a, b) { return a.start - b.start; });
    var html = ''; var last = 0;
    items.forEach(function (it) { html += esc(text.slice(last, it.start)) + '<mark>' + esc(replacer(it)) + '</mark>'; last = it.end; });
    html += esc(text.slice(last));
    return { html: html, count: items.length, items: items };
  }
  function maskOf(type) {
    var p = PII_PATTERNS.filter(function (x) { return x.type === type; })[0] || { mask: '[REDACTED]' };
    return p.mask;
  }
  function maskRow(value) { return wrapMarked(value || '', function (it) { return maskOf(it.type); }).html; }
  function parseCsvLine(line) {
    var out = [], cur = '', q = false, i, c;
    for (i = 0; i < line.length; i++) { c = line[i]; if (c === '"') { q = !q; } else if (c === ',' && !q) { out.push(cur); cur = ''; } else { cur += c; } }
    out.push(cur); return out.map(function (s) { return s.trim(); });
  }
  function parseCSV(text) {
    var rows = []; text.split(/\r?\n/).forEach(function (line) { if (line.trim() === '') { return; } rows.push(parseCsvLine(line)); });
    return rows;
  }

    function buildFileList(track, list) {
    var ul = document.getElementById(track + '-filelist');
    if (!ul) { return; }
    ul.innerHTML = list.map(function (f) {
      var icon = track === 'image' ? '🖼️' : '📄';
      return '<li><span aria-hidden="true">' + icon + '</span><span class="file__name">' + esc(f.name) +
             '</span><span class="file__meta">' + humanSize(f.size) + '</span></li>';
    }).join('');
  }

  function processCsv(file, text) {
    var rows = parseCSV(text);
    var header = rows[0] || [];
    var body = rows.slice(1);
    var method = (document.querySelector('input[name="csv-method"]:checked') || {}).value || 'smart';
    setText('#csv-rows', number(body.length));
    setText('#csv-cols', String(header.length));
    setText('#csv-size', humanSize(file.size));
    setText('#csv-filename', file.name);
    setText('#csv-caption', file.name + ' — original');

    document.getElementById('csv-preview-head').innerHTML = '<tr>' + header.map(function (h) { return '<th>' + esc(h) + '</th>'; }).join('') + '</tr>';
    document.getElementById('csv-preview-body').innerHTML = body.slice(0, 4).map(function (r) {
      return '<tr>' + header.map(function (_, i) { return '<td class="raw">' + esc(r[i] || '') + '</td>'; }).join('') + '</tr>';
    }).join('');

    var findings = {}; header.forEach(function (h) { findings[h] = {}; });
    body.forEach(function (r) {
      header.forEach(function (h, c) {
        detectPII(r[c] || '').forEach(function (it) { findings[h][it.type] = (findings[h][it.type] || 0) + 1; });
      });
    });
    var total = 0, affected = 0;
    header.forEach(function (h) {
      var cnt = 0; Object.keys(findings[h]).forEach(function (t) { cnt += findings[h][t]; });
      if (cnt) { affected++; total += cnt; }
    });
    setText('#csv-found', number(total));
    setText('#csv-affected', affected);
    setText('#csv-clean', header.length - affected);
    setText('#csv-changed', number(body.filter(function (r) {
      return header.some(function (_, c) { return detectPII(r[c] || '').length; });
    }).length));
    setText('#csv-aff-count', affected);
    setText('#csv-results-hint', file.name + ' · Smart Redaction · threshold 0.8');

    var action = method === 'smart' ? 'Pseudonymized' : method === 'mask' ? 'Masked ****' : 'Partially masked';
    var frows = [];
    header.forEach(function (h) {
      var types = Object.keys(findings[h]); if (!types.length) { return; }
      var cnt = 0; types.forEach(function (x) { cnt += findings[h][x]; });
      frows.push('<tr><td>' + esc(h) + '</td><td>' + esc(types.join(', ')) + '</td><td>' + cnt + '</td><td class="redacted">' + action + '</td></tr>');
    });
    var fb = document.getElementById('csv-findings-body');
    fb.innerHTML = frows.length ? frows.join('') : '<tr><td colspan="4">No PII detected — file was already clean</td></tr>';

    document.getElementById('csv-comp-before-head').innerHTML = '<tr>' + header.map(function (h) { return '<th>' + esc(h) + '</th>'; }).join('') + '</tr>';
    document.getElementById('csv-comp-after-head').innerHTML = '<tr>' + header.map(function (h) { return '<th>' + esc(h) + '</th>'; }).join('') + '</tr>';
    document.getElementById('csv-comp-before-body').innerHTML = body.slice(0, 4).map(function (r) {
      return '<tr>' + header.map(function (_, i) { return '<td>' + esc(r[i] || '') + '</td>'; }).join('') + '</tr>';
    }).join('');
    document.getElementById('csv-comp-after-body').innerHTML = body.slice(0, 4).map(function (r) {
      return '<tr>' + header.map(function (_, i) { return '<td class="redacted">' + maskRow(r[i] || '') + '</td>'; }).join('') + '</tr>';
    }).join('');

    var base = file.name.replace(/\.[^.]+$/, '');
    setText('#csv-clean-name', 'clean_' + base + '.csv · ' + humanSize(file.size));
    setText('#csv-report-name', 'privacy_report_' + base + '.txt');
    currentData.csv = { file: file, findings: findings, total: total };
  }

    function processText(list, text) {
    var file = list[0], base = file.name.replace(/\.[^.]+$/, '');
    setText('#text-block-heading', file.name);
    setText('#text-size', String(text.length));
    setText('#text-results-hint', list.length + ' files scanned · regex-based detection · threshold 0.8');
    var items = detectPII(text), types = {};
    items.forEach(function (it) { types[it.type] = (types[it.type] || 0) + 1; });
    setText('#text-total', number(items.length));
    setText('#text-types', String(Object.keys(types).length));
    setText('#text-versions-count', '2');
    var tc = document.getElementById('text-types-chips');
    tc.innerHTML = Object.keys(types).map(function (t) {
      return '<li class="chip">' + t + ' · ' + number(types[t]) + '</li>';
    }).join('') || '<li class="chip">No PII found</li>';
    document.getElementById('text-sources-chips').innerHTML =
      (items.length ? '<li class="chip chip--green">regex patterns · ' + number(items.length) + '</li>' : '<li class="chip chip--green">regex patterns — 0 found</li>') + '<li class="chip">Presidio (NLP) · full app</li>';

    var mk = wrapMarked(text, function (it) { return maskOf(it.type); });
    fillVersionPanel('panel-mask', base + '_mask.txt', mk.count, mk.html, dedupeItems(mk.items, 'mask', null));
    var seen = {}, n = 0;
    var pk = wrapMarked(text, function (it) {
      if (!seen[it.value]) { seen[it.value] = 'P-' + (++n).toString().padStart(4, '0'); }
      return seen[it.value];
    });
    fillVersionPanel('panel-pseudo', base + '_pseudonymize.txt', pk.count, pk.html, dedupeItems(pk.items, 'pseudonymize', seen));
    currentData.text = { file: file, items: items };
  }

  function dedupeItems(items, mode, map) {
    var seen = {}, rows = [];
    items.forEach(function (it) {
      if (seen[it.value]) { return; }
      seen[it.value] = true;
      var to = (map && map[it.value]) ? map[it.value] : maskOf(it.type);
      rows.push('<li><span class="kv__type">' + it.label + '</span><span class="kv__from">' + esc(it.value) +
        '</span><span class="kv__arrow">→</span><span class="kv__to">' + esc(to) + '</span>' +
        '<span class="small muted">confidence ' + it.score.toFixed(2) + '</span></li>');
    });
    return rows.join('');
  }

  function fillVersionPanel(id, filename, count, html, kv) {
    var pan = document.getElementById(id);
    if (!pan) { return; }
    if (pan.querySelector('.preview')) { pan.querySelector('.preview').innerHTML = html; }
    var note = pan.querySelector('.preview + p'); if (note) { note.textContent = 'Redactions applied: ' + count; }
    var kvEl = pan.querySelector('.kv'); if (kvEl) { kvEl.innerHTML = kv || ''; }
    var fn = pan.querySelector('.downloads strong + span'); if (fn) { fn.textContent = filename; }
  }

  function loadImagePreviews(list) {
    var frames = document.querySelectorAll('[data-results="image"] .shot__frame');
    if (!frames.length) { return; }
    var url = URL.createObjectURL(list[0]);
    frames[0].innerHTML = '<img src="' + url + '" alt="' + esc(list[0].name) + '">';
    currentData.image = { file: list[0], count: list.length };
  }

  function handleFiles(track, files) {
    var list = Array.prototype.slice.call(files);
    if (!list.length) { return; }
    var bad = list.filter(function (f) { return !fileAccepted(track, f); });
    var err = document.querySelector('[data-upload-error="' + track + '"]');
    if (err) { err.hidden = !!bad.length; }
    if (bad.length) { return; }
    currentFiles[track] = list;
    buildFileList(track, list);
    var loaded = document.querySelector('[data-' + track + '-loaded]');
    if (loaded) { loaded.hidden = false; }
    if (track === 'csv') {
      var fr = new FileReader();
      fr.onload = function () { processCsv(list[0], fr.result); };
      fr.readAsText(list[0]);
    } else if (track === 'text') {
      readFilesAsText(list).then(function (docs) {
        currentData.text = { docs: docs };
      }).catch(function () {});
    } else if (track === 'image') { loadImagePreviews(list); }
    setStep(2);
    if (loaded) { loaded.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  }

  ['csv', 'text', 'image'].forEach(function (t) {
    var inp = document.getElementById(t + '-input');
    if (inp) {
      inp.addEventListener('change', function (e) {
        if (e.target.files && e.target.files.length) { handleFiles(t, e.target.files); }
      });
    }
  });

  showView('welcome');
})();
