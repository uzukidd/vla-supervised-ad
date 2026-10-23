'use strict';
// Native Node >=22 CDP test; reuse the existing generator and browser.
// node generator_filter_browser.cjs http://127.0.0.1:8765 [screenshot.png] [http://127.0.0.1:9339]
// FILTER_RUN=1 also submits one frame with SOR -> DBSCAN; default is preview only.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');

async function main() {
  if (!process.argv[2]) throw Error('Usage: node generator_filter_browser.cjs URL [screenshot.png] [CDP endpoint]');
  if (typeof WebSocket !== 'function') throw Error('Node >=22 with native WebSocket is required');
  const url = new URL(process.argv[2]);
  assert.ok(['http:', 'https:'].includes(url.protocol));
  assert.ok(['localhost', '127.0.0.1', '[::1]'].includes(url.hostname));
  const screenshot = path.resolve(process.argv[3] || path.join(os.tmpdir(), 'generator-filter-browser.png'));
  const endpoint = (process.argv[4] || 'http://127.0.0.1:9339').replace(/\/$/, '');
  const response = await fetch(endpoint + '/json/new?about:blank', {method: 'PUT'});
  if (!response.ok) throw Error('Cannot open CDP target: HTTP ' + response.status);
  const target = await response.json();
  const ws = new WebSocket(target.webSocketDebuggerUrl), pending = new Map(), errors = [];
  let sequence = 0;
  await new Promise((resolve, reject) => {
    ws.addEventListener('open', resolve, {once: true});
    ws.addEventListener('error', reject, {once: true});
  });
  ws.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.id) {
      const entry = pending.get(message.id);
      if (!entry) return;
      pending.delete(message.id); clearTimeout(entry.timer);
      message.error ? entry.reject(Error(JSON.stringify(message.error))) : entry.resolve(message.result);
    } else if (message.method === 'Runtime.exceptionThrown') errors.push(JSON.stringify(message.params.exceptionDetails));
  });
  const send = (method, params = {}, timeout = 140000) => new Promise((resolve, reject) => {
    const id = ++sequence;
    const timer = setTimeout(() => { pending.delete(id); reject(Error('CDP timed out: ' + method)); }, timeout);
    pending.set(id, {resolve, reject, timer});
    ws.send(JSON.stringify({id, method, params}));
  });
  const evaluate = async expression => {
    const result = await send('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
    if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  const wait = (expression, timeout = 20000) => evaluate(`new Promise((resolve,reject)=>{const end=performance.now()+${timeout};function tick(){if(${expression})return resolve(true);if(performance.now()>end)return reject(Error('Wait timed out: '+${JSON.stringify(expression)}));setTimeout(tick,100);}tick();})`);
  const input = (id, value) => evaluate(`(()=>{const el=document.getElementById(${JSON.stringify(id)});el.value=${JSON.stringify(String(value))};el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  const click = id => evaluate(`document.getElementById(${JSON.stringify(id)}).click()`);
  const report = {algorithms: []};
  try {
    await send('Page.enable'); await send('Runtime.enable');
    await send('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1400, deviceScaleFactor: 1, mobile: false});
    await send('Page.addScriptToEvaluateOnNewDocument', {source: `window.__filterRequests=[];const originalFetch=window.fetch;window.fetch=async function(resource,options){const response=await originalFetch.apply(this,arguments);if(options?.method==='POST')window.__filterRequests.push({url:String(resource),body:JSON.parse(options.body),hasToken:!!new Headers(options.headers).get('X-Generator-Token'),status:response.status,result:await response.clone().json()});return response;};`});
    await send('Page.navigate', {url: url.href});
    await wait('window.__generatorUI?.getState().ready || (document.getElementById("bootError") && !document.getElementById("bootError").hidden)');
    assert.equal(await evaluate('document.getElementById("bootError").hidden'), true,
      await evaluate('document.getElementById("bootErrorText").textContent'));
    const boot = await evaluate('fetch("/api/config").then(r=>r.json())');
    const defaults = {algorithm: 'none', sor_k: 16, sor_std: 2, dbscan_eps: 1, dbscan_min_samples: 5, cluster_policy: 'significant', min_cluster_size: 10, cluster_ratio: .1};
    assert.deepEqual(boot.defaults.point_filter, defaults);
    assert.deepEqual(await evaluate('window.__generatorUI.getConfig().point_filter'), defaults);
    assert.deepEqual(await evaluate('Array.from(document.getElementById("pointFilter").options,o=>o.value)'), ['none', 'sor', 'dbscan', 'sor_dbscan']);
    const scene = boot.scenes.find(s => s.available !== false && /0103/.test(s.name)) || boot.scenes.find(s => s.available !== false);
    assert.ok(scene, 'An available scene is required');
    await click('clearScenes'); await input('sceneSearch', scene.name); await click('selectVisible');
    await input('samples', 1); await input('startIndex', 0);
    const values = {sorK: 24, sorStd: 1.5, dbscanEps: .7, dbscanMinSamples: 7, clusterPolicy: 'largest', minClusterSize: 12, clusterRatio: .25};
    const fields = {sor_k: 'sorK', sor_std: 'sorStd', dbscan_eps: 'dbscanEps', dbscan_min_samples: 'dbscanMinSamples', cluster_policy: 'clusterPolicy', min_cluster_size: 'minClusterSize', cluster_ratio: 'clusterRatio'};
    for (const algorithm of ['none', 'sor', 'dbscan', 'sor_dbscan']) {
      await input('pointFilter', algorithm);
      const usesSor = ['sor', 'sor_dbscan'].includes(algorithm), usesDbscan = ['dbscan', 'sor_dbscan'].includes(algorithm);
      assert.deepEqual(await evaluate(`['sorParams','dbscanParams'].map(id=>!document.getElementById(id).hidden)`), [usesSor, usesDbscan]);
      assert.equal(await evaluate(`Array.from(document.querySelectorAll('#sorParams input')).every(el=>el.disabled===${!usesSor})`), true);
      assert.equal(await evaluate(`Array.from(document.querySelectorAll('#dbscanParams input,#dbscanParams select')).every(el=>el.disabled===${!usesDbscan})`), true);
      for (const [id, value] of Object.entries(values)) await input(id, value);
      assert.equal(await evaluate('document.getElementById("previewButton").disabled'), false);
      const count = await evaluate('window.__filterRequests.length');
      await click('previewButton');
      await wait(`window.__filterRequests.length>${count} && !document.getElementById('previewButton').disabled`);
      const request = await evaluate('window.__filterRequests.at(-1)');
      assert.equal(request.url, '/api/preview'); assert.equal(request.status, 200); assert.equal(request.hasToken, true);
      const expected = {algorithm, ...Object.fromEntries(Object.entries(fields).map(([key, id]) => [key, values[id]]))};
      assert.deepEqual(request.body.point_filter, expected);
      assert.deepEqual(request.result.config.point_filter, expected);
      assert.ok(request.result.command.includes('--point-filter ' + algorithm));
      for (const [key, id] of Object.entries(fields)) assert.ok(request.result.command.includes('--' + key.replaceAll('_', '-') + ' ' + values[id]), key);
      assert.equal(await evaluate('document.getElementById("commandPreview").textContent'), request.result.command);
      report.algorithms.push({algorithm, point_filter: expected, command: request.result.command});
    }
    for (const [id, invalid] of [['sorK', 1.5], ['sorStd', ''], ['dbscanEps', 0], ['dbscanMinSamples', 2.5], ['minClusterSize', 0], ['clusterRatio', 0], ['clusterRatio', 1.1]]) {
      await input(id, invalid);
      assert.equal(await evaluate('document.getElementById("previewButton").disabled'), true, id + ' must block preview');
      assert.equal(await evaluate('document.getElementById("generateButton").disabled'), true, id + ' must block generation');
      await input(id, values[id]);
    }
    if (process.env.FILTER_RUN === '1') {
      await evaluate('document.getElementById("allSceneFrames").checked=false');
      await input('samples', 1);
      await click('generateButton');
      await wait('window.__filterRequests.some(r=>r.url==="/api/start")');
      const request = await evaluate('window.__filterRequests.findLast(r=>r.url==="/api/start")');
      assert.equal(request.status, 201, JSON.stringify(request.result));
      assert.equal(request.body.samples, 1); assert.equal(request.body.point_filter.algorithm, 'sor_dbscan');
      const runId = request.result.id;
      await wait(`window.__generatorUI.getState().runs.some(r=>r.id===${JSON.stringify(runId)}&&['completed','failed','stopped'].includes(r.status))`, 120000);
      const run = await evaluate(`window.__generatorUI.getState().runs.find(r=>r.id===${JSON.stringify(runId)})`);
      assert.equal(run.status, 'completed', await evaluate('document.getElementById("logPanel").textContent'));
      assert.deepEqual(run.config.point_filter, request.body.point_filter);
      report.run = {id: run.id, status: run.status, point_filter: run.config.point_filter, result_url: run.result_url};
    } else assert.equal(await evaluate('window.__filterRequests.some(r=>r.url==="/api/start")'), false, 'Default must remain preview only');
    await click('previewButton');
    await wait('!document.getElementById("commandPreview").hidden');
    assert.deepEqual(errors, []);
    await evaluate('document.getElementById("pointFilterSection").scrollIntoView({block:"center"})');
    const image = await send('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true});
    fs.mkdirSync(path.dirname(screenshot), {recursive: true});
    fs.writeFileSync(screenshot, Buffer.from(image.data, 'base64'));
    console.log(JSON.stringify({ok: true, url: url.href, screenshot, ...report}, null, 2));
  } finally {
    await send('Page.close', {}, 3000).catch(() => {});
    ws.close();
    for (const entry of pending.values()) clearTimeout(entry.timer);
  }
}
main().catch(error => { console.error(error.stack || String(error)); process.exitCode = 1; });
