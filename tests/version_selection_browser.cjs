'use strict';
// Native Node browser regression test. Requires Chrome with --remote-debugging-port=9341.
// 原生 Node 浏览器回归测试；需先启动启用 9341 调试端口的 Chrome。
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { once } = require('node:events');

const html = fs.readFileSync(path.join(__dirname, '../tasks/nuscenes_sam3/web/generator.html'), 'utf8');
const mini = 'v1.0-mini', trainval = 'v1.0-trainval';
const scenes = version => [
  { name: 'shared-scene', description: version, frames: 20, available: true, split: version === mini ? 'mini_train' : 'trainval' },
  { name: 'unavailable-scene', description: 'Missing local data', frames: 10, available: false, reason: 'Missing camera sample / 缺少相机文件', split: version === mini ? 'mini_val' : 'trainval' },
];
const posts = [], sceneRequests = [], queuedResponses = [];
let legacy = false, bootVersion = mini;
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const json = (value, status = 200) => { res.writeHead(status, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(value)); };
  if (url.pathname === '/') { res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' }); res.end(html); return; }
  if (url.pathname === '/api/config') {
    json({ ...(legacy ? {} : { version: bootVersion, versions: [mini, trainval] }), token: 'test', classes: [{ name: 'road', display_name: 'Road', prompts: 'road', plane: true, color: '#123456' }], scenes: scenes(bootVersion), defaults: {}, runs: [{ id: 'previous-run', status: 'completed', scenes: ['shared-scene'], log_tail: 'Saved history' }] }); return;
  }
  if (url.pathname === '/api/scenes') {
    const version = url.searchParams.get('version'); sceneRequests.push(version);
    const behavior = queuedResponses.shift();
    const send = () => behavior?.error ? json({ error: 'Mock scene failure' }, 503) : json({ version: behavior?.version || version, scenes: scenes(version) });
    if (behavior?.hold) behavior.release = send; else send();
    return;
  }
  if (url.pathname === '/api/state') { json({ runs: [{ id: 'previous-run', status: 'completed', scenes: ['shared-scene'], log_tail: 'Saved history' }] }); return; }
  if (req.method === 'POST') {
    let text = ''; for await (const chunk of req) text += chunk;
    const body = JSON.parse(text); posts.push({ path: url.pathname, body });
    if (url.pathname === '/api/classes') { json({ classes_path: 'test/classes.json' }); return; }
    if (url.pathname === '/api/preview') { json({ command: 'generate ' + body.version, selected_scenes: body.scenes, estimated_frames: 1, total_images: 6 }); return; }
    if (url.pathname === '/api/start') { json({ id: 'new-run', status: 'completed', scenes: body.scenes }); return; }
  }
  if (url.pathname.startsWith('/api/runs/')) { json({ id: 'previous-run', status: 'completed', scenes: ['shared-scene'], log_tail: 'Saved history' }); return; }
  json({ error: 'Unknown route' }, 404);
});

async function main() {
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const origin = 'http://127.0.0.1:' + server.address().port;
  const cdpOrigin = process.env.CDP_URL || 'http://127.0.0.1:9341';
  let socket, target, seq = 0; const pending = new Map();
  try {
    const response = await fetch(cdpOrigin + '/json/new?' + encodeURIComponent('about:blank'), { method: 'PUT' });
    assert.ok(response.ok, 'Chrome CDP endpoint must accept new targets'); target = await response.json();
    socket = new WebSocket(target.webSocketDebuggerUrl); await once(socket, 'open');
    socket.addEventListener('message', event => {
      const message = JSON.parse(event.data); if (!message.id) return;
      const waiter = pending.get(message.id); if (!waiter) return; pending.delete(message.id);
      message.error ? waiter.reject(Error(message.error.message)) : waiter.resolve(message.result);
    });
    const send = (method, params = {}) => new Promise((resolve, reject) => { const id = ++seq; pending.set(id, { resolve, reject }); socket.send(JSON.stringify({ id, method, params })); });
    const evaluate = async expression => {
      const result = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
      if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails)); return result.result.value;
    };
    const waitFor = async (expression, label) => {
      const deadline = Date.now() + 10000;
      while (Date.now() < deadline) { if (await evaluate(expression)) return; await new Promise(resolve => setTimeout(resolve, 25)); }
      throw Error('Timed out: ' + label);
    };
    const change = (id, value) => evaluate(`(() => { const el=document.getElementById(${JSON.stringify(id)}); el.value=${JSON.stringify(value)}; el.dispatchEvent(new Event('change',{bubbles:true})); })()`);
    const click = id => evaluate(`document.getElementById(${JSON.stringify(id)}).click()`);
    const choose = () => evaluate(`document.querySelector('[data-scene="shared-scene"]').click()`);
    const loaded = version => waitFor(`document.getElementById('datasetVersion').value===${JSON.stringify(version)} && document.getElementById('sceneList').getAttribute('aria-busy')==='false' && document.querySelector('[data-scene="shared-scene"]')`, 'scenes loaded: ' + version);
    await send('Page.navigate', { url: origin });
    await waitFor(`document.getElementById('datasetVersion') && !document.getElementById('datasetVersion').disabled && !document.getElementById('addClass').disabled`, 'bootstrap');
    await loaded(mini);
    assert.deepEqual(await evaluate(`Array.from(document.getElementById('datasetVersion').options,o=>o.value)`), [mini, trainval]);
    assert.deepEqual(await evaluate(`Array.from(document.getElementById('splitFilter').options,o=>o.value)`), ['all', 'mini_train', 'mini_val']);
    assert.equal(await evaluate(`document.getElementById('availableOnly').checked`), true);
    await click('availableOnly');
    assert.equal(await evaluate(`document.querySelector('[data-scene="unavailable-scene"]').disabled`), true);
    assert.match(await evaluate(`document.querySelector('[data-scene="unavailable-scene"]').closest('label').textContent`), /Missing camera sample/);
    await choose();
    await evaluate(`const cls=document.querySelector('[data-field="display_name"]');cls.value='Edited road';cls.dispatchEvent(new Event('input',{bubbles:true}));document.getElementById('startIndex').value=7;document.getElementById('samples').value=3;`);
    const held = { hold: true }; queuedResponses.push(held);
    await change('datasetVersion', trainval);
    await waitFor(`document.getElementById('sceneList').getAttribute('aria-busy')==='true'`, 'loading indicator');
    assert.equal(await evaluate(`document.getElementById('previewButton').disabled && document.getElementById('generateButton').disabled`), true);
    assert.match(await evaluate(`document.getElementById('sceneLoadStatus').textContent`), /Loading scenes/);
    assert.equal(await evaluate(`document.getElementById('startIndex').value`), '0');
    assert.equal(await evaluate(`document.querySelectorAll('[data-scene]').length`), 0);
    const heldDeadline = Date.now() + 10000;
    while (!held.release && Date.now() < heldDeadline) await new Promise(resolve => setTimeout(resolve, 10));
    assert.ok(held.release, 'held version request arrived'); held.release();
    await loaded(trainval);
    assert.equal(await evaluate(`document.querySelector('[data-scene="shared-scene"]').checked`), false, 'same scene name must not carry selection');
    assert.equal(await evaluate(`document.querySelector('[data-field="display_name"]').value`), 'Edited road');
    assert.equal(await evaluate(`document.getElementById('samples').value`), '3');
    assert.match(await evaluate(`document.getElementById('runHistory').textContent`), /previous-run/);
    assert.deepEqual(await evaluate(`Array.from(document.getElementById('splitFilter').options,o=>o.value)`), ['all']);
    await choose(); await click('previewButton');
    await waitFor(`!document.getElementById('commandPreview').hidden`, 'preview');
    await click('generateButton');
    await waitFor(`document.getElementById('jobId').textContent==='new-run'`, 'generation');
    for (const endpoint of ['/api/preview', '/api/start']) {
      const payload = posts.find(post => post.path === endpoint)?.body;
      assert.equal(payload?.version, trainval); assert.deepEqual(payload.scenes, ['shared-scene']);
      assert.equal(payload.classes[0].display_name, 'Edited road');
    }
    // Old success must not replace the newest version; old errors must not block it.
    for (const error of [false, true]) {
      const stale = { hold: true, error }; queuedResponses.push(stale);
      await change('datasetVersion', mini);
      const deadline = Date.now() + 10000;
      while (!stale.release && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 10));
      assert.ok(stale.release, 'held stale request arrived');
      await change('datasetVersion', trainval); await loaded(trainval); await choose(); stale.release();
      await new Promise(resolve => setTimeout(resolve, 100));
      assert.equal(await evaluate(`document.querySelector('[data-scene="shared-scene"]').closest('label').textContent.includes('v1.0-trainval')`), true);
      assert.equal(await evaluate(`document.querySelector('[data-scene="shared-scene"]').checked && !document.getElementById('generateButton').disabled && document.getElementById('sceneLoadStatus').hidden`), true);
    }
    queuedResponses.push({ error: true }); await change('datasetVersion', mini);
    await waitFor(`!document.getElementById('retryScenes').hidden`, 'local load error');
    assert.equal(await evaluate(`document.getElementById('previewButton').disabled && document.getElementById('generateButton').disabled && document.querySelectorAll('[data-scene]').length===0`), true);
    assert.match(await evaluate(`document.getElementById('sceneLoadStatus').textContent`), /Mock scene failure/);
    await click('retryScenes'); await loaded(mini);
    assert.equal(await evaluate(`document.querySelector('[data-scene="shared-scene"]').checked`), false);
    queuedResponses.push({ version: trainval }); await change('datasetVersion', trainval); await loaded(trainval);
    queuedResponses.push({ version: trainval }); await change('datasetVersion', mini);
    await waitFor(`!document.getElementById('retryScenes').hidden`, 'version mismatch rejection');
    assert.equal(await evaluate(`document.getElementById('generateButton').disabled`), true);
    legacy = true; await send('Page.navigate', { url: origin + '/?legacy=1' });
    await waitFor(`location.search==='?legacy=1' && document.getElementById('datasetVersion') && !document.getElementById('datasetVersion').disabled && !document.getElementById('addClass').disabled`, 'legacy bootstrap');
    assert.deepEqual(await evaluate(`Array.from(document.getElementById('datasetVersion').options,o=>o.value)`), [mini]);
    legacy = false; bootVersion = trainval; await send('Page.navigate', { url: origin + '/?default=trainval' });
    await waitFor(`location.search==='?default=trainval' && document.getElementById('datasetVersion') && !document.getElementById('datasetVersion').disabled && !document.getElementById('addClass').disabled`, 'trainval bootstrap');
    await loaded(trainval);
    assert.deepEqual(await evaluate(`Array.from(document.getElementById('splitFilter').options,o=>o.value)`), ['all']);
    assert.ok(sceneRequests.includes(mini) && sceneRequests.includes(trainval));
    if(process.env.GENERATOR_URL){
      await send('Page.navigate',{url:process.env.GENERATOR_URL});
      await waitFor(`window.__generatorUI?.getState().ready && document.getElementById('datasetVersion').options.length>=2`,'live version bootstrap');
      await change('datasetVersion',trainval);
      await waitFor(`document.getElementById('sceneList').getAttribute('aria-busy')==='false' && document.querySelector('[data-scene="scene-0001"]')`,'live trainval scenes');
      assert.equal(await evaluate(`document.querySelector('[data-scene="scene-0001"]').disabled`),false);
      const liveCount=await evaluate(`document.querySelectorAll('[data-scene]').length`);
      assert.ok(liveCount>10);
      await evaluate(`document.querySelector('[data-scene="scene-0001"]').click()`);await click('previewButton');
      await waitFor(`!document.getElementById('commandPreview').hidden`,'live trainval preview');
      assert.match(await evaluate(`document.getElementById('commandPreview').textContent`),/--version v1.0-trainval/);
      await change('datasetVersion',mini);
      await waitFor(`document.getElementById('sceneList').getAttribute('aria-busy')==='false' && document.querySelectorAll('[data-scene]').length===10`,'live mini scenes');
      console.log(`PASS actual generator: ${liveCount} available trainval scenes, versioned preview, switch back to 10 mini scenes`);
    }
    console.log('PASS version selector: versions, availability, selection reset, class/settings/history preservation, version payloads, stale success/error, load failure/retry, mismatch rejection, legacy fallback');
  } finally {
    for (const behavior of queuedResponses) behavior.release?.();
    if (socket) socket.close();
    if (target) await fetch(cdpOrigin + '/json/close/' + target.id).catch(() => {});
    server.closeAllConnections(); await new Promise(resolve => server.close(resolve));
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
