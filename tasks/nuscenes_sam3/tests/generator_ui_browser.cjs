'use strict';
// Native Node >=22 CDP test. Reuse a running browser; never start a server/model.
// node generator_ui_browser.cjs http://127.0.0.1:8765 [screenshot.png] [http://127.0.0.1:9339]
// GENERATOR_RUN=1 additionally starts one real frame and waits up to 120 seconds.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');

async function main() {
  if (!process.argv[2]) throw Error('Usage: node generator_ui_browser.cjs URL [screenshot.png] [CDP endpoint]');
  if (typeof WebSocket !== 'function') throw Error('Node >=22 with native WebSocket is required.');
  const url = new URL(process.argv[2]);
  assert.ok(['http:', 'https:'].includes(url.protocol), 'Generator must be served over HTTP(S)');
  assert.ok(['localhost', '127.0.0.1', '[::1]'].includes(url.hostname), 'Use the local generator server');
  const screenshot = path.resolve(process.argv[3] || path.join(os.tmpdir(), 'generator-ui-browser.png'));
  const endpoint = (process.argv[4] || 'http://127.0.0.1:9339').replace(/\/$/, '');
  const targetResponse = await fetch(endpoint + '/json/new?about:blank', {method: 'PUT'});
  if (!targetResponse.ok) throw Error('Cannot open CDP target: HTTP ' + targetResponse.status);
  const target = await targetResponse.json();
  const ws = new WebSocket(target.webSocketDebuggerUrl), pending = new Map(), errors = [], requests = [];
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
    } else if (message.method === 'Runtime.exceptionThrown') {
      errors.push(JSON.stringify(message.params.exceptionDetails));
    } else if (message.method === 'Network.requestWillBeSent') {
      requests.push(message.params.request.url);
    }
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
  const click = id => evaluate(`document.getElementById(${JSON.stringify(id)}).click()`);
  const input = (id, value) => evaluate(`(()=>{const el=document.getElementById(${JSON.stringify(id)});el.value=${JSON.stringify(String(value))};el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  const check = (id, value) => evaluate(`(()=>{const el=document.getElementById(${JSON.stringify(id)});el.checked=${Boolean(value)};el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  const getConfig = () => evaluate('window.__generatorUI.getConfig()');
  const report = {};
  try {
    await send('Page.enable'); await send('Runtime.enable'); await send('Network.enable');
    await send('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1200, deviceScaleFactor: 1, mobile: false});
    await send('Page.addScriptToEvaluateOnNewDocument', {source: `window.__generatorRequests=[];const originalFetch=window.fetch;window.fetch=async function(resource,options){const url=String(resource);if(options?.method==='POST'){const headers=new Headers(options.headers);window.__generatorRequests.push({url,body:JSON.parse(options.body),hasToken:!!headers.get('X-Generator-Token')});}return originalFetch.apply(this,arguments);};`});
    await send('Page.navigate', {url: url.href});
    await wait('window.__generatorUI?.getState().ready || (document.getElementById("bootError") && !document.getElementById("bootError").hidden)');
    assert.equal(await evaluate('document.getElementById("bootError").hidden'), true,
      await evaluate('document.getElementById("bootErrorText").textContent'));
    const boot = await evaluate('fetch("/api/config").then(r=>r.json())');
    assert.ok(boot.classes.length >= 7, 'Expected at least seven editable categories');
    assert.equal(boot.scenes.length, 10, 'Mini dataset should expose ten scenes');
    assert.equal(await evaluate('document.querySelectorAll("#classRows .class-row").length'), boot.classes.length);
    assert.equal(await evaluate('document.querySelectorAll("#sceneList input[data-scene]").length'), boot.scenes.length);
    const crosswalkIndex = boot.classes.findIndex(c => c.name === 'crosswalk');
    assert.ok(crosswalkIndex >= 0, 'Crosswalk must be editable independently');
    assert.equal((await getConfig()).samples, boot.defaults?.samples ?? 1);
    assert.equal(await evaluate('document.getElementById("generateButton").disabled'), true);

    await click('addClass');
    assert.equal((await getConfig()).classes.length, boot.classes.length + 1);
    await evaluate(`(()=>{const row=document.getElementById('classRows').lastElementChild;for(const [key,value]of Object.entries({name:'temporary_test',prompts:'test object | object'})){const el=row.querySelector('[data-field="'+key+'"]');el.value=value;el.dispatchEvent(new Event('input',{bubbles:true}));}row.querySelector('[data-action="remove"]').click();})()`);
    assert.equal((await getConfig()).classes.length, boot.classes.length, 'Row deletion must update submitted classes');
    const editedPrompt = boot.classes[crosswalkIndex].prompts + ' | zebra crossing';
    await evaluate(`(()=>{const row=document.getElementById('classRows').children[${crosswalkIndex}],prompt=row.querySelector('[data-field="prompts"]'),plane=row.querySelector('[data-field="plane"]');prompt.value=${JSON.stringify(editedPrompt)};prompt.dispatchEvent(new Event('input',{bubbles:true}));plane.checked=true;plane.dispatchEvent(new Event('change',{bubbles:true}));})()`);
    assert.equal((await getConfig()).classes[crosswalkIndex].prompts, editedPrompt);
    assert.equal((await getConfig()).classes[crosswalkIndex].plane, true);
    assert.equal(await evaluate('document.activeElement?.tagName === "BODY" || document.getElementById("classRows").contains(document.activeElement)'), true);

    await input('splitFilter', 'mini_val');
    assert.equal(await evaluate('document.querySelectorAll("#sceneList input[data-scene]").length'), boot.scenes.filter(s => s.split === 'mini_val').length);
    await input('splitFilter', 'all');
    await click('selectVisible');
    assert.equal((await getConfig()).scenes.length, boot.scenes.filter(s => s.available !== false).length);
    assert.equal(await evaluate('Array.from(document.querySelectorAll("#sceneList input:disabled")).every(el=>!el.checked)'), true);
    await click('clearScenes');
    assert.equal((await getConfig()).scenes.length, 0);
    const scene = boot.scenes.find(s => s.available !== false && /0103/.test(s.name)) || boot.scenes.find(s => s.available !== false);
    assert.ok(scene, 'At least one locally available scene is required');
    await input('sceneSearch', scene.name);
    await click('selectVisible');
    assert.deepEqual((await getConfig()).scenes, [scene.name]);
    await input('sceneSearch', 'no_matching_scene_12345');
    assert.equal(await evaluate('document.querySelectorAll("#sceneList input[data-scene]").length'), 0);
    assert.deepEqual((await getConfig()).scenes, [scene.name], 'Search must preserve selection');
    await input('sceneSearch', '');
    await input('startIndex', 0);
    await check('allSceneFrames', true);
    assert.equal((await getConfig()).samples, 0);
    assert.equal(await evaluate('document.getElementById("samples").disabled'), true);
    const expectedFrames = Number(scene.frames ?? scene.nbr_samples ?? 0);
    assert.ok((await evaluate('document.getElementById("selectedSummary").textContent')).includes('预计 ' + expectedFrames + ' 帧'), 'Full scene estimate must use scene metadata');
    await check('allSceneFrames', false); await input('samples', 1);
    assert.equal((await getConfig()).samples, 1);
    assert.equal(await evaluate('document.getElementById("generateButton").disabled'), false);
    await click('previewButton');
    await wait('!document.getElementById("commandPreview").hidden || document.getElementById("validationMessage").textContent.includes("失败")');
    assert.equal(await evaluate('document.getElementById("commandPreview").hidden'), false,
      await evaluate('document.getElementById("validationMessage").textContent'));
    const command = await evaluate('document.getElementById("commandPreview").textContent');
    assert.ok(command.length > 10, 'Preview must display the backend command');
    assert.ok(command.includes(scene.name), 'Command must include the selected scene');
    const preview = await evaluate('window.__generatorRequests.findLast(r=>r.url==="/api/preview")');
    assert.ok(preview.hasToken, 'POST must include X-Generator-Token');
    assert.deepEqual(preview.body.scenes, [scene.name]);
    assert.equal(preview.body.samples, 1); assert.equal(preview.body.start_index, 0);
    assert.equal(preview.body.classes[crosswalkIndex].plane, true);
    assert.equal(preview.body.classes[crosswalkIndex].prompts, editedPrompt);
    report.preview = {scene: scene.name, classes: preview.body.classes.length, samples: preview.body.samples, fullSceneFrames: expectedFrames, command};

    if (process.env.GENERATOR_RUN === '1') {
      await click('generateButton');
      await wait('window.__generatorRequests.some(r=>r.url==="/api/start")');
      await wait('window.__generatorUI.getState().currentId && !document.getElementById("generateButton").disabled || document.getElementById("validationMessage").textContent.includes("启动失败")');
      const submitted = await evaluate('window.__generatorRequests.findLast(r=>r.url==="/api/start")');
      assert.ok(submitted.hasToken); assert.equal(submitted.body.samples, 1);
      assert.deepEqual(submitted.body.scenes, [scene.name]);
      assert.ok(!(await evaluate('document.getElementById("validationMessage").textContent')).includes('启动失败'));
      const runId = await evaluate('window.__generatorUI.getState().currentId');
      assert.ok(runId, 'Start response must select a run');
      await wait(`(()=>{const run=window.__generatorUI.getState().runs.find(r=>r.id===${JSON.stringify(runId)});return run&&['completed','failed','stopped','cancelled','canceled'].includes(run.status);})()`, 120000);
      const run = await evaluate(`window.__generatorUI.getState().runs.find(r=>r.id===${JSON.stringify(runId)})`);
      assert.equal(run.status, 'completed', 'Run did not complete: ' + await evaluate('document.getElementById("logPanel").textContent'));
      assert.equal(await evaluate('document.getElementById("resultLink").hidden'), false);
      assert.ok(await evaluate('document.getElementById("resultLink").pathname.startsWith("/results/")'));
      assert.ok(await evaluate('document.getElementById("statusText").textContent.includes("completed")'));
      assert.ok(run.progress?.total > 0, 'Completed run must have progress totals');
      assert.equal(run.progress.current, run.progress.total);
      assert.ok((await evaluate('document.getElementById("logPanel").textContent')).trim().length > 0);
      assert.equal(await evaluate('document.getElementById("stopButton").disabled'), true);
      report.run = {id: run.id, status: run.status, progress: run.progress, result_url: run.result_url};
    } else {
      assert.equal(await evaluate('window.__generatorRequests.some(r=>r.url==="/api/start")'), false, 'Default test must not start inference');
      report.inference = 'skipped; set GENERATOR_RUN=1 to test one real frame';
    }
    const remoteRequests = requests.filter(request => /^https?:/.test(request) && new URL(request).origin !== url.origin);
    assert.deepEqual(remoteRequests, [], 'Standalone template must not request external resources');
    assert.deepEqual(errors, [], 'No uncaught page errors');
    await evaluate('window.scrollTo(0,0)');
    const image = await send('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true});
    fs.mkdirSync(path.dirname(screenshot), {recursive: true});
    fs.writeFileSync(screenshot, Buffer.from(image.data, 'base64'));
    console.log(JSON.stringify({ok: true, url: url.href, screenshot, ...report, checks: 'boot, editable classes, add/delete, pipe prompts, crosswalk plane checkbox, split/search, selection persistence, unavailable scenes, full-scene estimate, one-frame preview, token header, optional real run, result/progress/logs, no remote resources/errors'}, null, 2));
  } finally {
    await send('Page.close', {}, 3000).catch(() => {});
    ws.close();
    for (const entry of pending.values()) clearTimeout(entry.timer);
  }
}
main().catch(error => { console.error(error.stack || String(error)); process.exitCode = 1; });
