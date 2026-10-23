'use strict';

// Usage: node scene_viewer_browser.cjs scene.html screenshot.png
// PLAYWRIGHT_MODULE may name a package, an absolute package directory, or its entry file.
// No installation or network access is performed by this script.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

function loadPlaywright() {
  const requested = process.env.PLAYWRIGHT_MODULE;
  if (requested) return require(requested); // Explicit overrides must fail visibly if invalid.
  const candidates = ['playwright', 'playwright-core', '@playwright/test'];
  const failures = [];
  for (const name of candidates) {
    try { return require(name); } catch (error) {
      if (error.code !== 'MODULE_NOT_FOUND') throw error;
      failures.push(name);
    }
  }
  throw new Error(`Cannot require ${failures.join(', ')}. Set PLAYWRIGHT_MODULE to your existing Playwright module; this test never installs dependencies.`);
}

async function main() {
  const html = process.argv[2];
  if (!html) throw new Error('Usage: node scene_viewer_browser.cjs scene.html [screenshot.png]');
  const htmlPath = path.resolve(html);
  assert.ok(fs.statSync(htmlPath).isFile(), `Not a file: ${htmlPath}`);
  const screenshotPath = path.resolve(process.argv[3] || htmlPath.replace(/\.html?$/i, '') + '.browser.png');
  const { chromium } = loadPlaywright();
  assert.ok(chromium, 'The selected Playwright module must export chromium.');
  const browser = await chromium.launch({
    headless: true,
    args: ['--allow-file-access-from-files', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
    ...((process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || process.env.BROWSER_EXE) ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || process.env.BROWSER_EXE } : {}),
  });
  const errors = [];
  const remoteRequests = [];
  const gpuWarning = /(?:GPU stall due to ReadPixels|automatic fallback to software WebGL|deprecated.*software.*WebGL|GL Driver Message.*performance|Failed to query GLX|Passthrough is not supported|ContextResult::kTransientFailure)/i;
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error' && !gpuWarning.test(message.text())) errors.push(message.text()); });
    page.on('request', request => { if (/^https?:/i.test(request.url())) remoteRequests.push(request.url()); });
    await page.goto(pathToFileURL(htmlPath).href, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__sceneViewerReady === true || !document.getElementById('error').hidden, null, { timeout: 60000 });
    const errorText = await page.locator('#error').evaluate(element => element.hidden ? '' : element.textContent);
    assert.equal(errorText, '', `Viewer failed: ${errorText}`);
    await page.waitForFunction(() => window.__sceneViewerReady === true && window.__sceneViewer.getState().drawCalls > 0);
    const getState = () => page.evaluate(() => window.__sceneViewer.getState());
    const waitState = async (predicate, argument) => page.waitForFunction(({ source, arg }) => (0, eval)(`(${source})`)(window.__sceneViewer.getState(), arg), { source: predicate.toString(), arg: argument });
    const change = async (selector, checked) => {
      const before = (await getState()).renderCount;
      const target = page.locator(selector);
      if (await target.isChecked() === checked) return;
      await target.setChecked(checked);
      await waitState((s, n) => s.renderCount > n, before);
    };
    const slider = async (selector, value) => {
      const before = (await getState()).renderCount;
      await page.locator(selector).evaluate((element, n) => { element.value = String(n); element.dispatchEvent(new Event('input', { bubbles: true })); }, value);
      await waitState((s, n) => s.renderCount > n, before);
    };
    const clickAndRender = async selector => {
      const before = (await getState()).renderCount;
      await page.locator(selector).click();
      await waitState((s, n) => s.renderCount > n, before);
    };
    const info = await page.evaluate(() => ({
      scene: window.__sceneViewer.data.scene,
      frames: window.__sceneViewer.data.frames.length,
      webgl: !!document.getElementById('viewer').getContext('webgl'),
      totals: window.__sceneViewer.data.frames.map(frame => frame.points.count),
    }));
    assert.ok(info.frames >= 2, 'Browser contract test requires a scene with at least two frames.');
    assert.ok(info.webgl, 'A real WebGL context must be available.');
    assert.equal(await page.locator('#timeline').getAttribute('max'), String(info.frames - 1));
    const initial = await getState();
    assert.equal(await page.locator('#ipmLayer, #mapMethod').count(), 0, 'Removed IPM and map method controls must be absent.');
    assert.deepEqual(initial.methods, ['polygon']);
    assert.deepEqual(initial.classIds, [1, 2, 3, 4, 5, 6]);
    assert.equal(initial.allFrames, false);
    assert.equal(initial.points, true);
    assert.equal(initial.groundPoints, false);
    assert.ok(initial.drawCalls > 0);
    assert.ok(initial.pointCount > 0 && initial.pointDrawCalls > 0, 'Initial scene must draw semantic points.');
    assert.ok(initial.triangleCount > 0 && initial.surfaceDrawCalls > 0, 'Initial scene must draw polygon triangles.');

    // Frame changes must retain the camera and honor timeline selection.
    await slider('#timeline', 1);
    assert.equal((await getState()).frame, 1);
    assert.deepEqual((await getState()).camera, initial.camera);
    await clickAndRender('#prevFrame');
    assert.equal((await getState()).frame, 0);
    await clickAndRender('#nextFrame');
    assert.equal((await getState()).frame, 1);
    const beforeSetFrame = (await getState()).renderCount;
    await page.evaluate(() => window.__sceneViewer.setFrame(0));
    await waitState((s, n) => s.frame === 0 && s.renderCount > n, beforeSetFrame);
    const single = await getState();
    await change('#allFrames', true);
    const all = await getState();
    assert.equal(all.allFrames, true);
    assert.equal(all.playing, false);
    assert.ok(all.pointCount > single.pointCount, 'All-frame overlay must include more visible points than a single frame.');
    assert.ok(all.triangleCount >= single.triangleCount);
    assert.ok(await page.locator('#trailWarning').isVisible());
    assert.deepEqual(all.camera, single.camera);

    // Polygon visibility and selected classes must match exported triangle counts.
    const expectedTriangles = async () => page.evaluate(() => {
      const viewer = window.__sceneViewer, state = viewer.getState();
      const frames = state.allFrames ? viewer.data.frames : [viewer.data.frames[state.frame]];
      if (!state.methods.includes('polygon')) return 0;
      return frames.reduce((sum, frame) => sum + state.planeClassIds.reduce((classSum, id) => classSum + (state.classIds.includes(id) ? Number(frame.surfaces?.polygon?.[id]?.count || 0) / 3 : 0), 0), 0);
    });
    assert.equal((await getState()).triangleCount, await expectedTriangles());
    await change('#polygonLayer', false);
    assert.deepEqual((await getState()).methods, []);
    assert.equal((await getState()).triangleCount, 0);
    assert.equal((await getState()).surfaceDrawCalls, 0);
    await change('#polygonLayer', true);
    assert.deepEqual((await getState()).methods, ['polygon']);
    assert.equal((await getState()).triangleCount, await expectedTriangles());

    await clickAndRender('#selectNone');
    assert.deepEqual((await getState()).classIds, []);
    assert.equal((await getState()).pointCount, 0);
    assert.equal((await getState()).triangleCount, 0);
    assert.equal((await getState()).pointDrawCalls, 0);
    assert.equal((await getState()).surfaceDrawCalls, 0);
    for (const id of initial.availableClassIds) {
      await change(`[data-class="${id}"]`, true);
      assert.deepEqual((await getState()).classIds, [id]);
      assert.equal((await getState()).triangleCount, await expectedTriangles());
      if (!initial.planeClassIds.includes(id)) assert.equal((await getState()).triangleCount, 0);
      await change(`[data-class="${id}"]`, false);
      assert.deepEqual((await getState()).classIds, []);
    }
    await clickAndRender('#selectAll');
    assert.deepEqual((await getState()).classIds, [0, 1, 2, 3, 4, 5, 6]);
    await change('#groundPointLayer', true);
    const full = await getState();
    assert.equal(full.pointCount, info.totals.reduce((a, b) => a + b, 0), 'All classes + ground points + all frames must retain every exported point.');
    await change('#pointLayer', false);
    assert.equal((await getState()).pointCount, 0);
    assert.equal((await getState()).pointDrawCalls, 0);
    await change('#pointLayer', true);
    assert.equal((await getState()).pointCount, full.pointCount);
    await change('#groundPointLayer', false);
    assert.ok((await getState()).pointCount <= full.pointCount);
    await change('#trajectoryLayer', false);
    assert.equal((await getState()).trajectory, false);
    await change('#trajectoryLayer', true);
    await slider('#pointSize', 4);
    assert.equal((await getState()).pointSize, 4);
    await slider('#planeOpacity', 0.5);
    assert.equal((await getState()).planeOpacity, 0.5);

    // Playback advances through requestAnimationFrame; no fixed long sleeps.
    await change('#allFrames', false);
    await page.locator('#fps').selectOption('10');
    await page.locator('#playPause').click();
    await waitState(s => s.playing);
    const startingFrame = (await getState()).frame;
    await waitState((s, n) => s.frame !== n, startingFrame);
    await page.locator('#playPause').click();
    await waitState(s => !s.playing);
    await page.locator('#viewer').click({ position: { x: 120, y: 200 } });
    const keyFrame = (await getState()).frame;
    await page.keyboard.press('ArrowRight');
    await waitState((s, n) => s.frame === n, (keyFrame + 1) % info.frames);
    await page.keyboard.press('Space');
    await waitState(s => s.playing);
    await change('#allFrames', true);
    assert.equal((await getState()).playing, false);
    await page.keyboard.press('Space');
    assert.equal((await getState()).playing, false, 'Playback stays paused in all-frame mode.');

    // Camera controls must change a rendered frame, with fit restoring scene coverage.
    await clickAndRender('#topView');
    assert.ok((await getState()).camera.pitch > 1.5);
    await clickAndRender('#orbitView');
    assert.ok((await getState()).camera.pitch < 1.5);
    const distanceBeforeZoom = (await getState()).camera.distance;
    await page.locator('#viewer').hover({ position: { x: 200, y: 220 } });
    await page.mouse.wheel(0, -150);
    await waitState((s, distance) => s.camera.distance < distance, distanceBeforeZoom);
    await clickAndRender('#fitView');
    await change('#polygonLayer', true);
    await change('[data-class="0"]', false);
    await change('#groundPointLayer', false);
    await slider('#pointSize', 2.5);
    await slider('#planeOpacity', 0.7);
    await page.evaluate(() => window.__sceneViewer.render());
    assert.ok((await getState()).pointCount > 0 && (await getState()).triangleCount > 0);
    assert.ok(await page.locator('#stats').innerText());
    const pixels = await page.evaluate(() => {
      const canvas = document.getElementById('viewer'), gl = canvas.getContext('webgl');
      const bytes = new Uint8Array(canvas.width * canvas.height * 4);
      gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
      let colored = 0;
      for (let i = 0; i < bytes.length; i += 4) {
        const max = Math.max(bytes[i], bytes[i + 1], bytes[i + 2]);
        const min = Math.min(bytes[i], bytes[i + 1], bytes[i + 2]);
        if (max - min > 35 && min < 210) colored++;
      }
      return colored;
    });
    assert.ok(pixels > 10, `Rendered canvas should contain semantic colors; found ${pixels} colored pixels.`);
    assert.deepEqual(remoteRequests, [], 'Standalone HTML must make no HTTP(S) requests.');
    assert.deepEqual(errors, [], 'Browser console/page errors must be empty.');
    fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
    await page.screenshot({ path: screenshotPath, fullPage: true });
    console.log(JSON.stringify({ ok: true, scene: info.scene, frames: info.frames, coloredPixels: pixels, state: await getState(), screenshot: screenshotPath }, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error.stack || String(error)); process.exitCode = 1; });
