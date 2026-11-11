/* Reel Studio mascot: the 3D character that plays while a job runs.
   He is kept centered on his hips every frame, so walking and drifting animations stay in place. */
(function (root) {
'use strict';
const SRC = 'mascot/mascot.glb';
let T = null, loading = null, M = null;        // three, load promise, loaded model parts
let renderer = null, canvas = null, raf = 0, last = 0, ro = null, spin = 0, spinVel = 0, dragging = false, lastX = 0, gap = 33, busy = 0;

async function load() {
  T = await import('three');
  const [{ GLTFLoader }, { MeshoptDecoder }] = await Promise.all([import('three/addons/loaders/GLTFLoader.js'), import('three/addons/libs/meshopt_decoder.module.js')]);
  const gltf = await new GLTFLoader().setMeshoptDecoder(MeshoptDecoder).loadAsync(SRC);
  const scene = new T.Scene(), turn = new T.Group(), pivot = new T.Group();
  scene.add(turn); turn.add(pivot); pivot.add(gltf.scene);
  const mat = new T.MeshPhysicalMaterial({ color: 0xF4F2FA, roughness: 0.42, metalness: 0, clearcoat: 0.35, clearcoatRoughness: 0.45, side: T.DoubleSide });
  gltf.scene.traverse(o => { if (o.isMesh) { o.material = mat; o.frustumCulled = false; } });
  scene.add(new T.HemisphereLight(0xffffff, 0xaab6ff, 1.5));
  const key = new T.DirectionalLight(0xfff3ea, 2.3); key.position.set(2.2, 3, 4); scene.add(key);
  const rim = new T.DirectionalLight(0x9fdcff, 1.8); rim.position.set(-3, 2.2, -2.5); scene.add(rim);
  const warm = new T.DirectionalLight(0xffb79a, 0.9); warm.position.set(-2.5, 0.5, 3); scene.add(warm);
  // fixed camera: frames his full height with room for jumps and wide arms
  const cam = new T.PerspectiveCamera(26, 1, 0.1, 50), fit = 2.2, dist = fit / 2 / Math.tan(13 * Math.PI / 180);
  cam.position.set(0, 0.96, dist); cam.lookAt(0, 0.96, 0);
  const mixer = new T.AnimationMixer(gltf.scene), clips = gltf.animations.slice();
  const hips = gltf.scene.getObjectByName('Hips');
  const actions = clips.map(c => { const a = mixer.clipAction(c); a.setLoop(T.LoopOnce, 1); a.clampWhenFinished = true; return a; });
  let cur = 0; actions[0].play();
  mixer.addEventListener('finished', () => { const prev = actions[cur]; cur = (cur + 1) % actions.length; const next = actions[cur]; next.reset().play(); if (prev !== next) prev.crossFadeTo(next, 0.35, false); });
  const p = new T.Vector3();
  const center = () => {   // always centered: cancel any sideways or forward travel of the hips
    pivot.position.set(0, 0, 0); turn.updateMatrixWorld(true);
    if (!hips) return; hips.getWorldPosition(p); turn.worldToLocal(p); pivot.position.set(-p.x, 0, -p.z);
  };
  M = { scene, cam, mixer, turn, center };
  return M;
}
const warm = () => loading || (loading = load().catch(e => { loading = null; throw e; }));

function frame(now) {
  raf = requestAnimationFrame(frame);
  if (now - last < gap) return;                // 30 frames a second, fewer if the device is busy: the video job always comes first
  const dt = Math.min(0.1, (now - last) / 1000 || 0); last = now;
  const t0 = performance.now();
  M.mixer.update(dt);
  if (!dragging) { spin += spinVel * dt; spinVel *= Math.pow(0.02, dt); spin *= Math.pow(0.04, dt); }   // eases back to face you
  M.turn.rotation.y = spin; M.center();
  renderer.render(M.scene, M.cam);
  // adapt: if drawing him costs much, or frames arrive late because the job is busy, slow down; speed back up when it's easy
  const cost = performance.now() - t0, late = dt * 1000 > gap * 1.8;
  busy = busy ? busy * 0.8 + cost * 0.2 : cost;                       // average time to draw one frame
  const floor = Math.max(33, busy * 4);                                 // never more than a quarter of the device's time
  gap = late ? Math.min(1000, Math.max(floor, gap * 1.25)) : Math.max(floor, gap * 0.97);
}
function softwareGL(r) {
  if (root.__mascotForce3D) return false;
  try { const gl = r.getContext(), ext = gl.getExtension('WEBGL_debug_renderer_info'); const name = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    return /swiftshader|llvmpipe|software|basic render/i.test(String(name)); } catch (e) { return false; }
}
function size() {
  if (!renderer || !canvas) return;
  const w = Math.max(40, Math.round(canvas.clientWidth)), h = Math.max(40, Math.round(canvas.clientHeight));
  renderer.setPixelRatio(Math.min(root.devicePixelRatio || 1, 2)); renderer.setSize(w, h, false);
  M.cam.aspect = w / h; M.cam.updateProjectionMatrix();
}
/** Start playing on this canvas. Resolves true once he is on screen; false if 3D isn't available here. */
async function start(cv, opts = {}) {
  try {
    await warm();
    if (canvas !== cv || !renderer) {
      if (renderer) renderer.dispose();
      canvas = cv; renderer = new T.WebGLRenderer({ canvas: cv, antialias: true, alpha: true, powerPreference: 'high-performance', preserveDrawingBuffer: !!opts.snapshot });
      if (!opts.snapshot && softwareGL(renderer)) { renderer.dispose(); renderer = null; canvas = null; return false; }   // no real graphics chip: the drawn loop plays instead
      renderer.outputColorSpace = T.SRGBColorSpace; renderer.toneMapping = T.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05; renderer.setClearColor(0x000000, 0);
      cv.style.touchAction = 'pan-y';
      cv.onpointerdown = e => { dragging = true; lastX = e.clientX; spinVel = 0; try { cv.setPointerCapture(e.pointerId); } catch (x) { /* fine */ } };
      cv.onpointermove = e => { if (!dragging) return; const d = (e.clientX - lastX) / Math.max(60, cv.clientWidth) * 3.2; spin += d; spinVel = d * 30; lastX = e.clientX; };
      cv.onpointerup = cv.onpointercancel = () => { dragging = false; };
      if (ro) ro.disconnect(); if (root.ResizeObserver) { ro = new ResizeObserver(size); ro.observe(cv); }
    }
    size(); cancelAnimationFrame(raf); last = performance.now(); gap = 33; busy = 0; raf = requestAnimationFrame(frame);
    return true;
  } catch (e) { console.warn('3D mascot unavailable', e); return false; }
}
function stop() { cancelAnimationFrame(raf); raf = 0; }
root.Mascot3D = { warm: () => warm().catch(() => {}), start, stop, _m: () => M };
})(window);
