import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { VERT, FRAG } from './shaders.js';

const SEGMENTS = 384;   // mesh grid; independent of the 504^2 depth texture

/**
 * Depth as a Float32Array, whichever way the exporter encoded it.
 *
 * The off-nadir dataset ships depth as 16 bits packed into a PNG's R/G channels
 * (~360 KB vs ~1 MB raw, and 5200 scenes of float32 would be ~5 GB of fetches).
 * Decoding to float here rather than sampling the packed bytes in GLSL means
 * both datasets converge on one DataTexture path and the shader stays unchanged
 * -- and it sidesteps the trap that a packed-byte texture can never be linearly
 * filtered, since interpolating the high byte produces garbage between texels.
 */
async function loadDepthArray(THREE, baseDir, spec) {
  if (!spec.packed) {
    const res = await fetch(`${baseDir}/${spec.file}`);
    if (!res.ok) throw new Error(`${spec.file} ${res.status}`);
    return new Float32Array(await res.arrayBuffer());
  }

  const img = await new THREE.ImageLoader().loadAsync(`${baseDir}/${spec.file}`);
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const px = ctx.getImageData(0, 0, img.width, img.height).data;

  const out = new Float32Array(img.width * img.height);
  const lo = spec.lo;
  const span = spec.hi - spec.lo;
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = lo + ((px[p] * 256 + px[p + 1]) / 65535) * span;
  }
  return out;
}

export function initViewer(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  if (!renderer.capabilities.isWebGL2) {
    throw new Error('WebGL2 is required and is not available in this browser');
  }
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

  // Float textures cannot always be linearly filtered. Probe rather than
  // assume: nearest sampling is exact at texel centres and merely slightly
  // blocky between them, which beats a black screen.
  const canFilterFloat = !!renderer.getContext().getExtension('OES_texture_float_linear');

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e1116);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.005, 100);
  camera.position.set(0, 0.85, 1.15);

  const orbit = new OrbitControls(camera, canvas);
  orbit.enableDamping = true;
  orbit.target.set(0, 0, 0);

  const fly = new PointerLockControls(camera, canvas);
  const keys = new Set();
  addEventListener('keydown', (e) => keys.add(e.code));
  addEventListener('keyup', (e) => keys.delete(e.code));
  let navMode = 'orbit';

  const uniforms = {
    uDepth:    { value: null },
    uRgb:      { value: null },
    uPlane:    { value: new THREE.Vector3() },
    uRawRange: { value: new THREE.Vector2(0, 1) },
    uResRange: { value: new THREE.Vector2(0, 1) },
    uTexel:    { value: new THREE.Vector2(1 / 504, 1 / 504) },
    uExag:     { value: 0.15 },
    uDetrend:  { value: 0 },
    uCmap:     { value: 0 },
  };

  const geometry = new THREE.PlaneGeometry(1, 1, SEGMENTS, SEGMENTS);
  const material = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, uniforms });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.rotation.x = -Math.PI / 2;   // XY plane -> horizontal, local +Z is up
  mesh.visible = false;             // nothing to show until a scene loads
  scene.add(mesh);

  const state = { meta: null, depth: null, width: 0, height: 0, variant: 'depth', range: [0, 1] };

  /** Height in [0,1] at UV, from the DATA -- never the displaced geometry, so
   *  exaggeration cannot corrupt a reading. Mirrors the GLSL heightAt exactly. */
  function heightAt(u, v) {
    if (!state.depth) return NaN;
    const x = Math.min(state.width - 1, Math.max(0, Math.round(u * state.width)));
    // (1 - v) because the depth texture is uploaded with flipY=true, so v=1
    // samples array row 0. Without this the CPU probe and the GPU surface would
    // disagree by a vertical mirror and clicking a peak would report a trough.
    const y = Math.min(state.height - 1, Math.max(0, Math.round((1 - v) * state.height)));
    const d = state.depth[y * state.width + x];
    const m = state.meta;
    // Ranges come from the ACTIVE variant, not always from meta: the refined
    // field has its own min/max, and reading meta's here would make the CPU
    // probe disagree with the GPU surface.
    if (uniforms.uDetrend.value > 0.5 && state.variant !== 'refined') {
      const r = d - (m.plane.a + m.plane.b * u + m.plane.c * v);
      return (m.residual_max - r) / Math.max(m.residual_max - m.residual_min, 1e-8);
    }
    const lo = state.range[0], hi = state.range[1];
    return (hi - d) / Math.max(hi - lo, 1e-8);
  }

  function resize() {
    const { clientWidth: w, clientHeight: h } = canvas;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  // THREE.Clock is deprecated in r185 in favour of Timer.
  const timer = new THREE.Timer();
  renderer.setAnimationLoop(() => {
    timer.update();
    const dt = timer.getDelta();
    if (navMode === 'orbit') {
      orbit.update();
    } else if (fly.isLocked) {
      const speed = (keys.has('ShiftLeft') ? 1.6 : 0.55) * dt;
      if (keys.has('KeyW')) fly.moveForward(speed);
      if (keys.has('KeyS')) fly.moveForward(-speed);
      if (keys.has('KeyA')) fly.moveRight(-speed);
      if (keys.has('KeyD')) fly.moveRight(speed);
      if (keys.has('KeyE')) camera.position.y += speed;
      if (keys.has('KeyQ')) camera.position.y -= speed;
      // Clamp above terrain by sampling the height field (design doc 5.2).
      const u = camera.position.x + 0.5;
      const v = 0.5 - camera.position.z;
      if (u >= 0 && u <= 1 && v >= 0 && v <= 1) {
        const floor = heightAt(u, v) * uniforms.uExag.value + 0.02;
        if (camera.position.y < floor) camera.position.y = floor;
      }
    }
    renderer.render(scene, camera);
  });

  /**
   * @param base directory holding meta.json + depth.{bin,png} + rgb.jpg
   * @param opts.variant 'depth' (model output) or 'refined' (footprint-constrained)
   *
   * The refined surface is a separate height field, not a shader mode: it is
   * produced by flattening roofs against building footprints, which is a
   * per-pixel rewrite the GPU cannot derive from the raw depth on its own.
   */
  async function loadScene(base, opts = {}) {
    const metaRes = await fetch(`${base}/meta.json`);
    if (!metaRes.ok) throw new Error(`meta.json ${metaRes.status}`);
    const meta = await metaRes.json();

    const useRefined = opts.variant === 'refined' && !!meta.refined;
    const src = useRefined ? meta.refined : meta;
    const packed = useRefined || meta.encoding === 'rg16-png';
    const spec = {
      packed,
      file: useRefined ? 'refined.png' : (packed ? 'depth.png' : 'depth.bin'),
      lo: src.depth_lo, hi: src.depth_hi,
    };
    const depth = await loadDepthArray(THREE, base, spec);
    const expected = meta.width * meta.height;
    if (depth.length !== expected) {
      throw new Error(`depth has ${depth.length} values, meta declares ${expected}`);
    }

    const tex = new THREE.DataTexture(
      depth, meta.width, meta.height, THREE.RedFormat, THREE.FloatType,
    );
    tex.minFilter = tex.magFilter = canFilterFloat ? THREE.LinearFilter : THREE.NearestFilter;
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    // DataTexture defaults to flipY=false while TextureLoader defaults to true.
    // Left alone, depth renders vertically MIRRORED against its own RGB texture
    // -- relief lands on the wrong side of the image. Both must agree, and the
    // RGB convention (row 0 of the array = top of the image = v=1) is the one
    // to match, since the JPEG cannot be re-oriented. heightAt() below applies
    // the same flip on the CPU so probing and rendering stay in step.
    tex.flipY = true;
    tex.needsUpdate = true;
    uniforms.uDepth.value?.dispose();
    uniforms.uDepth.value = tex;

    const rgb = await new THREE.TextureLoader().loadAsync(`${base}/rgb.jpg`);
    rgb.colorSpace = THREE.SRGBColorSpace;
    uniforms.uRgb.value?.dispose();
    uniforms.uRgb.value = rgb;

    // The refined field is already ground-removed, so it carries no plane to
    // subtract; zeroing these keeps the detrend toggle a no-op rather than
    // double-detrending an nDSM.
    if (useRefined) {
      uniforms.uPlane.value.set(0, 0, 0);
      uniforms.uRawRange.value.set(src.depth_min, src.depth_max);
      uniforms.uResRange.value.set(src.depth_min, src.depth_max);
    } else {
      uniforms.uPlane.value.set(meta.plane.a, meta.plane.b, meta.plane.c);
      uniforms.uRawRange.value.set(meta.depth_min, meta.depth_max);
      uniforms.uResRange.value.set(meta.residual_min, meta.residual_max);
    }
    uniforms.uTexel.value.set(1 / meta.width, 1 / meta.height);

    Object.assign(state, { meta, depth, width: meta.width, height: meta.height,
                           variant: useRefined ? 'refined' : 'depth',
                           range: [src.depth_min, src.depth_max] });
    mesh.visible = true;
    return meta;
  }

  return {
    loadScene,
    heightAt,
    orbit,
    camera,
    mesh,
    get meta() { return state.meta; },
    get variant() { return state.variant; },
    get exaggeration() { return uniforms.uExag.value; },
    setMode: (m) => { uniforms.uDetrend.value = m === 'detrended' ? 1 : 0; },
    setExaggeration: (x) => { uniforms.uExag.value = x; },
    setColormap: (c) => { uniforms.uCmap.value = Number(c); },
    /** Returns the mode actually in effect -- pointer lock can be refused
     *  (headless browsers, embedded frames, missing user gesture), and an
     *  unhandled throw there would leave the UI claiming fly mode while orbit
     *  is disabled and nothing responds. */
    setNavMode: (m) => {
      if (m === 'fly') {
        try {
          fly.lock();
        } catch {
          navMode = 'orbit';
          orbit.enabled = true;
          return 'orbit';
        }
      } else {
        fly.unlock();
      }
      navMode = m;
      orbit.enabled = m === 'orbit';
      return m;
    },
  };
}
