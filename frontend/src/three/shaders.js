// Ported unchanged from viewer/web/js/shaders.js -- same GLSL, same reasoning.
// heightAt is needed in BOTH stages -- the vertex stage to displace, the
// fragment stage to derive normals by finite differences. GLSL has no shared
// includes here, so the source is defined once and injected into both.
export const HEIGHT_FN = /* glsl */`
uniform sampler2D uDepth;
uniform vec3  uPlane;      // a, b, c of z = a + b*u + c*v
uniform vec2  uRawRange;   // depth_min, depth_max
uniform vec2  uResRange;   // residual_min, residual_max
uniform float uExag;
uniform float uDetrend;    // 0 = raw, 1 = detrended

float heightAt(vec2 uv) {
  float d = texture2D(uDepth, uv).r;
  if (uDetrend > 0.5) {
    float r = d - (uPlane.x + uPlane.y * uv.x + uPlane.z * uv.y);
    return (uResRange.y - r) / max(uResRange.y - uResRange.x, 1e-8);
  }
  // Depth is distance from sensor: larger depth = lower ground. Hence max - d.
  return (uRawRange.y - d) / max(uRawRange.y - uRawRange.x, 1e-8);
}`;

export const VERT = /* glsl */`
${HEIGHT_FN}
varying vec2  vUv;
varying float vH;
void main() {
  vUv = uv;
  vH = heightAt(uv);
  // PlaneGeometry lies in XY; the mesh is rotated -90deg about X, so local +Z
  // becomes world up.
  vec3 p = vec3(position.xy, vH * uExag);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
}`;

export const FRAG = /* glsl */`
${HEIGHT_FN}
uniform sampler2D uRgb;
uniform vec2  uTexel;    // 1/width, 1/height
uniform float uCmap;     // 0 rgb, 1 turbo, 2 slope shade
varying vec2  vUv;
varying float vH;

// Polynomial turbo approximation -- compact and good enough for relief.
vec3 turbo(float t) {
  t = clamp(t, 0.0, 1.0);
  return clamp(vec3(
    0.13572138 + t*(4.61539260 + t*(-42.66032258 + t*(132.13108234 + t*(-152.94239396 + t*59.28637943)))),
    0.09140261 + t*(2.19418839 + t*(  4.84296658 + t*(-14.18503333 + t*(   4.27729857 + t* 2.82956604)))),
    0.10667330 + t*(12.6419460 + t*(-60.58204836 + t*(110.36276771 + t*( -89.90310912 + t*27.34824973))))
  ), 0.0, 1.0);
}

vec3 surfaceNormal() {
  // Horizontal extent is 1 world unit across the whole grid, so a one-texel
  // step is uTexel world units. Vertical scale is uExag.
  float hl = heightAt(vUv - vec2(uTexel.x, 0.0));
  float hr = heightAt(vUv + vec2(uTexel.x, 0.0));
  float hd = heightAt(vUv - vec2(0.0, uTexel.y));
  float hu = heightAt(vUv + vec2(0.0, uTexel.y));
  return normalize(vec3(
    (hl - hr) * uExag,
    2.0 * uTexel.x,
    (hd - hu) * uExag
  ));
}

void main() {
  vec3 n = surfaceNormal();
  vec3 lightDir = normalize(vec3(0.45, 0.8, 0.35));
  float lambert = 0.35 + 0.65 * max(dot(n, lightDir), 0.0);

  vec3 base;
  if (uCmap < 0.5)      base = texture2D(uRgb, vUv).rgb;
  else if (uCmap < 1.5) base = sRGBTransferEOTF(vec4(turbo(vH), 1.0)).rgb;
  else                  base = sRGBTransferEOTF(vec4(vec3(0.72), 1.0)).rgb;

  gl_FragColor = vec4(base * lambert, 1.0);

  // The RGB texture is tagged sRGB, so three converts it to linear on sample
  // and all the lighting above happens in linear space. Without converting
  // back for display the whole scene renders visibly dark. Built-in materials
  // get this chunk appended automatically; a ShaderMaterial must ask.
  #include <colorspace_fragment>
}`;
