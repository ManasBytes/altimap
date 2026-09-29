import * as THREE from "three";
// Numbered badge sprite so a waypoint's order in the flight path is legible
// at a glance instead of needing to count markers along the route.
function createWaypointLabel(number) {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgba(7, 17, 30, 0.92)";
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#ffb55e";
  ctx.lineWidth = 4;
  ctx.stroke();
  ctx.fillStyle = "#ffe4c2";
  ctx.font = "bold 30px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(number), size / 2, size / 2 + 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: texture, depthTest: false }),
  );
  sprite.scale.set(0.26, 0.26, 1);
  sprite.position.y = 0.34;
  sprite.renderOrder = 10;
  return sprite;
}

// A pin (stem + head + ground halo) reads far better than a bare sphere and
// gives waypoints a visible "footprint" on the terrain. A separate, larger,
// invisible hit-sphere makes markers easy to click/reselect without needing
// pixel-precise aim on the small visible head. Everything lives under one
// group so raycasting can walk back up to `userData.waypointIndex`.
function createWaypointMarker(index, point) {
  const group = new THREE.Group();
  group.position.copy(point);
  group.userData.waypointIndex = index;

  const stem = new THREE.Mesh(
    new THREE.CylinderGeometry(0.012, 0.012, 0.16, 8),
    new THREE.MeshBasicMaterial({ color: 0xffb55e }),
  );
  stem.position.y = 0.08;
  group.add(stem);

  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.055, 20, 16),
    new THREE.MeshStandardMaterial({
      color: 0xff9f43,
      emissive: 0x552300,
      emissiveIntensity: 0.4,
      roughness: 0.35,
      metalness: 0.1,
    }),
  );
  head.position.y = 0.16;
  group.add(head);

  const halo = new THREE.Mesh(
    new THREE.RingGeometry(0.075, 0.11, 28),
    new THREE.MeshBasicMaterial({
      color: 0xffb55e,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.65,
      depthWrite: false,
    }),
  );
  halo.rotation.x = -Math.PI / 2;
  halo.position.y = 0.004;
  group.add(halo);

  const hitArea = new THREE.Mesh(
    new THREE.SphereGeometry(0.22, 12, 10),
    new THREE.MeshBasicMaterial({ visible: false }),
  );
  hitArea.position.y = 0.1;
  group.add(hitArea);

  const label = createWaypointLabel(index + 1);
  group.add(label);

  group.userData.head = head;
  group.userData.halo = halo;
  group.userData.label = label;
  return group;
}

function setWaypointNumber(marker, number) {
  const old = marker.userData.label;
  if (old) {
    marker.remove(old);
    old.material.map.dispose();
    old.material.dispose();
  }
  const label = createWaypointLabel(number);
  marker.add(label);
  marker.userData.label = label;
}

function setWaypointSelected(marker, selected) {
  const { head, halo } = marker.userData;
  head.material.color.set(selected ? 0xfff3d6 : 0xff9f43);
  head.material.emissive.set(selected ? 0xffa64d : 0x552300);
  head.material.emissiveIntensity = selected ? 0.9 : 0.4;
  halo.material.opacity = selected ? 1 : 0.65;
  marker.scale.setScalar(selected ? 1.25 : 1);
}

function disposeWaypointMarker(marker) {
  marker.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (child.material.map) child.material.map.dispose();
      child.material.dispose();
    }
  });
}

// Finds the ancestor marker group for a raycast hit against any of a
// marker's child meshes (stem, head, halo, hit-sphere, label).
function resolveWaypointHit(hit) {
  let obj = hit?.object ?? null;
  while (obj && obj.userData.waypointIndex === undefined) obj = obj.parent;
  return obj;
}

export {
  createWaypointMarker,
  setWaypointNumber,
  setWaypointSelected,
  disposeWaypointMarker,
  resolveWaypointHit,
};
