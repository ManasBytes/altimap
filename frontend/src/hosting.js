// API and saved-scene assets can live on different hosts from the viewer.
export function resultAssets(data, baseUrl) {
  const url = (value) => value ? new URL(value, baseUrl).href : value;
  const relief = (value) => value && { ...value, png: url(value.png) };
  return {
    ...data,
    rgb: url(data.rgb),
    height: url(data.height),
    classes: url(data.classes),
    error: relief(data.error),
    terrain: data.terrain && {
      ...data.terrain,
      ground: relief(data.terrain.ground),
      dsm: relief(data.terrain.dsm),
    },
    downloads: Object.fromEntries(Object.entries(data.downloads ?? {}).map(([name, path]) => [name, url(path)])),
  };
}

export function demoCatalog(index, indexUrl) {
  if (!Array.isArray(index.scenes)) throw new Error("Invalid demo catalog");
  return index.scenes.map((scene) => {
    if (!scene.id || !scene.label || !scene.result) throw new Error("Invalid demo scene");
    return { ...scene, result: new URL(scene.result, indexUrl).href };
  });
}
