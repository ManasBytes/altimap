// Terrain detail controls. Keep samples one larger than segments because a grid
// with N segments contains N + 1 vertices along that axis.
// 512 segments keeps more than half a million triangles while avoiding the
// million-vertex CPU/GPU cost of the source grid during interactive flight.
const MESH_SEGMENTS_X = 512;
const MESH_SEGMENTS_Y = 512;
const HEIGHT_SAMPLE_WIDTH = MESH_SEGMENTS_X + 1;
const HEIGHT_SAMPLE_HEIGHT = MESH_SEGMENTS_Y + 1;

export {
  HEIGHT_SAMPLE_WIDTH,
  HEIGHT_SAMPLE_HEIGHT,
  MESH_SEGMENTS_X,
  MESH_SEGMENTS_Y,
};
