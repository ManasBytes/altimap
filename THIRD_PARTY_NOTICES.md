# Third-party model and data notices

AltiMap combines software, model weights, and geospatial data from several sources. Keep the
upstream notices with any distributed copy of the corresponding weights.

- **RS3DAda / SynRS3D**: source and model code from `JTRNEO/SynRS3D`, pinned in setup to commit
  `ab5a485`. The repository's bundled notice identifies the code as MIT. The SynRS3D training
  data is documented as CC BY-NC 4.0; treat checkpoints trained from it as non-commercial.
- **AltiMap fine-tuned v1/v2 checkpoints**: see `docs/model-card.md` and the model card at
  `https://huggingface.co/Dilavesh/altimap-height` for the checkpoint-specific terms.
- **Meta CHMv2 / DINOv3**: the optional forest specialist is distributed under Meta's DINOv3
  license. Use the exact license bundled with the downloaded checkpoint. When this specialist
  is active, the viewer displays `Built with DINOv3`.
- **Copernicus GLO-30**: used as the optional coarse elevation source for GeoTIFF absolute DSM
  composition. It is fetched from the AWS Open Data mirror or Planetary Computer at runtime;
  consult the provider's current terms when redistributing derived products.
- **GAMUS and external evaluation data**: follow each dataset's own license and citation
  requirements. Evaluation imagery and LiDAR are not bundled by this repository.

This file is an attribution checklist, not a replacement for the upstream license texts.
