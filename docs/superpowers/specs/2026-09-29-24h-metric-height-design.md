# 24-hour plan — metric height model on GAMUS

Date: 2026-09-29. Hard deadline: 24 h from start. Supersedes the training plan in
`2026-08-23-single-view-dsm-design.md` §4 (three-encoder ablation on NAIP + 3DEP) for this
submission; that doc's core decision (§2) stands.

## 1. What is being optimised

The brief (`docs/problem-statement.md`) grades 50% on RMSE / MAE / correlation against LiDAR, across
urban, sparse, hilly and forested scenes. The organisers' repo names **GAMUS**
(`earthflow/GAMUS`, CC-BY-4.0, 11,507 tiles, 5 US cities, RGB + nDSM in metres + 7 land-cover
classes, 1024×1024) as the reference dataset. Final evaluation uses ISRO RGB satellite imagery that
we cannot see in advance.

## 2. Decisions

1. **DSM = ground + object height.** Object height (nDSM, metres) comes from a learned model; ground
   comes from a public DEM (Copernicus GLO-30 with morphological opening, SRTM as fallback). One
   image cannot give ground elevation, and the DEM gives it for free. (Unchanged from the Aug design.)
2. **Model: RS3DAda** (`RS3DAda_vitl_DPT_height.pth`, DPT + ViT-L on the Depth Anything
   framework, trained on the 69k-image synthetic SynRS3D set, outputs nDSM). Chosen because it is
   already a remote-sensing height model, so fine-tuning adapts it rather than teaching it height
   from scratch. It also satisfies the brief's "pre-trained monocular depth backbone" wording.
3. **Fine-tune on a GAMUS subset** (§4) with two heads: height (metres) and a 7-class land-cover auxiliary
   head. The class head is the "semantic prior" idea, trained jointly so the network learns per-object
   height rather than a fixed per-class range. This reverses the Aug design's §13 cut of multi-task
   heads: GAMUS ships the class labels, so the extra head costs one conv layer and one loss term.
4. **Zero-shot RS3DAda is the first milestone and the safety net.** It runs through the same
   pipeline; if fine-tuning fails, it ships.
5. **Accepted risk: licence.** RS3DAda weights carry no stated licence and may inherit Depth
   Anything V2-Large's CC-BY-NC. Accepted for the hackathon, stated in the README. This overrides
   the Aug design's permissive-weights-only rule for this submission only.

## 3. Compute

| Machine | Job |
|---|---|
| Team VM, RTX A5000 24 GB (Ampere, bf16) | Everything, in sequence: zero-shot baseline, the fine-tuning run, evaluation. |
| Kaggle | Spare evaluation / inference while the A5000 trains. Too little disk and a 12 h session cap for training. |

No rented GPU. The A5000 is roughly a third to half of an A100 for this workload, so training uses
the reduced configuration in §4 and a GAMUS subset rather than the full 80 GB. Training runs inside
`tmux`; checkpoints are copied off the VM after every evaluation.

## 4. Training

- Data: `viewer/gamus_dataset.py` already reads the HF layout
  (`{images,heights,classes}/{split}/{id}_{RGB,AGL,CLS}.h5`). Download a subset (~20 GB): ~3,000
  `train` tiles sampled evenly across the 5 cities, a fixed 300-tile `val` subset, and all of
  `test`.
- GAMUS splits used as published: `train` to train, `val` to select checkpoints, `test` only for the
  numbers we report.
- Inputs: random 512×512 crops of 1024 tiles; flips and 90° rotations (heights are invariant);
  mild colour jitter. No scale augmentation in v1 (see GSD risk, §7).
- Loss: `L1(height) + 0.5·gradient-L1(height) + 0.2·CE(class)`. Heights clamped ≥ 0 at output.
- Optimiser: AdamW, bf16, gradient checkpointing, batch 4. Encoder frozen except biases (BitFit,
  lr 1e-4); decoder and heads fully trained (lr 1e-4); cosine with 500-step warmup. BitFit is the
  default on 24 GB, not a fallback: it cuts optimizer memory and was the best fine-tuning strategy
  in arXiv:2505.06905.
- Budget: stop at 8 h of wall time regardless of epoch count. Evaluate on a fixed 300-tile `val`
  subset every ~45 min; keep best by `val` RMSE.

## 5. Evaluation (what we report)

On GAMUS `test`, per city and overall: RMSE, MAE, Pearson r, plus RMSE on building pixels only
(class == buildings), because the all-pixel average hides tall-building error.

Baselines in the same table: predict-zero, predict-train-mean, zero-shot RS3DAda, zero-shot DA3
(affine-fitted per tile, as an upper bound). A result that doesn't beat predict-train-mean isn't
reported as a success.

Landscape stratification: GAMUS is 5 US cities, so "forested" and "hilly" are thin. Report per-tile
buckets by tree fraction and building fraction from the class labels, and say plainly that hilly
terrain is not covered by GAMUS.

## 6. Inference and outputs

- Tiled inference at 512 px, 25% overlap, cosine-feathered blending.
- GSD: GeoTIFF input is resampled to GAMUS GSD (to be measured in hour 0 from the GAMUS paper and
  metadata) before inference, and the output is resampled back. PNG/JPG takes a GSD field in the
  upload UI, defaulting to GAMUS GSD.
- **Georeferenced (GeoTIFF):** absolute DSM = nDSM + DEM ground, written as a float32 COG with the
  input CRS, NaN nodata, plus the `Sidecar` JSON.
- **Non-georeferenced (PNG/JPG):** the nDSM in metres, no CRS, `datum: "relative"`.
- Serving: `viewer/server.py`'s upload path swaps DA3 for this model. The frontend shows the height
  map, a height/slope probe, and a validation view that plots predicted against reference heights
  when a reference raster is uploaded.

## 7. Risks

| Risk | Mitigation |
|---|---|
| **Vertical datum.** GLO-30 / SRTM heights are geoid-based (EGM2008 / EGM96); the contract only allows `"ellipsoidal"` or `"relative"`. Over India the geoid–ellipsoid gap is tens of metres. | Add an `"orthometric"` datum value, record the geoid model in the sidecar, and match whatever datum the evaluation reference uses. One-line contract change plus a test. |
| ISRO imagery GSD or sensor differs from GAMUS | GSD resampling (§6). If time allows after training, a second short run with random rescale augmentation. |
| Fine-tuned model worse than zero-shot on some cities | Report both. Ship whichever wins on `val`. |
| VM or SSH session dies mid-run | Training in `tmux`; checkpoints copied off the VM after every evaluation (§3). |
| Hilly and forested terrain under-represented in GAMUS | Stated in results. Terrain relief comes from the DEM, not the model, so hilly scenes lean on ground-DEM accuracy. |

## 8. Out of scope for the 24 h

Ordinal/SID height loss and HTC-DC loss (the Aug design's Stage B/C), Prior2DSM-style spatial
scale-and-shift refinement against the DEM, extra datasets (SynRS3D real-data mixing, 3DEP), and
the Django backend path.
