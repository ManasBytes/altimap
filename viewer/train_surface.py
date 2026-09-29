"""Fine-tune classification and learn log-AGL on aligned RGB/CLS/AGL triplets.

Run: python -m viewer.train_surface --epochs 16 --batch-size 4
The existing classifier is never overwritten. best.pt is chosen on validation
only; test results are computed once after training. Checkpoints include full
optimizer/scaler state for reproducibility, and a JSON report records splits.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from viewer.classify import CLASS_NAMES, DEFAULT_CHECKPOINT, IMAGENET_MEAN, IMAGENET_STD, build_model
from viewer.gamus_dataset import find_tiles, load_tile
from viewer.train_classifier import _color_jitter, _per_class_iou
from viewer.surface import infer_tiled


class SurfaceDataset:
    def __init__(self, tiles, train, size=256):
        self.data = [load_tile(t) for t in tiles]
        self.train, self.size = train, size

    def __len__(self):
        return len(self.data) * 4

    def __getitem__(self, index):
        rgb, labels, heights = self.data[index // 4]
        h, w = labels.shape
        if self.train:
            y, x = random.randrange(h-self.size+1), random.randrange(w-self.size+1)
        else:
            # Four fixed, distributed crops per tile; final report also uses
            # every native pixel through overlap-tiled evaluation.
            y = round((h-self.size) * (0.2 if index % 4 < 2 else 0.8))
            x = round((w-self.size) * (0.2 if index % 2 == 0 else 0.8))
        arrays = [a[y:y+self.size, x:x+self.size].copy() for a in (rgb, labels, heights)]
        if self.train:
            k = random.randrange(4)
            arrays = [np.rot90(a, k).copy() for a in arrays]
            if random.random() < .5:
                arrays = [a[:, ::-1].copy() for a in arrays]
            arrays[0] = _color_jitter(arrays[0])
        rgb, labels, heights = arrays
        valid = np.isfinite(heights) & (heights >= 0) & (heights <= 150)
        logh = np.log1p(np.clip(np.nan_to_num(heights), 0, 150))
        rgb = ((rgb / 255. - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
        return torch.from_numpy(rgb).permute(2,0,1), torch.from_numpy(labels.astype(np.int64)), torch.from_numpy(logh), torch.from_numpy(valid)


def metrics(confusion, absolute, squared, count):
    ious = _per_class_iou(confusion)
    return {"miou": float(np.nanmean(ious)), "iou": {k: float(v) if np.isfinite(v) else None for k,v in zip(CLASS_NAMES, ious)},
            "agl_mae_m": absolute/max(1,count), "agl_rmse_m": (squared/max(1,count))**.5, "valid_height_pixels": int(count)}


def evaluate(model, loader, device):
    model.eval()
    conf = np.zeros((7,7), np.int64)
    absolute = squared = count = 0
    with torch.inference_mode():
        for x,y,h,valid in loader:
            with torch.autocast("cuda", enabled=device == "cuda"):
                logits, predh = model.surface(x.to(device))
            pred = logits.argmax(1).cpu().numpy()
            conf += np.bincount((y.numpy()*7+pred).ravel(), minlength=49).reshape(7,7)
            error = (predh.float().cpu().clamp(0, np.log1p(150)).expm1() - h.expm1())[valid]
            absolute += error.abs().sum().item(); squared += error.square().sum().item(); count += error.numel()
    return metrics(conf, absolute, squared, count)


def evaluate_full(model, tiles, device, with_height=True):
    conf = np.zeros((7,7), np.int64)
    absolute = squared = count = 0
    for tile in tiles:
        rgb, y, h = load_tile(tile)
        pred, ph, _ = infer_tiled(rgb, model, device, with_height=with_height)
        conf += np.bincount((y.astype(np.int64)*7+pred).ravel(), minlength=49).reshape(7,7)
        if with_height:
            valid = np.isfinite(h) & (h >= 0) & (h <= 150)
            error = ph[valid]-h[valid]
            absolute += float(np.abs(error).sum()); squared += float(np.square(error).sum()); count += error.size
    return metrics(conf, absolute, squared, count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('viewer/cache/surface_run'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--data-roots', nargs='+', type=Path)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tiles = find_tiles(tuple(args.data_roots)) if args.data_roots else find_tiles()
    splits = {s: [t for t in tiles if t.split == s] for s in ('train','val','test')}
    if not all(splits.values()):
        raise RuntimeError('Aligned train, val, and test triplets are required')
    args.output.mkdir(parents=True, exist_ok=True)
    model = build_model(pretrained=False, with_height=True).to(device)
    old = torch.load(DEFAULT_CHECKPOINT, map_location='cpu', weights_only=False)
    model.load_state_dict(old['model'], strict=False)
    # Preserve pretrained encoder statistics on this small local dataset.
    encoder_prefixes = ('stem.', 'layer1.', 'layer2.', 'layer3.', 'layer4.')
    for name, p in model.named_parameters():
        if name.startswith(encoder_prefixes): p.requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('height_head.')], 'lr':3e-5},
        {'params':model.height_head.parameters(), 'lr':5e-4}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=device=='cuda')
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler']); start=ck['epoch']+1
    train = DataLoader(SurfaceDataset(splits['train'], True), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val = DataLoader(SurfaceDataset(splits['val'], False), batch_size=args.batch_size, num_workers=0)
    counts = sum((np.bincount(y.ravel(), minlength=7) for _, y, _ in train.dataset.data), np.zeros(7))
    class_weights = np.minimum(np.sqrt(counts.sum()/np.maximum(counts,1)), 12.)
    class_weights /= class_weights.mean()
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    report = {'seed':args.seed, 'split_ids':{k:[t.scene_id for t in v] for k,v in splits.items()}, 'epochs':[], 'validation_protocol':'four fixed 256px crops per tile for selection; native full tiles for final evaluation'}
    baseline = build_model(pretrained=False).to(device).eval(); baseline.load_state_dict(old['model'])
    print('Evaluating original classifier on every validation pixel...', flush=True)
    report['baseline_validation'] = evaluate_full(baseline, splits['val'], device, False)
    del baseline
    print(report['baseline_validation'], flush=True)
    best = -float('inf')
    for epoch in range(start, args.epochs):
        t0=time.time(); model.train()
        # Keep all BN running statistics fixed; batches contain adjacent pixels.
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d): module.eval()
        total=0
        for x,y,h,valid in train:
            x,y,h,valid = [a.to(device) for a in (x,y,h,valid)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=device=='cuda'):
                logits, ph = model.surface(x)
                ce = torch.nn.functional.cross_entropy(logits, y, weight=class_weights)
                probabilities = logits.float().softmax(1)
                target = torch.nn.functional.one_hot(y, 7).permute(0,3,1,2).float()
                intersection = (probabilities*target).sum((0,2,3))
                mass = (probabilities+target).sum((0,2,3))
                present = target.sum((0,2,3)) > 0
                dice = (1-(2*intersection+1)/(mass+1))[present].mean()
                # Height loss balances ground and structures without arbitrary
                # class-specific height constants.
                weight = torch.where((y==3)|(y==6), 2., 1.) * valid
                height_loss = (torch.nn.functional.smooth_l1_loss(ph,h,reduction='none')*weight).sum()/weight.sum().clamp_min(1)
                loss = ce + .3*dice + .35*height_loss
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer); scaler.update(); total+=loss.item()
        result=evaluate(model,val,device)
        result.update(epoch=epoch+1,loss=total/len(train),seconds=round(time.time()-t0,2))
        report['epochs'].append(result)
        # Validation selection balances semantic fidelity and height accuracy.
        score=result['miou']-.01*result['agl_mae_m']
        ck={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'metrics':result,'seed':args.seed,'classes':CLASS_NAMES}
        torch.save(ck,args.output/'last.pt')
        if score>best:
            best=score; torch.save(ck,args.output/'best.pt')
        print(json.dumps(result),flush=True)
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
    ck=torch.load(args.output/'best.pt',map_location=device,weights_only=False); model.load_state_dict(ck['model'])
    for split in ('val','test'):
        print('Evaluating full '+split+' tiles...',flush=True)
        report[split]=evaluate_full(model,splits[split],device)
    report['promote_segmentation']=report['val']['miou']>=report['baseline_validation']['miou']
    # Height model may still be useful if the original classifier wins; serving
    # uses the original semantic model in that case.
    ck['metrics']=report
    torch.save(ck,args.output/'best.pt')
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('val','test','promote_segmentation')},indent=2),flush=True)


if __name__=='__main__': main()
