"""Geometric robustness experiment."""
import csv
import random

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import CFG
from ..dataset import TO_TENSOR, get_dataset
from ..factories import make_baseline_model, make_patch_model
from ..utils import out_path, save_json, already_done, compute_metrics
from .helpers import run_slug, run_config


def _apply_geometric_distortion(img_tensor, shift_px=5, angle_deg=2,
                                scale_range=(0.95, 1.05)):
    import torchvision.transforms.functional as TF
    pil = transforms.ToPILImage()(img_tensor.squeeze(0))
    _, h = pil.size[0], pil.size[1]
    w_px = pil.size[0]
    dx = random.uniform(-shift_px, shift_px)
    dy = random.uniform(-shift_px, shift_px)
    angle = random.uniform(-angle_deg, angle_deg)
    s = random.uniform(*scale_range)
    pil = TF.affine(pil, angle=angle, translate=(dx, dy), scale=s, shear=0)
    pil = TF.center_crop(pil, (h, w_px))
    return TO_TENSOR(pil).unsqueeze(0)


def experiment_geometric_robustness(datasets, num_workers, force, device):
    print("\n=== Phase 3: Geometric Robustness ===")
    feat_layer = CFG.get_feature_layer()
    wt_layer = CFG.get_weight_layer()
    slug = run_slug(CFG.backbone, feat_layer, wt_layer)
    cfg_dict = run_config(CFG.backbone, feat_layer, wt_layer)
    print(f"  backbone={CFG.backbone}  feat={feat_layer}  wt={wt_layer}"
          f"  pf={CFG.percent_features}  ws={CFG.window_size}  ps={CFG.patch_size}")
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    summary = {}

    for ds in datasets:
        model_global = make_baseline_model(device, feature_layer=feat_layer).eval()
        model_patch = make_patch_model(device, feature_layer=feat_layer,
                                       weight_layer=wt_layer).eval()
        csv_name = f"geom_robustness_{slug}_{ds}.csv"

        if already_done(csv_name, force):
            with open(out_path(csv_name), newline="") as f:
                rows = list(csv.DictReader(f))
            srcc_glob, _ = compute_metrics([float(r["global_gram"]) for r in rows],
                                           [float(r["mos_label"]) for r in rows])
            srcc_patch, _ = compute_metrics([float(r["patch_weighted"]) for r in rows],
                                            [float(r["mos_label"]) for r in rows])
            srcc_ssim, _ = compute_metrics([float(r["ssim"]) for r in rows],
                                           [float(r["mos_label"]) for r in rows])
            print(f"  {ds:10s}  Global={srcc_glob:.4f}  Patch={srcc_patch:.4f}"
                  f"  SSIM={srcc_ssim:.4f}  (cached)")
            summary[ds] = {"global_gram_srcc": srcc_glob,
                           "patch_weighted_srcc": srcc_patch,
                           "ssim_srcc": srcc_ssim, **cfg_dict}
            continue

        dataset = get_dataset(ds)
        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=num_workers)
        csv_file = out_path(csv_name)
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "global_gram", "patch_weighted", "ssim", "mos_label"])
            f.flush()
            gg_list, pw_list, ss_list, mos_list = [], [], [], []
            with torch.no_grad():
                for idx, batch in enumerate(tqdm(loader,
                                                 desc=f"GeoRobust/{ds}", leave=False)):
                    ref = batch["ref_img"].to(device)
                    dis = batch["dis_img"].to(device)
                    mos_val = float(batch["score"][0]) if "score" in batch else float("nan")
                    dis_geo = _apply_geometric_distortion(dis.cpu()).to(device)
                    s_glob = model_global(ref, dis_geo).item()
                    s_patch = model_patch(ref, dis_geo).item()
                    try:
                        s_ssim = ssim_fn(ref, dis_geo).item()
                    except Exception:
                        s_ssim = float("nan")
                    writer.writerow([idx, s_glob, s_patch, s_ssim, mos_val])
                    f.flush()
                    gg_list.append(s_glob)
                    pw_list.append(s_patch)
                    ss_list.append(s_ssim)
                    mos_list.append(mos_val)

        srcc_glob, _ = compute_metrics(gg_list, mos_list)
        srcc_patch, _ = compute_metrics(pw_list, mos_list)
        srcc_ssim, _ = compute_metrics(ss_list, mos_list)
        print(f"  {ds:10s}  Global={srcc_glob:.4f}  Patch={srcc_patch:.4f}"
              f"  SSIM={srcc_ssim:.4f}")
        summary[ds] = {"global_gram_srcc": srcc_glob,
                       "patch_weighted_srcc": srcc_patch,
                       "ssim_srcc": srcc_ssim, **cfg_dict}

    out_name = f"phase3_geometric_robustness_{slug}.json"
    save_json(summary, out_name)
    print(f"  Summary → {out_path(out_name)}")
    return summary
