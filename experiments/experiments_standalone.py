"""General experiments that apply across models."""
import time
import json
import csv
import random

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import CFG
from ..dataset import TO_TENSOR, get_dataset
from ..utils import out_path, save_json, already_done, compute_metrics
from ..helpers import run_slug, run_config
from ..registry import ExperimentBase, register_experiment
from .baseline import _build_baseline_model
from .patch import _build_patch_model


@register_experiment
class ComplexityExperiment(ExperimentBase):
    name = "complexity"
    description = "Measure inference complexity (latency, FLOPs)"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default=CFG.backbone)
        parser.add_argument("--feature-layer", type=str, default=None)
        parser.add_argument("--weight-layer", type=str, default=None)
        parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512])

    def run(self, args, datasets, num_workers, force, device):
        backbone = args.backbone
        feat_layer = args.feature_layer or CFG.get_feature_layer(backbone)
        wt_layer = args.weight_layer or CFG.get_weight_layer(backbone)
        slug = run_slug(backbone, feat_layer, wt_layer)
        cfg_dict = run_config(backbone, feat_layer, wt_layer)
        print(f"  backbone={backbone}  feat={feat_layer}  wt={wt_layer}"
              f"  pf={CFG.percent_features}  ws={CFG.window_size}  ps={CFG.patch_size}")
        out_file = f"complexity_{slug}.json"
        if already_done(out_file, force):
            print(f"  (cached) -> {out_path(out_file)}")
            with open(out_path(out_file)) as f:
                return json.load(f)

        h, w = args.image_size
        ref = torch.rand(1, 3, h, w).to(device)
        dis = torch.rand(1, 3, h, w).to(device)
        N = CFG.complexity_n_runs
        configs = {
            "baseline": _build_baseline_model(device, backbone=backbone,
                                              feature_layer=feat_layer),
            "patch_weighted": _build_patch_model(device, backbone=backbone,
                                                 feature_layer=feat_layer,
                                                 weight_layer=wt_layer),
        }
        results = {}
        for name, model in configs.items():
            model.eval()
            with torch.no_grad():
                for _ in range(3):
                    model(ref, dis)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(N):
                    model(ref, dis)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / N * 1000
            results[name] = {
                "avg_ms_per_pair": round(elapsed, 3),
                "device": device.type,
                "image_size": f"{h}x{w}",
                **cfg_dict,
            }
            print(f"  {name:20s}  {elapsed:.1f} ms/pair  (device={device.type})")

        save_json(results, out_file)
        print(f"  Summary -> {out_path(out_file)}")
        return results


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


@register_experiment
class GeometricRobustnessExperiment(ExperimentBase):
    name = "geometric_robustness"
    description = "Test robustness to geometric distortions"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default=CFG.backbone)
        parser.add_argument("--feature-layer", type=str, default=None)
        parser.add_argument("--weight-layer", type=str, default=None)
        parser.add_argument("--percent-features", type=float, default=CFG.percent_features)
        parser.add_argument("--window-size", type=int, default=CFG.window_size)
        parser.add_argument("--patch-size", type=int, default=CFG.patch_size)

    def run(self, args, datasets, num_workers, force, device):
        backbone = args.backbone
        feat_layer = args.feature_layer or CFG.get_feature_layer(backbone)
        wt_layer = args.weight_layer or CFG.get_weight_layer(backbone)
        slug = run_slug(backbone, feat_layer, wt_layer)
        cfg_dict = run_config(backbone, feat_layer, wt_layer)
        print(f"  backbone={backbone}  feat={feat_layer}  wt={wt_layer}"
              f"  pf={args.percent_features}  ws={args.window_size}  ps={args.patch_size}")
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        summary = {}

        for ds in datasets:
            model_global = _build_baseline_model(device, backbone=backbone,
                                                 feature_layer=feat_layer,
                                                 pf=args.percent_features,
                                                 ws=args.window_size).eval()
            model_patch = _build_patch_model(device, backbone=backbone,
                                             feature_layer=feat_layer,
                                             weight_layer=wt_layer,
                                             pf=args.percent_features,
                                             ws=args.window_size,
                                             ps=args.patch_size).eval()
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

        out_name = f"geometric_robustness_{slug}.json"
        save_json(summary, out_name)
        print(f"  Summary -> {out_path(out_name)}")
        return summary
