"""Complexity experiment."""
import time
import json

import torch

from ..config import CFG
from ..factories import make_baseline_model, make_patch_model
from ..utils import out_path, save_json, already_done
from .helpers import run_slug, run_config


def experiment_complexity(force, device):
    print("\n=== Phase 3: Complexity Analysis ===")
    feat_layer = CFG.get_feature_layer()
    wt_layer = CFG.get_weight_layer()
    slug = run_slug(CFG.backbone, feat_layer, wt_layer)
    cfg_dict = run_config(CFG.backbone, feat_layer, wt_layer)
    print(f"  backbone={CFG.backbone}  feat={feat_layer}  wt={wt_layer}"
          f"  pf={CFG.percent_features}  ws={CFG.window_size}  ps={CFG.patch_size}")
    out_file = f"phase3_complexity_{slug}.json"
    if already_done(out_file, force):
        print(f"  (cached) → {out_path(out_file)}")
        with open(out_path(out_file)) as f:
            return json.load(f)

    try:
        from thop import profile
        has_thop = True
    except ImportError:
        has_thop = False

    h, w = CFG.complexity_image_size
    ref = torch.rand(1, 3, h, w).to(device)
    dis = torch.rand(1, 3, h, w).to(device)
    N = CFG.complexity_n_runs
    configs = {
        "baseline": make_baseline_model(device),
        "patch_weighted": make_patch_model(device),
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
    print(f"  Summary → {out_path(out_file)}")
    return results
