"""Core evaluation loop."""
import os
import csv
import random as _random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .config import CFG
from .dataset import get_dataset
from .utils import out_path, compute_metrics


def run_evaluation(model, dataset_name: str, csv_filename: str,
                   num_workers: int = 2, force: bool = False,
                   desc: str = "Evaluating"):
    csv_file = out_path(csv_filename)
    device = (next(iter(model.parameters())).device
              if list(model.parameters())
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = get_dataset(dataset_name)

    if dataset_name in CFG.large_datasets and len(dataset) > CFG.large_ds_threshold:
        if os.path.exists(csv_file) and not force:
            with open(csv_file, newline="") as f:
                existing = list(csv.DictReader(f))
            p = [float(r["score"]) for r in existing]
            m = [float(r["mos_label"]) for r in existing]
            srcc, plcc = compute_metrics(p, m)
            print(f"    (cached, single-sample metrics) SRCC={srcc:.4f}  PLCC={plcc:.4f}")
            return srcc, plcc, p, m

        N_SAMPLES = CFG.large_ds_n_samples
        N_RUNS = CFG.large_ds_n_runs
        all_srcc, all_plcc = [], []
        last_preds, last_mos = [], []

        for run in range(N_RUNS):
            indices = _random.sample(range(len(dataset)), N_SAMPLES)
            subset = Subset(dataset, indices)
            loader = DataLoader(subset, batch_size=1, shuffle=False,
                                num_workers=num_workers)
            run_preds, run_mos = [], []
            with torch.no_grad():
                for batch in tqdm(loader,
                                  desc=f"{desc} run {run+1}/{N_RUNS}",
                                  leave=False):
                    ref = batch["ref_img"].to(device)
                    dis = batch["dis_img"].to(device)
                    mos_val = float(batch["score"][0]) if "score" in batch else float("nan")
                    run_preds.append(model(ref, dis).item())
                    run_mos.append(mos_val)
            rs, rp = compute_metrics(run_preds, run_mos)
            all_srcc.append(rs)
            all_plcc.append(rp)
            last_preds, last_mos = run_preds, run_mos

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "ref_img_path", "dis_img_path", "score", "mos_label"])
            for i, (s, mv) in enumerate(zip(last_preds, last_mos)):
                writer.writerow([i, "", "", s, mv])

        return float(np.mean(all_srcc)), float(np.mean(all_plcc)), last_preds, last_mos

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    start_idx = 0
    existing_rows = []
    if os.path.exists(csv_file) and not force:
        with open(csv_file, newline="") as f:
            existing_rows = list(csv.DictReader(f))
        start_idx = len(existing_rows)
        if start_idx >= len(dataset):
            preds = [float(r["score"]) for r in existing_rows]
            mos = [float(r["mos_label"]) for r in existing_rows]
            return compute_metrics(preds, mos) + (preds, mos)
        print(f"    Resuming {csv_filename} from index {start_idx}/{len(dataset)}")

    mode = "a" if start_idx > 0 else "w"
    with open(csv_file, mode, newline="") as f:
        writer = csv.writer(f)
        if start_idx == 0:
            writer.writerow(["idx", "ref_img_path", "dis_img_path", "score", "mos_label"])
            f.flush()

        all_preds = [float(r["score"]) for r in existing_rows]
        all_mos = [float(r["mos_label"]) for r in existing_rows]

        with torch.no_grad():
            for g_idx, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
                if g_idx < start_idx:
                    continue
                ref = batch["ref_img"].to(device)
                dis = batch["dis_img"].to(device)
                mos_val = float(batch["score"][0]) if "score" in batch else float("nan")
                score = model(ref, dis).item()
                ref_name = (os.path.basename(batch["ref_img_path"][0])
                            if "ref_img_path" in batch else f"img{g_idx}")
                dis_name = (os.path.basename(batch["dis_img_path"][0])
                            if "dis_img_path" in batch else "")
                writer.writerow([g_idx, ref_name, dis_name, score, mos_val])
                f.flush()
                all_preds.append(score)
                all_mos.append(mos_val)

    srcc, plcc = compute_metrics(all_preds, all_mos)
    return srcc, plcc, all_preds, all_mos
