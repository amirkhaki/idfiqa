"""Core evaluation loop."""
import os
import csv
import random as _random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .utils import tqdm

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
            extra_keys = [k for k in existing[0].keys() if k not in ["idx", "ref_img_path", "dis_img_path", "score", "mos_label"]]
            if extra_keys:
                print(f"    (cached) main: SRCC={srcc:.4f}  PLCC={plcc:.4f}")
                for k in extra_keys:
                    ek_p = [float(r[k]) for r in existing]
                    ek_s, ek_p_val = compute_metrics(ek_p, m)
                    print(f"    (cached) {k:12s} SRCC={ek_s:.4f}  PLCC={ek_p_val:.4f}")
            else:
                print(f"    (cached, single-sample metrics) SRCC={srcc:.4f}  PLCC={plcc:.4f}")
            return srcc, plcc, p, m

        N_SAMPLES = CFG.large_ds_n_samples
        N_RUNS = CFG.large_ds_n_runs
        all_srcc, all_plcc = [], []
        all_extra_srcc = {}
        all_extra_plcc = {}

        last_preds, last_mos = [], []
        last_refs, last_dists = [], []
        last_extra_preds = {}

        for run in range(N_RUNS):
            indices = _random.sample(range(len(dataset)), N_SAMPLES)
            subset = Subset(dataset, indices)
            loader = DataLoader(subset, batch_size=1, shuffle=False,
                                num_workers=num_workers)
            run_preds, run_mos = [], []
            run_refs, run_dists = [], []
            run_extra = {}
            with torch.no_grad():
                for i, batch in enumerate(tqdm(loader,
                                  desc=f"{desc} run {run+1}/{N_RUNS}",
                                  leave=False)):
                    ref = batch["ref_img"].to(device)
                    dis = batch["dis_img"].to(device)
                    mos_val = float(batch["score"][0]) if "score" in batch else float("nan")
                    
                    out = model(ref, dis)
                    if isinstance(out, dict):
                        out_dict = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
                    else:
                        out_dict = {"score": out.item() if isinstance(out, torch.Tensor) else out}
                    
                    run_preds.append(out_dict["score"])
                    for k, v in out_dict.items():
                        if k != "score":
                            if k not in run_extra:
                                run_extra[k] = []
                            run_extra[k].append(v)
                            
                    run_mos.append(mos_val)
                    ref_name = (os.path.basename(batch["ref_img_path"][0])
                                if "ref_img_path" in batch else f"img{i}")
                    dis_name = (os.path.basename(batch["dis_img_path"][0])
                                if "dis_img_path" in batch else "")
                    run_refs.append(ref_name)
                    run_dists.append(dis_name)
            
            rs, rp = compute_metrics(run_preds, run_mos)
            all_srcc.append(rs)
            all_plcc.append(rp)
            for k in run_extra:
                if k not in all_extra_srcc:
                    all_extra_srcc[k] = []
                    all_extra_plcc[k] = []
                es, ep = compute_metrics(run_extra[k], run_mos)
                all_extra_srcc[k].append(es)
                all_extra_plcc[k].append(ep)
                
            last_preds, last_mos = run_preds, run_mos
            last_refs, last_dists = run_refs, run_dists
            last_extra_preds = run_extra

        mean_srcc = float(np.mean(all_srcc))
        mean_plcc = float(np.mean(all_plcc))
        
        extra_keys = list(last_extra_preds.keys())
        if extra_keys:
            for k in extra_keys:
                ms = float(np.mean(all_extra_srcc[k]))
                mp = float(np.mean(all_extra_plcc[k]))
                print(f"    {k:12s} SRCC={ms:.4f}  PLCC={mp:.4f}")

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "ref_img_path", "dis_img_path", "score", "mos_label"] + extra_keys)
            for i, (s, mv, rn, dn) in enumerate(zip(last_preds, last_mos, last_refs, last_dists)):
                row = [i, rn, dn, s, mv]
                for k in extra_keys:
                    row.append(last_extra_preds[k][i])
                writer.writerow(row)

        return mean_srcc, mean_plcc, last_preds, last_mos

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
            srcc, plcc = compute_metrics(preds, mos)
            extra_keys = [k for k in existing_rows[0].keys() if k not in ["idx", "ref_img_path", "dis_img_path", "score", "mos_label"]]
            if extra_keys:
                print(f"    (cached) main: SRCC={srcc:.4f}  PLCC={plcc:.4f}")
                for k in extra_keys:
                    ek_p = [float(r[k]) for r in existing_rows]
                    ek_s, ek_p_val = compute_metrics(ek_p, mos)
                    print(f"    (cached) {k:12s} SRCC={ek_s:.4f}  PLCC={ek_p_val:.4f}")
            return (srcc, plcc, preds, mos)
        print(f"    Resuming {csv_filename} from index {start_idx}/{len(dataset)}")

    mode = "a" if start_idx > 0 else "w"
    extra_keys = []
    if start_idx > 0 and existing_rows:
        extra_keys = [k for k in existing_rows[0].keys() if k not in ["idx", "ref_img_path", "dis_img_path", "score", "mos_label"]]
        
    with open(csv_file, mode, newline="") as f:
        all_preds = [float(r["score"]) for r in existing_rows]
        all_mos = [float(r["mos_label"]) for r in existing_rows]
        all_extra = {k: [float(r[k]) for r in existing_rows] for k in extra_keys}
        
        writer = None

        with torch.no_grad():
            for g_idx, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
                if g_idx < start_idx:
                    continue
                ref = batch["ref_img"].to(device)
                dis = batch["dis_img"].to(device)
                mos_val = float(batch["score"][0]) if "score" in batch else float("nan")
                
                out = model(ref, dis)
                if isinstance(out, dict):
                    out_dict = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
                else:
                    out_dict = {"score": out.item() if isinstance(out, torch.Tensor) else out}
                
                if start_idx == 0 and g_idx == 0:
                    extra_keys = [k for k in out_dict.keys() if k != "score"]
                    all_extra = {k: [] for k in extra_keys}
                    writer = csv.writer(f)
                    writer.writerow(["idx", "ref_img_path", "dis_img_path", "score", "mos_label"] + extra_keys)
                    f.flush()
                elif writer is None:
                    writer = csv.writer(f)
                
                score = out_dict["score"]
                ref_name = (os.path.basename(batch["ref_img_path"][0])
                            if "ref_img_path" in batch else f"img{g_idx}")
                dis_name = (os.path.basename(batch["dis_img_path"][0])
                            if "dis_img_path" in batch else "")
                            
                row = [g_idx, ref_name, dis_name, score, mos_val]
                for k in extra_keys:
                    val = out_dict.get(k, 0.0)
                    row.append(val)
                    all_extra[k].append(val)
                    
                writer.writerow(row)
                f.flush()
                all_preds.append(score)
                all_mos.append(mos_val)

    srcc, plcc = compute_metrics(all_preds, all_mos)
    if extra_keys:
        for k in extra_keys:
            es, ep = compute_metrics(all_extra[k], all_mos)
            print(f"    {k:12s} SRCC={es:.4f}  PLCC={ep:.4f}")

    return srcc, plcc, all_preds, all_mos
