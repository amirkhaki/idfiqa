"""Trainable model with a regression head and varying feature modes."""
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..config import CFG
from ..extractors import make_single_extractor
from ..registry import ExperimentBase, register_experiment
from ..utils import out_path, compute_metrics, save_json, already_done
from ..dataset import get_dataset
from ..evaluation import run_evaluation


class TrainableRegressionHead(nn.Module):
    def __init__(self, in_features, hidden_dim=256):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.gap(x).view(x.size(0), -1)
        return self.fc(x).squeeze(-1)


class IDFIQA_Trainable(nn.Module):
    """
    Model with a frozen feature extractor and a trainable regression head.
    Allows for different 'feature modes' (e.g., diff, concat).
    """

    def __init__(self, feature_extractor, normalize, feature_node_key="features",
                 device=None, feature_mode="concat"):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.normalize = normalize
        self.feature_node_key = feature_node_key
        self.feature_mode = feature_mode

        # Determine in_channels by passing a dummy tensor
        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            feat = self._features(dummy)
            channels = feat.shape[1]

        if feature_mode in ["diff", "abs_diff"]:
            in_features = channels
        elif feature_mode == "concat":
            in_features = channels * 2
        elif feature_mode == "concat_diff":
            in_features = channels * 3
        else:
            raise ValueError(f"Unknown feature mode: {feature_mode}")

        self.head = TrainableRegressionHead(in_features).to(self.device)

    def _features(self, img):
        out = self.feature_extractor(self.normalize(img.to(self.device)))
        return out[self.feature_node_key] if isinstance(out, dict) else out

    def forward(self, ref, dist):
        fr = self._features(ref)
        fd = self._features(dist)

        if self.feature_mode == "diff":
            feat = fd - fr
        elif self.feature_mode == "abs_diff":
            feat = torch.abs(fd - fr)
        elif self.feature_mode == "concat":
            feat = torch.cat([fd, fr], dim=1)
        elif self.feature_mode == "concat_diff":
            feat = torch.cat([fd, fr, fd - fr], dim=1)
        else:
            raise ValueError(f"Unknown feature mode: {self.feature_mode}")

        return self.head(feat)


def _build_trainable_model(device, backbone=None, feature_layer=None, feature_mode="concat"):
    backbone = backbone or CFG.backbone
    feature_layer = feature_layer or CFG.get_feature_layer(backbone)
    ext, norm, key = make_single_extractor(backbone, feature_layer)
    return IDFIQA_Trainable(ext, norm, feature_node_key=key, device=device, feature_mode=feature_mode)


@register_experiment
class TrainableExperiment(ExperimentBase):
    name = "trainable"
    description = "Trainable model with regression head and varying feature modes"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default=CFG.backbone)
        parser.add_argument("--feature-layer", type=str, default=None)
        parser.add_argument("--feature-mode", type=str, default="concat",
                            choices=["diff", "abs_diff", "concat", "concat_diff"])
        parser.add_argument("--loss", type=str, default="mse", choices=["mse", "l1"])
        
        # Training arguments
        parser.add_argument("--train-dataset", type=str, default="LIVE", choices=CFG.all_datasets)
        parser.add_argument("--train-percent", type=float, default=0.8)
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--batch-size", type=int, default=4)

        sub = parser.add_subparsers(dest="action")
        sub.add_parser("train", help="Train the model on the train-dataset")
        sub.add_parser("feature_mode_search", help="Sweep over all feature modes to find the best configuration")

    def _get_slug(self, args, mode_override=None):
        mode = mode_override or args.feature_mode
        feat_layer = args.feature_layer or CFG.get_feature_layer(args.backbone)
        safe_feat = feat_layer.replace(".", "_")
        return f"trainable_{args.backbone}_{safe_feat}_{mode}_{args.loss}_{args.train_dataset}"

    def run(self, args, datasets, num_workers, force, device):
        if hasattr(args, "action") and args.action == "feature_mode_search":
            self._feature_mode_search(args, datasets, num_workers, force, device)
        else:
            # Default action is train and then evaluate
            self._train_and_evaluate(args, datasets, num_workers, force, device)

    def _train_and_evaluate(self, args, datasets, num_workers, force, device, mode_override=None):
        mode = mode_override or args.feature_mode
        slug = self._get_slug(args, mode_override=mode)
        weights_path = out_path(f"{slug}_best.pt")

        model = _build_trainable_model(device, backbone=args.backbone, 
                                       feature_layer=args.feature_layer, 
                                       feature_mode=mode)

        if not os.path.exists(weights_path) or force:
            self._train(model, args, num_workers, device, weights_path)
        else:
            print(f"  [cached] Loading trained weights from {weights_path}")
            model.load_state_dict(torch.load(weights_path, map_location=device))

        # Evaluate on test datasets
        model.eval()
        results = {}
        for ds in datasets:
            csv_name = f"{slug}_{ds}.csv"
            
            if already_done(csv_name, force):
                with open(out_path(csv_name), newline="") as f:
                    rows = list(csv.DictReader(f))
                srcc, plcc = compute_metrics([float(r["score"]) for r in rows],
                                             [float(r["mos_label"]) for r in rows])
                print(f"  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}  (cached)")
            else:
                srcc, plcc, _, _ = run_evaluation(model, ds, csv_name,
                                                  num_workers=num_workers, force=force,
                                                  desc=f"Eval {ds}")
                print(f"  {ds:10s}  SRCC={srcc:.4f}  PLCC={plcc:.4f}")
            results[ds] = {"srcc": srcc, "plcc": plcc}
            
        summary_file = f"{slug}_summary.json"
        save_json(results, summary_file)
        print(f"  Summary -> {out_path(summary_file)}")
        return results

    def _train(self, model, args, num_workers, device, weights_path):
        print(f"\n=== Training ===")
        print(f"  Dataset: {args.train_dataset}")
        print(f"  Mode: {model.feature_mode}, Loss: {args.loss}")
        dataset = get_dataset(args.train_dataset)
        
        train_len = int(len(dataset) * args.train_percent)
        val_len = len(dataset) - train_len
        
        # Set generator for reproducibility across runs if needed, or omit for randomness
        train_ds, val_ds = random_split(dataset, [train_len, val_len])
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

        optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr)
        criterion = nn.MSELoss() if args.loss == "mse" else nn.L1Loss()
        
        best_val_loss = float("inf")

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
            for batch in pbar:
                ref = batch["ref_img"].to(device)
                dis = batch["dis_img"].to(device)
                score = batch["score"].to(device).float()
                
                optimizer.zero_grad()
                preds = model(ref, dis)
                loss = criterion(preds, score)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * ref.size(0)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            train_loss /= len(train_ds)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_preds, val_mos = [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False):
                    ref = batch["ref_img"].to(device)
                    dis = batch["dis_img"].to(device)
                    score = batch["score"].to(device).float()
                    
                    preds = model(ref, dis)
                    loss = criterion(preds, score)
                    val_loss += loss.item() * ref.size(0)
                    
                    val_preds.extend(preds.cpu().numpy())
                    val_mos.extend(score.cpu().numpy())
                    
            val_loss /= len(val_ds)
            srcc, plcc = compute_metrics(val_preds, val_mos)
            
            print(f"  Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val SRCC: {srcc:.4f} | Val PLCC: {plcc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), weights_path)
                print(f"    [Saved best weights]")

        print(f"=== Training Complete ===")

    def _feature_mode_search(self, args, datasets, num_workers, force, device):
        modes = ["diff", "abs_diff", "concat", "concat_diff"]
        print(f"\n=== Feature Mode Search ===")
        all_results = {}
        for mode in modes:
            print(f"\n--- Mode: {mode} ---")
            res = self._train_and_evaluate(args, datasets, num_workers, force, device, mode_override=mode)
            all_results[mode] = res

        summary_path = f"trainable_{args.backbone}_feature_mode_search.json"
        save_json(all_results, summary_path)
        print(f"\nFeature mode search complete -> {out_path(summary_path)}")
