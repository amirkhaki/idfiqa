"""Trainable model with a regression head and varying feature modes."""
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from ..utils import tqdm

from ..config import CFG
from ..extractors import make_single_extractor
from ..registry import ExperimentBase, register_experiment
from ..utils import out_path, compute_metrics, save_json, already_done
from ..dataset import get_dataset
from ..evaluation import run_evaluation


class TrainableRegressionHead(nn.Module):
    def __init__(self, in_features, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.fc(x).squeeze(-1)


class IDFIQA_Trainable(nn.Module):
    """
    Model with a frozen feature extractor and a trainable regression head.
    Allows for different 'feature modes' (e.g., diff, concat) and aggregations.
    """

    def __init__(self, feature_extractor, normalize, feature_node_key="features",
                 device=None, feature_mode="concat", aggregation="gap"):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.normalize = normalize
        self.feature_node_key = feature_node_key
        self.feature_mode = feature_mode
        self.aggregation = aggregation

        # Determine in_features by passing a dummy tensor
        dummy_ref = torch.randn(1, 3, 224, 224).to(self.device)
        dummy_dist = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            feat = self.forward_features(dummy_ref, dummy_dist)
            in_features = feat.shape[1]

        self.head = TrainableRegressionHead(in_features).to(self.device)

    def _features(self, img):
        out = self.feature_extractor(self.normalize(img.to(self.device)))
        return out[self.feature_node_key] if isinstance(out, dict) else out

    def _aggregate(self, feat):
        if self.aggregation == "gap":
            return feat.mean(dim=[2, 3])
        elif self.aggregation == "gram":
            B, C, H, W = feat.size()
            feat_flat = feat.view(B, C, H * W)
            gram = torch.bmm(feat_flat, feat_flat.transpose(1, 2)) / (H * W)
            return gram.view(B, -1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def forward_features(self, ref, dist):
        fr_raw = self._features(ref)
        fd_raw = self._features(dist)

        fr = self._aggregate(fr_raw)
        fd = self._aggregate(fd_raw)

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
        return feat

    def forward(self, ref, dist):
        feat = self.forward_features(ref, dist)
        return self.head(feat)


def _build_trainable_model(device, backbone=None, feature_layer=None, feature_mode="concat", aggregation="gap"):
    backbone = backbone or CFG.backbone
    feature_layer = feature_layer or CFG.get_feature_layer(backbone)
    ext, norm, key = make_single_extractor(backbone, feature_layer)
    return IDFIQA_Trainable(ext, norm, feature_node_key=key, device=device, feature_mode=feature_mode, aggregation=aggregation)


@register_experiment
class TrainableExperiment(ExperimentBase):
    name = "trainable"
    description = "Trainable model with regression head and varying feature modes / aggregations"

    def add_arguments(self, parser):
        parser.add_argument("--backbone", type=str, default=CFG.backbone)
        parser.add_argument("--feature-layer", type=str, default=None)
        parser.add_argument("--feature-mode", type=str, default="concat",
                            choices=["diff", "abs_diff", "concat", "concat_diff"])
        parser.add_argument("--aggregation", type=str, default="gap", choices=["gap", "gram"])
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

    def _get_slug(self, args, mode_override=None, agg_override=None):
        mode = mode_override or args.feature_mode
        agg = agg_override or args.aggregation
        feat_layer = args.feature_layer or CFG.get_feature_layer(args.backbone)
        safe_feat = feat_layer.replace(".", "_")
        return f"trainable_{args.backbone}_{safe_feat}_{agg}_{mode}_{args.loss}_{args.train_dataset}"

    def run(self, args, datasets, num_workers, force, device):
        if hasattr(args, "action") and args.action == "feature_mode_search":
            self._feature_mode_search(args, datasets, num_workers, force, device)
        else:
            # Default action is train and then evaluate
            self._train_and_evaluate(args, datasets, num_workers, force, device)

    def _train_and_evaluate(self, args, datasets, num_workers, force, device, mode_override=None, agg_override=None):
        mode = mode_override or args.feature_mode
        agg = agg_override or args.aggregation
        slug = self._get_slug(args, mode_override=mode, agg_override=agg)
        weights_path = out_path(f"{slug}_best.pt")

        model = _build_trainable_model(device, backbone=args.backbone, 
                                       feature_layer=args.feature_layer, 
                                       feature_mode=mode, aggregation=agg)

        if not os.path.exists(weights_path) or force:
            self._train(model, args, num_workers, device, weights_path, slug)
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

    def _train(self, model, args, num_workers, device, weights_path, slug):
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
        
        history_train_loss = []
        history_val_loss = []
        history_val_srcc = []
        history_val_plcc = []

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
                
            history_train_loss.append(train_loss)
            history_val_loss.append(val_loss)
            history_val_srcc.append(srcc)
            history_val_plcc.append(plcc)

        print(f"=== Training Complete ===")
        
        try:
            import matplotlib.pyplot as plt
            epochs_range = range(1, args.epochs + 1)
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, history_train_loss, label='Train Loss')
            plt.plot(epochs_range, history_val_loss, label='Val Loss')
            plt.title('Loss over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, history_val_srcc, label='Val SRCC')
            plt.plot(epochs_range, history_val_plcc, label='Val PLCC')
            plt.title('Metrics over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            
            plt.tight_layout()
            plot_path = out_path(f"{slug}_training_curves.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"  Saved training curves to {plot_path}")
        except ImportError:
            print("  matplotlib not installed, skipping training curves plot.")

    def _feature_mode_search(self, args, datasets, num_workers, force, device):
        modes = ["diff", "abs_diff", "concat", "concat_diff"]
        aggregations = ["gap", "gram"]
        print(f"\n=== Feature Mode Search ===")
        all_results = {}
        for agg in aggregations:
            for mode in modes:
                print(f"\n--- Aggregation: {agg} | Mode: {mode} ---")
                res = self._train_and_evaluate(args, datasets, num_workers, force, device, mode_override=mode, agg_override=agg)
                all_results[f"{agg}_{mode}"] = res

        summary_path = f"trainable_{args.backbone}_feature_mode_search.json"
        save_json(all_results, summary_path)
        print(f"\nFeature mode search complete -> {out_path(summary_path)}")
