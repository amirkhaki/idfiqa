"""CLI entrypoint for IDFIQA experiments."""
import os
import argparse
from datetime import datetime

import torch

from experiments.config import CFG
from experiments.registry import list_experiments


def main():
    parser = argparse.ArgumentParser(
        description="IDFIQA experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-d", "--datasets", nargs="+", default=None,
                        choices=CFG.all_datasets)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-w", "--num-workers", type=int, default=2)
    parser.add_argument("-o", "--output-dir", type=str, default=CFG.output_dir)

    subparsers = parser.add_subparsers(dest="experiment")

    for name, exp_cls in sorted(list_experiments().items()):
        sub = subparsers.add_parser(name, help=exp_cls.description)
        exp_cls().add_arguments(sub)

    args = parser.parse_args()

    if not args.experiment:
        parser.print_help()
        print("\nAvailable experiments:")
        for name, exp_cls in sorted(list_experiments().items()):
            print(f"  {name:25s} {exp_cls.description}")
        return

    CFG.output_dir = args.output_dir
    os.makedirs(CFG.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = args.datasets or CFG.all_datasets

    print(f"Device:     {device}")
    print(f"Experiment: {args.experiment}")
    print(f"Output:     {os.path.abspath(CFG.output_dir)}")
    print(f"Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    experiments = list_experiments()
    exp = experiments[args.experiment]()
    exp.run(args, datasets, args.num_workers, args.force, device)

    print(f"\nDone. All outputs in: {os.path.abspath(CFG.output_dir)}/")


if __name__ == "__main__":
    main()
