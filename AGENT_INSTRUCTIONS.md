# Running Experiments via Self-Hosted Runner

This repository is configured with a self-hosted GitHub Actions runner to automate the execution of IDFIQA experiments. The runner is connected to a powerful server with GPU access, and handles dataset mounting and environment setup automatically.

## Triggering an Experiment

You can trigger an experiment manually using the GitHub Actions `workflow_dispatch` event. If you have the GitHub CLI (`gh`) installed, you can trigger it directly from the terminal.

```bash
# General format
gh workflow run experiment.yml -f command="<YOUR_COMMAND>"

# Example: Run the default help command to see available experiments
gh workflow run experiment.yml -f command="python run.py --help"

# Example: Run the baseline experiment on the LIVE dataset
gh workflow run experiment.yml -f command="python run.py -d LIVE baseline"
```

### What happens in the background?
1. The action checks out the repository.
2. It activates the pre-configured `dd` Conda environment which contains PyTorch and CUDA.
3. It sets the `IDFIQA_DATASET_ROOT` environment variable to point to `$HOME/idfiqa/datasets`.
4. It executes the command you provided.
5. It captures the standard output/error and outputs generated in the `results/` folder.

## Retrieving Results

Once the run completes, all results (plus a log of the execution output) are uploaded as a GitHub Actions artifact named `experiment-results`.

You can view the status of the runs and download the artifact via the GitHub UI, or use the `gh` CLI:

```bash
# Check the status of recent runs
gh run list --workflow=experiment.yml

# Download the artifact from the latest run
gh run download -n experiment-results -D ./downloaded_results
```

The downloaded artifact will contain:
- `run_output.log`: The captured stdout and stderr from your command.
- The contents of the `results/` directory containing any files created by the experiment.
