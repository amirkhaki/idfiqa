#!/usr/bin/env bash

# 1. Get all Run IDs for the experiment.yml workflow (adjust limit as needed)
RUN_IDS=$(gh run list --workflow experiment.yml --limit 50 --json databaseId --jq '.[].databaseId')

for RUN_ID in $RUN_IDS; do
  # 2. Get all Job IDs associated with this Run ID
  JOB_IDS=$(gh run view "$RUN_ID" --json jobs --jq '.jobs[].databaseId')
  
  for JOB_ID in $JOB_IDS; do
    # 3. View the log for the specific job and grep for "PIPAL"
    # stderr is redirected to /dev/null to hide errors for expired/missing logs
    MATCHES=$(gh run view --log --job="$JOB_ID" 2>/dev/null | grep "PIPAL")
    
    # 4. If a match is found, print the Job ID and the matched lines
    if [ -n "$MATCHES" ]; then
      echo "Job ID: $JOB_ID"
      echo "$MATCHES"
      echo "----------------------------------------"
    fi
  done
done
