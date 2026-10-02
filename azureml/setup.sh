#!/usr/bin/env bash
# One-time Azure ML setup, then submit the training job.
# Usage: bash azureml/setup.sh            (create resources + submit)
#        SKIP_SETUP=1 bash azureml/setup.sh (just submit again)
set -euo pipefail

RG=${RG:-visual-search-rg}
WS=${WS:-visual-search-ws}
LOCATION=${LOCATION:-eastus}
cd "$(dirname "$0")"

if [ -z "${SKIP_SETUP:-}" ]; then
  az group create -n "$RG" -l "$LOCATION" -o none
  az ml workspace create -n "$WS" -g "$RG" -l "$LOCATION" -o none
  az ml compute create -f compute-gpu.yml -g "$RG" -w "$WS" -o none
fi

RUN=$(az ml job create -f train-job.yml -g "$RG" -w "$WS" --query name -o tsv)
echo "submitted job: $RUN"
echo "watch:    az ml job stream -n $RUN -g $RG -w $WS"
echo "download: az ml job download -n $RUN -g $RG -w $WS --all --download-path ../azure_outputs"
