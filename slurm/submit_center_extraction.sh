#!/usr/bin/env bash

set -euo pipefail

# Parameterized wrapper to submit center point extraction array jobs
# Usage:
#   ./slurm/submit_center_extraction.sh [--methods cellpose_sam,cellpose3] [--dry-run]

METHODS="cellpose_sam,cellpose3"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --methods)
      METHODS="$2"; shift 2;;
    --dry-run)
      DRY_RUN=1; shift;;
    -h|--help)
      echo "Usage: $0 [--methods cellpose_sam,cellpose3] [--dry-run]"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# Resolve repo root as directory containing this script, two levels up
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IFS=',' read -r -a METHOD_LIST <<< "$METHODS"

echo "Submitting center point extraction array jobs for methods: ${METHOD_LIST[*]}"

declare -A method_dirs
method_dirs["cellpose_sam"]="$REPO_ROOT/scripts/methods/cellpose_sam"
method_dirs["cellpose3"]="$REPO_ROOT/scripts/methods/cellpose3"

submit_method() {
  local method="$1"
  local mdir="${method_dirs[$method]:-}"
  if [[ -z "$mdir" || ! -d "$mdir" ]]; then
    echo "[WARN] Method '$method' not found at expected path: $mdir" >&2
    return 1
  fi
  local slurm_file="$mdir/scripts/run_center_point_extraction_array.slurm"
  if [[ ! -f "$slurm_file" ]]; then
    echo "[WARN] Slurm script missing for method '$method': $slurm_file" >&2
    return 1
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] cd $mdir && sbatch $slurm_file"
    return 0
  fi
  pushd "$mdir" >/dev/null
  local job_id
  job_id="$(sbatch "$slurm_file" | awk '{print $4}')"
  popd >/dev/null
  echo "$method array job submitted with ID: $job_id"
}

any_submitted=0
for method in "${METHOD_LIST[@]}"; do
  if submit_method "$method"; then
    any_submitted=1
  fi
done

if [[ "$DRY_RUN" -eq 0 && "$any_submitted" -eq 1 ]]; then
  echo "All requested array jobs submitted."
  echo "Monitor with: squeue -u $USER"
  echo "Check logs in: /projects/weilab/gohaina/logs/"
fi


