#!/bin/bash
#
# Sync/async rewiring campaign (manuscript Campaign 2, app:sync_async).
#
# Single parameter point (P2 = CRS, kappa=1, Delta_A=0.05). For each of 50
# (W_bar, S^(0)) pairs, run 1 synchronous + 50 asynchronous trajectories =
# 51 runs/pair = 2,550 runs total. At n=50 each run is fast (~1-2 s); one
# SLURM job covers it. The user can split with --n_pairs into batches if
# more parallelism is desired (not the default).
#
# Output dir: results/sync_async/ (gitignored). The final figure sync_async.png
# lands here too once campaigns/sync_async/plot.py is invoked.
# BASE_SEED default = 1800.
#
# Usage:
#     bash campaigns/sync_async/launch.sh                 # 1 job, BASE_SEED=1800
#     bash campaigns/sync_async/launch.sh 1800 --dry-run

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results/sync_async"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED=${1:-1800}
DRY_RUN=false
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        *) shift ;;
    esac
done

if ! $DRY_RUN; then
    mkdir -p "$OUTPUT_DIR" "$SLURM_LOG_DIR"
fi

TIME_LIMIT="48:00:00"
MEM="2G"
NB_ROUNDS=200
N_PAIRS=50
N_ASYNC=50

out="${OUTPUT_DIR}/sync_async_seed${BASE_SEED}.csv"
job="sync_async_s${BASE_SEED}"
cmd="sbatch \
    --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
    --job-name=${job} \
    --output=${SLURM_LOG_DIR}/${job}.%j.out \
    --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/campaigns/sync_async/study.py \
    --n_pairs ${N_PAIRS} --n_async ${N_ASYNC} \
    --nb_rounds ${NB_ROUNDS} \
    --base_seed ${BASE_SEED} --output ${out}'\""

if $DRY_RUN; then
    echo "[1] sync_async seed=${BASE_SEED}  ->  ${out##*/}"
else
    eval "$cmd"
    echo "[1] queued: sync_async (BASE_SEED=${BASE_SEED})"
fi

echo
echo "Done: 1 job queued (BASE_SEED=${BASE_SEED}, "
echo "      N_PAIRS=${N_PAIRS}, N_ASYNC=${N_ASYNC}, "
echo "      total runs = $((N_PAIRS * (1 + N_ASYNC))), "
echo "      MEM=${MEM}, time=${TIME_LIMIT})"
