#!/bin/bash
#
# Static-gap study extension: the (tier_mean, tier_std) = (1, 1) cell that
# was not in launch_cost_static.sh (which covered m=2 s=2 and m=3 s=3 only).
# Adds the leftmost interior point on the panel-(c) bar_tau axis of
# results/visibility/plot_visibility.py.
#
# Cell: opC operating point
#     n=100, cc=4, ms=1, mode=limited
#     a=hom 0.5, b=hom 0.9, z=hom 1.0   (homogeneous DRS, matches R3)
#     aisi=0, sigma_w=0
#     tier_mean=1, tier_std=1 (lognormal hetero, CV=1)
#
# Sample budget: TECH_PER_JOB=3 x INITS_PER_TECH=40 = 120 trials, matching
# the launch_cost_static.sh budget so coverage is comparable to the existing
# m=2 s=2 and m=3 s=3 cells.
#
# BASE_SEED=1410 is reserved here (launch_cost_static.sh used 1400).
#
# Usage:
#     bash launch_cost_static_tau1.sh                  # 1 job, BASE_SEED=1410
#     bash launch_cost_static_tau1.sh 1410 --dry-run

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_cost_static"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED=${1:-1410}
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
TECH_PER_JOB=${TECH_PER_JOB:-3}
INITS_PER_TECH=${INITS_PER_TECH:-40}

N=100
CC=4
MS=1
AISI=0.0
SW=0.0
A_CFG="homogeneous:0.5"
B_CFG="homogeneous:0.9"
Z_CFG="homogeneous:1.0"
TIER_MEAN=1
TIER_STD=1
TAG="opC_hetero_m1s1"

count=1
out="${OUTPUT_DIR}/coststat_n${N}_${TAG}_seed${BASE_SEED}.csv"
job="coststat_${TAG}_s${BASE_SEED}"
cmd="sbatch \
    --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
    --job-name=${job} \
    --output=${SLURM_LOG_DIR}/${job}.%j.out \
    --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/scripts/cost_reduction_study.py \
    --n ${N} --cc ${CC} --max_swaps ${MS} \
    --aisi_spread ${AISI} --sigma_w ${SW} \
    --a_config ${A_CFG} --b_config ${B_CFG} --z_config ${Z_CFG} \
    --mode limited --tier_mean ${TIER_MEAN} --tier_std ${TIER_STD} \
    --tech_per_job ${TECH_PER_JOB} --inits_per_tech ${INITS_PER_TECH} \
    --nb_rounds ${NB_ROUNDS} \
    --base_seed ${BASE_SEED} --output ${out}'\""

if $DRY_RUN; then
    echo "[$count] n=${N} mode=limited tau=(${TIER_MEAN},${TIER_STD})  tag=${TAG}"
else
    eval "$cmd"
    echo "[$count] queued: ${TAG} (BASE_SEED=${BASE_SEED})"
fi

echo
echo "Done: $count job queued (BASE_SEED=${BASE_SEED}, "
echo "      TECH_PER_JOB=${TECH_PER_JOB}, INITS_PER_TECH=${INITS_PER_TECH}, "
echo "      total trials = $((TECH_PER_JOB * INITS_PER_TECH)), "
echo "      MEM=${MEM}, time=${TIME_LIMIT})"
