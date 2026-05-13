#!/bin/bash
#
# Cost-reduction study with the OPTION-B "static gap" query enabled, on the
# THREE operating points used in panels (a, b) of
# results/visibility/figure_visibility.png:
#
#   R1 -- Full heterogeneity         : a~U[0.4,0.6], b~U[0.9,1.1],
#                                      z~U[0.9,1.1], aisi=0.05, sigma_w=0.05
#   R2 -- Firm-level heterogeneity   : same a/b/z, aisi=0,    sigma_w=0
#   R3 -- Homogeneous DRS            : a=hom 0.5, b=hom 0.9, z=hom 1.0,
#                                      aisi=0,    sigma_w=0
#
# All at n=100, cc=4, ms=1, mode=limited.  Three tier cells per series:
#
#   bar_tau = 0:  (tier_mean=0, tier_std=0)  -- homogeneous tau=0 (matches
#                                               the leftmost point spliced
#                                               into panel (b) at tau=0)
#   bar_tau = 1:  (tier_mean=1, tier_std=1)  -- lognormal hetero (CV=1)
#   bar_tau = 2:  (tier_mean=2, tier_std=2)  -- lognormal hetero (CV=1)
#
# Static-gap definition (in cost_reduction_study.py):
#   p_best_static_i = min P[i] over candidate supplier sets reachable by up
#     to static_max_swaps simultaneous swaps. Default static_max_swaps =
#     min(c, cc) = 4, so the enumeration covers EVERY size-c subset of
#     firm i's pool ("full visibility + unlimited swap").
#
# Cell grid: 3 series x 3 tier cells = 9 cells.
# Sample budget per cell: TECH_PER_JOB=3 x INITS_PER_TECH=40 = 120 trials.
# Output dir: results_cost_static/
#
# BASE_SEED default = 1500 (1400 was the previous ms-reachable-only schema).
#
# Usage:
#     bash launch_cost_static.sh                 # 9 jobs, BASE_SEED=1500
#     bash launch_cost_static.sh 1500 --dry-run

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_cost_static"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED=${1:-1500}
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

# Shared across all three series.
N=100
CC=4
MS=1

count=0
submit() {
    # aisi  sw  a_cfg  b_cfg  z_cfg  tier_mean  tier_std  tag
    local aisi=$1 sw=$2 a_cfg=$3 b_cfg=$4 z_cfg=$5
    local tier_mean=$6 tier_std=$7 tag=$8

    count=$((count + 1))
    local out="${OUTPUT_DIR}/coststat_n${N}_${tag}_seed${BASE_SEED}.csv"
    local job="coststat_${tag}_s${BASE_SEED}"
    local cmd="sbatch \
        --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
        --job-name=${job} \
        --output=${SLURM_LOG_DIR}/${job}.%j.out \
        --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/scripts/cost_reduction_study.py \
    --n ${N} --cc ${CC} --max_swaps ${MS} \
    --aisi_spread ${aisi} --sigma_w ${sw} \
    --a_config ${a_cfg} --b_config ${b_cfg} --z_config ${z_cfg} \
    --mode limited --tier_mean ${tier_mean} --tier_std ${tier_std} \
    --tech_per_job ${TECH_PER_JOB} --inits_per_tech ${INITS_PER_TECH} \
    --nb_rounds ${NB_ROUNDS} \
    --base_seed ${BASE_SEED} --output ${out}'\""

    if $DRY_RUN; then
        echo "[$count] tau=(${tier_mean},${tier_std})  tag=${tag}"
    else
        eval "$cmd"
        echo "[$count] queued: ${tag}"
    fi
}

submit_series() {
    # aisi  sw  a_cfg  b_cfg  z_cfg  series_tag
    local aisi=$1 sw=$2 a_cfg=$3 b_cfg=$4 z_cfg=$5 stag=$6
    for m in 0 1 2; do
        submit "${aisi}" "${sw}" "${a_cfg}" "${b_cfg}" "${z_cfg}" \
               ${m} ${m} "${stag}_m${m}s${m}"
    done
}

# =============================================================================
# R1: Full heterogeneity  (a~U[0.4,0.6], b~U[0.9,1.1], z~U[0.9,1.1],
#                          aisi=0.05, sigma_w=0.05)
# =============================================================================
echo "=== R1: Full heterogeneity ==="
submit_series 0.05 0.05 \
    uniform:0.4:0.6 uniform:0.9:1.1 uniform:0.9:1.1 "R1"

# =============================================================================
# R2: Firm-level heterogeneity only  (same a/b/z, aisi=0, sigma_w=0)
# =============================================================================
echo "=== R2: Firm-level heterogeneity only ==="
submit_series 0.0 0.0 \
    uniform:0.4:0.6 uniform:0.9:1.1 uniform:0.9:1.1 "R2"

# =============================================================================
# R3: Homogeneous DRS  (a=hom 0.5, b=hom 0.9, z=hom 1.0, aisi=0, sigma_w=0)
# =============================================================================
echo "=== R3: Homogeneous DRS ==="
submit_series 0.0 0.0 \
    homogeneous:0.5 homogeneous:0.9 homogeneous:1.0 "R3"

echo
echo "Done: $count jobs queued (BASE_SEED=${BASE_SEED}, "
echo "      TECH_PER_JOB=${TECH_PER_JOB}, INITS_PER_TECH=${INITS_PER_TECH}, "
echo "      total trials per cell = $((TECH_PER_JOB * INITS_PER_TECH)), "
echo "      MEM=${MEM}, time=${TIME_LIMIT})"
