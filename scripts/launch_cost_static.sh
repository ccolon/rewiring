#!/bin/bash
#
# Cost-reduction study with the OPTION-B "static gap" query enabled.
#
# Each trial now also computes, at its FINAL state:
#   - p_current_i  = P[i] under the full GE at the final supplier configuration
#   - p_best_static_i = min over ms-reachable single-firm swaps of P[i],
#                       evaluated as a full-GE counterfactual with all other
#                       firms held fixed
#   - theta_static_i  = p_current_i - p_best_static_i
#   - theta_static    = (sum p_current - sum p_best_static) / sum p_current
#
# In addition, per-firm rewire-event counts in three windows are written
# (swaps_first_R, swaps_last_R, swaps_in_cycle) using trace=True. R defaults
# to 10 (R_WINDOW in cost_reduction_study.py).
#
# Cell grid (3 op-points x 5 tau cells = 15 cells):
#   tau scan over: homo {0, 1, 2}, hetero {m=2 s=2, m=3 s=3}.
# Op-points (same as launch_cost_reduction.sh):
#   (A) old-paper baseline:  n=50, hom a/b/z, b=0.9, aisi=0, sw=0
#   (B) realistic:           n=50, uniform a/b/z, aisi=0.05, sw=0
#   (C) larger-n:            n=100, hom a/b/z, b=0.9, aisi=0, sw=0
#
# Sample budget per cell: TECH_PER_JOB=3 x INITS_PER_TECH=40 = 120 trials.
# This is smaller than the 300/cell of launch_cost_reduction.sh because the
# static-gap query is expensive (1 full-GE solve per candidate swap per firm
# at the final state).
#
# Output dir:  results_cost_static/   (separate from results_cost to avoid
# confusion with the older runs that lacked the static-gap columns).
#
# Usage:
#     bash launch_cost_static.sh                  # 15 jobs, BASE_SEED=1400
#     bash launch_cost_static.sh 1400 --dry-run

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_cost_static"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED=${1:-1400}
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

count=0
submit() {
    # n  cc  ms  aisi  sw  a_cfg  b_cfg  z_cfg  mode  tier_mean  tier_std  tag
    local n=$1 cc=$2 ms=$3 aisi=$4 sw=$5
    local a_cfg=$6 b_cfg=$7 z_cfg=$8
    local mode=$9 tier_mean=${10} tier_std=${11} tag=${12}

    count=$((count + 1))
    local out="${OUTPUT_DIR}/coststat_n${n}_${tag}_seed${BASE_SEED}.csv"
    local job="coststat_${tag}_s${BASE_SEED}"
    local cmd="sbatch \
        --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
        --job-name=${job} \
        --output=${SLURM_LOG_DIR}/${job}.%j.out \
        --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/scripts/cost_reduction_study.py \
    --n ${n} --cc ${cc} --max_swaps ${ms} \
    --aisi_spread ${aisi} --sigma_w ${sw} \
    --a_config ${a_cfg} --b_config ${b_cfg} --z_config ${z_cfg} \
    --mode ${mode} --tier_mean ${tier_mean} --tier_std ${tier_std} \
    --tech_per_job ${TECH_PER_JOB} --inits_per_tech ${INITS_PER_TECH} \
    --nb_rounds ${NB_ROUNDS} \
    --base_seed ${BASE_SEED} --output ${out}'\""

    if $DRY_RUN; then
        echo "[$count] n=${n} mode=${mode} tau=(${tier_mean},${tier_std}) tag=${tag}"
    else
        eval "$cmd"
        echo "[$count] queued: ${tag}"
    fi
}

submit_op_point() {
    # n  cc  ms  aisi  sw  a_cfg  b_cfg  z_cfg  op_tag
    local n=$1 cc=$2 ms=$3 aisi=$4 sw=$5
    local a_cfg=$6 b_cfg=$7 z_cfg=$8 op_tag=$9

    # Homogeneous tau scan (3 cells: tau in {0, 1, 2}).
    for tau in 0 1 2; do
        submit ${n} ${cc} ${ms} ${aisi} ${sw} "${a_cfg}" "${b_cfg}" "${z_cfg}" \
               limited ${tau} 0 "${op_tag}_homo_t${tau}"
    done
    # Heterogeneous tau (mean=std => coefficient of variation = 1).
    submit ${n} ${cc} ${ms} ${aisi} ${sw} "${a_cfg}" "${b_cfg}" "${z_cfg}" \
           limited 2 2 "${op_tag}_hetero_m2s2"
    submit ${n} ${cc} ${ms} ${aisi} ${sw} "${a_cfg}" "${b_cfg}" "${z_cfg}" \
           limited 3 3 "${op_tag}_hetero_m3s3"
}

# =============================================================================
# Op-point A: old-paper baseline (n=50, fully hom, b=0.9, no AiSi, no sw)
# =============================================================================
echo "=== Op-point A: old-paper baseline (n=50, b=hom 0.9, fully hom) ==="
submit_op_point 50 4 1 0.0 0.0 \
    homogeneous:0.5 homogeneous:0.9 homogeneous:1.0 "opA"

# =============================================================================
# Op-point B: realistic (n=50, uniform a/b/z, aisi=0.05)
# =============================================================================
echo "=== Op-point B: realistic (n=50, uniform a/b/z, aisi=0.05) ==="
submit_op_point 50 4 1 0.05 0.0 \
    uniform:0.4:0.6 uniform:0.9:1.1 uniform:0.9:1.1 "opB"

# =============================================================================
# Op-point C: larger-n (n=100, hom a/b/z, b=0.9)
# =============================================================================
echo "=== Op-point C: larger-n (n=100, b=hom 0.9, fully hom) ==="
submit_op_point 100 4 1 0.0 0.0 \
    homogeneous:0.5 homogeneous:0.9 homogeneous:1.0 "opC"

echo
echo "Done: $count jobs queued (BASE_SEED=${BASE_SEED}, "
echo "      TECH_PER_JOB=${TECH_PER_JOB}, INITS_PER_TECH=${INITS_PER_TECH}, "
echo "      total trials per cell = $((TECH_PER_JOB * INITS_PER_TECH)), "
echo "      MEM=${MEM}, time=${TIME_LIMIT})"
