#!/bin/bash
#
# Static-gap study: heterogeneous-tau cells for opA and opB, the two
# operating points that are NOT yet represented in panel (c) of
# results/visibility/figure_visibility.png (only opC is plotted today).
#
# Produces the three hetero cells per op-point used by plot_visibility.py
# (after dropping the opA / opB plotting branch in):
#
#     (tier_mean, tier_std) in { (1, 1), (2, 2), (3, 3) }
#
# Op-points (same as launch_cost_static.sh):
#     opA  -- old-paper baseline:  n=50, hom a/b/z, b=0.9, aisi=0, sw=0
#     opB  -- full heterogeneity:  n=50, uniform a/b/z, aisi=0.05, sw=0
#
# Cell budget: TECH_PER_JOB=3 x INITS_PER_TECH=40 = 120 trials per cell,
# matching the existing opC cells. 2 op-points x 3 hetero cells = 6 jobs.
#
# Output dir: results_cost_static/  (drop into results/cost_gap/ on the
# local machine to populate panel (c) for opA and opB).
#
# BASE_SEED=1420 is reserved here (1400 was launch_cost_static.sh,
# 1410 was launch_cost_static_tau1.sh).
#
# Usage:
#     bash launch_cost_static_opAB.sh                  # 6 jobs, BASE_SEED=1420
#     bash launch_cost_static_opAB.sh 1420 --dry-run

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_cost_static"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED=${1:-1420}
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
    # n  cc  ms  aisi  sw  a_cfg  b_cfg  z_cfg  tier_mean  tier_std  tag
    local n=$1 cc=$2 ms=$3 aisi=$4 sw=$5
    local a_cfg=$6 b_cfg=$7 z_cfg=$8
    local tier_mean=$9 tier_std=${10} tag=${11}

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
    --mode limited --tier_mean ${tier_mean} --tier_std ${tier_std} \
    --tech_per_job ${TECH_PER_JOB} --inits_per_tech ${INITS_PER_TECH} \
    --nb_rounds ${NB_ROUNDS} \
    --base_seed ${BASE_SEED} --output ${out}'\""

    if $DRY_RUN; then
        echo "[$count] n=${n} tau=(${tier_mean},${tier_std})  tag=${tag}"
    else
        eval "$cmd"
        echo "[$count] queued: ${tag}"
    fi
}

submit_op_hetero() {
    # n  cc  ms  aisi  sw  a_cfg  b_cfg  z_cfg  op_tag
    local n=$1 cc=$2 ms=$3 aisi=$4 sw=$5
    local a_cfg=$6 b_cfg=$7 z_cfg=$8 op_tag=$9

    submit ${n} ${cc} ${ms} ${aisi} ${sw} "${a_cfg}" "${b_cfg}" "${z_cfg}" \
           1 1 "${op_tag}_hetero_m1s1"
    submit ${n} ${cc} ${ms} ${aisi} ${sw} "${a_cfg}" "${b_cfg}" "${z_cfg}" \
           2 2 "${op_tag}_hetero_m2s2"
    submit ${n} ${cc} ${ms} ${aisi} ${sw} "${a_cfg}" "${b_cfg}" "${z_cfg}" \
           3 3 "${op_tag}_hetero_m3s3"
}

# =============================================================================
# Op-point A: old-paper baseline (n=50, fully hom DRS, b=0.9)
# =============================================================================
echo "=== Op-point A: old-paper baseline (n=50, b=hom 0.9, fully hom) -- hetero tau ==="
submit_op_hetero 50 4 1 0.0 0.0 \
    homogeneous:0.5 homogeneous:0.9 homogeneous:1.0 "opA"

# =============================================================================
# Op-point B: realistic full heterogeneity (n=50, uniform a/b/z, aisi=0.05)
# =============================================================================
echo "=== Op-point B: realistic (n=50, uniform a/b/z, aisi=0.05) -- hetero tau ==="
submit_op_hetero 50 4 1 0.05 0.0 \
    uniform:0.4:0.6 uniform:0.9:1.1 uniform:0.9:1.1 "opB"

echo
echo "Done: $count jobs queued (BASE_SEED=${BASE_SEED}, "
echo "      TECH_PER_JOB=${TECH_PER_JOB}, INITS_PER_TECH=${INITS_PER_TECH}, "
echo "      total trials per cell = $((TECH_PER_JOB * INITS_PER_TECH)), "
echo "      MEM=${MEM}, time=${TIME_LIMIT})"
