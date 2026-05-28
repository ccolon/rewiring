#!/bin/bash
#
# Cost-reduction (theta_T vs tau) study.
#
# Three operating points x ten cells per op-point:
#   homo tau scan {0,1,2,3,4,5,6}  (7 cells)
#   tau = infinity reference: mode=full                  (1 cell)
#   hetero tau:  (mean, std) in {(2,2), (3,3)}            (2 cells)
#
# Op-points:
#   (A) old-paper baseline:  n=50, b=hom 0.9, fully hom, aisi=0, sigma_w=0
#   (B) realistic:           n=50, uniform a/b/z (0.4-0.6 / 0.9-1.1 / 0.9-1.1), aisi=0.05
#   (C) larger-n:            n=100, same as (A)
#
# Sample budget per cell: TECH_PER_JOB=10 tech matrices x INITS_PER_TECH=30
# random initial networks = 300 trials/cell. Per-firm rows = 300 * n.
#
# Per-job runtime estimate at n=50, mode=limited: ~2 min/trial (varies by tau).
# For 300 trials / job at n=50: ~10 hours/job. Comfortable under 48h.
# At n=100: ~3-5x longer; split into 3 batches.
#
# 10 cells x 3 op-points = 30 jobs (n=50) + n=100 with 3 batches = 30 extras.
# Simpler: split each n=100 cell into 3 sub-jobs by tech_idx.
#
# Usage:
#     bash launch_cost_reduction.sh                      # 30 jobs, BASE_SEED 1100-1102
#     bash launch_cost_reduction.sh 1100 --dry-run
#     bash launch_cost_reduction.sh 1110 --num-batches 2

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_cost"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED_START=${1:-1100}
DRY_RUN=false
NUM_BATCHES=1
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --num-batches) NUM_BATCHES=$2; shift 2 ;;
        *) shift ;;
    esac
done

if ! $DRY_RUN; then
    mkdir -p "$OUTPUT_DIR" "$SLURM_LOG_DIR"
fi

TIME_LIMIT="48:00:00"
MEM="6G"
NB_ROUNDS=200
TECH_PER_JOB=10
INITS_PER_TECH=30

count=0
submit() {
    # n  cc  ms  aisi  sw  a_cfg  b_cfg  z_cfg  mode  tier_mean  tier_std  tag
    local n=$1 cc=$2 ms=$3 aisi=$4 sw=$5
    local a_cfg=$6 b_cfg=$7 z_cfg=$8
    local mode=$9 tier_mean=${10} tier_std=${11} tag=${12}

    for (( i=0; i<NUM_BATCHES; i++ )); do
        local seed=$(( BASE_SEED_START + i ))
        count=$((count + 1))
        local out="${OUTPUT_DIR}/cost_n${n}_${tag}_seed${seed}.csv"
        local job="cost_${tag}_s${seed}"
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
    --base_seed ${seed} --output ${out}'\""

        if $DRY_RUN; then
            echo "[$count] n=${n} mode=${mode} tau=(${tier_mean},${tier_std}) seed=${seed}  ->  ${tag}"
        else
            eval "$cmd"
            echo "[$count] queued: ${tag} (seed=${seed})"
        fi
    done
}

submit_op_point() {
    # n  cc  ms  aisi  sw  a_cfg  b_cfg  z_cfg  op_tag
    local n=$1 cc=$2 ms=$3 aisi=$4 sw=$5
    local a_cfg=$6 b_cfg=$7 z_cfg=$8 op_tag=$9

    # Homogeneous tau scan.
    for tau in 0 1 2 3 4 5 6; do
        submit ${n} ${cc} ${ms} ${aisi} ${sw} "${a_cfg}" "${b_cfg}" "${z_cfg}" \
               limited ${tau} 0 "${op_tag}_homo_t${tau}"
    done
    # Full mode reference (tau = infinity).
    submit ${n} ${cc} ${ms} ${aisi} ${sw} "${a_cfg}" "${b_cfg}" "${z_cfg}" \
           full 0 0 "${op_tag}_full"
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
# Op-point C: larger-n version of A (n=100, b=hom 0.9, fully hom)
# =============================================================================
echo "=== Op-point C: larger-n (n=100, b=hom 0.9, fully hom) ==="
submit_op_point 100 4 1 0.0 0.0 \
    homogeneous:0.5 homogeneous:0.9 homogeneous:1.0 "opC"

echo
echo "Done: $count jobs queued (BASE_SEED in [${BASE_SEED_START}, $((BASE_SEED_START + NUM_BATCHES - 1))], "
echo "      TECH_PER_JOB=${TECH_PER_JOB}, INITS_PER_TECH=${INITS_PER_TECH}, "
echo "      total trials per cell = $((TECH_PER_JOB * INITS_PER_TECH * NUM_BATCHES)), "
echo "      time=${TIME_LIMIT})"
