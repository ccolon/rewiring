#!/bin/bash
#
# Top-up for the cost-reduction (theta_T vs tau) study to fill cells the
# original launch_cost_reduction.sh missed.
#
# Gaps filled:
#   opB (n=50, realistic, aisi=0.05):
#       - homo tau in {4, 5, 6}        (3 cells; we have 0..3)
#       - full-mode reference           (1 cell)
#       - hetero (m=2,std=2), (m=3,std=3)  (2 cells)
#       = 6 cells.
#
#   opC (n=100, b=hom 0.9, no AiSi):
#       - homo tau in {0, 1, 2}        (3 cells; we have 3..6)
#       = 3 cells.
#
# Total: 9 jobs, BASE_SEED=1110 (clear of prior 1100 used by main launcher).
#
# Sample budget per cell: TECH_PER_JOB=10 x INITS_PER_TECH=30 = 300 trials,
# matching the original launch.
#
# Usage:
#     bash launch_cost_reduction_topup.sh                 # 9 jobs, BASE_SEED=1110
#     bash launch_cost_reduction_topup.sh 1110 --dry-run

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_cost"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED=${1:-1110}
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

    count=$((count + 1))
    local out="${OUTPUT_DIR}/cost_n${n}_${tag}_seed${BASE_SEED}.csv"
    local job="costtu_${tag}_s${BASE_SEED}"
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
        echo "[$count] n=${n} mode=${mode} tau=(${tier_mean},${tier_std}) aisi=${aisi}  ${tag}"
    else
        eval "$cmd"
        echo "[$count] queued: ${tag}"
    fi
}

# =============================================================================
# opB top-up: n=50, realistic (uniform a/b/z), aisi=0.05
#   homo tau in {4, 5, 6} + full + hetero (m2s2, m3s3)
# =============================================================================
echo "=== opB top-up: n=50, realistic, aisi=0.05 ==="
A_B="uniform:0.4:0.6"; B_B="uniform:0.9:1.1"; Z_B="uniform:0.9:1.1"
for tau in 4 5 6; do
    submit 50 4 1 0.05 0.0 "${A_B}" "${B_B}" "${Z_B}" \
           limited ${tau} 0 "opB_homo_t${tau}"
done
submit 50 4 1 0.05 0.0 "${A_B}" "${B_B}" "${Z_B}" full 0 0 "opB_full"
submit 50 4 1 0.05 0.0 "${A_B}" "${B_B}" "${Z_B}" limited 2 2 "opB_hetero_m2s2"
submit 50 4 1 0.05 0.0 "${A_B}" "${B_B}" "${Z_B}" limited 3 3 "opB_hetero_m3s3"

# =============================================================================
# opC top-up: n=100, b=hom 0.9, fully hom, no AiSi, no sigma_w
#   homo tau in {0, 1, 2}
# =============================================================================
echo "=== opC top-up: n=100, b=hom 0.9, no AiSi ==="
A_C="homogeneous:0.5"; B_C="homogeneous:0.9"; Z_C="homogeneous:1.0"
for tau in 0 1 2; do
    submit 100 4 1 0.0 0.0 "${A_C}" "${B_C}" "${Z_C}" \
           limited ${tau} 0 "opC_homo_t${tau}"
done

echo
echo "Done: $count jobs queued (BASE_SEED=${BASE_SEED}, "
echo "      TECH_PER_JOB=${TECH_PER_JOB}, INITS_PER_TECH=${INITS_PER_TECH}, "
echo "      total trials per cell = $((TECH_PER_JOB * INITS_PER_TECH)), "
echo "      time=${TIME_LIMIT})"
