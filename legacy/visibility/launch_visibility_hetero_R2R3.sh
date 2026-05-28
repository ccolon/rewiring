#!/bin/bash
#
# Visibility study, heterogeneous-tau cells for the two operating points
# that the original `launch_visibility.sh` did NOT cover in hetero mode:
#
#   R2 -- "Firm-level heterogeneity only"  : a~U[0.4,0.6], b~U[0.9,1.1],
#                                            z~U[0.9,1.1], aisi=0, sigma_w=0
#   R3 -- "Homogenous DRS"                  : a=hom 0.5, b=hom 0.9, z=hom 1.0,
#                                            aisi=0, sigma_w=0
#
# Both at n=100, cc=4, ms=1, mode=limited. tau_mode=hetero with
# (mean = std) in {0, 1, 2, 3, 4, 5, 6} (mean=0 collapses to homogeneous
# tau=0, kept for the leftmost panel-(b) endpoint to match R1).
#
# 2 cells x NUM_BATCHES jobs.  Default: TECH_PER_JOB=50, NUM_BATCHES=2 ->
# 100 tech matrices per cell, 7 tau levels each = 700 sims per job. Comparable
# to the existing R1 hetero coverage (~107 trials per (tau_mean, tau_std)).
#
# Hetero distribution: Poisson(lambda = bar_tau). BASE_SEED bumped to 960+
# so the new Poisson CSVs are distinguishable from prior lognormal ones.
#
# Usage:
#     bash launch_visibility_hetero_R2R3.sh                 # 4 jobs, BASE_SEED 960-961
#     bash launch_visibility_hetero_R2R3.sh 960 --dry-run
#     bash launch_visibility_hetero_R2R3.sh 962 --num-batches 4  # +200 more tech/cell

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_sweep"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED_START=${1:-960}
DRY_RUN=false
NUM_BATCHES=2
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
MEM="2G"
NB_ROUNDS=200
TECH_PER_JOB=${TECH_PER_JOB:-50}
TAU_VALUES="0,1,2,3,4,5,6"
N=100
CC=4

count=0
submit_cell() {
    # tag  a_cfg  b_cfg  z_cfg
    local tag=$1 a_cfg=$2 b_cfg=$3 z_cfg=$4

    for (( i=0; i<NUM_BATCHES; i++ )); do
        local seed=$(( BASE_SEED_START + i ))
        count=$((count + 1))
        local out="${OUTPUT_DIR}/visibility_n${N}_${tag}_hetero_seed${seed}.csv"
        local job="vis_${tag}_hetero_s${seed}"
        local cmd="sbatch \
            --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
            --job-name=${job} \
            --output=${SLURM_LOG_DIR}/${job}.%j.out \
            --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/scripts/visibility_study.py \
    --n ${N} --tech_per_job ${TECH_PER_JOB} --nb_rounds ${NB_ROUNDS} \
    --cc ${CC} --max_swaps 1 \
    --aisi_spread 0.0 --sigma_w 0.0 \
    --a_config ${a_cfg} --b_config ${b_cfg} --z_config ${z_cfg} \
    --tau_values ${TAU_VALUES} --tau_mode hetero \
    --tier_dist poisson \
    --base_seed ${seed} --output ${out}'\""

        if $DRY_RUN; then
            echo "[$count] ${tag} hetero  seed=${seed}  ->  ${out##*/}"
        else
            eval "$cmd"
            echo "[$count] queued: ${tag} hetero (seed=${seed})"
        fi
    done
}

echo "=== R2: firm-level heterogeneity only (uniform a/b/z, aisi=0, sw=0) ==="
submit_cell "R2_uniformAbz" \
            "uniform:0.4:0.6" "uniform:0.9:1.1" "uniform:0.9:1.1"

echo "=== R3: fully homogeneous DRS (hom 0.5 / 0.9 / 1.0) ==="
submit_cell "R3_homDRS" \
            "homogeneous:0.5" "homogeneous:0.9" "homogeneous:1.0"

echo
echo "Done: $count jobs queued (BASE_SEED in [${BASE_SEED_START}, $((BASE_SEED_START + NUM_BATCHES - 1))], "
echo "      TECH_PER_JOB=${TECH_PER_JOB}, total tech/cell = $((TECH_PER_JOB * NUM_BATCHES)), "
echo "      MEM=${MEM}, time=${TIME_LIMIT})"
