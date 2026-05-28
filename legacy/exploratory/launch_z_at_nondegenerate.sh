#!/bin/bash
#
# z-heterogeneity at a non-degenerate baseline.
#
# Closes the gap left by the n=30 z test, which was run at b = 1.0 (degenerate
# corner: no rewires happen, "diversity" = 1.0 in dif_init is an artefact).
# Here we test z heterogeneity against a non-degenerate baseline by setting
# b = hom(0.9) -- DRS, where the model rewires and converges (verified by
# tests/test_homogeneous_corner.py).
#
# Operating point:
#     n=100, cc=4, ms=1, aisi=0, sigma_w=0
#     a = hom(0.5),  b = hom(0.9),  z varied
# Sweep: z_config in
#     hom 1.0           (z_spread = 0,   baseline)
#     uniform 0.95-1.05 (z_spread = 0.1, mild)
#     uniform 0.875-1.125 (z_spread = 0.25, moderate)
#     uniform 0.75-1.25 (z_spread = 0.5, large)
#
# Sample budget:
#     TECH_PER_JOB=5, NUM_BATCHES=5, N_TRIALS=30
#     => 25 tech matrices per cell, 30 trials per (tech, series), 2 series
#     => 1500 sims per cell, bootstrap CI on diversity ~0.04 per cell.
# 4 cells x 5 batches = 20 jobs. Per-job estimate ~13 h at n=100, mode=full,
# ms=1 -- comfortable under the 48 h time limit.
#
# Decisive answer: does z heterogeneity (in isolation) raise per-tech diversity
# above the b=hom(0.9), z=hom(1.0) baseline? If yes, the "z does not create
# diversity" claim is refuted. If no, it survives at non-degenerate b.
#
# Usage:
#     bash launch_z_at_nondegenerate.sh                    # 20 jobs, BASE_SEED 800-804
#     bash launch_z_at_nondegenerate.sh 805 --num-batches 5 --dry-run

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_sweep"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED_START=${1:-800}
DRY_RUN=false
NUM_BATCHES=5
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
TECH_PER_JOB=5
N_TRIALS=30

N=100
CC=4
MS=1
AISI=0.0
SW=0.0
A_CONFIG="homogeneous:0.5"
B_CONFIG="homogeneous:0.9"

count=0
submit_zcell() {
    # z_config_str  tag
    local z_cfg=$1 tag=$2

    for (( i=0; i<NUM_BATCHES; i++ )); do
        local seed=$(( BASE_SEED_START + i ))
        count=$((count + 1))
        local out="${OUTPUT_DIR}/zNonDeg_n${N}_${tag}_seed${seed}.csv"
        local job="zND_${tag}_s${seed}"
        local cmd="sbatch \
            --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
            --job-name=${job} \
            --output=${SLURM_LOG_DIR}/${job}.%j.out \
            --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/scripts/diversity_study.py \
    --n_min ${N} --n_max ${N} --n_tech ${TECH_PER_JOB} \
    --nb_rounds ${NB_ROUNDS} --n_trials ${N_TRIALS} \
    --b_config ${B_CONFIG} --a_config ${A_CONFIG} --z_config ${z_cfg} \
    --cc ${CC} --max_swaps ${MS} \
    --aisi_spread ${AISI} --sigma_w ${SW} \
    --base_seed ${seed} --output ${out}'\""

        if $DRY_RUN; then
            echo "[$count] z=${z_cfg}  seed=${seed}  ->  ${tag}"
        else
            eval "$cmd"
            echo "[$count] queued: ${tag} (seed=${seed})"
        fi
    done
}

# 4 z-spread values, baseline included.
echo "=== z_spread = 0     (hom 1.0,           baseline) ==="
submit_zcell "homogeneous:1.0"     "zspread0"
echo "=== z_spread = 0.10  (uniform 0.95-1.05, mild) ==="
submit_zcell "uniform:0.95:1.05"   "zspread010"
echo "=== z_spread = 0.25  (uniform 0.875-1.125, moderate) ==="
submit_zcell "uniform:0.875:1.125" "zspread025"
echo "=== z_spread = 0.50  (uniform 0.75-1.25,  large) ==="
submit_zcell "uniform:0.75:1.25"   "zspread050"

echo
echo "Done: $count jobs queued (BASE_SEED in [${BASE_SEED_START}, $((BASE_SEED_START + NUM_BATCHES - 1))], "
echo "      TECH_PER_JOB=${TECH_PER_JOB}, NUM_BATCHES=${NUM_BATCHES}, "
echo "      total tech matrices per cell = $((TECH_PER_JOB * NUM_BATCHES)), "
echo "      N_TRIALS=${N_TRIALS}, time=${TIME_LIMIT})"
