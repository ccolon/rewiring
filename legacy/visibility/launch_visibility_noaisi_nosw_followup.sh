#!/bin/bash
#
# Followup to launch_visibility_noaisi_nosw.sh: complete the n-scan at the
# AiSi-off, sigma_w-off operating point, in two blocks.
#
# Block 1 (realistic operating point, no AiSi, no sw):
#     n in {20, 100, 200}, cc=4, ms=1, homo tau in {0..6}
#     a ~ U[0.4,0.6], b ~ U[0.9,1.1], z ~ U[0.9,1.1]
#     aisi = 0, sigma_w = 0
#   (n=60 already done by launch_visibility_noaisi_nosw.sh -- skipped here.)
#
# Block 2 (fully homogeneous):
#     n in {20, 60, 100, 200}, cc=4, ms=1, homo tau in {0..6}
#     a = hom 0.5, b = hom 0.9, z = hom 1.0
#     aisi = 0, sigma_w = 0
#
# Both at 100 tech matrices per cell (TECH_PER_JOB=50, NUM_BATCHES=2). 7 cells
# x 2 batches = 14 jobs.
#
# Per-job estimate: 50 tech * 7 tau ~= 350 sims; ranges from ~12 min (n=20)
# to ~6 h (n=200). All comfortably under 48 h.
#
# Usage:
#     bash launch_visibility_noaisi_nosw_followup.sh                       # 14 jobs, BASE_SEED 920-921
#     bash launch_visibility_noaisi_nosw_followup.sh 920 --dry-run         # print only
#     bash launch_visibility_noaisi_nosw_followup.sh 922 --num-batches 4   # +200 tech matrices/cell

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_sweep"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED_START=${1:-920}
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
MEM="6G"
NB_ROUNDS=200
TECH_PER_JOB=50
TAU_VALUES="0,1,2,3,4,5,6"
CC=4
AISI=0.0
SW=0.0

count=0
submit_cell() {
    # n  a_cfg  b_cfg  z_cfg  tag
    local n=$1 a_cfg=$2 b_cfg=$3 z_cfg=$4 tag=$5

    for (( i=0; i<NUM_BATCHES; i++ )); do
        local seed=$(( BASE_SEED_START + i ))
        count=$((count + 1))
        local out="${OUTPUT_DIR}/visibility_n${n}_${tag}_seed${seed}.csv"
        local job="visNH_${tag}_s${seed}"
        local cmd="sbatch \
            --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
            --job-name=${job} \
            --output=${SLURM_LOG_DIR}/${job}.%j.out \
            --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/scripts/visibility_study.py \
    --n ${n} --tech_per_job ${TECH_PER_JOB} --nb_rounds ${NB_ROUNDS} \
    --cc ${CC} --max_swaps 1 \
    --aisi_spread ${AISI} --sigma_w ${SW} \
    --a_config ${a_cfg} --b_config ${b_cfg} --z_config ${z_cfg} \
    --tau_values ${TAU_VALUES} --tau_mode homo \
    --base_seed ${seed} --output ${out}'\""

        if $DRY_RUN; then
            echo "[$count] n=${n} a=${a_cfg} b=${b_cfg} z=${z_cfg}  seed=${seed}  ->  ${tag}"
        else
            eval "$cmd"
            echo "[$count] queued: ${tag} (seed=${seed})"
        fi
    done
}

# =============================================================================
# Block 1: realistic (uniform a, b, z) -- top up for n in {20, 100, 200}
# =============================================================================
A_REAL="uniform:0.4:0.6"
B_REAL="uniform:0.9:1.1"
Z_REAL="uniform:0.9:1.1"

echo "=== Block 1: realistic operating point, no AiSi, no sw ==="
for n in 20 100 200; do
    submit_cell ${n} "${A_REAL}" "${B_REAL}" "${Z_REAL}" "visNoHeter_realistic_n${n}"
done

# =============================================================================
# Block 2: fully homogeneous a, b, z
# =============================================================================
A_HOM="homogeneous:0.5"
B_HOM="homogeneous:0.9"
Z_HOM="homogeneous:1.0"

echo "=== Block 2: fully homogeneous (a=0.5, b=0.9, z=1.0), no AiSi, no sw ==="
for n in 20 60 100 200; do
    submit_cell ${n} "${A_HOM}" "${B_HOM}" "${Z_HOM}" "visNoHeter_fullhom_n${n}"
done

echo
echo "Done: $count jobs queued (BASE_SEED in [${BASE_SEED_START}, $((BASE_SEED_START + NUM_BATCHES - 1))], "
echo "      TECH_PER_JOB=${TECH_PER_JOB}, total tech matrices per cell = $((TECH_PER_JOB * NUM_BATCHES)), time=${TIME_LIMIT})"
