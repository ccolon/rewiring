#!/bin/bash
#
# Limited-visibility study (Section "limited visibility leads to endless
# rewiring"). Operating point: AiSi-driven trap (aisi=0.05, sigma_w=0.05),
# ms=1, mode=limited; uniform a/b/z. For each cell (n, cc, tau_mode), a
# single SLURM job sweeps tau in {0..6} on TECH_PER_JOB independent tech
# matrices. NUM_BATCHES separate jobs use BASE_SEED offsets to add more
# independent tech matrices.
#
# 4 cell types (n=100 cc=4 homo/hetero, n=20 cc=4 homo, n=60 cc=4 homo,
# n=20 cc=2 homo, n=60 cc=2 homo) -> 6 cells.
#
# Default first-batch budget:
#     TECH_PER_JOB=2 x NUM_BATCHES=5 = 10 tech matrices per cell.
#     6 cells x 5 batches = 30 jobs.
#
# To top up to 100 tech matrices per cell, re-launch later with a higher
# BASE_SEED_START and NUM_BATCHES=45 (or whatever you need).
#
# Hetero distribution: Poisson(lambda = bar_tau). BASE_SEED bumped to 950+
# so the new Poisson CSVs don't collide with the old lognormal ones.
#
# Usage:
#     bash launch_visibility.sh                             # 30 jobs, BASE_SEED 950-954
#     bash launch_visibility.sh 950 --dry-run               # print only
#     bash launch_visibility.sh 955 --num-batches 45        # top-up

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_sweep"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED_START=${1:-950}
DRY_RUN=false
NUM_BATCHES=5
shift || true   # consume the BASE_SEED_START arg if present
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
TECH_PER_JOB=2
TAU_VALUES="0,1,2,3,4,5,6"

# Operating point: trap-phase, mild a/b/z heterogeneity.
A_CONFIG="uniform:0.4:0.6"
B_CONFIG="uniform:0.9:1.1"
Z_CONFIG="uniform:0.9:1.1"
AISI=0.05
SW=0.05

# ----- one sbatch submission --------------------------------------------------
count=0
submit_cell() {
    # n cc tau_mode tag
    local n=$1 cc=$2 tau_mode=$3 tag=$4

    for (( i=0; i<NUM_BATCHES; i++ )); do
        local seed=$(( BASE_SEED_START + i ))
        count=$((count + 1))
        local out="${OUTPUT_DIR}/visibility_n${n}_${tag}_seed${seed}.csv"
        local job="vis_${tag}_s${seed}"
        local cmd="sbatch \
            --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
            --job-name=${job} \
            --output=${SLURM_LOG_DIR}/${job}.%j.out \
            --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/scripts/visibility_study.py \
    --n ${n} --tech_per_job ${TECH_PER_JOB} --nb_rounds ${NB_ROUNDS} \
    --cc ${cc} --max_swaps 1 \
    --aisi_spread ${AISI} --sigma_w ${SW} \
    --a_config ${A_CONFIG} --b_config ${B_CONFIG} --z_config ${Z_CONFIG} \
    --tau_values ${TAU_VALUES} --tau_mode ${tau_mode} \
    --tier_dist poisson \
    --base_seed ${seed} --output ${out}'\""

        if $DRY_RUN; then
            echo "[$count] n=${n} cc=${cc} tau_mode=${tau_mode}  seed=${seed}  ->  ${tag}"
        else
            eval "$cmd"
            echo "[$count] queued: ${tag} (seed=${seed})"
        fi
    done
}

# 6 cells. Each is one SLURM job per BASE_SEED offset.
echo "=== n=100 cc=4 homo ===";   submit_cell 100 4 homo   "homoCc4_n100"
echo "=== n=100 cc=4 hetero ==="; submit_cell 100 4 hetero "heteroCc4_n100"
echo "=== n=20  cc=4 homo ==="; submit_cell 20  4 homo "homoCc4_n20"
echo "=== n=60  cc=4 homo ==="; submit_cell 60  4 homo "homoCc4_n60"
echo "=== n=20  cc=2 homo ==="; submit_cell 20  2 homo "homoCc2_n20"
echo "=== n=60  cc=2 homo ==="; submit_cell 60  2 homo "homoCc2_n60"

echo
echo "Done: $count jobs queued (BASE_SEED in [${BASE_SEED_START}, $((BASE_SEED_START + NUM_BATCHES - 1))], "
echo "      TECH_PER_JOB=${TECH_PER_JOB}, total tech matrices per cell = $((TECH_PER_JOB * NUM_BATCHES)), time=${TIME_LIMIT})"
