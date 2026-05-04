#!/bin/bash
#
# Visibility study at the *old-paper-like* operating point: aisi = 0 and
# sigma_w = 0, so combinatorial and link-level heterogeneity are both off.
# Only the firm-level dispersions (uniform a, b, z) drive the cost landscape,
# matching as closely as our framework allows the smoothness of the old
# paper's setup.
#
# Hypothesis: removing the AiSi trap shifts the bifurcation tier *up*.
# Concretely we expect tau* > 1 at n=60 here, vs tau* ~ 1 in the
# AiSi=0.05 / sigma_w=0.05 regime.
#
# Single cell:
#     n=60, cc=4, ms=1, homo tau in {0..6}
#     a ~ U[0.4,0.6],  b ~ U[0.9,1.1],  z ~ U[0.9,1.1]
#     aisi = 0,  sigma_w = 0,  mode = limited
#
# Sample budget:
#     TECH_PER_JOB=50, NUM_BATCHES=2  ->  100 tech matrices, 2 jobs
#     7 tau values x 50 tech/job = 350 sims/job, ~3 h/job at n=60.
#
# Usage:
#     bash launch_visibility_noaisi_nosw.sh                       # 2 jobs, BASE_SEED 900-901
#     bash launch_visibility_noaisi_nosw.sh 900 --dry-run         # print only
#     bash launch_visibility_noaisi_nosw.sh 902 --num-batches 4   # +200 more tech

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_sweep"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED_START=${1:-900}
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

A_CONFIG="uniform:0.4:0.6"
B_CONFIG="uniform:0.9:1.1"
Z_CONFIG="uniform:0.9:1.1"
AISI=0.0
SW=0.0
N=60
CC=4
TAG="visNoHeter_homoCc4_n60"

count=0
for (( i=0; i<NUM_BATCHES; i++ )); do
    seed=$(( BASE_SEED_START + i ))
    count=$((count + 1))
    out="${OUTPUT_DIR}/visibility_n${N}_${TAG}_seed${seed}.csv"
    job="visNH_${TAG}_s${seed}"
    cmd="sbatch \
        --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
        --job-name=${job} \
        --output=${SLURM_LOG_DIR}/${job}.%j.out \
        --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/scripts/visibility_study.py \
    --n ${N} --tech_per_job ${TECH_PER_JOB} --nb_rounds ${NB_ROUNDS} \
    --cc ${CC} --max_swaps 1 \
    --aisi_spread ${AISI} --sigma_w ${SW} \
    --a_config ${A_CONFIG} --b_config ${B_CONFIG} --z_config ${Z_CONFIG} \
    --tau_values ${TAU_VALUES} --tau_mode homo \
    --base_seed ${seed} --output ${out}'\""

    if $DRY_RUN; then
        echo "[$count] n=${N} cc=${CC}  aisi=0  sw=0  seed=${seed}  ->  ${TAG}"
    else
        eval "$cmd"
        echo "[$count] queued: ${TAG} (seed=${seed})"
    fi
done

echo
echo "Done: $count jobs queued (BASE_SEED in [${BASE_SEED_START}, $((BASE_SEED_START + NUM_BATCHES - 1))], "
echo "      TECH_PER_JOB=${TECH_PER_JOB}, total tech matrices = $((TECH_PER_JOB * NUM_BATCHES)), time=${TIME_LIMIT})"
