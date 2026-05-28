#!/bin/bash
#
# Switching-cost campaign (manuscript appendix: fig:switchcost).
#
# Two parameter points P2 (CRS, kappa=1, Delta_A=0.05) and P4 (HRS, kappa=1,
# compounded) -- labels match welfare_dispersion_study.py. Each (point, chi)
# cell runs 50 tech matrices x 50 initial networks = 2500 trials, split into
# 5 batches of (10 tech x 50 inits = 500 trials) to fit SLURM time limits.
#
# 2 points x 7 chi values x 5 batches = 70 jobs.
#
# Seed scheme: tech_seed = BASE_SEED * 10000 + tech_idx is shared across chi
# values within a point (and across points), so trials at different chi
# values use the SAME (Wbar, AiSi, a, b, z, S^(0)) configurations.
# BASE_SEED_START = 2700 by default (offsets from welfare 1700 and
# sync_async 1800).
#
# Tracked code in campaigns/switchcosts/   (launcher, orchestrator, plotter).
# Untracked outputs in results/switchcosts/ (per-trial CSVs and the final figure).
#
# Usage:
#     bash campaigns/switchcosts/launch.sh                            # full campaign
#     bash campaigns/switchcosts/launch.sh 2700 --dry-run             # preview
#     bash campaigns/switchcosts/launch.sh 2700 --num-batches 1       # pilot: 500 trials/cell
#     bash campaigns/switchcosts/launch.sh 2700 --tech_per_job 5 --inits_per_tech 10 --num-batches 1
#                                                                     # mini-pilot: 50 trials/cell
#     bash campaigns/switchcosts/launch.sh 2700 --chis "0.0,0.005,0.02"
#                                                                     # subset chi sweep
#     bash campaigns/switchcosts/launch.sh 2700 --points "P2"         # only P2

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
# CSVs land flat in results/switchcosts/ (gitignored). The final figure lands
# here too once campaigns/switchcosts/plot.py is invoked.
OUTPUT_DIR="${SCRIPT_DIR}/results/switchcosts"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED_START=${1:-2700}
DRY_RUN=false
NUM_BATCHES=5
TECH_PER_JOB=10
INITS_PER_TECH=50
TIME_LIMIT="48:00:00"
MEM="2G"
NB_ROUNDS=200
N=100
POINTS_STR="P2,P4"
CHIS_STR="0.0,0.0005,0.001,0.002,0.005,0.01,0.02"

shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --num-batches) NUM_BATCHES=$2; shift 2 ;;
        --tech_per_job) TECH_PER_JOB=$2; shift 2 ;;
        --inits_per_tech) INITS_PER_TECH=$2; shift 2 ;;
        --nb_rounds) NB_ROUNDS=$2; shift 2 ;;
        --n) N=$2; shift 2 ;;
        --time) TIME_LIMIT=$2; shift 2 ;;
        --mem) MEM=$2; shift 2 ;;
        --points) POINTS_STR=$2; shift 2 ;;
        --chis) CHIS_STR=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if ! $DRY_RUN; then
    mkdir -p "$OUTPUT_DIR" "$SLURM_LOG_DIR"
fi

IFS=',' read -ra POINTS <<< "$POINTS_STR"
IFS=',' read -ra CHIS   <<< "$CHIS_STR"

count=0
for point in "${POINTS[@]}"; do
    for chi in "${CHIS[@]}"; do
        for (( i=0; i<NUM_BATCHES; i++ )); do
            seed=$(( BASE_SEED_START + i ))
            count=$((count + 1))
            out="${OUTPUT_DIR}/switch_${point}_chi${chi}_seed${seed}.csv"
            job="switch_${point}_chi${chi}_s${seed}"
            cmd="sbatch \
                --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
                --job-name=${job} \
                --output=${SLURM_LOG_DIR}/${job}.%j.out \
                --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/campaigns/switchcosts/study.py \
    --point ${point} --chi ${chi} \
    --tech_per_job ${TECH_PER_JOB} --inits_per_tech ${INITS_PER_TECH} \
    --nb_rounds ${NB_ROUNDS} --n ${N} \
    --base_seed ${seed} --output ${out}'\""

            if $DRY_RUN; then
                echo "[$count] point=${point} chi=${chi} seed=${seed}  ->  ${out##*/}"
            else
                eval "$cmd"
                echo "[$count] queued: ${point} chi=${chi} seed=${seed}"
            fi
        done
    done
done

trials_per_cell=$((TECH_PER_JOB * INITS_PER_TECH * NUM_BATCHES))
echo
echo "Done: ${count} jobs queued."
echo "  points       : ${POINTS[*]}"
echo "  chis         : ${CHIS[*]}"
echo "  base seeds   : ${BASE_SEED_START}..$((BASE_SEED_START + NUM_BATCHES - 1))"
echo "  per-cell     : ${TECH_PER_JOB} tech x ${INITS_PER_TECH} inits x ${NUM_BATCHES} batches = ${trials_per_cell} trials"
echo "  n=${N}, nb_rounds=${NB_ROUNDS}, mem=${MEM}, time=${TIME_LIMIT}"
