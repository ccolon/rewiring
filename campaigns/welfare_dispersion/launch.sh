#!/bin/bash
#
# Welfare-dispersion campaign (manuscript Campaign 1, tab:welfare_dispersion).
#
# Five parameter points P0..P4, each with 50 tech matrices x 50 initial
# networks = 2500 trials. To fit comfortably under SLURM time limits and to
# parallelise wall-clock time, every point is split into 5 batches of
# (10 tech x 50 inits = 500 trials). Total: 5 points x 5 batches = 25 jobs.
#
# Seed scheme: tech_seed = BASE_SEED * 10000 + tech_idx is shared across
# points, so the analyzer can join each point's per-pair U_T against P0's U_AA
# on (tech_seed, init_seed). To split 50 tech matrices into 5 batches we use
# BASE_SEED_START + batch_idx, with each batch contributing tech_per_job=10
# matrices indexed within that batch.
#
# Output dir: results_welfare/    (drop into results/welfare_dispersion/ locally)
#
# BASE_SEED default = 1700.  P0 batches use seeds 1700..1704; P1 1700..1704;
# etc. The point label distinguishes the resulting CSVs even when seeds
# overlap, since the seed * point_offset pattern only matters within a point.
# (We want the SAME (tech_seed, init_seed) across points to enable the
# P0-as-U_AA join in the analyzer.)
#
# Usage:
#     bash launch_welfare_dispersion.sh                 # 25 jobs, BASE_SEED 1700-1704
#     bash launch_welfare_dispersion.sh 1700 --dry-run

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
# CSVs land flat in results/welfare_dispersion/ (gitignored). The final table
# tab_welfare_dispersion.tex lands here too once campaigns/welfare_dispersion/analyze.py
# is invoked.
OUTPUT_DIR="${SCRIPT_DIR}/results/welfare_dispersion"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED_START=${1:-1700}
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
MEM="2G"
NB_ROUNDS=200
TECH_PER_JOB=10
INITS_PER_TECH=50

POINTS=(P0 P1 P2 P3 P4)

count=0
for point in "${POINTS[@]}"; do
    for (( i=0; i<NUM_BATCHES; i++ )); do
        seed=$(( BASE_SEED_START + i ))
        count=$((count + 1))
        out="${OUTPUT_DIR}/welfare_${point}_seed${seed}.csv"
        job="welfare_${point}_s${seed}"
        cmd="sbatch \
            --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
            --job-name=${job} \
            --output=${SLURM_LOG_DIR}/${job}.%j.out \
            --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/campaigns/welfare_dispersion/study.py \
    --point ${point} \
    --tech_per_job ${TECH_PER_JOB} --inits_per_tech ${INITS_PER_TECH} \
    --nb_rounds ${NB_ROUNDS} \
    --base_seed ${seed} --output ${out}'\""

        if $DRY_RUN; then
            echo "[$count] point=${point} seed=${seed}  ->  ${out##*/}"
        else
            eval "$cmd"
            echo "[$count] queued: ${point} seed=${seed}"
        fi
    done
done

echo
echo "Done: $count jobs queued (BASE_SEED in [${BASE_SEED_START}, "
echo "      $((BASE_SEED_START + NUM_BATCHES - 1))],"
echo "      TECH_PER_JOB=${TECH_PER_JOB}, INITS_PER_TECH=${INITS_PER_TECH}, "
echo "      total trials per point = $((TECH_PER_JOB * INITS_PER_TECH * NUM_BATCHES)), "
echo "      MEM=${MEM}, time=${TIME_LIMIT})"
