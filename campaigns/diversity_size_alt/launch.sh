#!/bin/bash
#
# Diversity scaling campaign (alternate operating points).
#
# Companion to launch_diversity_size.sh.  Sweeps n in {10, 20, 50, 100,
# 200, 500} at the SIX operating points listed below.  Each row of the
# manuscript's fig:diversity_size_alt is one of these series.
#
# Series:
#     tag                kappa  b              Delta_A  sigma_w  Delta_z
#     HRS_dz010          1      unif 0.9:1.1   0        0        0.1
#     IRS_dz030          1      hom  1.1       0        0        0.3
#     DRS_dz030          1      hom  0.9       0        0        0.3
#     HRS_sw010_dz050    1      unif 0.9:1.1   0        0.1      0.5
#     HRS_dA0005_dz010   1      unif 0.9:1.1   0.005    0        0.1
#     HRS_dA005_k4       4      unif 0.9:1.1   0.05     0        0
#
# Common fixed parameters (per spec):
#     full-GE anticipation, a_i = 0.5, c = 4, c' = 4.
#
# Per cell: 50 tech matrices x 50 initial networks = 2,500 trials,
# split into 5 batches of (TECH_PER_JOB=10, INITS_PER_TECH=50).
# 6 series x 6 n-values x 5 batches = 180 SLURM jobs at BASE_SEED 2000-2004.
#
# IRS at kappa=1 is known to converge less reliably than CRS/DRS; the
# downstream analyzer is responsible for flagging per-cell non-convergence
# rate > 5% (per the spec).
#
# Output dir: results_diversity_size_alt/  (drop into
# results/diversity_size_alt/ locally; the existing plotter
# results/diversity_size/plot_diversity_size.py is hard-coded to one
# operating point and will need an alt-companion that accepts a list
# of operating points before this campaign's figure can be rendered).
#
# Usage:
#     bash campaigns/diversity_size_alt/launch.sh                 # 180 jobs (n up to 500)
#     bash campaigns/diversity_size_alt/launch.sh 2000 --dry-run
#     bash campaigns/diversity_size_alt/launch.sh 2000 --max-n 200    # 150 jobs (drop n=500)

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
# CSVs land flat in results/diversity_size_alt/ (gitignored). The final figure
# lands here too once campaigns/diversity_size_alt/plot.py is invoked.
OUTPUT_DIR="${SCRIPT_DIR}/results/diversity_size_alt"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED_START=${1:-2000}
DRY_RUN=false
NUM_BATCHES=5
MAX_N=500
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --num-batches) NUM_BATCHES=$2; shift 2 ;;
        --max-n) MAX_N=$2; shift 2 ;;
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

C=4
CC=4
A_CFG="homogeneous:0.5"

ALL_N=(10 20 50 100 200 500)

# Format per row:
#   tag  b_config         aisi   sigma_w  z_config         kappa
SERIES=(
    "HRS_dz010         uniform:0.9:1.1  0.0    0.0   uniform:0.9:1.1  1"
    "IRS_dz030         homogeneous:1.1  0.0    0.0   uniform:0.7:1.3  1"
    "DRS_dz030         homogeneous:0.9  0.0    0.0   uniform:0.7:1.3  1"
    "HRS_sw010_dz050   uniform:0.9:1.1  0.0    0.1   uniform:0.5:1.5  1"
    "HRS_dA0005_dz010  uniform:0.9:1.1  0.005  0.0   uniform:0.9:1.1  1"
    "HRS_dA005_k4      uniform:0.9:1.1  0.05   0.0   homogeneous:1.0  4"
)

count=0
for entry in "${SERIES[@]}"; do
    read -r tag b_cfg aisi sw z_cfg kappa <<< "$entry"
    for n in "${ALL_N[@]}"; do
        if (( n > MAX_N )); then
            continue
        fi
        for (( i=0; i<NUM_BATCHES; i++ )); do
            seed=$(( BASE_SEED_START + i ))
            count=$((count + 1))
            out="${OUTPUT_DIR}/divsizealt_${tag}_n${n}_seed${seed}.csv"
            job="divsizealt_${tag}_n${n}_s${seed}"
            cmd="sbatch \
                --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
                --job-name=${job} \
                --output=${SLURM_LOG_DIR}/${job}.%j.out \
                --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/campaigns/diversity/study.py \
    --n_min ${n} --n_max ${n} \
    --n_tech ${TECH_PER_JOB} --n_trials ${INITS_PER_TECH} \
    --nb_rounds ${NB_ROUNDS} \
    --cc ${CC} --max_swaps ${kappa} \
    --aisi_spread ${aisi} --sigma_w ${sw} \
    --a_config ${A_CFG} --b_config ${b_cfg} --z_config ${z_cfg} \
    --mode full --series_filter dif_init_only \
    --base_seed ${seed} --output ${out}'\""

            if $DRY_RUN; then
                echo "[$count] ${tag}  n=${n}  kappa=${kappa}  seed=${seed}  ->  ${out##*/}"
            else
                eval "$cmd"
                echo "[$count] queued: ${tag} n=${n} (seed=${seed})"
            fi
        done
    done
done

echo
echo "Done: $count jobs queued (BASE_SEED in [${BASE_SEED_START}, "
echo "      $((BASE_SEED_START + NUM_BATCHES - 1))], "
echo "      TECH_PER_JOB=${TECH_PER_JOB}, INITS_PER_TECH=${INITS_PER_TECH}, "
echo "      total trials per cell = $((TECH_PER_JOB * INITS_PER_TECH * NUM_BATCHES)), "
echo "      MEM=${MEM}, time=${TIME_LIMIT}, max_n=${MAX_N})"
