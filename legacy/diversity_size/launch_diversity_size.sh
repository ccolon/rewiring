#!/bin/bash
#
# Diversity scaling campaign (manuscript fig:diversity_size).
#
# Two-panel figure showing how diversity scales with economy size n:
#   Panel (a): n in {10, 20, 50, 100, 200, 500} x RTS in {CRS, DRS, HRS}
#              at fixed c' = 4.
#   Panel (b): n in {10, 20, 50, 100, 200, 500} x c' in {2, 4, 8}
#              at fixed CRS.
# Overlap (CRS, c' = 4) x 6 n is computed once and shared between panels.
#
# Series (5 unique tags):
#   CRS_cc4  (b = hom 1.0,    c' = 4)   <- shared between (a) and (b)
#   DRS_cc4  (b = hom 0.9,    c' = 4)
#   HRS_cc4  (b = unif 0.9:1.1, c' = 4)
#   CRS_cc2  (b = hom 1.0,    c' = 2)
#   CRS_cc8  (b = hom 1.0,    c' = 8)
#
# Common fixed parameters (per spec):
#   Delta_A = 0.05, sigma_w = 0, Delta_z = 0,
#   kappa = 1, full-GE anticipation, a_i = 0.5, z_i = 1, c = 4.
#
# Per cell: 50 tech matrices x 50 initial networks = 2500 trials.
# Split into 5 batches of (tech_per_job=10, inits_per_tech=50) = 500 trials
# each, giving 5 batches/cell x 30 cells = 150 SLURM jobs at BASE_SEED
# 1900-1904.
#
# The diversity_study.py CSV per row already has every column the
# downstream plotter needs (diversity, frac_converged, n, c, cc, b_config,
# tech_seed, etc.). Only the dif_init series is computed (--series_filter
# dif_init_only) since fig:diversity_size only needs nu_P.
#
# Output dir: results_diversity_size/  (drop into results/diversity_size/
# locally for the plotter).
#
# Usage:
#     bash launch_diversity_size.sh                # 150 jobs (n up to 500)
#     bash launch_diversity_size.sh 1900 --dry-run
#     bash launch_diversity_size.sh 1900 --max-n 200   # 125 jobs (drop n=500)

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_diversity_size"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED_START=${1:-1900}
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
AISI=0.05
SW=0.0
A_CFG="homogeneous:0.5"
Z_CFG="homogeneous:1.0"
KAPPA=1

ALL_N=(10 20 50 100 200 500)

# series tag => (b_config, cc)
SERIES=(
    "CRS_cc4 homogeneous:1.0  4"
    "DRS_cc4 homogeneous:0.9  4"
    "HRS_cc4 uniform:0.9:1.1  4"
    "CRS_cc2 homogeneous:1.0  2"
    "CRS_cc8 homogeneous:1.0  8"
)

count=0
for entry in "${SERIES[@]}"; do
    read -r tag b_cfg cc <<< "$entry"
    for n in "${ALL_N[@]}"; do
        if (( n > MAX_N )); then
            continue
        fi
        for (( i=0; i<NUM_BATCHES; i++ )); do
            seed=$(( BASE_SEED_START + i ))
            count=$((count + 1))
            out="${OUTPUT_DIR}/divsize_${tag}_n${n}_seed${seed}.csv"
            job="divsize_${tag}_n${n}_s${seed}"
            cmd="sbatch \
                --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
                --job-name=${job} \
                --output=${SLURM_LOG_DIR}/${job}.%j.out \
                --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/scripts/diversity_study.py \
    --n_min ${n} --n_max ${n} \
    --n_tech ${TECH_PER_JOB} --n_trials ${INITS_PER_TECH} \
    --nb_rounds ${NB_ROUNDS} \
    --cc ${cc} --max_swaps ${KAPPA} \
    --aisi_spread ${AISI} --sigma_w ${SW} \
    --a_config ${A_CFG} --b_config ${b_cfg} --z_config ${Z_CFG} \
    --mode full --series_filter dif_init_only \
    --base_seed ${seed} --output ${out}'\""

            if $DRY_RUN; then
                echo "[$count] ${tag}  n=${n}  seed=${seed}  ->  ${out##*/}"
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
