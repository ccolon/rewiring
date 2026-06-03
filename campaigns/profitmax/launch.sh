#!/bin/bash
#
# Profit-maximisation diversity-scaling campaign
# (manuscript appendix "Profit maximisation" / fig:profitmax).
#
# Sweeps n in {10, 20, 50, 100, 200} for TWO objectives:
#     mode=full              cost-minimisation baseline (Section ssec:AA_baseline)
#     mode=full_profitmax    profit-maximisation variant (Eqs. price_profitmax--
#                            profit_profitmax in the appendix)
#
# Both curves share the *same* DRS+sigma_w calibration:
#     a = hom 0.5,  b = hom 0.9,  z = hom 1.0   (Delta_z = 0),
#     c = c' = 4,  kappa = 1,  sigma_w = 0.2,   Delta_A = 0.
# Rationale: pure-homogeneous (sigma_w = 0) saturates both curves at ~100%
# diversity from n=10 (uninformative); Delta_z > 0 leaves profit-max
# saturated since z_i does not enter the profit objective. sigma_w > 0
# instead introduces *link-weight* heterogeneity that BOTH objectives see
# (it enters alpha_i = a_i + (1-a_i) sum_j W_ji and through W directly in
# the sales and price systems), so neither curve trivially saturates and
# their growth-with-n can be compared on the same axis.
# Profit-max is restricted to DRS upstream (the simulation driver guards
# b*alpha < 1); CRS and IRS are inoperative under that objective (see appendix).
#
# Per cell: 50 tech matrices x 50 initial networks = 2,500 trials,
# split into 5 batches of (TECH_PER_JOB=10, INITS_PER_TECH=50).
# 2 modes x 5 n-values x 5 batches = 50 SLURM jobs at BASE_SEED 2300-2304.
#
# Seed scheme (BASE_SEED_START = 2300 by default, chosen to avoid collisions
# with the diversity main sweep (2200) and the switchcosts campaign (2700)):
#     batch i -> seed = BASE_SEED_START + i, tech_seed = seed * 10000 + tech_idx.
#
# Output dir: results/profitmax/  (gitignored). campaigns/profitmax/plot.py
# reads every CSV in that dir.
#
# Usage:
#     bash campaigns/profitmax/launch.sh                  # 50 jobs
#     bash campaigns/profitmax/launch.sh 2300 --dry-run   # preview
#     bash campaigns/profitmax/launch.sh 2300 --max-n 100 # 40 jobs (drop n=200)

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results/profitmax"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED_START=${1:-2300}
DRY_RUN=false
NUM_BATCHES=5
MAX_N=200
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
KAPPA=1
A_CFG="homogeneous:0.5"
B_CFG="homogeneous:0.9"
Z_CFG="homogeneous:1.0"   # Delta_z = 0
AISI=0.0
SIGMA_W=0.2               # link-weight heterogeneity (operative under both objectives)

ALL_N=(10 20 50 100 200)

# (mode_tag, --mode value)
MODES=(
    "costmin    full"
    "profitmax  full_profitmax"
)

count=0
for entry in "${MODES[@]}"; do
    read -r tag mode_arg <<< "$entry"
    for n in "${ALL_N[@]}"; do
        if (( n > MAX_N )); then
            continue
        fi
        for (( i=0; i<NUM_BATCHES; i++ )); do
            seed=$(( BASE_SEED_START + i ))
            count=$((count + 1))
            out="${OUTPUT_DIR}/profitmax_${tag}_n${n}_seed${seed}.csv"
            job="profitmax_${tag}_n${n}_s${seed}"
            cmd="sbatch \
                --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
                --job-name=${job} \
                --output=${SLURM_LOG_DIR}/${job}.%j.out \
                --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/campaigns/diversity/study.py \
    --n_min ${n} --n_max ${n} \
    --n_tech ${TECH_PER_JOB} --n_trials ${INITS_PER_TECH} \
    --nb_rounds ${NB_ROUNDS} \
    --cc ${CC} --max_swaps ${KAPPA} \
    --aisi_spread ${AISI} --sigma_w ${SIGMA_W} \
    --a_config ${A_CFG} --b_config ${B_CFG} --z_config ${Z_CFG} \
    --mode ${mode_arg} --series_filter dif_init_only \
    --base_seed ${seed} --output ${out}'\""

            if $DRY_RUN; then
                echo "[$count] ${tag}  n=${n}  mode=${mode_arg}  seed=${seed}  ->  ${out##*/}"
            else
                eval "$cmd"
                echo "[$count] queued: ${tag} n=${n} (seed=${seed})"
            fi
        done
    done
done

echo
echo "Done: $count jobs queued"
echo "  BASE_SEED in [${BASE_SEED_START}, $((BASE_SEED_START + NUM_BATCHES - 1))]"
echo "  TECH_PER_JOB=${TECH_PER_JOB}, INITS_PER_TECH=${INITS_PER_TECH}"
echo "  total trials per cell = $((TECH_PER_JOB * INITS_PER_TECH * NUM_BATCHES))"
echo "  MEM=${MEM}, time=${TIME_LIMIT}, max_n=${MAX_N}"
