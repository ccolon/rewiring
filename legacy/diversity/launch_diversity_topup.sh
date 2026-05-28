#!/bin/bash
#
# Top-up launcher for plot_diversity.py's bottom row.
#
# launch_6panel_v1.sh ran HRS-base extras with TWO heterogeneity sources
# fixed simultaneously (aisi+sigma_w, aisi+z, sigma_w+z, all at z_width=0.1).
# It did NOT run the "HRS + 1 fixed het" cells needed to populate the
# bottom-row series of the new diversity figure.
#
# This top-up adds the 6 missing (sweep, fixed-het) combinations at
# Z_HALF_WIDTH_FIXED = 0.1:
#
#     (left   panel) aisi sweep  with sigma_w=0.1, z=0           --> series  HRS + sigma_w=0.1
#     (left   panel) aisi sweep  with sigma_w=0,   z=0.1         --> series  HRS + z=0.1
#     (center panel) sw   sweep  with aisi=0.05,   z=0           --> series  HRS + aisi=0.05
#     (center panel) sw   sweep  with aisi=0,      z=0.1         --> series  HRS + z=0.1
#     (right  panel) z    sweep  with aisi=0.05,   sigma_w=0     --> series  HRS + aisi=0.05
#     (right  panel) z    sweep  with aisi=0,      sigma_w=0.1   --> series  HRS + sigma_w=0.1
#
# Each combination runs at 6 sweep levels (matching launch_6panel_v1's grid):
#     aisi    in {0.0, 0.005, 0.01, 0.02, 0.05, 0.1}
#     sigma_w in {0.0, 0.025, 0.05, 0.1, 0.2, 0.3}
#     z_width in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}
#
# Total: 6 missing series x 6 levels = 36 jobs.  Only ms=1 (n=100, cc=4).
# Sample budget per cell: 5 tech x 50 dif_init inits = 250 sims.
#
# Output dir: results_diversity_topup/
#
# Usage:
#     bash launch_diversity_topup.sh                  # 36 jobs, BASE_SEED=1310
#     bash launch_diversity_topup.sh 1310 --dry-run   # print only

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_diversity_topup"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED=${1:-1310}
DRY_RUN=false
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        *) shift ;;
    esac
done

if ! $DRY_RUN; then
    mkdir -p "$OUTPUT_DIR" "$SLURM_LOG_DIR"
fi

TIME_LIMIT="48:00:00"
MEM="2G"
NB_ROUNDS=200
N_TECH=${N_TECH:-5}
N_TRIALS=${N_TRIALS:-50}
N=100
CC=4
MS=1

# Sweep grids (must match launch_6panel_v1's per-axis grids).
LEVELS_AISI=(0.0  0.005 0.01 0.02 0.05 0.10)
LEVELS_SW=(  0.0  0.025 0.05 0.10 0.20 0.30)
LEVELS_ZW=(  0.0  0.10  0.20 0.30 0.40 0.50)

# z_width to z_config string.
zcfg_for_width() {
    local w=$1
    if (( $(awk -v x="$w" 'BEGIN{print (x==0)}') )); then
        echo "homogeneous:1.0"
    else
        local lo hi
        lo=$(awk -v x="$w" 'BEGIN{printf "%.4g", 1.0 - x}')
        hi=$(awk -v x="$w" 'BEGIN{printf "%.4g", 1.0 + x}')
        echo "uniform:${lo}:${hi}"
    fi
}

count=0
submit_cell() {
    # aisi sw zw tag
    local aisi=$1 sw=$2 zw=$3 tag=$4
    local z_cfg
    z_cfg=$(zcfg_for_width "$zw")

    count=$((count + 1))
    local out="${OUTPUT_DIR}/divtu_${tag}_n${N}_cc${CC}_ms${MS}_a${aisi}_w${sw}_zw${zw}_seed${BASE_SEED}.csv"
    local job="divtu_${tag}_a${aisi}_w${sw}_zw${zw}"
    local cmd="sbatch \
        --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
        --job-name=${job} \
        --output=${SLURM_LOG_DIR}/${job}.%j.out \
        --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/scripts/diversity_study.py \
    --n_min ${N} --n_max ${N} --n_tech ${N_TECH} --n_trials ${N_TRIALS} \
    --nb_rounds ${NB_ROUNDS} --cc ${CC} --max_swaps ${MS} \
    --aisi_spread ${aisi} --sigma_w ${sw} \
    --a_config homogeneous:0.5 --b_config uniform:0.9:1.1 --z_config ${z_cfg} \
    --mode full --series_filter dif_init_only \
    --base_seed ${BASE_SEED} --output ${out}'\""

    if $DRY_RUN; then
        echo "[$count] aisi=${aisi}  sw=${sw}  zw=${zw}  z=${z_cfg}  tag=${tag}"
    else
        eval "$cmd"
        echo "[$count] queued: ${tag}  aisi=${aisi} sw=${sw} zw=${zw}"
    fi
}

# =============================================================================
# (left panel) aisi sweep
# =============================================================================
echo "=== (left panel) aisi sweep cells ==="
# HRS + sigma_w=0.1, varying aisi, z=0
for v in "${LEVELS_AISI[@]}"; do
    submit_cell ${v} 0.1 0.0 "HRS_sw01_aisiSweep"
done
# HRS + z=0.1, varying aisi, sigma_w=0
for v in "${LEVELS_AISI[@]}"; do
    submit_cell ${v} 0.0 0.1 "HRS_z01_aisiSweep"
done

# =============================================================================
# (center panel) sigma_w sweep
# =============================================================================
echo "=== (center panel) sigma_w sweep cells ==="
# HRS + aisi=0.05, varying sigma_w, z=0
for v in "${LEVELS_SW[@]}"; do
    submit_cell 0.05 ${v} 0.0 "HRS_aisi005_swSweep"
done
# HRS + z=0.1, varying sigma_w, aisi=0
for v in "${LEVELS_SW[@]}"; do
    submit_cell 0.0 ${v} 0.1 "HRS_z01_swSweep"
done

# =============================================================================
# (right panel) z_width sweep
# =============================================================================
echo "=== (right panel) z_width sweep cells ==="
# HRS + aisi=0.05, varying z, sigma_w=0
for v in "${LEVELS_ZW[@]}"; do
    submit_cell 0.05 0.0 ${v} "HRS_aisi005_zSweep"
done
# HRS + sigma_w=0.1, varying z, aisi=0
for v in "${LEVELS_ZW[@]}"; do
    submit_cell 0.0 0.1 ${v} "HRS_sw01_zSweep"
done

echo
echo "Done: $count jobs queued (BASE_SEED=${BASE_SEED}, "
echo "      N_TECH=${N_TECH}, N_TRIALS=${N_TRIALS} (dif_init only), "
echo "      MEM=${MEM}, time=${TIME_LIMIT})"
