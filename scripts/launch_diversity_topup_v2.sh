#!/bin/bash
#
# Focused top-up for plot_diversity.py (n=100, cc=4, ms=1, BASE_SEED=1320).
#
# Identifies exactly the 16 cells still missing after the existing 4panel_v1
# and 6panel_v1 data, dropping any series that has already saturated at
# diversity ~= 1 across what's been measured.
#
# Series populated by this top-up:
#   HRS + sigma_w=0.1, aisi panel: missing aisi in {0.005, 0.01, 0.02, 0.1}
#   HRS + sigma_w=0.1, z    panel: missing z    in {0.2, 0.3, 0.4, 0.5}
#   HRS + z=0.1,       aisi panel: missing aisi in {0.005, 0.01, 0.02, 0.1}
#   HRS + z=0.1,       sw   panel: missing sw   in {0.025, 0.05, 0.2, 0.3}
#
# Skipped (already saturated at ~1 across measured levels):
#   HRS + AiSi=0.05  on both sw and z panels
#   (All TOP-row aisi sweeps -- CRS/DRS/IRS already at ~1.)
#
# Total: 16 jobs.  Per cell: 5 tech x 50 dif_init inits = 250 sims.
#
# Output dir: results_diversity_topup_v2/
#
# Usage:
#     bash launch_diversity_topup_v2.sh                  # 16 jobs, BASE_SEED=1320
#     bash launch_diversity_topup_v2.sh 1320 --dry-run

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_diversity_topup_v2"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED=${1:-1320}
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
    local out="${OUTPUT_DIR}/divtu2_${tag}_n${N}_cc${CC}_ms${MS}_a${aisi}_w${sw}_zw${zw}_seed${BASE_SEED}.csv"
    local job="divtu2_${tag}_a${aisi}_w${sw}_zw${zw}"
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
# Series HRS + sigma_w=0.1
# =============================================================================
echo "=== HRS + sigma_w=0.1, aisi panel: aisi in {0.005, 0.01, 0.02, 0.1} ==="
for v in 0.005 0.01 0.02 0.10; do
    submit_cell ${v} 0.1 0.0 "HRS_sw01_aisiSweep"
done

echo "=== HRS + sigma_w=0.1, z panel: z in {0.2, 0.3, 0.4, 0.5} ==="
for v in 0.20 0.30 0.40 0.50; do
    submit_cell 0.0 0.1 ${v} "HRS_sw01_zSweep"
done

# =============================================================================
# Series HRS + z=0.1
# =============================================================================
echo "=== HRS + z=0.1, aisi panel: aisi in {0.005, 0.01, 0.02, 0.1} ==="
for v in 0.005 0.01 0.02 0.10; do
    submit_cell ${v} 0.0 0.1 "HRS_z01_aisiSweep"
done

echo "=== HRS + z=0.1, sw panel: sw in {0.025, 0.05, 0.2, 0.3} ==="
for v in 0.025 0.05 0.20 0.30; do
    submit_cell 0.0 ${v} 0.1 "HRS_z01_swSweep"
done

echo
echo "Done: $count jobs queued (BASE_SEED=${BASE_SEED}, "
echo "      N_TECH=${N_TECH}, N_TRIALS=${N_TRIALS} (dif_init only), "
echo "      MEM=${MEM}, time=${TIME_LIMIT})"
