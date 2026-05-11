#!/bin/bash
#
# Top-up to plot the diversity figure at ms=4 (appendix companion to the
# main ms=1 figure).
#
# Top row (CRS / DRS / IRS / HRS, ms=4) is COMPLETE in the existing data.
# Bottom row (HRS + 1 het fixed) is partial: each (series, panel) has only
# the two endpoint cells (aisi=0 or aisi=0.05, etc.) -- we run the
# remaining 4 levels per panel.
#
# Missing cells (24 total, all at n=100, cc=4, ms=4, HRS economy
# a=hom 0.5, b=unif 0.9:1.1):
#   HRS + AiSi=0.05, sigma_w panel: sigma_w in {0.025, 0.05, 0.2, 0.3}
#   HRS + AiSi=0.05, z       panel: z       in {0.2, 0.3, 0.4, 0.5}
#   HRS + sw=0.1,    aisi    panel: aisi    in {0.005, 0.01, 0.02, 0.1}
#   HRS + sw=0.1,    z       panel: z       in {0.2, 0.3, 0.4, 0.5}
#   HRS + z=0.1,     aisi    panel: aisi    in {0.005, 0.01, 0.02, 0.1}
#   HRS + z=0.1,     sigma_w panel: sigma_w in {0.025, 0.05, 0.2, 0.3}
#
# Per cell: 5 tech x 50 dif_init inits = 250 sims.
# Output dir: results_diversity_topup_ms4/
#
# Usage:
#     bash launch_diversity_topup_ms4.sh                  # 24 jobs, BASE_SEED=1330
#     bash launch_diversity_topup_ms4.sh 1330 --dry-run

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_diversity_topup_ms4"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED=${1:-1330}
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
MS=4

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
    local aisi=$1 sw=$2 zw=$3 tag=$4
    local z_cfg
    z_cfg=$(zcfg_for_width "$zw")

    count=$((count + 1))
    local out="${OUTPUT_DIR}/divtu_ms4_${tag}_n${N}_cc${CC}_ms${MS}_a${aisi}_w${sw}_zw${zw}_seed${BASE_SEED}.csv"
    local job="divtu_ms4_${tag}_a${aisi}_w${sw}_zw${zw}"
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
# Series HRS + AiSi=0.05  (top sw and z panels)
# =============================================================================
echo "=== HRS + AiSi=0.05, sigma_w panel: sw in {0.025, 0.05, 0.2, 0.3} ==="
for v in 0.025 0.05 0.20 0.30; do
    submit_cell 0.05 ${v} 0.0 "HRSAisi_swSweep"
done

echo "=== HRS + AiSi=0.05, z panel: z in {0.2, 0.3, 0.4, 0.5} ==="
for v in 0.20 0.30 0.40 0.50; do
    submit_cell 0.05 0.0 ${v} "HRSAisi_zSweep"
done

# =============================================================================
# Series HRS + sw=0.1  (aisi and z panels)
# =============================================================================
echo "=== HRS + sw=0.1, aisi panel: aisi in {0.005, 0.01, 0.02, 0.1} ==="
for v in 0.005 0.01 0.02 0.10; do
    submit_cell ${v} 0.1 0.0 "HRSsw_aisiSweep"
done

echo "=== HRS + sw=0.1, z panel: z in {0.2, 0.3, 0.4, 0.5} ==="
for v in 0.20 0.30 0.40 0.50; do
    submit_cell 0.0 0.1 ${v} "HRSsw_zSweep"
done

# =============================================================================
# Series HRS + z=0.1  (aisi and sigma_w panels)
# =============================================================================
echo "=== HRS + z=0.1, aisi panel: aisi in {0.005, 0.01, 0.02, 0.1} ==="
for v in 0.005 0.01 0.02 0.10; do
    submit_cell ${v} 0.0 0.1 "HRSz_aisiSweep"
done

echo "=== HRS + z=0.1, sigma_w panel: sw in {0.025, 0.05, 0.2, 0.3} ==="
for v in 0.025 0.05 0.20 0.30; do
    submit_cell 0.0 ${v} 0.1 "HRSz_swSweep"
done

echo
echo "Done: $count jobs queued (BASE_SEED=${BASE_SEED}, "
echo "      N_TECH=${N_TECH}, N_TRIALS=${N_TRIALS} (dif_init only), "
echo "      MS=${MS}, MEM=${MEM}, time=${TIME_LIMIT})"
