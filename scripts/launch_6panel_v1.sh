#!/bin/bash
#
# 6-panel manuscript figure: diversity vs three heterogeneity sources
# at two action regimes.
#
# Layout (rows = ms, cols = swept axis):
#                  aisi sweep      sigma_w sweep    z_width sweep
#     ms=1         (left, top)     (center, top)    (right, top)
#     ms=cc=4      (left, bot)     (center, bot)    (right, bot)
#
# Series per panel: 4 b-regimes + 1 extra
#   1. CRS hom:  a=hom 0.5, b=hom 1.0, z=hom 1.0
#   2. DRS hom:  a=hom 0.5, b=hom 0.9, z=hom 1.0
#   3. IRS hom:  a=hom 0.5, b=hom 1.1, z=hom 1.0
#   4. HRS:      a=hom 0.5, b=unif 0.9:1.1, z=hom 1.0
#   5. extras (HRS base, plus 2 fixed heterogeneities):
#       left  : aisi varies, sw=0.1, z_width=0.1
#       center: sw varies,   aisi=0.05, z_width=0.1
#       right : z_width vrs, aisi=0.05, sw=0.1
#
# Sweep grid (per-axis, 6 levels, focused on each axis's regime structure):
#   aisi    : {0, 0.005, 0.01, 0.02, 0.05, 0.1}      (threshold ~0.02, saturation by 0.05)
#   sigma_w : {0, 0.025, 0.05, 0.1, 0.2, 0.3}        (threshold + plateau + dropoff)
#   z_width : {0, 0.1, 0.2, 0.3, 0.4, 0.5}           (small-effect zone + suppression zone)
#
# Sample budget: 5 tech matrices x 50 dif_init inits = 250 sims per cell.
# Cells:
#   4 base series x 2 ms x (1 origin + 5 aisi-only + 5 sw-only + 5 z-only)
#       = 4 x 2 x 16 = 128
#   3 extras x 2 ms x 6 sweep levels = 36
#   Total = 164 jobs.
#
# Usage:
#     bash launch_6panel_v1.sh                 # 314 jobs, BASE_SEED=1300
#     bash launch_6panel_v1.sh 1300 --dry-run

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_6panel_v1"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED=${1:-1300}
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
N_TECH=5
N_TRIALS=50
N=100
CC=4

# Per-axis sweep grids (6 levels each, focused on each axis's interesting regime).
LEVELS_AISI=(0.0    0.005 0.01  0.02  0.05  0.10)
LEVELS_SW=(  0.0    0.025 0.05  0.10  0.20  0.30)
LEVELS_ZW=(  0.0    0.10  0.20  0.30  0.40  0.50)

# z_width to z_config: hom 1.0 if 0, else unif (1-w):(1+w) (formatted as "low:high").
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
    # n cc ms aisi sw z_width a_cfg b_cfg series_tag
    local n=$1 cc=$2 ms=$3 aisi=$4 sw=$5 zw=$6
    local a_cfg=$7 b_cfg=$8 tag=$9
    local z_cfg
    z_cfg=$(zcfg_for_width "$zw")

    count=$((count + 1))
    local out="${OUTPUT_DIR}/6panel_${tag}_n${n}_cc${cc}_ms${ms}_a${aisi}_w${sw}_zw${zw}_seed${BASE_SEED}.csv"
    local job="6p_${tag}_ms${ms}_a${aisi}_w${sw}_zw${zw}"
    local cmd="sbatch \
        --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
        --job-name=${job} \
        --output=${SLURM_LOG_DIR}/${job}.%j.out \
        --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/scripts/diversity_study.py \
    --n_min ${n} --n_max ${n} --n_tech ${N_TECH} --n_trials ${N_TRIALS} \
    --nb_rounds ${NB_ROUNDS} --cc ${cc} --max_swaps ${ms} \
    --aisi_spread ${aisi} --sigma_w ${sw} \
    --a_config ${a_cfg} --b_config ${b_cfg} --z_config ${z_cfg} \
    --mode full --series_filter dif_init_only \
    --base_seed ${BASE_SEED} --output ${out}'\""

    if $DRY_RUN; then
        echo "[$count] ms=${ms} aisi=${aisi} sw=${sw} zw=${zw}  z=${z_cfg}  ${tag}"
    else
        eval "$cmd"
        echo "[$count] queued: ${tag}  ms=${ms} aisi=${aisi} sw=${sw} zw=${zw}"
    fi
}

# 16 unique cells per (base series, ms): origin + 3 axes x 5 sweep levels each.
submit_base() {
    # ms a_cfg b_cfg tag
    local ms=$1 a_cfg=$2 b_cfg=$3 tag=$4

    # Origin (0, 0, 0)
    submit_cell ${N} ${CC} ${ms} 0.0 0.0 0.0 "${a_cfg}" "${b_cfg}" "${tag}"

    # aisi sweep at sw=0, z_width=0  (skip the 0 already in origin)
    for v in "${LEVELS_AISI[@]:1}"; do
        submit_cell ${N} ${CC} ${ms} ${v} 0.0 0.0 "${a_cfg}" "${b_cfg}" "${tag}"
    done
    # sw sweep at aisi=0, z_width=0
    for v in "${LEVELS_SW[@]:1}"; do
        submit_cell ${N} ${CC} ${ms} 0.0 ${v} 0.0 "${a_cfg}" "${b_cfg}" "${tag}"
    done
    # z_width sweep at aisi=0, sw=0
    for v in "${LEVELS_ZW[@]:1}"; do
        submit_cell ${N} ${CC} ${ms} 0.0 0.0 ${v} "${a_cfg}" "${b_cfg}" "${tag}"
    done
}

# Each extra is a single 6-level sweep on the panel's axis (HRS base).
submit_extra_left() {
    # aisi varies, sw=0.1, z_width=0.1
    local ms=$1
    for v in "${LEVELS_AISI[@]}"; do
        submit_cell ${N} ${CC} ${ms} ${v} 0.1 0.1 \
            homogeneous:0.5 uniform:0.9:1.1 "extra_aisi"
    done
}
submit_extra_center() {
    # sw varies, aisi=0.05, z_width=0.1
    local ms=$1
    for v in "${LEVELS_SW[@]}"; do
        submit_cell ${N} ${CC} ${ms} 0.05 ${v} 0.1 \
            homogeneous:0.5 uniform:0.9:1.1 "extra_sw"
    done
}
submit_extra_right() {
    # z_width varies, aisi=0.05, sw=0.1
    local ms=$1
    for v in "${LEVELS_ZW[@]}"; do
        submit_cell ${N} ${CC} ${ms} 0.05 0.1 ${v} \
            homogeneous:0.5 uniform:0.9:1.1 "extra_zw"
    done
}

# =============================================================================
# 4 base series x 2 ms = 8 series-ms combos x 31 cells = 248 jobs
# =============================================================================
echo "=== Base series ==="
for ms in 1 4; do
    submit_base ${ms} homogeneous:0.5 homogeneous:1.0 "CRShom"
    submit_base ${ms} homogeneous:0.5 homogeneous:0.9 "DRShom"
    submit_base ${ms} homogeneous:0.5 homogeneous:1.1 "IRShom"
    submit_base ${ms} homogeneous:0.5 uniform:0.9:1.1 "HRS"
done

# =============================================================================
# Extras: 3 panels x 2 ms x 11 levels = 66 jobs
# =============================================================================
echo "=== Extras (HRS base) ==="
for ms in 1 4; do
    submit_extra_left   ${ms}
    submit_extra_center ${ms}
    submit_extra_right  ${ms}
done

echo
echo "Done: $count jobs queued (BASE_SEED=${BASE_SEED}, "
echo "      N_TECH=${N_TECH}, N_TRIALS=${N_TRIALS} (dif_init only), "
echo "      6-level focused sweeps per axis, MEM=${MEM}, time=${TIME_LIMIT})"
