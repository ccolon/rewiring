#!/bin/bash
#
# Quick first-pass data for the 4-panel manuscript figure.
#
# Y axis: diversity (same_tech_dif_init only).
# Panels:
#   top-left  ms=1, x=aisi  (sw=0, aisi in {0, 0.05, 0.1})
#   top-right ms=1, x=sw    (aisi=0, sw in {0, 0.05, 0.1})
#   bot-left  ms=4, x=aisi  (same)
#   bot-right ms=4, x=sw    (same)
#
# Series (each plotted on every panel, except 5 and 6 — see below):
#   1: fully hom CRS    a=hom 0.5, b=hom 1.0, z=hom 1.0       (n=100, cc=4)
#   2: hom + b unif     a=hom 0.5, b=unif 0.9:1.1, z=hom 1.0  (n=100, cc=4)
#   3: realistic        a=unif 0.4:0.6, b=unif 0.9:1.1,
#                       z=unif 0.9:1.1                         (n=100, cc=4)
#   4: realistic+wide z a=unif 0.4:0.6, b=unif 0.9:1.1,
#                       z=unif 0.5:1.5                         (n=100, cc=4)
#   5: realistic, cc=2, ms=1 only (top-row panels)             (n=100, cc=2)
#   6: realistic, n=200                                        (n=200, cc=4)
#
# Sample budget per cell: 5 tech matrices x 50 inits = 250 dif_init sims.
#
# Cell count: 4 series x 2 ms x 5 unique (aisi,sw) cells = 40
#           + series 5: 1 x 1 ms x 5 cells = 5
#           + series 6: 1 x 2 ms x 5 cells = 10
#           = 55 cells = 55 jobs.
#
# (aisi, sw) grid per (series, ms):
#   (0,0)  (0.05,0)  (0.1,0)  (0,0.05)  (0,0.1)   -- 5 unique cells.
#
# Usage:
#     bash launch_4panel_v1.sh                  # 55 jobs, BASE_SEED=1200
#     bash launch_4panel_v1.sh 1200 --dry-run

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_4panel_v1"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED=${1:-1200}
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
MEM="6G"
NB_ROUNDS=200
N_TECH=5
N_TRIALS=50

count=0
submit_cell() {
    # n cc ms aisi sw a_cfg b_cfg z_cfg series_tag
    local n=$1 cc=$2 ms=$3 aisi=$4 sw=$5
    local a_cfg=$6 b_cfg=$7 z_cfg=$8 tag=$9

    count=$((count + 1))
    local out="${OUTPUT_DIR}/4panel_${tag}_n${n}_cc${cc}_ms${ms}_a${aisi}_w${sw}_seed${BASE_SEED}.csv"
    local job="4p_${tag}_n${n}_ms${ms}_a${aisi}_w${sw}"
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
        echo "[$count] n=${n} cc=${cc} ms=${ms} aisi=${aisi} sw=${sw}  ${tag}"
    else
        eval "$cmd"
        echo "[$count] queued: ${tag}  n=${n} cc=${cc} ms=${ms} aisi=${aisi} sw=${sw}"
    fi
}

# 5 cells per (series, ms): (0,0), (0.05,0), (0.1,0), (0,0.05), (0,0.1)
submit_grid() {
    # n cc ms a_cfg b_cfg z_cfg tag
    local n=$1 cc=$2 ms=$3 a_cfg=$4 b_cfg=$5 z_cfg=$6 tag=$7
    submit_cell ${n} ${cc} ${ms} 0.0  0.0  ${a_cfg} ${b_cfg} ${z_cfg} ${tag}
    submit_cell ${n} ${cc} ${ms} 0.05 0.0  ${a_cfg} ${b_cfg} ${z_cfg} ${tag}
    submit_cell ${n} ${cc} ${ms} 0.1  0.0  ${a_cfg} ${b_cfg} ${z_cfg} ${tag}
    submit_cell ${n} ${cc} ${ms} 0.0  0.05 ${a_cfg} ${b_cfg} ${z_cfg} ${tag}
    submit_cell ${n} ${cc} ${ms} 0.0  0.1  ${a_cfg} ${b_cfg} ${z_cfg} ${tag}
}

# =============================================================================
# Series 1: fully hom CRS
# =============================================================================
echo "=== Series 1: fully hom CRS (a=hom 0.5, b=hom 1.0, z=hom 1.0) ==="
for ms in 1 4; do
    submit_grid 100 4 ${ms} homogeneous:0.5 homogeneous:1.0 homogeneous:1.0 "s1_homCRS"
done

# =============================================================================
# Series 2: hom + b uniform
# =============================================================================
echo "=== Series 2: hom + b unif 0.9:1.1 (a=hom 0.5, z=hom 1.0) ==="
for ms in 1 4; do
    submit_grid 100 4 ${ms} homogeneous:0.5 uniform:0.9:1.1 homogeneous:1.0 "s2_bunif"
done

# =============================================================================
# Series 3: realistic, z=[0.9, 1.1]
# =============================================================================
echo "=== Series 3: realistic (a=unif 0.4:0.6, b=unif 0.9:1.1, z=unif 0.9:1.1) ==="
for ms in 1 4; do
    submit_grid 100 4 ${ms} uniform:0.4:0.6 uniform:0.9:1.1 uniform:0.9:1.1 "s3_realistic"
done

# =============================================================================
# Series 4: realistic + wide z = [0.5, 1.5]
# =============================================================================
echo "=== Series 4: realistic + wide z (a=unif 0.4:0.6, b=unif 0.9:1.1, z=unif 0.5:1.5) ==="
for ms in 1 4; do
    submit_grid 100 4 ${ms} uniform:0.4:0.6 uniform:0.9:1.1 uniform:0.5:1.5 "s4_widez"
done

# =============================================================================
# Series 5: realistic, cc=2, ms=1 (top row only)
# =============================================================================
echo "=== Series 5: realistic, cc=2, ms=1 only ==="
submit_grid 100 2 1 uniform:0.4:0.6 uniform:0.9:1.1 uniform:0.9:1.1 "s5_cc2"

# =============================================================================
# Series 6: realistic, n=200, cc=4, both ms
# =============================================================================
echo "=== Series 6: realistic, n=200, cc=4 ==="
for ms in 1 4; do
    submit_grid 200 4 ${ms} uniform:0.4:0.6 uniform:0.9:1.1 uniform:0.9:1.1 "s6_n200"
done

echo
echo "Done: $count jobs queued (BASE_SEED=${BASE_SEED}, "
echo "      N_TECH=${N_TECH}, N_TRIALS=${N_TRIALS} (dif_init only), time=${TIME_LIMIT})"
