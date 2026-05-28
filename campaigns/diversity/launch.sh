#!/bin/bash
#
# Diversity campaign (manuscript figs:diversity, figs:diversity_ms4).
#
# Single canonical launcher that enumerates EVERY cell needed by
# campaigns/diversity/plot.py to render figure_diversity.png (ms=1) and
# figure_diversity_ms4.png (ms=4). Filter on load: series='same_tech_dif_init',
# n=100, cc=4.
#
# Layout per ms in {1, 4} -- 6 panels each:
#
#                       aisi sweep         sigma_w sweep        z_width sweep
#     Top (homo. b):       (a)                (b)                  (c)
#     Bottom (HRS-based):  (d)                (e)                  (f)
#
# TOP ROW (4 b-regimes; the other two het sources are pinned at 0):
#     CRS  : b = hom 1.0
#     DRS  : b = hom 0.9
#     IRS  : b = hom 1.1
#     HRS  : b = unif 0.9:1.1
#
# BOTTOM ROW (HRS economy + 1 or 2 additional fixed heterogeneities).
# Each series only appears on the panel(s) whose swept axis differs from
# its fixed axes -- otherwise the "fixed" value would conflict with the sweep.
#     HRS + AiSi=0.05         : on (b)/(e) sigma_w panel, (c)/(f) z_width panel
#     HRS + sigma_w=0.1       : on (a)/(d) aisi panel,    (c)/(f) z_width panel
#     HRS + z_width=0.1       : on (a)/(d) aisi panel,    (b)/(e) sigma_w panel
#     HRS + AiSi=0.05+sw=0.1  : on (c)/(f) z_width panel only
#     HRS + AiSi=0.05+z=0.1   : on (b)/(e) sigma_w panel only
#     HRS + sw=0.1+z=0.1      : on (a)/(d) aisi panel only
#
# Sweep grids (matching the 6panel_v1 plot conventions):
#     aisi    in {0, 0.005, 0.01, 0.02, 0.05, 0.1}        (6 levels)
#     sigma_w in {0, 0.025, 0.05, 0.1, 0.2, 0.3}          (6 levels)
#     z_width in {0, 0.1, 0.2, 0.3, 0.4, 0.5}             (6 levels)
#                                  (encoded as b_config = unif 1-w:1+w)
#
# Cell count:
#     Top    : 2 ms x 4 series x 3 axes x 6 levels = 144 cells
#     Bottom : 2 ms x [2+2+2+1+1+1 series-panel pairs] x 6 levels = 108 cells
#     TOTAL  : 252 cells (one sbatch job each).
#
# Per cell: 50 tech matrices x 50 dif_init inits = 2500 sims. The launcher
# splits into NUM_BATCHES batches of (tech_per_job=10, inits_per_tech=50);
# default NUM_BATCHES=5 -> 5 sub-jobs per cell -> 1260 sbatch jobs total.
# Set --num-batches 1 to get a smaller first-pass with 500 sims/cell.
#
# Output dir: results/diversity/  (gitignored). The plot.py invocation
# reads every diversity-format CSV in that dir.
#
# Seed scheme: BASE_SEED_START = 2200 by default (offsets from welfare 1700,
# sync_async 1800, diversity_size_alt 2000, switchcosts 2700). Per cell the
# 5 batches use seeds 2200..2204; tech_seed = base_seed * 10000 + tech_idx.
#
# Usage:
#     bash campaigns/diversity/launch.sh                          # full 1260 jobs
#     bash campaigns/diversity/launch.sh 2200 --dry-run           # preview
#     bash campaigns/diversity/launch.sh 2200 --num-batches 1     # 252 jobs, 500 sims/cell
#     bash campaigns/diversity/launch.sh 2200 --ms 1              # only ms=1 (126 cells)
#     bash campaigns/diversity/launch.sh 2200 --ms 4 --num-batches 1
#                                                                  # ms=4 first-pass
#
# (To use only the existing results CSVs without launching anything, just
#  run python campaigns/diversity/plot.py directly.)

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results/diversity"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED_START=${1:-2200}
DRY_RUN=false
NUM_BATCHES=5
TECH_PER_JOB=10
INITS_PER_TECH=50
TIME_LIMIT="48:00:00"
MEM="2G"
NB_ROUNDS=20
N=100
C=4
CC=4
MS_FILTER=""   # empty = both 1 and 4; "1" or "4" restrict

shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --num-batches) NUM_BATCHES=$2; shift 2 ;;
        --tech_per_job) TECH_PER_JOB=$2; shift 2 ;;
        --inits_per_tech) INITS_PER_TECH=$2; shift 2 ;;
        --nb_rounds) NB_ROUNDS=$2; shift 2 ;;
        --time) TIME_LIMIT=$2; shift 2 ;;
        --mem) MEM=$2; shift 2 ;;
        --ms) MS_FILTER=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if ! $DRY_RUN; then
    mkdir -p "$OUTPUT_DIR" "$SLURM_LOG_DIR"
fi

# Sweep grids (kept aligned with plot.py's expectations).
SWEEP_AISI=(0 0.005 0.01 0.02 0.05 0.1)
SWEEP_SW=(0 0.025 0.05 0.1 0.2 0.3)
SWEEP_Z=(0 0.1 0.2 0.3 0.4 0.5)

# z_width -> z_config CLI string.
z_to_cfg() {
    local w="$1"
    if awk "BEGIN {exit !($w == 0)}"; then
        echo "homogeneous:1.0"
    else
        local lo=$(awk "BEGIN {printf \"%.4g\", 1.0 - $w}")
        local hi=$(awk "BEGIN {printf \"%.4g\", 1.0 + $w}")
        echo "uniform:${lo}:${hi}"
    fi
}

count=0
submit_cell() {
    local label="$1"
    local b_cfg="$2"
    local z_cfg="$3"
    local aisi="$4"
    local sw="$5"
    local ms="$6"

    for (( i=0; i<NUM_BATCHES; i++ )); do
        local seed=$(( BASE_SEED_START + i ))
        count=$((count + 1))
        local out="${OUTPUT_DIR}/div_${label}_ms${ms}_aisi${aisi}_sw${sw}_z${z_cfg//:/-}_seed${seed}.csv"
        local job="div_${label}_ms${ms}_s${seed}"
        local cmd="sbatch \
            --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
            --job-name=${job} \
            --output=${SLURM_LOG_DIR}/${job}.%j.out \
            --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/campaigns/diversity/study.py \
    --n_min ${N} --n_max ${N} \
    --n_tech ${TECH_PER_JOB} --n_trials ${INITS_PER_TECH} \
    --nb_rounds ${NB_ROUNDS} \
    --cc ${CC} \
    --aisi_spread ${aisi} --sigma_w ${sw} \
    --a_config homogeneous:0.5 \
    --b_config ${b_cfg} \
    --z_config ${z_cfg} \
    --max_swaps ${ms} \
    --series_filter dif_init_only \
    --base_seed ${seed} --output ${out}'\""

        if $DRY_RUN; then
            echo "[$count] ${label}  ms=${ms}  b=${b_cfg}  z=${z_cfg}  aisi=${aisi}  sw=${sw}  seed=${seed}"
        else
            eval "$cmd"
        fi
    done
}

ms_active() {
    [[ -z "$MS_FILTER" ]] || [[ "$MS_FILTER" == "$1" ]]
}

# Top row: 4 b-regimes x 3 axis sweeps x 6 values.
TOP_LABELS=(CRS DRS IRS HRS)
TOP_B=("homogeneous:1.0" "homogeneous:0.9" "homogeneous:1.1" "uniform:0.9:1.1")

for ms in 1 4; do
    ms_active "$ms" || continue
    for (( idx=0; idx<${#TOP_LABELS[@]}; idx++ )); do
        lbl="${TOP_LABELS[$idx]}"
        bcfg="${TOP_B[$idx]}"
        # Panel (a)/(d): aisi sweep, sw=0, z=hom 1.0
        for v in "${SWEEP_AISI[@]}"; do
            submit_cell "top_${lbl}_panAisi" "$bcfg" "homogeneous:1.0" "$v" 0 "$ms"
        done
        # Panel (b)/(e): sw sweep, aisi=0, z=hom 1.0
        for v in "${SWEEP_SW[@]}"; do
            submit_cell "top_${lbl}_panSw" "$bcfg" "homogeneous:1.0" 0 "$v" "$ms"
        done
        # Panel (c)/(f): z sweep, aisi=0, sw=0
        for v in "${SWEEP_Z[@]}"; do
            zcfg=$(z_to_cfg "$v")
            submit_cell "top_${lbl}_panZ" "$bcfg" "$zcfg" 0 0 "$ms"
        done
    done
done

# Bottom row: HRS economy, six (series, panel) combinations.
HRS_B="uniform:0.9:1.1"
Z_FIXED_CFG="uniform:0.9:1.1"   # z_width = 0.1

for ms in 1 4; do
    ms_active "$ms" || continue

    # HRS+AiSi=0.05 on sigma_w panel
    for v in "${SWEEP_SW[@]}"; do
        submit_cell "bot_AiSi_panSw" "$HRS_B" "homogeneous:1.0" 0.05 "$v" "$ms"
    done
    # HRS+AiSi=0.05 on z_width panel
    for v in "${SWEEP_Z[@]}"; do
        zcfg=$(z_to_cfg "$v")
        submit_cell "bot_AiSi_panZ" "$HRS_B" "$zcfg" 0.05 0 "$ms"
    done

    # HRS+sw=0.1 on aisi panel
    for v in "${SWEEP_AISI[@]}"; do
        submit_cell "bot_Sw_panAisi" "$HRS_B" "homogeneous:1.0" "$v" 0.1 "$ms"
    done
    # HRS+sw=0.1 on z_width panel
    for v in "${SWEEP_Z[@]}"; do
        zcfg=$(z_to_cfg "$v")
        submit_cell "bot_Sw_panZ" "$HRS_B" "$zcfg" 0 0.1 "$ms"
    done

    # HRS+z=0.1 on aisi panel
    for v in "${SWEEP_AISI[@]}"; do
        submit_cell "bot_Z_panAisi" "$HRS_B" "$Z_FIXED_CFG" "$v" 0 "$ms"
    done
    # HRS+z=0.1 on sigma_w panel
    for v in "${SWEEP_SW[@]}"; do
        submit_cell "bot_Z_panSw" "$HRS_B" "$Z_FIXED_CFG" 0 "$v" "$ms"
    done

    # HRS+AiSi+sw on z_width panel
    for v in "${SWEEP_Z[@]}"; do
        zcfg=$(z_to_cfg "$v")
        submit_cell "bot_AiSiSw_panZ" "$HRS_B" "$zcfg" 0.05 0.1 "$ms"
    done

    # HRS+AiSi+z on sigma_w panel
    for v in "${SWEEP_SW[@]}"; do
        submit_cell "bot_AiSiZ_panSw" "$HRS_B" "$Z_FIXED_CFG" 0.05 "$v" "$ms"
    done

    # HRS+sw+z on aisi panel
    for v in "${SWEEP_AISI[@]}"; do
        submit_cell "bot_SwZ_panAisi" "$HRS_B" "$Z_FIXED_CFG" "$v" 0.1 "$ms"
    done
done

cells_per_ms=126
if [[ -n "$MS_FILTER" ]]; then
    total_cells=$cells_per_ms
else
    total_cells=$((2 * cells_per_ms))
fi
echo
echo "Done: ${count} jobs queued."
echo "  ms values    : ${MS_FILTER:-1, 4}"
echo "  base seeds   : ${BASE_SEED_START}..$((BASE_SEED_START + NUM_BATCHES - 1))"
echo "  per cell     : ${TECH_PER_JOB} tech x ${INITS_PER_TECH} inits x ${NUM_BATCHES} batches"
echo "  expected     : ${total_cells} cells (=${cells_per_ms} per ms)"
echo "  n=${N}, c=${C}, cc=${CC}, nb_rounds=${NB_ROUNDS}, mem=${MEM}, time=${TIME_LIMIT}"
