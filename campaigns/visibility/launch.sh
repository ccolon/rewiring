#!/bin/bash
#
# Visibility campaign (manuscript fig:visibility).
#
# Single canonical launcher that fans out EVERY cell needed for
# campaigns/visibility/plot.py to render figure_visibility.png (3 panels):
#
#   (a) Homo tau   : convergence + cycling fractions vs tau in {0..6}
#                    for R1 (full het), R2 (firm-level only), R3 (homo DRS)
#   (b) Hetero tau : same metrics, hetero-tau Poisson(lambda = tau_mean)
#   (c) Per-firm cost gap: theta_rel_i binned by firm's tau_i,
#                          R1 / R2 / R3 hetero-tau cells
#
# All cells share: n=100, c=c'=4, ms=1, mode=limited.
#
# === PHASE 1 -- visibility (panels a, b) ===
# Output dir: results/visibility/. Orchestrator: campaigns/visibility/study.py.
#
#   3 series x 2 tau_modes = 6 cells. Each cell sweeps tau in {0..6} internally
#   via --tau_values "0,1,2,3,4,5,6". Per-cell budget = TECH_PER_JOB tech
#   matrices, split over NUM_BATCHES sbatch jobs.
#
#   Series:
#     R1 -- aisi=0.05, sw=0.05, uniform a/b/z
#     R2 -- aisi=0,    sw=0,    uniform a/b/z
#     R3 -- aisi=0,    sw=0,    homo DRS (a=0.5, b=0.9, z=1.0)
#
# === PHASE 2 -- cost-gap (panel c) ===
# Output dir: results/cost_gap/. Orchestrator: campaigns/visibility/cost_study.py.
#
#   3 series x 3 hetero-tau cells (lam in {0, 1, 2}) = 9 cells. The static-gap
#   query (theta_static + per-firm columns) runs at static_max_swaps = min(c, c')
#   = 4 -- the unconstrained best deviation per firm at the terminal state.
#
# Total: 6 + 9 = 15 cells; with NUM_BATCHES=2 = 30 sbatch jobs.
#
# Seed scheme: PHASE_1_BASE_SEED = 950 (visibility), PHASE_2_BASE_SEED = 1700
# (cost-gap). Both offset from BASE_SEED_START via PHASE offsets so a single
# CLI knob controls both phases.
#
# Usage:
#     bash campaigns/visibility/launch.sh                          # full
#     bash campaigns/visibility/launch.sh 950 --dry-run            # preview
#     bash campaigns/visibility/launch.sh 950 --num-batches 1      # 15 jobs
#     bash campaigns/visibility/launch.sh 950 --phase visibility   # phase 1 only
#     bash campaigns/visibility/launch.sh 950 --phase cost         # phase 2 only

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
VIS_OUTPUT_DIR="${SCRIPT_DIR}/results/visibility"
COST_OUTPUT_DIR="${SCRIPT_DIR}/results/cost_gap"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

PHASE_1_BASE_SEED=${1:-950}
PHASE_2_BASE_SEED=$(( PHASE_1_BASE_SEED + 750 ))  # 950 -> 1700
DRY_RUN=false
NUM_BATCHES=2
TECH_PER_JOB=50           # phase 1 (visibility); phase 2 overrides below
INITS_PER_TECH=40         # phase 2 only
PHASE_2_TECH=3            # phase 2 only
NB_ROUNDS_VIS=200
NB_ROUNDS_COST=200
TIME_LIMIT="48:00:00"
MEM="2G"
N=100
C=4
CC=4
MS=1
PHASE_FILTER=""           # empty = both; "visibility" or "cost" restricts

shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --num-batches) NUM_BATCHES=$2; shift 2 ;;
        --tech_per_job) TECH_PER_JOB=$2; shift 2 ;;
        --inits_per_tech) INITS_PER_TECH=$2; shift 2 ;;
        --nb_rounds_vis) NB_ROUNDS_VIS=$2; shift 2 ;;
        --nb_rounds_cost) NB_ROUNDS_COST=$2; shift 2 ;;
        --time) TIME_LIMIT=$2; shift 2 ;;
        --mem) MEM=$2; shift 2 ;;
        --phase) PHASE_FILTER=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if ! $DRY_RUN; then
    mkdir -p "$VIS_OUTPUT_DIR" "$COST_OUTPUT_DIR" "$SLURM_LOG_DIR"
fi

# Series definitions (R1, R2, R3) -- shared between phases.
# Each entry: tag a_cfg b_cfg z_cfg aisi sw
R1=(R1 uniform:0.4:0.6 uniform:0.9:1.1 uniform:0.9:1.1 0.05 0.05)
R2=(R2 uniform:0.4:0.6 uniform:0.9:1.1 uniform:0.9:1.1 0    0)
R3=(R3 homogeneous:0.5 homogeneous:0.9 homogeneous:1.0 0    0)
SERIES_KEYS=(R1 R2 R3)

phase_active() {
    [[ -z "$PHASE_FILTER" ]] || [[ "$PHASE_FILTER" == "$1" ]]
}

count=0

# ============================================================================
# PHASE 1 -- visibility (panels a, b)
# ============================================================================
submit_visibility_cell() {
    # series_key tau_mode
    local key="$1"
    local tau_mode="$2"

    local -n S="$key"
    local tag="${S[0]}" a_cfg="${S[1]}" b_cfg="${S[2]}" z_cfg="${S[3]}"
    local aisi="${S[4]}" sw="${S[5]}"

    for (( i=0; i<NUM_BATCHES; i++ )); do
        local seed=$(( PHASE_1_BASE_SEED + i ))
        count=$((count + 1))
        local out="${VIS_OUTPUT_DIR}/vis_${tag}_${tau_mode}_seed${seed}.csv"
        local job="vis_${tag}_${tau_mode}_s${seed}"
        local cmd="sbatch \
            --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
            --job-name=${job} \
            --output=${SLURM_LOG_DIR}/${job}.%j.out \
            --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/campaigns/visibility/study.py \
    --n ${N} --tech_per_job ${TECH_PER_JOB} --nb_rounds ${NB_ROUNDS_VIS} \
    --cc ${CC} --max_swaps ${MS} \
    --aisi_spread ${aisi} --sigma_w ${sw} \
    --a_config ${a_cfg} --b_config ${b_cfg} --z_config ${z_cfg} \
    --tau_values 0,1,2,3,4,5,6 --tau_mode ${tau_mode} \
    --base_seed ${seed} --output ${out}'\""

        if $DRY_RUN; then
            echo "[$count] phase=visibility  ${tag}  tau_mode=${tau_mode}  seed=${seed}"
        else
            eval "$cmd"
        fi
    done
}

if phase_active visibility; then
    for key in "${SERIES_KEYS[@]}"; do
        submit_visibility_cell "$key" homo
        submit_visibility_cell "$key" hetero
    done
fi

# ============================================================================
# PHASE 2 -- cost-gap (panel c)
# ============================================================================
submit_cost_cell() {
    # series_key tier_mean tier_std
    local key="$1"
    local tier_mean="$2"
    local tier_std="$3"

    local -n S="$key"
    local tag="${S[0]}" a_cfg="${S[1]}" b_cfg="${S[2]}" z_cfg="${S[3]}"
    local aisi="${S[4]}" sw="${S[5]}"

    for (( i=0; i<NUM_BATCHES; i++ )); do
        local seed=$(( PHASE_2_BASE_SEED + i ))
        count=$((count + 1))
        local out="${COST_OUTPUT_DIR}/coststat_n${N}_${tag}_lam${tier_mean}_seed${seed}.csv"
        local job="cost_${tag}_lam${tier_mean}_s${seed}"
        local mode_kwargs=""
        if [[ "$tier_mean" == "0" ]]; then
            mode_kwargs="--tier_mean 0 --tier_std 0"
        else
            mode_kwargs="--tier_mean ${tier_mean} --tier_std ${tier_std} --tier_dist poisson"
        fi
        local cmd="sbatch \
            --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
            --job-name=${job} \
            --output=${SLURM_LOG_DIR}/${job}.%j.out \
            --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/campaigns/visibility/cost_study.py \
    --n ${N} --c ${C} --cc ${CC} \
    --max_swaps ${MS} \
    --aisi_spread ${aisi} --sigma_w ${sw} \
    --a_config ${a_cfg} --b_config ${b_cfg} --z_config ${z_cfg} \
    --mode limited ${mode_kwargs} \
    --tech_per_job ${PHASE_2_TECH} --inits_per_tech ${INITS_PER_TECH} \
    --nb_rounds ${NB_ROUNDS_COST} \
    --base_seed ${seed} --output ${out}'\""

        if $DRY_RUN; then
            echo "[$count] phase=cost  ${tag}  lam=${tier_mean}  seed=${seed}"
        else
            eval "$cmd"
        fi
    done
}

if phase_active cost; then
    for key in "${SERIES_KEYS[@]}"; do
        for lam in 0 1 2; do
            submit_cost_cell "$key" "$lam" "$lam"
        done
    done
fi

echo
echo "Done: ${count} jobs queued."
echo "  phase filter : ${PHASE_FILTER:-visibility + cost}"
echo "  visibility base seeds : ${PHASE_1_BASE_SEED}..$((PHASE_1_BASE_SEED + NUM_BATCHES - 1))"
echo "  cost base seeds       : ${PHASE_2_BASE_SEED}..$((PHASE_2_BASE_SEED + NUM_BATCHES - 1))"
echo "  visibility per cell   : ${TECH_PER_JOB} tech x 7 tau values (homo or hetero)"
echo "  cost       per cell   : ${PHASE_2_TECH} tech x ${INITS_PER_TECH} inits = $((PHASE_2_TECH * INITS_PER_TECH)) trials"
echo "  expected cells: 6 visibility + 9 cost = 15"
echo "  n=${N}, c=${C}, cc=${CC}, ms=${MS}, mode=limited, mem=${MEM}, time=${TIME_LIMIT}"
