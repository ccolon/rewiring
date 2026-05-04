#!/bin/bash
#
# Limited-visibility study (Section "limited visibility leads to endless
# rewiring"). At a fixed operating point in the AiSi-driven trap phase
# (aisi=0.05, sigma_w=0.05, ms=1, uniform a/b/z), sweep the per-firm tier
# visibility tau and characterise three regimes:
#   - converged       (rewirings == 0 in a full round, cycle_period = 1)
#   - period-k cycle  (k in [2, MAX_CYCLE_PERIOD], cycle_period = k)
#   - unstable        (hit nb_rounds without either)
#
# Four blocks:
#   A. Homogeneous tau at n=100, cc=4: tau in {0..6}.
#   B. Heterogeneous tau at n=100, cc=4: lognormal mean=tau, std=tau.
#   C. n-dependence (homogeneous tau, cc=4): n in {20, 60}, tau in {0..6}.
#   D. cc-dependence at small n (homogeneous tau, cc=2): n in {20, 60}.
#
# Operating point: aisi=0.05, sigma_w=0.05, ms=1, mode=limited;
# a~U[0.4,0.6], b~U[0.9,1.1], z~U[0.9,1.1].
#
# Usage:
#   bash launch_visibility.sh                  # BASE_SEED=600, submits jobs
#   bash launch_visibility.sh 605              # BASE_SEED=605
#   bash launch_visibility.sh 600 --dry-run    # print sbatch commands only

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_sweep"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED_START=${1:-600}
DRY_RUN=false
[[ "$2" == "--dry-run" ]] && DRY_RUN=true

if ! $DRY_RUN; then
    mkdir -p "$OUTPUT_DIR" "$SLURM_LOG_DIR"
fi

TIME_LIMIT="48:00:00"
MEM="6G"
NB_ROUNDS=200

# Mild a / b / z heterogeneity (matches the operating point in the manuscript).
A_CONFIG="uniform:0.4:0.6"
B_CONFIG="uniform:0.9:1.1"
Z_CONFIG="uniform:0.9:1.1"

# Trap-phase fixed parameters.
AISI=0.05
SW=0.05
MS=1

# ----- one sbatch submission with full per-call control ----------------------
count=0
submit() {
    # cc aisi sw tag nb_rounds n a_cfg b_cfg z_cfg n_trials tech_per_job num_batches mode tier_mean tier_std
    local cc=$1 aisi=$2 sw=$3 tag=$4
    local nb_rounds=${5:-${NB_ROUNDS}}
    local n=${6:-100}
    local a_cfg=${7:-${A_CONFIG}}
    local b_cfg=${8:-${B_CONFIG}}
    local z_cfg=${9:-${Z_CONFIG}}
    local n_trials=${10:-50}
    local tech_per_job=${11:-10}
    local num_batches=${12:-5}
    local mode=${13:-limited}
    local tier_mean=${14:-0}
    local tier_std=${15:-0}

    for (( i=0; i<num_batches; i++ )); do
        local seed=$(( BASE_SEED_START + i ))
        count=$((count + 1))
        local out="${OUTPUT_DIR}/diversity_n${n}_${tag}_seed${seed}.csv"
        local job="vis_${tag}_s${seed}"
        local cmd="sbatch \
            --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
            --job-name=${job} \
            --output=${SLURM_LOG_DIR}/${job}.%j.out \
            --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/scripts/diversity_study.py \
    --n_min ${n} --n_max ${n} --n_tech ${tech_per_job} \
    --nb_rounds ${nb_rounds} --n_trials ${n_trials} \
    --b_config ${b_cfg} --a_config ${a_cfg} --z_config ${z_cfg} \
    --cc ${cc} --max_swaps ${MS} \
    --aisi_spread ${aisi} --sigma_w ${sw} \
    --mode ${mode} --tier_mean ${tier_mean} --tier_std ${tier_std} \
    --base_seed ${seed} --output ${out}'\""

        if $DRY_RUN; then
            echo "[$count] n=${n} cc=${cc} tau=${tier_mean}+/-${tier_std} mode=${mode}  seed=${seed}  ->  ${tag}"
        else
            eval "$cmd"
            echo "[$count] queued: ${tag} (seed=${seed})"
        fi
    done
}

# Helper: a tau scan at fixed (n, cc, tier_std) with prefix.
# tier_std=0 -> homogeneous; tier_std=tau -> lognormal hetero with std=mean.
scan_tau_homo() {
    local n=$1 cc=$2 tpj=$3 nb=$4 prefix=$5
    for tau in 0 1 2 3 4 5 6; do
        submit ${cc} ${AISI} ${SW} "${prefix}_t${tau}" \
               ${NB_ROUNDS} ${n} \
               "${A_CONFIG}" "${B_CONFIG}" "${Z_CONFIG}" \
               50 ${tpj} ${nb} \
               limited ${tau} 0
    done
}

scan_tau_hetero() {
    local n=$1 cc=$2 tpj=$3 nb=$4 prefix=$5
    for tau in 0 1 2 3 4 5 6; do
        # tau=0: skip the heterogeneous draw (degenerate, all firms tau=0).
        local std=${tau}
        submit ${cc} ${AISI} ${SW} "${prefix}_t${tau}" \
               ${NB_ROUNDS} ${n} \
               "${A_CONFIG}" "${B_CONFIG}" "${Z_CONFIG}" \
               50 ${tpj} ${nb} \
               limited ${tau} ${std}
    done
}

# =============================================================================
# Block A -- homogeneous tau at n=100, cc=4 (headline figure)
# =============================================================================
echo "=== Block A: homo tau, n=100, cc=4 ==="
scan_tau_homo 100 4 5 10 "visibilityHomo_cc4"

# =============================================================================
# Block B -- heterogeneous (lognormal) tau at n=100, cc=4
# =============================================================================
echo "=== Block B: hetero tau (lognormal mean=std), n=100, cc=4 ==="
scan_tau_hetero 100 4 5 10 "visibilityHetero_cc4"

# =============================================================================
# Block C -- n-dependence of the bifurcation (homo tau, cc=4) at n in {20, 60}
# =============================================================================
echo "=== Block C: homo tau, cc=4, n in {20, 60} (n-dependence) ==="
scan_tau_homo 20 4 10 5 "visibilityHomo_cc4_n20"
scan_tau_homo 60 4 10 5 "visibilityHomo_cc4_n60"

# =============================================================================
# Block D -- cc-dependence (homo tau, cc=2) at n in {20, 60}
# =============================================================================
echo "=== Block D: homo tau, cc=2, n in {20, 60} (cc-dependence) ==="
scan_tau_homo 20 2 10 5 "visibilityHomo_cc2_n20"
scan_tau_homo 60 2 10 5 "visibilityHomo_cc2_n60"

echo
echo "Done: $count jobs queued (BASE_SEED in [${BASE_SEED_START}, $((BASE_SEED_START + 9))], time=${TIME_LIMIT})"
