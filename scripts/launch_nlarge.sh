#!/bin/bash
#
# Round-3 follow-up: n-scan ("size effects") at two complementary operating
# points. Both at ms=1 (realistic single-swap), uniform a/b/z (the same mild
# heterogeneity used in the original n=100 main sweep). 1-D scans over n.
#
#   A) aisi-trap onset (cc=4, ms=1, aisi=0.005, sigma_w=0)
#      Diversity at n=100 is ~0.38 (dif) / ~0.11 (same) -- transitional in
#      both series, so growth or decline with n is visible without ceiling
#      effects. Probes whether the trap saturation threshold shifts with n.
#
#   B) sigma_w residue (cc=4, ms=1, aisi=0, sigma_w=0.10)
#      Diversity at n=100 is ~0.41 (dif) / ~0.36 (same) -- intermediate.
#      Probes the link-noise-driven multiplicity in the action-restricted
#      regime, complementary to Goal 4c at ms=4.
#
# For each operating point, sweep n in {20, 50, 100, 200, 500}.
#
# Per-cell sample-size budget (TECH_PER_JOB x NUM_BATCHES = total tech):
#   n=20  : 10 x 5  = 50 tech, n_trials=50, ~100 min/job   (cheap)
#   n=50  : 10 x 5  = 50 tech, n_trials=50, ~7 h/job       (safe)
#   n=100 : 5  x 10 = 50 tech, n_trials=50, ~22 h/job      (split for margin)
#   n=200 : 1  x 10 = 10 tech, n_trials=50, ~27 h/job      (tighter CIs)
#   n=500 : 1  x 10 = 10 tech, n_trials=15, ~33 h/job      (n_trials reduced
#                                                           to fit 48 h)
#
# Usage:
#   bash launch_followup3.sh                 # BASE_SEED=400, submits jobs
#   bash launch_followup3.sh 405             # BASE_SEED=405
#   bash launch_followup3.sh 400 --dry-run   # print sbatch commands only

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_sweep"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED_START=${1:-400}
DRY_RUN=false
[[ "$2" == "--dry-run" ]] && DRY_RUN=true

if ! $DRY_RUN; then
    mkdir -p "$OUTPUT_DIR" "$SLURM_LOG_DIR"
fi

TIME_LIMIT="48:00:00"
MEM="6G"
NB_ROUNDS=200

# Original n=100 main-sweep configs (mild a, b, z heterogeneity).
A_CONFIG="uniform:0.4:0.6"
B_CONFIG="uniform:0.9:1.1"
Z_CONFIG="uniform:0.8:1.2"

# ----- one sbatch submission with full per-call control ----------------------
count=0
submit() {
    # cc ms aisi sw tag [nb_rounds] [n] [a_cfg] [b_cfg] [z_cfg] [n_trials] [tech_per_job] [num_batches]
    local cc=$1 ms=$2 aisi=$3 sw=$4 tag=$5
    local nb_rounds_override=${6:-${NB_ROUNDS}}
    local n_override=${7:-100}
    local a_cfg=${8:-${A_CONFIG}}
    local b_cfg=${9:-${B_CONFIG}}
    local z_cfg=${10:-${Z_CONFIG}}
    local n_trials_override=${11:-50}
    local tech_per_job=${12:-10}
    local num_batches=${13:-5}

    for (( i=0; i<num_batches; i++ )); do
        local seed=$(( BASE_SEED_START + i ))
        count=$((count + 1))
        local out="${OUTPUT_DIR}/diversity_n${n_override}_${tag}_seed${seed}.csv"
        local job="nlarge_${tag}_s${seed}"
        local cmd="sbatch \
            --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
            --job-name=${job} \
            --output=${SLURM_LOG_DIR}/${job}.%j.out \
            --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/scripts/diversity_study.py \
    --n_min ${n_override} --n_max ${n_override} --n_tech ${tech_per_job} \
    --nb_rounds ${nb_rounds_override} --n_trials ${n_trials_override} \
    --b_config ${b_cfg} --a_config ${a_cfg} --z_config ${z_cfg} \
    --cc ${cc} --max_swaps ${ms} \
    --aisi_spread ${aisi} --sigma_w ${sw} \
    --base_seed ${seed} --output ${out}'\""

        if $DRY_RUN; then
            echo "[$count] n=${n_override} cc=${cc} ms=${ms} aisi=${aisi} sw=${sw}  tpj=${tech_per_job} nb=${num_batches} ntr=${n_trials_override}  seed=${seed}  ->  ${tag}"
        else
            eval "$cmd"
            echo "[$count] queued: ${tag} (seed=${seed})"
        fi
    done
}

# Per-n scaling helper. Calls submit() with the right (TECH_PER_JOB, NUM_BATCHES)
# pair for each n, given a single tag prefix.
scan_over_n() {
    local cc=$1 ms=$2 aisi=$3 sw=$4 prefix=$5
    # n=20  : 10 x 5  = 50 tech matrices
    submit ${cc} ${ms} ${aisi} ${sw} "${prefix}_n20"  ${NB_ROUNDS}  20 \
           "${A_CONFIG}" "${B_CONFIG}" "${Z_CONFIG}" 50 10 5
    # n=50  : 10 x 5  = 50 tech matrices
    submit ${cc} ${ms} ${aisi} ${sw} "${prefix}_n50"  ${NB_ROUNDS}  50 \
           "${A_CONFIG}" "${B_CONFIG}" "${Z_CONFIG}" 50 10 5
    # n=100 : 5  x 10 = 50 tech matrices (split for time-budget safety)
    submit ${cc} ${ms} ${aisi} ${sw} "${prefix}_n100" ${NB_ROUNDS} 100 \
           "${A_CONFIG}" "${B_CONFIG}" "${Z_CONFIG}" 50 5  10
    # n=200 : 1  x 10 = 10 tech matrices
    submit ${cc} ${ms} ${aisi} ${sw} "${prefix}_n200" ${NB_ROUNDS} 200 \
           "${A_CONFIG}" "${B_CONFIG}" "${Z_CONFIG}" 50 1  10
    # n=500 : 1  x 10 = 10 tech matrices, n_trials reduced to 15
    # Per-eigsolve cost at n=500 is ~25x larger than at n=100, so 50 trials
    # blow past the 48 h budget; 15 keeps each job under ~33 h. The per-tech-
    # matrix diversity estimate is coarser (granularity 1/14 instead of 1/49)
    # but the cell mean across 10 tech matrices stays usable for the curve.
    submit ${cc} ${ms} ${aisi} ${sw} "${prefix}_n500" ${NB_ROUNDS} 500 \
           "${A_CONFIG}" "${B_CONFIG}" "${Z_CONFIG}" 15 1  10
}

# =============================================================================
# A) aisi-trap onset n-scan
# =============================================================================
echo "=== Block A: n-scan at (cc=4, ms=1, aisi=0.005, sigma_w=0, uniform a/b/z) ==="
scan_over_n 4 1 0.005 0.0 "aisiOnset"

# =============================================================================
# B) sigma_w residue n-scan
# =============================================================================
echo "=== Block B: n-scan at (cc=4, ms=1, aisi=0, sigma_w=0.10, uniform a/b/z) ==="
scan_over_n 4 1 0.0 0.1 "swResidue"

echo
echo "Done: $count jobs queued (BASE_SEED in [${BASE_SEED_START}, $((BASE_SEED_START + 9))], time=${TIME_LIMIT})"
