#!/bin/bash
#
# Follow-up sweep that addresses three gaps identified by the post-processing
# of `diversity_n100.csv`:
#
#   1) AISI threshold scan
#      Fixed (cc=4, max_swaps=1, sigma_w=0); aisi in {0, 0.005, 0.01, 0.02, 0.03}
#      with 50 tech matrices each. The aisi=0.05 point is intentionally omitted
#      here because Goal 2 (max_swaps=1) already produces the same parameter
#      cell with the same BASE_SEED -- skipping the duplicate avoids redundant
#      compute and avoids inflating `sample_size` if the aggregator does not
#      deduplicate by `tech_seed`.
#
#   2) MAX_SWAPS release scan
#      Fixed (cc=4, aisi=0.05, sigma_w=0); max_swaps in {1, 2, 3, 4} with 50
#      tech matrices each. ms=3 and ms=4 already exceed 50 in the existing CSV,
#      but we re-run them here so the four points share the same Monte Carlo
#      stream and can be compared directly.
#
#   3) Top-ups for under-sampled cells (sample_size < 10 in diversity_n100.csv)
#      Adds 30 fresh tech matrices per cell. Cells already covered by Goal 2
#      are NOT re-listed here.
#
# Usage:
#   bash launch_followup.sh                  # BASE_SEED=100, submits jobs
#   bash launch_followup.sh 101              # BASE_SEED=101
#   bash launch_followup.sh 100 --dry-run    # print sbatch commands only
#
# Use a BASE_SEED that has never been used in the original sweep
# (the original launcher used 0..9). Recommended: BASE_SEED >= 100.

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_sweep"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED=${1:-100}
DRY_RUN=false
[[ "$2" == "--dry-run" ]] && DRY_RUN=true

if ! $DRY_RUN; then
    mkdir -p "$OUTPUT_DIR" "$SLURM_LOG_DIR"
fi

# ----- fixed background parameters (match the original sweep) ----------------
N=100
A_CONFIG="uniform:0.4:0.6"
B_CONFIG="uniform:0.9:1.1"
Z_CONFIG="uniform:0.8:1.2"

# ----- one sbatch submission -------------------------------------------------
count=0
submit() {
    local n_tech=$1 cc=$2 ms=$3 aisi=$4 sw=$5 tag=$6
    count=$((count + 1))
    local out="${OUTPUT_DIR}/diversity_n${N}_${tag}_seed${BASE_SEED}.csv"
    local job="fu_${tag}_s${BASE_SEED}"
    local cmd="sbatch \
        --nodes=1 --time=4:00:00 --mem=4G --ntasks=1 \
        --job-name=${job} \
        --output=${SLURM_LOG_DIR}/${job}.%j.out \
        --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/scripts/diversity_study.py \
    --n_min ${N} --n_max ${N} --n_tech ${n_tech} \
    --b_config ${B_CONFIG} --a_config ${A_CONFIG} --z_config ${Z_CONFIG} \
    --cc ${cc} --max_swaps ${ms} \
    --aisi_spread ${aisi} --sigma_w ${sw} \
    --base_seed ${BASE_SEED} --output ${out}'\""

    if $DRY_RUN; then
        echo "[$count] cc=${cc} ms=${ms} aisi=${aisi} sw=${sw} n_tech=${n_tech}  ->  ${tag}"
    else
        eval "$cmd"
        echo "[$count] queued: ${tag}"
    fi
}

# =============================================================================
# Goal 1 -- AISI THRESHOLD
# =============================================================================
echo "=== Goal 1: aisi threshold (cc=4, ms=1, sw=0) ==="
for aisi in 0.0 0.005 0.01 0.02 0.03; do
    submit 50 4 1 ${aisi} 0.0 "aisiThr_a${aisi}"
done

# =============================================================================
# Goal 2 -- MAX_SWAPS RELEASE
# =============================================================================
echo "=== Goal 2: max_swaps release (cc=4, aisi=0.05, sw=0) ==="
for ms in 1 2 3 4; do
    submit 50 4 ${ms} 0.05 0.0 "msRel_m${ms}"
done

# =============================================================================
# Goal 3 -- TOP-UPS for cells with sample_size < 10
# Cells covered by Goal 2 (cc=4, ms in {1,2}, aisi=0.05, sw=0) are NOT here.
# Format: cc  ms  aisi  sw
# =============================================================================
echo "=== Goal 3: top-ups for thin cells ==="
while IFS=' ' read -r cc ms aisi sw; do
    [[ -z "$cc" || "${cc:0:1}" == "#" ]] && continue
    submit 30 ${cc} ${ms} ${aisi} ${sw} "topup_cc${cc}m${ms}a${aisi}w${sw}"
done <<'CELLS'
# cc ms aisi  sw
2  2  0.0   0.1
3  2  0.0   0.0
3  2  0.0   0.05
3  1  0.0   0.1
3  2  0.0   0.1
3  2  0.05  0.0
3  2  0.05  0.05
3  1  0.05  0.1
3  2  0.05  0.1
3  2  0.1   0.0
3  2  0.1   0.05
3  2  0.1   0.1
4  2  0.0   0.0
4  1  0.0   0.05
4  2  0.0   0.05
4  1  0.0   0.1
4  2  0.0   0.1
4  1  0.05  0.05
4  2  0.05  0.05
4  1  0.05  0.1
4  2  0.05  0.1
4  1  0.1   0.0
4  2  0.1   0.0
4  1  0.1   0.05
4  2  0.1   0.05
4  1  0.1   0.1
4  2  0.1   0.1
CELLS

echo
echo "Done: $count jobs queued (BASE_SEED=${BASE_SEED})"
