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
#   bash launch_followup.sh                  # BASE_SEED_START=100, NUM_BATCHES=5
#   bash launch_followup.sh 200              # BASE_SEED_START=200
#   bash launch_followup.sh 100 --dry-run    # print sbatch commands only
#
# Each cell is submitted NUM_BATCHES times, each with a distinct BASE_SEED in
# {BASE_SEED_START, ..., BASE_SEED_START + NUM_BATCHES - 1}. Each batch runs
# TECH_PER_JOB tech matrices, so each cell gets NUM_BATCHES * TECH_PER_JOB tech
# matrices in total. Splitting keeps individual jobs comfortably under the
# SLURM wall-time cap.
#
# Use BASE_SEED_START values not consumed by previous launches (the original
# sweep used 0..9; the first followup attempt likely used 100). Pick e.g. 200.

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_sweep"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED_START=${1:-100}
DRY_RUN=false
[[ "$2" == "--dry-run" ]] && DRY_RUN=true

# Per-job work; tune TECH_PER_JOB down if jobs still hit the wall.
# Cluster wall-time caps (sinfo -o "%P %l"):
#   generic / generic_plus / matlab_batch  : 10 days
#   highfreq / hi9 / gpu                   : 5 days
#   matlab_interactive                     : 8 h
#   jup_kernel                             : 4 h    <-- the old 4h limit was here
# We use the default `generic` partition; 48 h is a safe envelope for the
# heavier (ms=2/3, cc=4) cells while staying well under the 10-day cap.
TECH_PER_JOB=10
NUM_BATCHES=5            # total tech matrices per cell = TECH_PER_JOB * NUM_BATCHES
TIME_LIMIT="48:00:00"
MEM="6G"
NB_ROUNDS=30             # bumped from the diversity_study.py default of 20

if ! $DRY_RUN; then
    mkdir -p "$OUTPUT_DIR" "$SLURM_LOG_DIR"
fi

# ----- fixed background parameters (match the original sweep) ----------------
N=100
A_CONFIG="uniform:0.4:0.6"
B_CONFIG="uniform:0.9:1.1"
Z_CONFIG="uniform:0.8:1.2"

# ----- one sbatch submission, looped over NUM_BATCHES BASE_SEEDs --------------
count=0
submit() {
    # cc ms aisi sw n_tech_target tag [nb_rounds_override] [n_override] [a_cfg b_cfg z_cfg]
    # n_tech_target is the *total* tech matrices wanted for this cell;
    # we split it into NUM_BATCHES jobs of TECH_PER_JOB matrices each.
    local cc=$1 ms=$2 aisi=$3 sw=$4 n_tech_target=$5 tag=$6
    local nb_rounds_override=${7:-${NB_ROUNDS}}
    local n_override=${8:-${N}}
    local a_cfg=${9:-${A_CONFIG}}
    local b_cfg=${10:-${B_CONFIG}}
    local z_cfg=${11:-${Z_CONFIG}}
    local n_tech_per_batch=${TECH_PER_JOB}
    local n_batches=${NUM_BATCHES}
    # If caller asks for fewer than the standard target, scale batches down.
    if (( n_tech_target < TECH_PER_JOB * NUM_BATCHES )); then
        n_batches=$(( (n_tech_target + TECH_PER_JOB - 1) / TECH_PER_JOB ))
    fi

    for (( i=0; i<n_batches; i++ )); do
        local seed=$(( BASE_SEED_START + i ))
        count=$((count + 1))
        local out="${OUTPUT_DIR}/diversity_n${n_override}_${tag}_seed${seed}.csv"
        local job="fu_${tag}_s${seed}"
        local cmd="sbatch \
            --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
            --job-name=${job} \
            --output=${SLURM_LOG_DIR}/${job}.%j.out \
            --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/scripts/diversity_study.py \
    --n_min ${n_override} --n_max ${n_override} --n_tech ${n_tech_per_batch} --nb_rounds ${nb_rounds_override} \
    --b_config ${b_cfg} --a_config ${a_cfg} --z_config ${z_cfg} \
    --cc ${cc} --max_swaps ${ms} \
    --aisi_spread ${aisi} --sigma_w ${sw} \
    --base_seed ${seed} --output ${out}'\""

        if $DRY_RUN; then
            echo "[$count] n=${n_override} cc=${cc} ms=${ms} aisi=${aisi} sw=${sw} nb_rounds=${nb_rounds_override} batch=${i}/${n_batches} seed=${seed}  ->  ${tag}"
        else
            eval "$cmd"
            echo "[$count] queued: ${tag} (seed=${seed})"
        fi
    done
}

# Configs used for the homogeneous-economy follow-up sweeps (Goals 4 and 5).
# Goals 1-3 keep the original sweep's uniform a/b/z.
HOM_A="homogeneous:0.5"
HOM_B="homogeneous:1.0"
HOM_Z="homogeneous:1.0"

# =============================================================================
# Goal 1 -- AISI THRESHOLD
# =============================================================================
echo "=== Goal 1: aisi threshold (cc=4, ms=1, sw=0) ==="
for aisi in 0.0 0.005 0.01 0.02 0.03; do
    submit 4 1 ${aisi} 0.0 50 "aisiThr_a${aisi}"
done

# =============================================================================
# Goal 2 -- MAX_SWAPS RELEASE
# =============================================================================
echo "=== Goal 2: max_swaps release (cc=4, aisi=0.05, sw=0) ==="
for ms in 1 2 3 4; do
    submit 4 ${ms} 0.05 0.0 50 "msRel_m${ms}"
done

# =============================================================================
# Goal 3 -- TOP-UPS for cells with sample_size < 10
# Cells covered by Goal 2 (cc=4, ms in {1,2}, aisi=0.05, sw=0) are NOT here.
# Format: cc  ms  aisi  sw
# =============================================================================
echo "=== Goal 3: top-ups for thin cells ==="
while IFS=' ' read -r cc ms aisi sw; do
    [[ -z "$cc" || "${cc:0:1}" == "#" ]] && continue
    submit ${cc} ${ms} ${aisi} ${sw} 30 "topup_cc${cc}m${ms}a${aisi}w${sw}"
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

# =============================================================================
# Goal 4 -- "Is ms=cc always diversity=0?" residue characterisation
# Always: max_swaps == cc, aisi=0, fully-homogeneous a/b/z, dif_init+same_init.
# nb_rounds bumped to 200 so cycle detection has room to fire.
# =============================================================================
echo "=== Goal 4a: ms=cc residue, cc-scan along the diagonal ==="
# 4a: cc-scan at ms=cc, with σ_w in {0.0, 0.1}.
# Capped at cc=6 because ms=cc=8 enumerates >12000 candidates per firm-decision.
for cc in 2 3 4 5 6; do
    submit ${cc} ${cc} 0.0 0.0 50 "msEqCc_cc${cc}_w0.0" 200 100 "${HOM_A}" "${HOM_B}" "${HOM_Z}"
    submit ${cc} ${cc} 0.0 0.1 50 "msEqCc_cc${cc}_w0.1" 200 100 "${HOM_A}" "${HOM_B}" "${HOM_Z}"
done

echo "=== Goal 4b: ms=cc residue, σ_w fine-grain at cc=4 ==="
# 4b: σ_w fine grain at (cc=4, ms=4, aisi=0, n=100).
for sw in 0.0 0.025 0.05 0.075 0.1 0.15 0.2 0.3; do
    submit 4 4 0.0 ${sw} 50 "msEqCc_swScan_w${sw}" 200 100 "${HOM_A}" "${HOM_B}" "${HOM_Z}"
done

echo "=== Goal 4c: ms=cc residue, n-scan at σ_w=0.1 ==="
# 4c: network-size scan at the σ_w-driven residue corner.
# n=500 omitted: full GE solve scaling makes it >24h per trial.
for nval in 20 50 100 200; do
    submit 4 4 0.0 0.1 50 "msEqCc_nScan_n${nval}" 200 ${nval} "${HOM_A}" "${HOM_B}" "${HOM_Z}"
done

# =============================================================================
# Goal 5 -- Long nb_rounds confirmation on the non-convergence corner
# Cells where conv_below_1.csv showed frac_converged < 1: re-run with
# nb_rounds=200 and the new cycle detector to separate "long transient" from
# "limit cycle" from "true non-convergence".
# Targets exactly the cells flagged in the user's diagnostic CSV.
# =============================================================================
echo "=== Goal 5: long-rounds re-run on the (aisi=0, σ_w>0, ms=cc) corner ==="
for nval in 20 50 100; do
    for sw in 0.1 0.15 0.2; do
        submit 4 4 0.0 ${sw} 50 "longRound_n${nval}_cc4_w${sw}" 200 ${nval} "${HOM_A}" "${HOM_B}" "${HOM_Z}"
    done
done
# Plus the (cc=2, ms=2) cell that showed frac_converged=0.92 in the diagnostic.
submit 2 2 0.0 0.1 50 "longRound_n100_cc2_w0.1" 200 100 "${HOM_A}" "${HOM_B}" "${HOM_Z}"

echo
echo "Done: $count jobs queued (BASE_SEED in [${BASE_SEED_START}, $((BASE_SEED_START + NUM_BATCHES - 1))], TECH_PER_JOB=${TECH_PER_JOB}, time=${TIME_LIMIT})"
