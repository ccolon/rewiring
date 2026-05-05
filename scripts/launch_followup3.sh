#!/bin/bash
#
# Follow-up 3: closes four open questions left after the visibility study and
# the homogeneous-corner / zNonDeg sweeps.
#
# All blocks run mode=full, NB_ROUNDS=200 (modern cycle detector), uniform-W
# baseline (sigma_w from each block), aisi from each block.
#
# -----------------------------------------------------------------------------
# Block A -- ms=cc + fully homogeneous + b varied (NEW question)
# -----------------------------------------------------------------------------
# Goal 4a only ran b=hom 1.0 at cc=4, ms=4, fully hom -- confirmed degenerate.
# We never measured (ms=cc, fully hom, sigma_w=0) at b!=1, so we don't know
# whether the path-dependent partial-diversity seen at small n,cc=2 (~0.40 at
# b=hom 0.9) survives at the canonical study point.
#
# Cells: n in {50, 100}, cc=4, ms=4, b in {hom 0.9, hom 1.0, hom 1.1}, fully
# homogeneous a/z, sigma_w=0, aisi=0.  -> 6 cells x 3 batches = 18 jobs.
#
# -----------------------------------------------------------------------------
# Block B -- Open question 1: IRS at n>=100 hom (DRS/IRS asymmetry persistence)
# -----------------------------------------------------------------------------
# n=30 hom 1.1 saturated only partially (0.30-0.77 across cc); n=100 hom 1.1
# never tested.  Closes the IRS gap with two cells.
#
# Cells: (n=100, b=hom 1.1, ms=1), (n=200, b=hom 1.1, ms=1), fully homogeneous.
# -> 2 cells x 2 batches = 4 jobs.
#
# -----------------------------------------------------------------------------
# Block C -- Open question 2: sigma_w in isolation at strict CRS, ms=1
# -----------------------------------------------------------------------------
# Goal 4b showed sigma_w produces residue at ms=cc; at ms=1 we only have data
# at b!=1 or aisi>0.  Mechanism predicts sigma_w alone (b=1, aisi=0, ms=1)
# should already produce diversity.
#
# Cells: n=100, cc=4, ms=1, b=hom 1.0, fully hom a/z, aisi=0, sigma_w in
#        {0.025, 0.05, 0.075, 0.1, 0.2}.  -> 5 cells x 1 batch = 5 jobs.
#
# -----------------------------------------------------------------------------
# Block D -- Open question 7: b heterogeneity near 1 (anchoring vs dispersion)
# -----------------------------------------------------------------------------
# b ~ U[0.5, 1.5] suppresses diversity (~0.15-0.35 at n=30).  Is the mechanism
# "firms near b=1 anchor the system" (then bands not crossing 1 should give
# *more* diversity) or "dispersion creates conflicting preferences" (then any
# wide band suppresses)?  We test two narrow bands offset from 1:
#   - b ~ U[0.85, 0.95]: pure DRS, no firm at CRS
#   - b ~ U[1.05, 1.15]: pure IRS, no firm at CRS
# Compared to the existing data on b ~ U[0.9, 1.1] (mixed, crosses 1).
#
# Cells: n=100, cc=4, ms=1, fully hom a/z, sigma_w=0, aisi=0, b varied as above.
# -> 2 cells x 3 batches = 6 jobs.
#
# -----------------------------------------------------------------------------
# Total: 18 + 4 + 5 + 6 = 33 jobs.  Per-job estimate: <= 6 h at n=100, ms=1;
# <= 12 h at n=200; ms=cc=4 cells finish quickly because dynamics terminate
# in <= 1-3 rounds at the homogeneous corner.  All comfortably under 48 h.
#
# Usage:
#     bash launch_followup3.sh                        # 33 jobs, BASE_SEED 1000-1002
#     bash launch_followup3.sh 1000 --dry-run         # print only
#     bash launch_followup3.sh 1010 --num-batches 5   # extend to 5 batches/cell

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_sweep"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED_START=${1:-1000}
DRY_RUN=false
NUM_BATCHES=3
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --num-batches) NUM_BATCHES=$2; shift 2 ;;
        *) shift ;;
    esac
done

if ! $DRY_RUN; then
    mkdir -p "$OUTPUT_DIR" "$SLURM_LOG_DIR"
fi

TIME_LIMIT="48:00:00"
MEM="6G"
NB_ROUNDS=200
TECH_PER_JOB=10
N_TRIALS=30

# Default fully-homogeneous parameters; per-block overrides happen via submit args.
HOM_A="homogeneous:0.5"
HOM_B="homogeneous:1.0"
HOM_Z="homogeneous:1.0"

count=0
submit() {
    # n  cc  ms  aisi  sw  a_cfg  b_cfg  z_cfg  n_tech_target  tag
    local n=$1 cc=$2 ms=$3 aisi=$4 sw=$5
    local a_cfg=$6 b_cfg=$7 z_cfg=$8
    local n_tech_target=$9 tag=${10}

    local n_tech_per_batch=${TECH_PER_JOB}
    local n_batches=${NUM_BATCHES}
    if (( n_tech_target < TECH_PER_JOB * NUM_BATCHES )); then
        n_batches=$(( (n_tech_target + TECH_PER_JOB - 1) / TECH_PER_JOB ))
    fi

    for (( i=0; i<n_batches; i++ )); do
        local seed=$(( BASE_SEED_START + i ))
        count=$((count + 1))
        local out="${OUTPUT_DIR}/fu3_n${n}_${tag}_seed${seed}.csv"
        local job="fu3_${tag}_s${seed}"
        local cmd="sbatch \
            --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
            --job-name=${job} \
            --output=${SLURM_LOG_DIR}/${job}.%j.out \
            --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/scripts/diversity_study.py \
    --n_min ${n} --n_max ${n} --n_tech ${n_tech_per_batch} \
    --nb_rounds ${NB_ROUNDS} --n_trials ${N_TRIALS} \
    --b_config ${b_cfg} --a_config ${a_cfg} --z_config ${z_cfg} \
    --cc ${cc} --max_swaps ${ms} \
    --aisi_spread ${aisi} --sigma_w ${sw} \
    --base_seed ${seed} --output ${out}'\""

        if $DRY_RUN; then
            echo "[$count] n=${n} cc=${cc} ms=${ms} aisi=${aisi} sw=${sw} b=${b_cfg} batch=${i}/${n_batches} seed=${seed}  ->  ${tag}"
        else
            eval "$cmd"
            echo "[$count] queued: ${tag} (seed=${seed})"
        fi
    done
}

# =============================================================================
# Block A -- ms=cc + fully homogeneous + b varied  (NEW)
# =============================================================================
echo "=== Block A: ms=cc=4, fully hom, sigma_w=0, b in {0.9, 1.0, 1.1}, n in {50, 100} ==="
for n in 50 100; do
    for b_val in 0.9 1.0 1.1; do
        b_cfg="homogeneous:${b_val}"
        tag="msEqCc_fullhom_b${b_val}"
        submit ${n} 4 4 0.0 0.0 "${HOM_A}" "${b_cfg}" "${HOM_Z}" 30 "${tag}"
    done
done

# =============================================================================
# Block B -- Open question 1: IRS at n>=100 hom
# =============================================================================
echo "=== Block B (Q1): b=hom 1.1, ms=1, fully hom, n in {100, 200} ==="
for n in 100 200; do
    submit ${n} 4 1 0.0 0.0 "${HOM_A}" "homogeneous:1.1" "${HOM_Z}" 20 "Q1_IRS_b1.1"
done

# =============================================================================
# Block C -- Open question 2: sigma_w in isolation at strict CRS, ms=1
# =============================================================================
echo "=== Block C (Q2): b=hom 1.0, ms=1, fully hom, aisi=0, sigma_w scan ==="
for sw in 0.025 0.05 0.075 0.1 0.2; do
    submit 100 4 1 0.0 ${sw} "${HOM_A}" "${HOM_B}" "${HOM_Z}" 10 "Q2_swCRS_w${sw}"
done

# =============================================================================
# Block D -- Open question 7: b heterogeneity near 1
# =============================================================================
echo "=== Block D (Q7): b uniform in narrow bands offset from 1, ms=1, fully hom ==="
submit 100 4 1 0.0 0.0 "${HOM_A}" "uniform:0.85:0.95" "${HOM_Z}" 30 "Q7_bU_DRS_0.85_0.95"
submit 100 4 1 0.0 0.0 "${HOM_A}" "uniform:1.05:1.15" "${HOM_Z}" 30 "Q7_bU_IRS_1.05_1.15"

echo
echo "Done: $count jobs queued (BASE_SEED in [${BASE_SEED_START}, $((BASE_SEED_START + NUM_BATCHES - 1))], "
echo "      TECH_PER_JOB=${TECH_PER_JOB}, NUM_BATCHES=${NUM_BATCHES}, N_TRIALS=${N_TRIALS}, time=${TIME_LIMIT})"
