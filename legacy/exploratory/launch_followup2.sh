#!/bin/bash
#
# Round-2 follow-up sweep. Three targeted blocks:
#
#   A) z-scan -- Section 3b
#      Fine-grain scan of z heterogeneity at the aisi-trap-saturated operating
#      point (cc=4, ms=1, aisi=0.05, sigma_w=0, n=100, hom a/b). Tests whether
#      firm-level productivity heterogeneity monotonically suppresses diversity,
#      analogous to (and opposite to) the sigma_w story.
#
#   B) sigma_w top-up -- Section 3a
#      Adds 30+ tech matrices per cell to the seven sigma_w > 0 cells in the
#      original Goal 4b scan (cc=4, ms=4, aisi=0, n=100, hom a/b/z). Currently
#      those cells have only 5-11 tech matrices each, leaving CIs ~+/-0.07.
#
#   C) cc-scan at ms=cc -- Section 1
#      Confirm "ms=cc kills the aisi trap" across cc values. cc=3 at n=100;
#      cc=5 at n=50 because at n=100 the candidate enumeration (251 per firm-
#      decision) blows past the 48h time limit. cc=6 omitted: even at n=30 it
#      requires reduced n_trials to fit.
#
# Usage:
#   bash launch_followup2.sh                 # BASE_SEED=300, submits jobs
#   bash launch_followup2.sh 305             # BASE_SEED=305
#   bash launch_followup2.sh 300 --dry-run   # print sbatch commands only
#
# Recommended BASE_SEED: any value not used by the original sweep (0..9) or
# round-1 follow-up (200..204). Default 300.

set -e

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_sweep"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED_START=${1:-300}
DRY_RUN=false
[[ "$2" == "--dry-run" ]] && DRY_RUN=true

if ! $DRY_RUN; then
    mkdir -p "$OUTPUT_DIR" "$SLURM_LOG_DIR"
fi

# Per-job budget.
TECH_PER_JOB=10
NUM_BATCHES=5            # tech matrices per cell = TECH_PER_JOB * NUM_BATCHES
TIME_LIMIT="48:00:00"
MEM="6G"
NB_ROUNDS=200            # generous; cycle detection lets most trials exit fast
N=100

# Default uniform configs (kept for compatibility with submit's positional API,
# but every block in this script overrides them with HOM_* configs).
A_CONFIG="uniform:0.4:0.6"
B_CONFIG="uniform:0.9:1.1"
Z_CONFIG="uniform:0.8:1.2"

# Homogeneous baselines reused in all three blocks below.
HOM_A="homogeneous:0.5"
HOM_B="homogeneous:1.0"
HOM_Z="homogeneous:1.0"

# ----- one sbatch submission, looped over NUM_BATCHES BASE_SEEDs --------------
count=0
submit() {
    # cc ms aisi sw n_tech_target tag [nb_rounds] [n] [a_cfg b_cfg z_cfg] [n_trials]
    local cc=$1 ms=$2 aisi=$3 sw=$4 n_tech_target=$5 tag=$6
    local nb_rounds_override=${7:-${NB_ROUNDS}}
    local n_override=${8:-${N}}
    local a_cfg=${9:-${A_CONFIG}}
    local b_cfg=${10:-${B_CONFIG}}
    local z_cfg=${11:-${Z_CONFIG}}
    local n_trials_override=${12:-50}
    local n_tech_per_batch=${TECH_PER_JOB}
    local n_batches=${NUM_BATCHES}
    if (( n_tech_target < TECH_PER_JOB * NUM_BATCHES )); then
        n_batches=$(( (n_tech_target + TECH_PER_JOB - 1) / TECH_PER_JOB ))
    fi

    for (( i=0; i<n_batches; i++ )); do
        local seed=$(( BASE_SEED_START + i ))
        count=$((count + 1))
        local out="${OUTPUT_DIR}/diversity_n${n_override}_${tag}_seed${seed}.csv"
        local job="fu2_${tag}_s${seed}"
        local cmd="sbatch \
            --nodes=1 --time=${TIME_LIMIT} --mem=${MEM} --ntasks=1 \
            --job-name=${job} \
            --output=${SLURM_LOG_DIR}/${job}.%j.out \
            --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/scripts/diversity_study.py \
    --n_min ${n_override} --n_max ${n_override} --n_tech ${n_tech_per_batch} \
    --nb_rounds ${nb_rounds_override} --n_trials ${n_trials_override} \
    --b_config ${b_cfg} --a_config ${a_cfg} --z_config ${z_cfg} \
    --cc ${cc} --max_swaps ${ms} \
    --aisi_spread ${aisi} --sigma_w ${sw} \
    --base_seed ${seed} --output ${out}'\""

        if $DRY_RUN; then
            echo "[$count] n=${n_override} cc=${cc} ms=${ms} aisi=${aisi} sw=${sw} z=${z_cfg} nb_rounds=${nb_rounds_override} ntr=${n_trials_override} batch=${i}/${n_batches} seed=${seed}  ->  ${tag}"
        else
            eval "$cmd"
            echo "[$count] queued: ${tag} (seed=${seed})"
        fi
    done
}

# =============================================================================
# A) z-scan: (cc=4, ms=1, aisi=0.05, sigma_w=0, n=100, hom a/b) x z_spread
#    Tests whether firm-level z heterogeneity breaks the aisi trap.
# =============================================================================
echo "=== Block A: z fine-grain scan at the aisi-trap-saturated operating point ==="
declare -a Z_SPECS=(
    "homogeneous:1.0"      # z_spread = 0
    "uniform:0.95:1.05"    # z_spread = 0.05
    "uniform:0.9:1.1"      # z_spread = 0.10
    "uniform:0.75:1.25"    # z_spread = 0.25
    "uniform:0.5:1.5"      # z_spread = 0.50
    "uniform:0.25:1.75"    # z_spread = 0.75
)
for z_spec in "${Z_SPECS[@]}"; do
    # build a filename-safe tag
    z_tag=$(echo "$z_spec" | tr ':.' '__')
    submit 4 1 0.05 0.0 50 "zScan_${z_tag}" 200 100 \
           "${HOM_A}" "${HOM_B}" "${z_spec}" 50
done

# =============================================================================
# B) sigma_w top-up: (cc=4, ms=4, aisi=0, n=100, hom a/b/z) x sigma_w
#    Brings each Goal-4b cell to ~80 tech matrices total (existing ~5-11 plus
#    50 fresh). Tightens the plateau / sigma_w=0.3 transition story.
# =============================================================================
echo "=== Block B: sigma_w fine-grain top-up at (cc=4, ms=4, aisi=0, n=100) ==="
for sw in 0.025 0.05 0.075 0.1 0.15 0.2 0.3; do
    submit 4 4 0.0 ${sw} 50 "swTopup_w${sw}" 200 100 \
           "${HOM_A}" "${HOM_B}" "${HOM_Z}" 50
done

# =============================================================================
# C) cc-scan at ms=cc: confirm "ms=cc kills the aisi trap" across cc.
#    cc=3 at n=100 (cheap), cc=5 at n=50 (expensive at n=100).
#    cc=6 omitted: too expensive at any n we'd consider with full enumeration.
# =============================================================================
echo "=== Block C: cc-scan along ms=cc at (aisi=0.05, sigma_w=0) ==="
# cc=3 at n=100 -- standard cost
submit 3 3 0.05 0.0 50 "ccScan_cc3_n100" 200 100 \
       "${HOM_A}" "${HOM_B}" "${HOM_Z}" 50
# cc=5 at n=50 -- enumeration is heavy; reduce n to fit the time budget
submit 5 5 0.05 0.0 50 "ccScan_cc5_n50" 200 50 \
       "${HOM_A}" "${HOM_B}" "${HOM_Z}" 50

echo
echo "Done: $count jobs queued (BASE_SEED in [${BASE_SEED_START}, $((BASE_SEED_START + NUM_BATCHES - 1))], TECH_PER_JOB=${TECH_PER_JOB}, time=${TIME_LIMIT})"
