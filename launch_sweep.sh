#!/bin/bash
#
# Orchestrator: submits one sbatch job per parameter combination.
# 384 jobs = 4(b) x 3(cc) x 2(max_swaps) x 2(aisi_spread) x 2(z) x 2(a) x 2(sigma_w)
#
# Usage:  bash launch_sweep.sh          (submits all jobs)
#         bash launch_sweep.sh --dry-run (prints commands without submitting)

SCRIPT_DIR=/projects/disruptsc/rewiring_vAA
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_sweep"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

mkdir -p "$OUTPUT_DIR" "$SLURM_LOG_DIR"

count=0

for b_config in "homogeneous:0.9" "homogeneous:1.0" "homogeneous:1.1" "uniform:0.5:1.5"; do
for cc in 1 2 4; do
for max_swaps in 1 2; do
for aisi_spread in 0.0 0.1; do
for z_config in "homogeneous:1.0" "uniform:0.5:2.0"; do
for a_config in "homogeneous:0.5" "uniform:0.3:0.7"; do
for sigma_w in 0.0 0.1; do

    # Build a short tag for filenames
    b_tag=$(echo "$b_config" | tr ':' '_')
    z_tag=$(echo "$z_config" | tr ':' '_')
    a_tag=$(echo "$a_config" | tr ':' '_')
    tag="b${b_tag}_cc${cc}_ms${max_swaps}_as${aisi_spread}_z${z_tag}_a${a_tag}_sw${sigma_w}"

    output_csv="${OUTPUT_DIR}/diversity_n20_${tag}.csv"
    job_name="div_${tag}"

    cmd="sbatch \
        --nodes=1 \
        --time=48:00:00 \
        --mem=8G \
        --ntasks=1 \
        --job-name=${job_name} \
        --output=${SLURM_LOG_DIR}/${job_name}.%j.out \
        --wrap=\"module purge && \
source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/diversity_study.py \
    --n_min 20 --n_max 20 \
    --b_config ${b_config} \
    --a_config ${a_config} \
    --z_config ${z_config} \
    --cc ${cc} \
    --max_swaps ${max_swaps} \
    --aisi_spread ${aisi_spread} \
    --sigma_w ${sigma_w} \
    --output ${output_csv}\""

    count=$((count + 1))

    if $DRY_RUN; then
        echo "[$count] $cmd"
        echo ""
    else
        eval "$cmd"
        echo "[$count] Submitted: $job_name"
    fi

done
done
done
done
done
done
done

echo ""
echo "Total jobs: $count"
