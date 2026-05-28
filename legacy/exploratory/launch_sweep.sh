#!/bin/bash
#
# Orchestrator: submits one sbatch job per parameter combination.
#
# Usage:  bash launch_sweep.sh                  (BASE_SEED=0)
#         bash launch_sweep.sh 3                (BASE_SEED=3)
#         bash launch_sweep.sh 0 --dry-run      (print without submitting)
#
# Monte Carlo: run repeatedly with different BASE_SEED values, e.g.
#         for s in 0 1 2 3 4 5 6 7 8 9; do bash launch_sweep.sh $s; done
# Each base_seed gives an independent RNG stream and writes distinct output files.

SCRIPT_DIR="/projects/disruptsc/rewiring_vAA"
PYTHON_ENV="/projects/disruptsc/miniforge3/envs/rewiring"
OUTPUT_DIR="${SCRIPT_DIR}/results_sweep"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm_logs"

BASE_SEED=${1:-0}

DRY_RUN=false
if [[ "$2" == "--dry-run" ]]; then
    DRY_RUN=true
fi

mkdir -p "$OUTPUT_DIR" "$SLURM_LOG_DIR"

count=0

#"homogeneous:0.9" "homogeneous:1.0" "homogeneous:1.1"
for b_config in  "uniform:0.9:1.1"; do
for cc in 1 2 3 4; do
for max_swaps in 1 2 3 4; do
for aisi_spread in 0.0 0.05 0.1; do
for z_config in "uniform:0.8:1.2"; do
for a_config in "uniform:0.4:0.6"; do
for sigma_w in 0.0 0.05 0.1; do
for n in 100; do
    if [ "$max_swaps" -gt "$cc" ]; then
        continue
    fi
    # Build a short tag for filenames
    b_tag=$(echo "$b_config" | tr ':' '_')
    z_tag=$(echo "$z_config" | tr ':' '_')
    a_tag=$(echo "$a_config" | tr ':' '_')
    tag="b${b_tag}_cc${cc}_ms${max_swaps}_as${aisi_spread}_z${z_tag}_a${a_tag}_sw${sigma_w}"

    output_csv="${OUTPUT_DIR}/diversity_n20_${tag}_seed${BASE_SEED}.csv"
    job_name="div_${tag}_s${BASE_SEED}"

    cmd="sbatch \
        --nodes=1 \
        --time=4:00:00 \
        --mem=4G \
        --ntasks=1 \
        --job-name=${job_name} \
        --output=${SLURM_LOG_DIR}/${job_name}.%j.out \
        --wrap=\"bash -c 'source /projects/disruptsc/miniforge3/bin/activate ${PYTHON_ENV} && \
python ${SCRIPT_DIR}/scripts/diversity_study.py \
    --n_min ${n} --n_max ${n} \
    --b_config ${b_config} \
    --a_config ${a_config} \
    --z_config ${z_config} \
    --cc ${cc} \
    --max_swaps ${max_swaps} \
    --aisi_spread ${aisi_spread} \
    --sigma_w ${sigma_w} \
    --base_seed ${BASE_SEED} \
    --output ${output_csv}'\""

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
done

echo ""
echo "Total jobs: $count  (BASE_SEED=${BASE_SEED})"
