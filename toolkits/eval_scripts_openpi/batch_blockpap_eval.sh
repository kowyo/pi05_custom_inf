#!/bin/bash
cd /workspace1/zhijun/RLinf

# ── Config ────────────────────────────────────────────────────────────────────
GPUS=(0 5)               # GPUs to use; one process per GPU runs in parallel
EPISODES_PER_GPU=16             # episodes per GPU → total = len(GPUS) * EPISODES_PER_GPU

TILE_VIDEOS=true             # true: tile all saved videos into one grid video
TILE_COLS=4                  # videos per row in the tiled output

# CKPT_DIR=/workspace1/zhijun/pi-StepNFT/logs/20260322-04:04:16-arc_blockpap_nft_actor_openpi_pi05/blockpap_nft_openpi_pi05
# STEP=20
CKPT_DIR=/workspace1/zhijun/RLinf/logs/20260317-14:18:32/new_norm_stats_stride2
STEP=15000
NORM_STATS=../hf_download/models/pi05_base
# 0、15、25、40、45 或 random
TRAJ_ID=random
BIAS=true
LOG_DIR=/workspace1/zhijun/RLinf/eval/blockpap
BIAS_TAG=$([ "${BIAS,,}" = "true" ] && echo "bias" || echo "no_bias")
EXP_NAME=0323_pi05_${BIAS_TAG}_${TRAJ_ID}_${STEP}_stride2_chunk5


# ── Launch one eval process per GPU in parallel ───────────────────────────────
PIDS=()
GPU_LOG_DIRS=()

for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    SEED=$((0 + i * 1000))          # different seed per GPU → different episodes
    GPU_LOG_DIR="${LOG_DIR}/${EXP_NAME}/gpu${GPU}"
    GPU_LOG_DIRS+=("${GPU_LOG_DIR}")
    mkdir -p "${GPU_LOG_DIR}"

    CUDA_VISIBLE_DEVICES=${GPU} python toolkits/eval_scripts_openpi/blockpap_eval.py \
        --exp_name      "gpu${GPU}" \
        --log_dir       "${GPU_LOG_DIR}" \
        --pretrained_path "${CKPT_DIR}/checkpoints/global_step_${STEP}/actor/model_state_dict/full_weights.pt" \
        --norm_stats_path "${NORM_STATS}" \
        --num_episodes  ${EPISODES_PER_GPU} \
        --max_steps     600 \
        --action_chunk  5 \
        --num_steps     5 \
        --traj_id       "${TRAJ_ID}" \
        --num_save_videos ${EPISODES_PER_GPU} \
        --log_interval  50 \
        --seed          ${SEED} \
        --state_bias    "${BIAS}" \
        > "${GPU_LOG_DIR}/run.log" 2>&1 &

    PIDS+=($!)
    echo "[GPU ${GPU}] pid=$! seed=${SEED} log=${GPU_LOG_DIR}/run.log"
done

# ── Wait for all to finish ────────────────────────────────────────────────────
echo "Waiting for ${#PIDS[@]} eval processes …"
FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    CODE=$?
    if [ $CODE -ne 0 ]; then
        echo "[WARN] GPU ${GPUS[$i]} process exited with code ${CODE}"
        FAILED=$((FAILED + 1))
    fi
done
echo "All processes done (${FAILED} failed)."


# ── Aggregate success rates ───────────────────────────────────────────────────
TOTAL_SUCCESS=0
TOTAL_EPISODES=0
echo ""
echo "Per-GPU results:"
for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    LOG="${GPU_LOG_DIRS[$i]}/run.log"
    LINE=$(grep "Success rate" "${LOG}" 2>/dev/null | tail -1)
    if [ -n "${LINE}" ]; then
        FRAC=$(echo "${LINE}" | grep -oP '\d+/\d+')
        S=$(echo "${FRAC}" | cut -d/ -f1)
        T=$(echo "${FRAC}" | cut -d/ -f2)
        TOTAL_SUCCESS=$((TOTAL_SUCCESS + S))
        TOTAL_EPISODES=$((TOTAL_EPISODES + T))
        echo "  [GPU ${GPU}] ${FRAC}"
    else
        echo "  [GPU ${GPU}] no result found in ${LOG}"
    fi
done

if [ "${TOTAL_EPISODES}" -gt 0 ]; then
    SR=$(awk "BEGIN {printf \"%.1f\", ${TOTAL_SUCCESS}/${TOTAL_EPISODES}*100}")
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "  OVERALL  ${TOTAL_SUCCESS} / ${TOTAL_EPISODES} = ${SR}%"
    echo "  EXP: ${EXP_NAME}"
    echo "╚══════════════════════════════════════╝"
fi


# ── Tile videos ───────────────────────────────────────────────────────────────
if [ "${TILE_VIDEOS,,}" = "true" ]; then
    TILE_OUT="${LOG_DIR}/${EXP_NAME}/tiled_${TILE_COLS}col.mp4"
    mkdir -p "$(dirname "${TILE_OUT}")"

    # collect all .mp4 files across GPU subdirs, sorted by name
    SEARCH_DIRS="${GPU_LOG_DIRS[*]}"
    python - "${TILE_OUT}" "${TILE_COLS}" ${SEARCH_DIRS} <<'PYEOF'
import sys, glob, pathlib
import numpy as np
import imageio

output   = sys.argv[1]
cols     = int(sys.argv[2])
dirs     = sys.argv[3:]

videos = []
for d in dirs:
    videos += sorted(glob.glob(str(pathlib.Path(d) / "**" / "*.mp4"), recursive=True))

if not videos:
    print("No videos found – skipping tiling.")
    sys.exit(0)

print(f"Tiling {len(videos)} videos into {cols}-column grid → {output}")

# Read all videos into memory; pad shorter ones with their last frame
all_frames = []
fps = 20
for v in videos:
    reader = imageio.get_reader(v)
    meta = reader.get_meta_data()
    fps = int(round(meta.get("fps", 20)))
    frames = [f for f in reader]
    reader.close()
    all_frames.append(frames)

max_len = max(len(f) for f in all_frames)
for f in all_frames:
    while len(f) < max_len:
        f.append(f[-1].copy())

# Pad to a multiple of cols with black clips
while len(all_frames) % cols != 0:
    h, w, c = all_frames[0][0].shape
    all_frames.append([np.zeros((h, w, c), dtype=np.uint8)] * max_len)

rows = len(all_frames) // cols
writer = imageio.get_writer(output, fps=fps, macro_block_size=None)
for t in range(max_len):
    row_imgs = []
    for r in range(rows):
        col_imgs = [all_frames[r * cols + c][t] for c in range(cols)]
        row_imgs.append(np.concatenate(col_imgs, axis=1))
    writer.append_data(np.concatenate(row_imgs, axis=0))
writer.close()
print(f"Saved → {output}")
PYEOF

fi


# ── Base model baseline (uncomment to run) ────────────────────────────────────
# GPUS=(0) EPISODES_PER_GPU=5 TRAJ_ID=random BIAS=false STEP=0
# CKPT_DIR=../hf_download/models/pi05_base
