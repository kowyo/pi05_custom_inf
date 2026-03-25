CKPT_DIR=/workspace1/zhijun/RLinf/logs/20260317-14:18:32/new_norm_stats_stride2/checkpoints
REPO=aaroncaozj/pi05_aligned_co-sft_blockpap
TYPE=model
NUM_WORKERS=32

cd /workspace1/zhijun/RLinf

# Throughput-oriented settings (full-folder upload, no file filtering).
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1

# # Upload the norm stats
# python hf_upload.py \
#   --path /workspace1/zhijun/hf_download/models/pi05_base/BlockPAP-v1_Mix \
#   --repo aaroncaozj/pi05_norm_stats_collection \
#   --type model \
#   --path-in-repo BlockPAP-v1_Mix

# python hf_upload.py \
#   --path ${CKPT_DIR}/global_step_5000 \
#   --repo ${REPO} \
#   --type ${TYPE} \
#   --path-in-repo global_step_5000

python hf_upload.py \
  --path ${CKPT_DIR}/global_step_15000 \
  --repo ${REPO} \
  --type ${TYPE} \
  --path-in-repo stride2_global_step_15000 \
  --num-workers ${NUM_WORKERS}
