uv run toolkits/eval_scripts_openpi/real_franka_eval.py \
    --config_name pi05_blockpap_mix \
    --pretrained_path /home/showlab/Users/zifeng/RLinf/ckpt/model.safetensors \
    --nuc_ip 192.168.1.112 \
    --external_camera_serial 317222075319 \
    --show_camera \
    --num_episodes 30 --action_chunk 8
