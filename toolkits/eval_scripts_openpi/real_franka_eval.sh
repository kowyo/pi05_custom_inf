uv run /home/showlab/Users/zifeng/RLinf/toolkits/eval_scripts_openpi/real_franka_eval.py \
    --config_name pi05_blockpap_mix \
    --pretrained_path /home/showlab/Users/zifeng/RLinf/ckpt/model.safetensors \
    --hf_stats_path /home/showlab/Users/zifeng/RLinf/ckpt/meta/stats.json \
    --nuc_ip 192.168.1.112 \
    --external_camera_serial 317222075319 \
    --side_camera_serial 336222073740 \
    --wrist_camera_serial 218622273043 \
    --show_camera \
    --num_episodes 30 --action_chunk 8
