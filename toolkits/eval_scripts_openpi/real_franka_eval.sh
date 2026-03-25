cd /home/showlab/Users/zhijun/RLinf

python toolkits/eval_scripts_openpi/real_franka_eval.py \
    --config_name pi05_blockpap_mix \
    --pretrained_path /home/showlab/Users/zhijun/my_ckpt/models/pi05_aligned_co-sft_blockpap/stride2_global_step_15000/actor/model_state_dict/full_weights.pt \
    --nuc_ip 192.168.1.112 \
    --external_camera_serial 317222075319 \
    --show_camera \
    --num_episodes 30 --action_chunk 8

# 无硬件测试：
# python toolkits/eval_scripts_openpi/real_franka_eval.py \
#     --config_name pi05_blockpap_mix \
#     --pretrained_path /home/showlab/Users/zhijun/my_ckpt/models/pi05_aligned_co-sft_blockpap/global_step_15000/actor/model_state_dict/full_weights.pt \
#     --nuc_ip 192.168.1.112 \
#     --external_camera_serial 123456789 \
#     --num_episodes 5 --action_chunk 8 --use_mock_robot --use_mock_camera

    # --camera_exposure 60 \
    # --camera_gain 64 \
    # --wrist_camera_serial 218622273043 \