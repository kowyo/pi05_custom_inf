from huggingface_hub import snapshot_download

# download RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT model
# save_dir = "/workspace1/zhijun/hf_download/models/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT"
# repo_id = "RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT"
# repo_type="model"
# allow_patterns="*"

# download RLinf-Pi05-LIBERO-SFT model
# save_dir = "../hf_download/models/RLinf-Pi05-LIBERO-SFT"
# repo_id = "RLinf/RLinf-Pi05-LIBERO-SFT"
# repo_type="model"
# allow_patterns="*"

# download RLinf-Pi05-ManiSkill-25Main-SFT model
# save_dir = "../hf_download/models/RLinf-Pi05-ManiSkill-25Main-SFT"
# repo_id = "RLinf/RLinf-Pi05-ManiSkill-25Main-SFT"
# repo_type="model"
# allow_patterns="*"

# download pi05_base model
# save_dir = "../hf_download/models/pi05_base"
# repo_id = "lerobot/pi05_base"
# repo_type="model"
# allow_patterns="*"

# download libero dataset
# save_dir = "../hf_download/datasets/libero"
# repo_id = "physical-intelligence/libero"
# repo_type="dataset"
# allow_patterns="*"

# download RLCo-maniskill-assets dataset
# save_dir = "rlinf/envs/maniskill/assets"
# repo_id = "RLinf/RLCo-maniskill-assets"
# repo_type="dataset"
# allow_patterns="custom_assets/*"

# download RLCo-maniskill-assets dataset
# save_dir = "rlinf/envs/maniskill/assets"
# repo_id = "RLinf/maniskill_assets"
# repo_type="dataset"
# allow_patterns="*"

# download RLCo-Example-Mix-Data
# save_dir = "../hf_download/datasets/RLCo-Example-Mix-Data"
# repo_id = "RLinf/RLCo-Example-Mix-Data"
# repo_type="dataset"
# allow_patterns="*"

# download 0323 put block on block dataset
save_dir = "../hf_download/datasets/0323_put_block_on_block"
repo_id = "kowyo/pick-and-place"
repo_type = "dataset"
allow_patterns = "*"

snapshot_download(
    local_dir=save_dir,
    repo_id=repo_id,
    repo_type=repo_type,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=[allow_patterns],
)
