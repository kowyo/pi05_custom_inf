#!/usr/bin/env python3
"""
Upload a local directory to Hugging Face Hub (dataset/model/space).
Token is read from HF_TOKEN.

Default behavior is throughput-oriented:
- use upload_large_folder
- allow worker parallelism
- preserve subdirectory uploads via temporary staging when path_in_repo is set
"""

import argparse
import os
import pathlib
import shutil
import tempfile
from contextlib import contextmanager

from huggingface_hub import HfApi


def _fast_copytree(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copy a directory tree quickly using hardlinks when possible."""
    try:
        shutil.copytree(src, dst, copy_function=os.link)
    except (OSError, shutil.Error):
        # Fallback for cross-device filesystems that do not support hardlinks.
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


@contextmanager
def _stage_with_path_in_repo(local_path: str, path_in_repo: str | None):
    """Map local_path under path_in_repo for upload_large_folder.

    upload_large_folder has no path_in_repo argument. To upload to a subdirectory,
    we stage a temporary root and mirror local_path into root/path_in_repo.
    """
    if not path_in_repo:
        yield local_path
        return

    src = pathlib.Path(local_path).resolve()
    # Keep staging on the same filesystem as src so hardlinks can succeed.
    stage_dir = str(src.parent)
    with tempfile.TemporaryDirectory(prefix="hf_upload_stage_", dir=stage_dir) as tmpdir:
        stage_root = pathlib.Path(tmpdir)
        dst = stage_root / path_in_repo
        dst.parent.mkdir(parents=True, exist_ok=True)
        _fast_copytree(src, dst)
        yield str(stage_root)


def upload_folder(
    local_path: str,
    repo_id: str,
    repo_type: str = "dataset",
    private: bool = False,
    path_in_repo: str | None = None,
    num_workers: int = 16,
    use_large_folder: bool = True,
) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable not set.")

    api = HfApi(token=token)

    print(f"Creating repo: {repo_id} (type={repo_type}, private={private})")
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )

    dest = f"{repo_id}/{path_in_repo}" if path_in_repo else repo_id
    print(f"Uploading {local_path} -> {dest} ...")
    print(
        f"Upload mode: {'upload_large_folder' if use_large_folder else 'upload_folder'}, "
        f"num_workers={num_workers}"
    )

    if use_large_folder:
        with _stage_with_path_in_repo(local_path, path_in_repo) as upload_root:
            api.upload_large_folder(
                repo_id=repo_id,
                folder_path=upload_root,
                repo_type=repo_type,
                num_workers=max(1, int(num_workers)),
            )
    else:
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=path_in_repo,
        )

    prefix = "datasets" if repo_type == "dataset" else "models" if repo_type == "model" else repo_type
    print(f"\nDone! Available at: https://huggingface.co/{prefix}/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a local folder to Hugging Face Hub")
    parser.add_argument(
        "--path",
        type=str,
        default="/workspace1/zhijun/mg_dataset/blockpap_cleaned_mimicgen",
        help="Local directory to upload",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="aaroncaozj/BlockPAP-v1_MimicGen",
        help="Hugging Face repo id (e.g. aaroncaozj/my-model)",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Repo type",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        default=None,
        help="Subdirectory inside the repo (e.g. global_step_25000).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Worker count for upload_large_folder.",
    )
    parser.add_argument(
        "--no-large-folder",
        action="store_true",
        help="Use upload_folder instead of upload_large_folder.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"[ERROR] Directory not found: {args.path}")
        return

    upload_folder(
        args.path,
        args.repo,
        repo_type=args.type,
        private=args.private,
        path_in_repo=args.path_in_repo,
        num_workers=args.num_workers,
        use_large_folder=not args.no_large_folder,
    )


if __name__ == "__main__":
    main()
