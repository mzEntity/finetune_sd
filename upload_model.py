from huggingface_hub import HfFolder, HfApi, upload_folder
import os

# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

HfFolder.save_token("hf_...")

api = HfApi()
repo_id = "username/UTKFace_finetuned_sd"
api.create_repo(repo_id=repo_id, repo_type="model", private=False)

folder_path = "/root/shared-storage/finetuned_sd"

upload_folder(
    repo_id=repo_id,
    folder_path=folder_path,
    commit_message="upload finetuned sd model"
)
