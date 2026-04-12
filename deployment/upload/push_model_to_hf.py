from huggingface_hub import create_repo, HfApi

HF_TOKEN = ""   # <-- Add your token

# 1. create repository
hf_name = ""
create_repo(
    hf_name,
    repo_type="dataset",
    exist_ok=True,
    token=HF_TOKEN,
)

# 2. initialize API with token
api = HfApi(token=HF_TOKEN)

# 3. upload large folder
folder_path = ""

# 4. upload large folder
api.upload_large_folder(
    folder_path=folder_path,
    repo_id=hf_name,
    repo_type="dataset",
)