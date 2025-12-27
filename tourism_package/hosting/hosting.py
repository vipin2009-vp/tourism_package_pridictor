from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("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"))
api.upload_folder(
    folder_path="tourism_package/deployment",     # the local folder containing files
    repo_id="Vipin0287/tourism-package-space",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
