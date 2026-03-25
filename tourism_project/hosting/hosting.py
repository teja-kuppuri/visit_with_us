#import streamlit as st
from huggingface_hub import HfApi
import os

# Target Hugging Face Space (Docker/Streamlit). Use hyphens in the Space name.
SPACE_REPO = "treddy333/visit-with-us-prediction"

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo="",
)
