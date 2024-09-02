from huggingface_hub import HfApi
import os

def push_folder_to_hub(local_folder, repo_name, repo_type="model"):
    # Get the token from the environment variable
    # token = os.environ.get("HF_TOKEN")
    
    # if not token:
    #     raise ValueError("HF_TOKEN environment variable is not set")

    # Initialize the Hugging Face API
    api = HfApi()

    api.create_repo(repo_id=repo_name, repo_type=repo_type, private=False)

    try:
        # Upload the folder
        api.upload_folder(
            folder_path=local_folder,
            repo_id=repo_name,
            repo_type=repo_type,
            ignore_patterns="**/logs/*.txt"  # Adjust this pattern as needed
        )

        print(f"Successfully pushed {local_folder} to {repo_name} on the Hugging Face Hub")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


local_folder = ".../out/finetune/lora/logs/step-0140000/"
repo_name = "ChallengerSpaceShuttle/finetuned-qlora-bafoGPT-2"

push_folder_to_hub(local_folder, repo_name)