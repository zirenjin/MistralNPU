from huggingface_hub import snapshot_download

model_id = "OpenVINO/llama-3.1-8b-instruct-int4-ov"
local_dir = "llama3.1_npu_ov"

my_token = "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"

print(f"Downloading to {local_dir} from HuggingFace...")

try:
    snapshot_download(
        repo_id=model_id, 
        local_dir=local_dir,
        token=my_token, 
        local_dir_use_symlinks=False
    )
    print("✅ Done！")
except Exception as e:
    print(f"❌Error: {e}")