from huggingface_hub import snapshot_download

# 这个地址是 Intel 官方转好的 DeepSeek-R1 1.5B 优化版
# 无需申请权限，直接下载
model_id = "OpenVINO/deepseek-r1-distill-qwen-1.5b-int4-ov"
local_dir = "deepseek_npu_ov"

print(f"正在下载 DeepSeek-R1 (1.5B) 到 {local_dir}...")
snapshot_download(repo_id=model_id, local_dir=local_dir)
print("✅ 下载完成！")