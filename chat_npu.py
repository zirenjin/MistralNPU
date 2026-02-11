import openvino_genai as ov_genai
import os
import sys
from pathlib import Path

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent

# 清屏函数
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_screen()

print("=" * 50)
print("  Mistral-7B NPU Chat (Unlocked Edition)")
print("  Intel AI Boost Accelerated")
print("=" * 50)
print("\nLoading model on NPU...\n")

model_path = str(SCRIPT_DIR / "mistral_npu_cw")

# System prompt to ensure consistent language
SYSTEM_PROMPT = "You are a helpful assistant. Always answer in the same language as the user's question. Do not translate your response unless explicitly asked."

try:
    # 1. 初始化管道
    pipe = ov_genai.LLMPipeline(model_path, "NPU")

    # 2. 开启连续对话模式，注入系统提示
    pipe.start_chat(SYSTEM_PROMPT)

except Exception as e:
    print(f"[!] Failed to load model: {e}")
    print("Hint: Try changing device to 'GPU' if NPU fails.")
    sys.exit(1)

print("[OK] NPU ready! Limit removed.\n")
print("Commands: /exit to quit, /clear to clear screen, /reset to forget history\n")
print("-" * 50)

while True:
    try:
        user_input = input("\nYou > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\nBye!")
        break

    if not user_input:
        continue

    # 指令处理
    if user_input.lower() in ['/exit', 'quit', 'exit']:
        print("\nBye!")
        pipe.finish_chat() # 优雅关闭
        break

    if user_input.lower() == '/clear':
        clear_screen()
        continue

    if user_input.lower() == '/reset':
        pipe.finish_chat()
        pipe.start_chat(SYSTEM_PROMPT) # 重置记忆，重新注入系统提示
        print("\n[!] Memory cleared.")
        continue

    # 3. 配置生成参数 (给足额度)
    config = ov_genai.GenerationConfig()

    # 直接给 2048 或更高。不用担心，模型说完话自己会停，不会强行凑字数。
    config.max_new_tokens = 2048

    config.do_sample = True
    config.temperature = 0.7   # 0.7 适合兼顾代码准确性和聊天创造性
    config.top_p = 0.9
    config.repetition_penalty = 1.1 # 防止复读机

    print("\nAI > ", end="", flush=True)

    # 4. 生成回复
    # 因为用了 start_chat()，这里只需要传 input，不需要手动拼历史
    pipe.generate(
        user_input,
        config,
        streamer=lambda x: print(x, end="", flush=True)
    )
    print()
