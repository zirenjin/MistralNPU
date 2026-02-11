# NPU Chat with Mistral

A conversational AI chatbot powered by **Intel NPU** (Neural Processing Unit) using OpenVINO GenAI. Run large language models locally with hardware acceleration on Intel Core Ultra processors.

## Features

- **NPU Acceleration**: Leverage Intel AI Boost (up to 47 TOPS) for fast inference
- **Multi-turn Conversation**: Maintains chat history for contextual responses
- **Auto Memory Reset**: Automatically handles context length limits
- **Multiple Models**: Support for Mistral, DeepSeek, Qwen, and more
- **Configurable**: Easy configuration via environment variables

## Requirements

- **Hardware**: Intel Core Ultra processor with NPU (Meteor Lake or newer)
- **OS**: Windows 11 / Linux
- **Python**: 3.10+
- **Intel NPU Driver**: Latest version from Intel

## Project Structure

```
MistralNPU/
├── src/
│   ├── chat.py          # Main chat application
│   └── download.py      # Model downloader
├── models/              # Downloaded models (git-ignored)
├── run_chat.bat         # Windows launcher
├── run_chat.sh          # Linux/Mac launcher
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
├── .env                 # Your config (git-ignored)
└── README.md
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/zirenjin/MistralNPU.git
cd MistralNPU
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
# Copy the example config
cp .env.example .env

# Edit .env with your settings (optional)
```

### 4. Download a model

```bash
# List available models
python src/download.py --list

# Download Mistral-7B (recommended)
python src/download.py mistral-7b mistral_npu_cw
```

### 5. Run the chat

**Windows:**
```batch
run_chat.bat
```

**Linux/Mac:**
```bash
chmod +x run_chat.sh
./run_chat.sh
```

Or directly:
```bash
python src/chat.py
```

## Configuration

All settings can be configured via the `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | (empty) | HuggingFace token (optional for public models) |
| `MODEL_NAME` | `mistral_npu_cw` | Model folder name in `models/` |
| `DEVICE` | `NPU` | Compute device: `NPU`, `GPU`, or `CPU` |
| `MAX_PROMPT_LEN` | `4096` | Maximum prompt/context length |
| `MAX_NEW_TOKENS` | `2048` | Maximum tokens to generate |
| `TEMPERATURE` | `0.7` | Sampling temperature (0.0-1.0) |
| `SYSTEM_PROMPT` | (see .env) | System instruction for the AI |

## Chat Commands

| Command | Description |
|---------|-------------|
| `/exit` | Exit the chat |
| `/clear` | Clear the screen |
| `/reset` | Reset conversation history |

## Available Models

NPU-optimized models from [OpenVINO HuggingFace](https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu):

| Model | Size | Command |
|-------|------|---------|
| Mistral-7B-Instruct | ~4GB | `python src/download.py mistral-7b` |
| DeepSeek-R1-1.5B | ~1GB | `python src/download.py deepseek-1.5b` |
| DeepSeek-R1-7B | ~4GB | `python src/download.py deepseek-7b` |
| Qwen3-8B | ~5GB | `python src/download.py qwen3-8b` |
| Phi-3-mini | ~2GB | `python src/download.py phi3-mini` |

## Troubleshooting

### Model fails to load on NPU

Try using GPU or CPU instead:
```bash
# Edit .env
DEVICE=GPU  # or CPU
```

### Out of memory / Context too long

The chat will automatically reset when context is too long. You can also manually reset:
```
You > /reset
```

### Model not found

Make sure you've downloaded the model first:
```bash
python src/download.py mistral-7b mistral_npu_cw
```

## License

MIT License

## Acknowledgements

- [OpenVINO](https://github.com/openvinotoolkit/openvino) - Intel's AI inference toolkit
- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) - GenAI extension
- [HuggingFace](https://huggingface.co/OpenVINO) - Model hosting
