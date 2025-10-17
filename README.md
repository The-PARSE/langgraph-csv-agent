# LangGraph CSV Agent

Multi-model CSV analysis agent with cost tracking.

## Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .


# Run
python main.py
```

## Supported Models

- GPT-5, GPT-5-mini
- Gemini 2.5 Pro, Gemini 2.5 Flash
- Claude Sonnet 4.5, Claude Haiku 4.5
