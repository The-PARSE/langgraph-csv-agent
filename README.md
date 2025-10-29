# LangGraph CSV Agent

Multi-model CSV analysis agent with DuckDB-powered operations and cost tracking.

## Features

- **DuckDB Integration**: Fast SQL queries on CSV files with automatic type detection
- **Multi-Model Support**: GPT-5, Gemini 2.5, Claude 4.5
- **Persistent Shell**: Maintains state across commands (working directory, env variables)
- **Cost Tracking**: Real-time token usage and cost monitoring with prompt caching
- **Autonomous**: No user prompts - fully automated CSV operations

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

## Key Technologies

- **DuckDB**: High-performance SQL analytics on CSV files
- **LangGraph**: Stateful agent workflow orchestration
- **Persistent Shell**: State-preserving bash subprocess for faster operations
