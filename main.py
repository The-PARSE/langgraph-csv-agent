#!/usr/bin/env python3
"""
Main entry point for LangGraph CSV Agent.
"""
import argparse
import os
from csv_agent import run_csv_agent_session


def select_model():
    """Interactive model selection."""
    models = {
        "1": ("gpt-5", "openai", "GPT-5 ($1.25 input, $10 output)"),
        "2": ("gpt-5-mini", "openai", "GPT-5-mini ($0.25 input, $2 output)"),
        "3": ("gemini-2.5-pro", "google", "Gemini 2.5 Pro ($1.25 input, $10 output)"),
        "4": ("gemini-2.5-flash", "google", "Gemini 2.5 Flash ($0.30 input, $2.50 output)"),
        "5": ("claude-sonnet-4-5-20250929", "anthropic", "Claude Sonnet 4.5 ($3 input, $15 output)"),
        "6": ("claude-haiku-4-5-20251001", "anthropic", "Claude Haiku 4.5 ($1 input, $5 output)"),
    }

    print("\n🤖 Select AI Model:")
    print("─" * 60)
    for key, (_, _, description) in models.items():
        print(f"  {key}. {description}")
    print("─" * 60)

    while True:
        choice = input("\nEnter choice (1-6): ").strip()
        if choice in models:
            model_name, provider = models[choice][:2]
            os.environ["MODEL"] = model_name
            os.environ["MODEL_PROVIDER"] = provider
            print(f"✓ Selected: {models[choice][2]}\n")
            return
        print("❌ Invalid choice. Please enter 1-6.")


def main():
    parser = argparse.ArgumentParser(
        description="LangGraph CSV Agent - Autonomous copilot for CSV operations"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="./csv_data",
        help="Path to folder containing CSV files (default: ./csv_data)"
    )

    args = parser.parse_args()

    # Interactive model selection if not set in .env
    if not os.environ.get("MODEL"):
        select_model()

    # Run the agent session
    run_csv_agent_session(csv_folder=args.folder)


if __name__ == "__main__":
    main()
