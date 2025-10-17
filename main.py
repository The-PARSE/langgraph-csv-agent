#!/usr/bin/env python3
"""
Main entry point for LangGraph CSV Agent.
"""
import argparse
from csv_agent import run_csv_agent_session


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

    # Run the agent session
    run_csv_agent_session(csv_folder=args.folder)


if __name__ == "__main__":
    main()
