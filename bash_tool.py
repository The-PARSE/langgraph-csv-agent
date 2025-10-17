"""
Bash execution tool for LangGraph agent.
"""
import subprocess
from typing import Annotated
from langchain_core.tools import tool


@tool
def execute_bash_command(
    command: Annotated[str, "The bash command to execute. REQUIRED parameter - must provide actual command string."]
) -> str:
    """
    Execute a bash command and return the output.
    Use this to run shell commands, process CSV files, list directories, or perform any shell operations.

    Args:
        command: The bash command to execute (REQUIRED - must provide a valid command string, cannot be empty)

    Returns:
        String containing the command output (stdout and stderr combined)
    """
    # Handle empty command (Claude bug - should never happen with required param)
    if not command or command.strip() == "":
        return "ERROR: command parameter is REQUIRED and cannot be empty. You must provide an actual bash command string."

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"

        output += f"Return Code: {result.returncode}"

        return output

    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 120 seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"
