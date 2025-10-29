"""
Bash execution tool for LangGraph agent with persistent subprocess.
"""
import subprocess
import threading
import queue
import time
from typing import Annotated, Optional
from langchain_core.tools import tool


class PersistentShell:
    """Persistent bash shell that maintains state across commands."""

    _instance: Optional['PersistentShell'] = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize persistent bash shell."""
        self.process = subprocess.Popen(
            ['/bin/bash'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Use queues for non-blocking reads
        self.stdout_queue = queue.Queue()
        self.stderr_queue = queue.Queue()

        # Start threads to read stdout/stderr
        self.stdout_thread = threading.Thread(target=self._enqueue_output, args=(self.process.stdout, self.stdout_queue))
        self.stderr_thread = threading.Thread(target=self._enqueue_output, args=(self.process.stderr, self.stderr_queue))
        self.stdout_thread.daemon = True
        self.stderr_thread.daemon = True
        self.stdout_thread.start()
        self.stderr_thread.start()

        # Unique marker for command completion
        self.marker = "<<<COMMAND_COMPLETE_MARKER_123>>>"

    @classmethod
    def get_instance(cls) -> 'PersistentShell':
        """Get or create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the shell instance (cleanup)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.cleanup()
                cls._instance = None

    def _enqueue_output(self, stream, queue):
        """Read lines from stream and put in queue."""
        for line in iter(stream.readline, ''):
            queue.put(line)
        stream.close()

    def execute(self, command: str, timeout: float = 120) -> str:
        """
        Execute command in persistent shell.

        Args:
            command: Bash command to execute
            timeout: Timeout in seconds

        Returns:
            Command output (stdout and stderr combined)
        """
        if not self.process or self.process.poll() is not None:
            return "ERROR: Shell process is not running. Please restart the agent."

        try:
            # Clear queues before executing command
            while not self.stdout_queue.empty():
                try:
                    self.stdout_queue.get_nowait()
                except queue.Empty:
                    break

            while not self.stderr_queue.empty():
                try:
                    self.stderr_queue.get_nowait()
                except queue.Empty:
                    break

            # Write command with marker to detect completion
            full_command = f"{command}\necho '{self.marker}' && echo '{self.marker}' >&2\n"
            self.process.stdin.write(full_command)
            self.process.stdin.flush()

            # Collect output until marker is found
            stdout_lines = []
            stderr_lines = []
            start_time = time.time()
            stdout_marker_found = False
            stderr_marker_found = False

            while not (stdout_marker_found and stderr_marker_found):
                # Check timeout
                if time.time() - start_time > timeout:
                    return f"Error: Command timed out after {timeout} seconds"

                # Read from stdout
                try:
                    line = self.stdout_queue.get(timeout=0.1)
                    if self.marker in line:
                        stdout_marker_found = True
                    else:
                        stdout_lines.append(line)
                except queue.Empty:
                    pass

                # Read from stderr
                try:
                    line = self.stderr_queue.get(timeout=0.1)
                    if self.marker in line:
                        stderr_marker_found = True
                    else:
                        stderr_lines.append(line)
                except queue.Empty:
                    pass

            # Format output
            output = ""
            if stdout_lines:
                output += "STDOUT:\n" + "".join(stdout_lines) + "\n"
            if stderr_lines:
                output += "STDERR:\n" + "".join(stderr_lines) + "\n"

            # Get return code
            return_code_cmd = "echo $?\n"
            self.process.stdin.write(return_code_cmd)
            self.process.stdin.flush()

            # Read return code (with short timeout)
            try:
                return_code_line = self.stdout_queue.get(timeout=1.0)
                return_code = return_code_line.strip()
                output += f"Return Code: {return_code}"
            except queue.Empty:
                output += "Return Code: Unknown"

            return output if output else "Command executed (no output)"

        except Exception as e:
            return f"Error executing command: {str(e)}"

    def cleanup(self):
        """Cleanup shell process."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


@tool
def execute_bash_command(
    command: Annotated[str, "The bash command to execute. REQUIRED parameter - must provide actual command string."]
) -> str:
    """
    Execute a bash command in a persistent shell and return the output.
    The shell maintains state across commands (working directory, environment variables, etc.).
    Use this to run shell commands, process CSV files, list directories, or perform any shell operations.

    Args:
        command: The bash command to execute (REQUIRED - must provide a valid command string, cannot be empty)

    Returns:
        String containing the command output (stdout and stderr combined)
    """
    # Handle empty command
    if not command or command.strip() == "":
        return "ERROR: command parameter is REQUIRED and cannot be empty. You must provide an actual bash command string."

    # Get persistent shell instance
    shell = PersistentShell.get_instance()

    # Execute command
    return shell.execute(command, timeout=120)


def cleanup_shell():
    """Cleanup function to be called when agent session ends."""
    PersistentShell.reset_instance()
