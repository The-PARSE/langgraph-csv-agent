"""
Markdown file search and read tools for context retrieval.
Designed to work like Claude Code's Grep and Read tools.
"""
import os
import re
from typing import Annotated, Optional, List, Dict
from langchain_core.tools import tool


@tool
def search_md_files(
    pattern: Annotated[str, "The regex pattern to search for in markdown files. Supports full Python regex syntax."],
    file_pattern: Annotated[Optional[str], "Optional glob pattern to filter files (e.g., '*.md', 'docs/*.md'). Defaults to '*.md'"] = "*.md",
    case_sensitive: Annotated[Optional[bool], "Case sensitive search. Defaults to False"] = False,
    max_results: Annotated[Optional[int], "Maximum number of results to return. Defaults to 100"] = 100,
    context_lines: Annotated[Optional[int], "Number of context lines before and after match. Defaults to 0"] = 0
) -> str:
    """
    Search for a regex pattern in markdown files and return matching lines with line numbers.

    This tool works like ripgrep/grep but specifically for markdown context files.
    Returns results in format: "filename:line_number: matched_line"

    Args:
        pattern: Regex pattern to search (Python regex syntax)
        file_pattern: Glob pattern for files (default: *.md)
        case_sensitive: Whether search is case sensitive (default: False)
        max_results: Maximum results to return (default: 100)
        context_lines: Lines of context before/after match (default: 0)

    Returns:
        Search results with filename, line numbers, and matched content

    Examples:
        - search_md_files("customer.*data", "*.md")
        - search_md_files("SAR[0-9]+", "requirements/*.md", case_sensitive=True)
        - search_md_files("transaction.*threshold", context_lines=2)
    """
    import glob

    if not pattern or pattern.strip() == "":
        return "ERROR: pattern parameter is REQUIRED and cannot be empty."

    try:
        # Compile regex pattern
        regex_flags = 0 if case_sensitive else re.IGNORECASE
        compiled_pattern = re.compile(pattern, regex_flags)
    except re.error as e:
        return f"ERROR: Invalid regex pattern: {str(e)}"

    try:
        # Get current working directory
        cwd = os.getcwd()

        # Find matching files
        search_pattern = os.path.join(cwd, file_pattern)
        md_files = glob.glob(search_pattern, recursive=True)

        if not md_files:
            return f"No markdown files found matching pattern: {file_pattern}"

        results = []
        total_matches = 0

        for filepath in md_files:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                rel_path = os.path.relpath(filepath, cwd)

                # Search for matches
                for line_num, line in enumerate(lines, start=1):
                    if compiled_pattern.search(line):
                        if total_matches >= max_results:
                            results.append(f"\n... (truncated: {max_results} results limit reached)")
                            break

                        # Add context lines if requested
                        if context_lines > 0:
                            start_idx = max(0, line_num - 1 - context_lines)
                            end_idx = min(len(lines), line_num + context_lines)

                            for ctx_num in range(start_idx, end_idx):
                                ctx_line = lines[ctx_num].rstrip('\n')
                                marker = ">" if ctx_num == line_num - 1 else " "
                                results.append(f"{rel_path}:{ctx_num + 1}:{marker} {ctx_line}")
                            results.append("")  # Blank line between matches
                        else:
                            # Just the matching line
                            matched_line = line.rstrip('\n')
                            results.append(f"{rel_path}:{line_num}: {matched_line}")

                        total_matches += 1

                if total_matches >= max_results:
                    break

            except Exception as e:
                results.append(f"Error reading {filepath}: {str(e)}")

        if not results:
            return f"No matches found for pattern: {pattern}"

        header = f"Found {total_matches} match(es) in {len([r for r in results if ':' in r])} file(s)\n"
        header += "="*80 + "\n"

        return header + "\n".join(results)

    except Exception as e:
        return f"Error during search: {str(e)}"


@tool
def read_md_file(
    filepath: Annotated[str, "Path to the markdown file to read (relative to current directory)"],
    start_line: Annotated[Optional[int], "Starting line number (1-indexed). Defaults to 1"] = 1,
    end_line: Annotated[Optional[int], "Ending line number (inclusive). If not specified, reads to end of file"] = None,
    max_lines: Annotated[Optional[int], "Maximum number of lines to read. Defaults to 500"] = 500
) -> str:
    """
    Read contents of a markdown file, optionally specifying line ranges.

    This tool works like Claude Code's Read tool but for markdown context files.
    Returns file contents with line numbers in format: "line_number: content"

    Args:
        filepath: Path to markdown file (relative to current directory)
        start_line: Starting line number (1-indexed, default: 1)
        end_line: Ending line number (inclusive, default: end of file)
        max_lines: Maximum lines to read (default: 500)

    Returns:
        File contents with line numbers

    Examples:
        - read_md_file("requirements.md")
        - read_md_file("docs/SAR_definitions.md", start_line=10, end_line=50)
        - read_md_file("context.md", start_line=100, max_lines=20)
    """
    if not filepath or filepath.strip() == "":
        return "ERROR: filepath parameter is REQUIRED and cannot be empty."

    try:
        # Get current working directory and construct full path
        cwd = os.getcwd()
        full_path = os.path.join(cwd, filepath)

        # Security check - ensure file is within working directory
        real_path = os.path.realpath(full_path)
        real_cwd = os.path.realpath(cwd)
        if not real_path.startswith(real_cwd):
            return f"ERROR: Cannot access files outside working directory: {filepath}"

        if not os.path.exists(full_path):
            return f"ERROR: File not found: {filepath}"

        if not os.path.isfile(full_path):
            return f"ERROR: Path is not a file: {filepath}"

        # Read file
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()

        total_lines = len(all_lines)

        # Validate start_line
        if start_line < 1:
            start_line = 1
        if start_line > total_lines:
            return f"ERROR: start_line {start_line} exceeds file length ({total_lines} lines)"

        # Calculate end_line
        if end_line is None:
            end_line = min(start_line + max_lines - 1, total_lines)
        else:
            if end_line < start_line:
                return f"ERROR: end_line ({end_line}) must be >= start_line ({start_line})"
            end_line = min(end_line, total_lines)

        # Enforce max_lines limit
        if end_line - start_line + 1 > max_lines:
            end_line = start_line + max_lines - 1

        # Extract lines (convert to 0-indexed)
        selected_lines = all_lines[start_line - 1:end_line]

        # Format output with line numbers
        output_lines = []
        for idx, line in enumerate(selected_lines, start=start_line):
            output_lines.append(f"{idx:6d}: {line.rstrip()}")

        # Header
        header = f"File: {filepath}\n"
        header += f"Lines: {start_line}-{end_line} of {total_lines}\n"
        header += "="*80 + "\n"

        result = header + "\n".join(output_lines)

        # Footer if truncated
        if end_line < total_lines:
            result += f"\n\n... ({total_lines - end_line} more lines not shown)"

        return result

    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def list_md_files(
    directory: Annotated[Optional[str], "Directory to search (relative to current dir). Defaults to current directory"] = ".",
    pattern: Annotated[Optional[str], "Glob pattern to filter files. Defaults to '*.md'"] = "*.md",
    recursive: Annotated[Optional[bool], "Search recursively in subdirectories. Defaults to True"] = True
) -> str:
    """
    List all markdown files in the specified directory.

    Useful for discovering available context files before searching/reading.

    Args:
        directory: Directory to search (default: current directory)
        pattern: Glob pattern for files (default: *.md)
        recursive: Search subdirectories recursively (default: True)

    Returns:
        List of markdown files with sizes and paths

    Examples:
        - list_md_files()
        - list_md_files("docs", "*.md")
        - list_md_files("requirements", "SAR*.md", recursive=False)
    """
    import glob

    try:
        cwd = os.getcwd()
        search_dir = os.path.join(cwd, directory)

        if not os.path.exists(search_dir):
            return f"ERROR: Directory not found: {directory}"

        # Build search pattern
        if recursive:
            search_pattern = os.path.join(search_dir, "**", pattern)
        else:
            search_pattern = os.path.join(search_dir, pattern)

        # Find files
        md_files = glob.glob(search_pattern, recursive=recursive)

        if not md_files:
            return f"No markdown files found in {directory} matching {pattern}"

        # Sort files
        md_files.sort()

        # Format output
        results = []
        results.append(f"Found {len(md_files)} markdown file(s) in {directory}")
        results.append("="*80)

        for filepath in md_files:
            rel_path = os.path.relpath(filepath, cwd)
            try:
                file_size = os.path.getsize(filepath)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    line_count = sum(1 for _ in f)

                results.append(f"{rel_path:60s} ({line_count:5d} lines, {file_size:8d} bytes)")
            except Exception as e:
                results.append(f"{rel_path:60s} (error: {str(e)})")

        return "\n".join(results)

    except Exception as e:
        return f"Error listing files: {str(e)}"
