"""
Session management system for handling context length limits and session continuity.
Supports automatic and manual compaction with searchable session history.
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


class SessionManager:
    """Manages session history, compaction, and continuity."""

    def __init__(self, csv_folder: str, model_provider: str, model_name: str, api_key: str):
        """
        Initialize session manager.

        Args:
            csv_folder: Working directory for CSV files
            model_provider: LLM provider (openai, google, anthropic)
            model_name: Model name
            api_key: API key for the provider
        """
        self.csv_folder = os.path.abspath(csv_folder)
        self.sessions_dir = os.path.join(self.csv_folder, ".sessions")
        self.model_provider = model_provider
        self.model_name = model_name
        self.api_key = api_key

        # Ensure sessions directory exists
        os.makedirs(self.sessions_dir, exist_ok=True)

        # Current session metadata
        self.current_session_id = self._generate_session_id()
        self.current_session_file = os.path.join(self.sessions_dir, f"{self.current_session_id}.md")
        self.token_count = 0
        self.max_tokens = 180000  # 90% of 200K limit for safety

        # Create summarizer model (use fast model for summaries)
        self.summarizer = self._create_summarizer()

    def _generate_session_id(self) -> str:
        """Generate unique session ID with timestamp."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _create_summarizer(self):
        """Create a fast LLM for summarizing sessions."""
        if self.model_provider == "openai":
            return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=self.api_key)
        elif self.model_provider == "google":
            return ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0, google_api_key=self.api_key)
        elif self.model_provider == "anthropic":
            return ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0, api_key=self.api_key)
        else:
            # Fallback to openai
            return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=self.api_key)

    def update_token_count(self, input_tokens: int, output_tokens: int):
        """Update running token count for current session."""
        self.token_count += input_tokens + output_tokens

    def should_compact(self) -> bool:
        """Check if session should be compacted (80% of max tokens)."""
        return self.token_count >= (self.max_tokens * 0.8)

    def format_message_for_history(self, message) -> str:
        """Format a single message for session history MD file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if isinstance(message, HumanMessage):
            return f"### [{timestamp}] USER REQUEST\n```\n{message.content}\n```\n"

        elif isinstance(message, AIMessage):
            output = f"### [{timestamp}] AGENT RESPONSE\n"
            if message.content:
                output += f"**Reasoning/Response:**\n{message.content}\n\n"
            if message.tool_calls:
                output += "**Tool Calls:**\n"
                for tool_call in message.tool_calls:
                    output += f"- `{tool_call['name']}` with args: `{tool_call['args']}`\n"
            output += "\n"
            return output

        elif isinstance(message, ToolMessage):
            # Truncate long tool outputs
            content = message.content
            if len(content) > 1000:
                content = content[:1000] + "\n... (truncated)"
            return f"**Tool Output ({message.name}):**\n```\n{content}\n```\n\n"

        return ""

    def append_to_session_history(self, messages: List):
        """Append messages to current session history MD file."""
        with open(self.current_session_file, 'a', encoding='utf-8') as f:
            for message in messages:
                formatted = self.format_message_for_history(message)
                if formatted:
                    f.write(formatted)
            f.write("---\n\n")

    def summarize_session(self, messages: List) -> str:
        """
        Summarize entire session into shorthand notes.

        Args:
            messages: List of conversation messages

        Returns:
            Compressed summary in note-taking format
        """
        # Build conversation history for summarizer
        conversation_text = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                conversation_text += f"\nUSER: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                if msg.content:
                    conversation_text += f"AGENT: {msg.content}\n"
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        conversation_text += f"  → Used {tc['name']}: {tc['args']}\n"
            elif isinstance(msg, ToolMessage):
                # Truncate tool output
                output = msg.content[:200] if len(msg.content) > 200 else msg.content
                conversation_text += f"  → Output: {output}\n"

        # Prompt for summarization
        summary_prompt = f"""Summarize this CSV agent session into concise shorthand notes. Focus on:
- What user requested
- What data was explored/analyzed
- Key findings or insights
- Files created/modified
- Important context for future sessions

Use bullet points, abbreviations, and compact format. Max 500 words.

SESSION TRANSCRIPT:
{conversation_text}

SHORTHAND NOTES:"""

        try:
            response = self.summarizer.invoke([HumanMessage(content=summary_prompt)])
            return response.content
        except Exception as e:
            # Fallback: simple summary
            return f"Session summary failed: {str(e)}\nMessage count: {len(messages)}"

    def compact_session(self, messages: List) -> Tuple[str, str]:
        """
        Compact current session by summarizing and saving history.

        Args:
            messages: Current session messages

        Returns:
            Tuple of (summary, history_file_path)
        """
        print("\n🗜️  Compacting session...")

        # 1. Save full history to MD file
        print(f"📝 Saving full history to: {self.current_session_file}")
        with open(self.current_session_file, 'w', encoding='utf-8') as f:
            f.write(f"# Session History: {self.current_session_id}\n\n")
            f.write(f"**Started:** {self.current_session_id.replace('session_', '').replace('_', ' ')}\n")
            f.write(f"**Total Messages:** {len(messages)}\n")
            f.write(f"**Token Count:** ~{self.token_count:,}\n\n")
            f.write("---\n\n")

            for message in messages:
                formatted = self.format_message_for_history(message)
                if formatted:
                    f.write(formatted)

        # 2. Generate compressed summary
        print("🤖 Generating summary...")
        summary = self.summarize_session(messages)

        # 3. Save summary separately
        summary_file = os.path.join(self.sessions_dir, f"{self.current_session_id}_SUMMARY.md")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"# Session Summary: {self.current_session_id}\n\n")
            f.write(summary)

        print(f"✅ Session compacted!")
        print(f"   Full history: {os.path.basename(self.current_session_file)}")
        print(f"   Summary: {os.path.basename(summary_file)}")

        return summary, self.current_session_file

    def start_new_session(self, previous_summary: Optional[str] = None) -> str:
        """
        Start a new session, optionally continuing from previous.

        Args:
            previous_summary: Optional summary from previous session

        Returns:
            System message for new session with context
        """
        # Generate new session ID
        self.current_session_id = self._generate_session_id()
        self.current_session_file = os.path.join(self.sessions_dir, f"{self.current_session_id}.md")
        self.token_count = 0

        print(f"\n🆕 Starting new session: {self.current_session_id}")

        # Build context message
        context_message = ""

        if previous_summary:
            context_message = f"""
📋 PREVIOUS SESSION CONTEXT:
{previous_summary}

Note: Full session history is available in .sessions/ directory for reference.
Use search_md_files() to search previous sessions if needed.
"""

        # List all available session histories
        session_files = self.list_session_histories()
        if session_files:
            context_message += f"\n📚 Available Session Histories ({len(session_files)} sessions):\n"
            for session_file in session_files[-5:]:  # Show last 5
                context_message += f"   - {os.path.basename(session_file)}\n"

        return context_message

    def list_session_histories(self) -> List[str]:
        """List all saved session history files."""
        import glob
        pattern = os.path.join(self.sessions_dir, "session_*.md")
        sessions = glob.glob(pattern)
        # Exclude SUMMARY files
        sessions = [s for s in sessions if "_SUMMARY.md" not in s]
        sessions.sort()
        return sessions

    def get_session_summary_file(self, session_file: str) -> Optional[str]:
        """Get the summary file for a given session history file."""
        base = session_file.replace(".md", "_SUMMARY.md")
        return base if os.path.exists(base) else None

    def display_sessions_menu(self) -> Optional[str]:
        """Display interactive menu for selecting previous sessions."""
        sessions = self.list_session_histories()

        if not sessions:
            print("\n📭 No previous sessions found.")
            return None

        print("\n📚 Previous Sessions:")
        print("=" * 60)

        for idx, session_file in enumerate(sessions[-10:], 1):  # Show last 10
            session_name = os.path.basename(session_file).replace("session_", "").replace(".md", "")

            # Try to read first few lines for preview
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_lines = len(lines)
                    # Find first user request
                    first_request = "..."
                    for line in lines:
                        if "USER REQUEST" in line:
                            idx_req = lines.index(line)
                            if idx_req + 1 < len(lines):
                                first_request = lines[idx_req + 2].strip()[:50]
                            break

                print(f"  {idx}. {session_name}")
                print(f"     └─ {total_lines} lines | First request: {first_request}")

            except Exception:
                print(f"  {idx}. {session_name}")

        print("=" * 60)
        choice = input("Select session (number) or press Enter to skip: ").strip()

        if choice.isdigit():
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(sessions[-10:]):
                return sessions[-10:][choice_idx]

        return None
