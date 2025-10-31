"""
LangGraph CSV Agent - Autonomous copilot for CSV operations.
Works exactly like Claude SDK with natural ending and session continuity.
"""
import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
# ToolNode not used - using custom tool executor to handle Claude's empty args bug
from bash_tool import execute_bash_command, cleanup_shell
from md_tools import search_md_files, read_md_file, list_md_files
from session_manager import SessionManager

# Load environment variables
load_dotenv()


# Define the state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def create_csv_agent(csv_folder: str):
    """
    Create a LangGraph CSV agent with natural ending and session continuity.

    Args:
        csv_folder: Path to folder containing CSV files

    Returns:
        Compiled LangGraph workflow
    """

    # Change to CSV folder - agent locked in this directory
    csv_folder_abs = os.path.abspath(csv_folder)
    os.chdir(csv_folder_abs)

    # Get model selection from environment
    model_name = os.environ.get("MODEL", "gpt-5")
    model_provider = os.environ.get("MODEL_PROVIDER", "openai")  # openai, google, anthropic

    # Create the appropriate LLM based on provider
    if model_provider == "openai":
        # OpenAI models (GPT-5, GPT-4, etc.)
        llm_kwargs = {"model": model_name, "api_key": os.environ.get("OPENAI_API_KEY")}

        # GPT-5 specific: minimal reasoning for speed
        if "gpt-5" in model_name.lower():
            llm_kwargs["model_kwargs"] = {"reasoning_effort": "minimal"}
        else:
            llm_kwargs["temperature"] = 0.7

        llm = ChatOpenAI(**llm_kwargs)

    elif model_provider == "google":
        # Google models (Gemini 2.5 Flash, Pro, etc.)
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.7,
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )

    elif model_provider == "anthropic":
        # Anthropic models (Claude Haiku 4.5, Sonnet, etc.)
        llm = ChatAnthropic(
            model=model_name,
            temperature=0.7,
            max_tokens=8192,
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            default_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )

    else:
        raise ValueError(f"Unsupported provider: {model_provider}. Choose from: openai, google, anthropic")

    # Bind tools to LLM
    tools = [execute_bash_command, search_md_files, read_md_file, list_md_files]

    # For Claude models, use specific configuration to prevent empty args bug
    if model_provider == "anthropic":
        from langchain_anthropic import convert_to_anthropic_tool

        # Convert all tools and add cache_control for prompt caching
        cached_tools = []
        for tool in tools:
            cached_tool = convert_to_anthropic_tool(tool)
            cached_tool["cache_control"] = {"type": "ephemeral"}
            cached_tools.append(cached_tool)

        llm_with_tools = llm.bind_tools(
            cached_tools,
            parallel_tool_calls=False,
            tool_choice="auto"
        )
    else:
        llm_with_tools = llm.bind_tools(tools)

    # System message for the agent (with cache control for Claude)
    system_content = f"""You are an expert CSV data analyst and manipulator with access to powerful tools.
You are currently working in the directory: {csv_folder_abs}
All CSV files are in the CURRENT directory (.).

AVAILABLE TOOLS:
1. execute_bash_command - Execute shell commands for CSV operations
2. search_md_files - Search markdown context files using regex patterns
3. read_md_file - Read specific sections of markdown files
4. list_md_files - List all available markdown context files

📚 MARKDOWN CONTEXT FILES & SESSION HISTORY (MANDATORY CONTEXT SOURCES):

⚠️ BEFORE ANY BASH EXPLORATION, YOU MUST AGGRESSIVELY SEARCH TWO LOCATIONS:

1. CURRENT DIRECTORY MD FILES:
   - ALWAYS check for markdown files using list_md_files()
   - Use search_md_files() to find relevant information (definitions, requirements, thresholds, rules)
   - Use read_md_file() to read specific sections after finding relevant line numbers
   - May contain: Business rules, SAR definitions, regulatory requirements, data schemas

2. PREVIOUS SESSION HISTORIES (.sessions/ directory):
   - MANDATORY: Check .sessions/ directory using list_md_files(directory=".sessions")
   - Search session history files aggressively for past analyses, findings, and context
   - Use search_md_files() with relevant keywords to find prior work
   - May contain: Past conversations, analyses performed, files created, previous findings

⚠️ CRITICAL ENFORCEMENT:
- You MUST search ALL .md files in current directory
- You MUST search ALL .sessions/*.md files for previous session context
- ONLY AFTER exhausting BOTH sources should you proceed to bash exploration
- DO NOT skip this step - context from these files is ESSENTIAL for avoiding duplicate work

CRITICAL RULES:
- You can ONLY access files in the current directory. Do not try to access parent directories or absolute paths.
- NEVER ask the user questions. You must be fully autonomous and complete tasks without any user clarification.
- If information is unclear, CHECK MARKDOWN FILES FIRST for context and definitions
- Always complete the task fully - don't stop midway.
- Be proactive and thorough in your analysis.

🎯 OPERATION MODES - DETECT AND EXECUTE 🎯

You operate in TWO distinct modes based on the user's intent. ALWAYS identify the mode first:

📊 ASK MODE (Analysis & Insights):
- User wants to query, analyze, or understand data
- Intent: Get insights, answers, or information from the data
- Output: Display results as TEXT/TABLE in terminal - DO NOT create any files
- Use: DuckDB queries that print to stdout, analysis scripts
- Final output: Print clear, formatted results directly to terminal

✏️ EDIT MODE (Modify & Create Files):
- User wants to create new files or modify existing CSV files
- Intent: Persist changes or generate output files
- Output: Create new CSV files or modify existing ones
- Use: DuckDB COPY TO commands, Python scripts that write files, in-place editing
- Final output: Confirm which files were created/modified with row counts

Your workflow for EVERY request (DO ALL STEPS AUTOMATICALLY):

1. DETECT MODE
   - Read the user's request carefully and understand their intent
   - Determine: Do they want INFORMATION (Ask Mode) or FILE OUTPUT (Edit Mode)?
   - Mode detection is CRITICAL - it determines your entire approach

2. CHECK FOR CONTEXT FILES (MANDATORY FIRST STEP - BEFORE ANY BASH EXPLORATION)

   ⚠️ CRITICAL: ITERATIVE AND GRADUAL CONTEXT BUILDING APPROACH

   MANDATORY ALTERNATING PATTERN:
   ⚠️ You MUST strictly alternate: search → read → search → read → search → read
   ⚠️ NEVER do multiple searches in a row without reading in between
   ⚠️ NEVER do multiple reads in a row without searching in between

   PRINCIPLES FOR CONTEXT GATHERING:
   - Search finds WHERE information is (returns file names + line numbers)
   - Read extracts WHAT the information says (reads those exact lines)
   - Next search uses what you learned to find MORE related information
   - Each tool call must inform the next tool call
   - Balance between current directory context files and .sessions/ history files

   WORKFLOW PATTERN:
   1. Search for specific term → Get file locations + line numbers
   2. Read ONLY those specific lines from search results (5-10 lines max)
   3. Search for new related term discovered in what you just read
   4. Read ONLY the specific sections where that new term appears (5-10 lines max)
   5. Repeat: search → read → search → read → search → read

   CRITICAL CONSTRAINTS:

   PRECISION OVER BREADTH:
   - Use SPECIFIC, NARROW search patterns (NOT broad regex with many ORs)
   - Search for ONE specific term or concept at a time
   - Keep max_results LOW (10-30 matches max) to avoid overwhelming output
   - Use context_lines sparingly (0-2 lines, not more)
   - If you need multiple concepts, do MULTIPLE targeted searches, not one giant search

   BAD EXAMPLE (TOO BROAD):
   ❌ search_md_files("threshold|limit|CTR|SAR|cash|transaction", max_results=50)
      → This returns 100+ lines of noise!

   GOOD EXAMPLE (TARGETED):
   ✅ search_md_files("\\$10,000", max_results=15, context_lines=1)
      → Find specific $10k threshold mentions
   ✅ search_md_files("SAR.*filing", max_results=20, context_lines=0)
      → Find SAR filing procedures
   ✅ search_md_files("CTR.*requirement", max_results=15, context_lines=1)
      → Find CTR requirements specifically

   READ TOOL CONSTRAINTS:
   - ALWAYS use start_line and end_line parameters (REQUIRED, not optional)
   - Read 5-10 lines MAXIMUM per read call
   - Use line numbers from search results to guide exactly what to read
   - NEVER read entire files without specific line ranges
   - Read ONLY the targeted areas that search identified
   - If you need more information, do another search → read cycle

   NATURAL ITERATIVE FLOW:
   - User asks: "What are the transaction thresholds?"
   - Search for "$10,000" (narrow, specific) → Find in SAR_requirements.md line 9
   - Read SAR_requirements.md lines 8-14 (just 7 lines around it)
   - Search for "$5,000" (next specific amount) → Find in multiple files
   - Read specific sections where $5,000 appears
   - Search .sessions/ for "threshold" to check past work
   - Continue with targeted searches for each specific threshold

   KEY PRINCIPLE: Ask yourself "What is the MOST SPECIFIC term I can search for right now?"

   ⚠️ Start with list_md_files() to discover available files, then begin iterative search→read pattern
   ⚠️ ONLY proceed to bash exploration after gathering sufficient context through multiple tool calls

3. UNDERSTAND THE QUESTION
   - Carefully parse what the user is asking for
   - Extract ALL criteria, conditions, and requirements from their message
   - Combine user request with context from markdown files AND previous session history
   - For ASK MODE: What insights do they want? How should results be presented?
   - For EDIT MODE: What files to create/modify? What transformations to apply?

4. EXPLORE THE CSVs (use bash tool multiple times from different angles)
   ⚠️ PREREQUISITE: Steps 2a and 2b MUST be completed before this step
   - List all CSV files: ls -la *.csv
   - Check file sizes and basic info
   - View first 10-20 rows: head -20 file.csv
   - View last few rows: tail -10 file.csv
   - Count total rows: wc -l file.csv
   - Check column headers: head -1 file.csv
   - Sample random rows if needed: shuf -n 5 file.csv
   - Check for data patterns, nulls, unique values in key columns
   - Identify column names that match the criteria (dates, amounts, countries, customer IDs, etc.)

4. ANALYZE
   - Based on exploration, understand the data structure
   - Map user's criteria to actual column names in the CSV
   - Identify relevant columns for the user's request
   - Determine the best approach based on MODE and complexity
   - Plan your solution step by step

5. EXECUTE BASED ON MODE

   📊 FOR ASK MODE (Analysis/Query):
   - Use DuckDB queries or scripts that output to STDOUT (terminal)
   - DO NOT create any files - results should only be displayed
   - Print formatted results directly to the terminal
   - Provide clear, readable output with appropriate formatting
   - Include summary statistics, counts, or insights as needed
   - Example: python3 -c "import duckdb; print(duckdb.sql('SELECT ...').df())"

   ✏️ FOR EDIT MODE (File Creation/Modification):
   - Use DuckDB COPY TO or Python scripts to create/modify CSV files
   - Explicitly name the output files (following user's request or logical naming)
   - Validate the results by checking row counts and sampling output
   - Confirm file creation with: ls -la *.csv
   - Report: "Created [filename] with [X] rows" or "Modified [filename], [X] rows affected"
   - Clean up any temporary files before finishing

TOOL USAGE GUIDELINES:

For ASK MODE (display results only):
- DuckDB with Python: python3 -c "import duckdb; print(duckdb.sql('SELECT ...').df())"
- DuckDB direct output: python3 -c "import duckdb; duckdb.sql('SELECT ...').show()"
- Simple queries: awk, grep, cut with pipes to stdout

For EDIT MODE (create/modify files):
- DuckDB COPY TO: python3 -c "import duckdb; duckdb.sql(\\\"COPY (SELECT ...) TO 'output.csv' WITH (HEADER, DELIMITER ',')\\\")"
- Python csv module: python3 -c "import csv; with open('output.csv', 'w') as f: writer = csv.writer(f); ..."
- In-place modification: sed -i, awk with redirection

DuckDB Features (HIGHLY RECOMMENDED):
- Query CSVs directly: SELECT * FROM 'file.csv' WHERE amount > 1000
- Multi-file JOINs: SELECT * FROM 'file1.csv' a JOIN 'file2.csv' b ON a.id = b.id
- Aggregations: SELECT category, SUM(amount) FROM 'data.csv' GROUP BY category
- Glob patterns: SELECT * FROM 'transactions_*.csv'
- Full SQL: JOINs, window functions, CTEs, subqueries
- Auto type detection, NULL handling, parallel processing

Available tools: head, tail, cat, wc, awk, sed, cut, sort, uniq, grep, python3, duckdb
NOT available: pandas (do NOT use)

IMPORTANT RULES:
- Always explore CSVs thoroughly before performing operations
- Use only relative paths (like file.csv or ./file.csv)
- DO NOT ask questions - just do the work autonomously
- Complete the entire task fully - no partial work
- When task is complete, provide a clear summary and naturally end your response
- DO NOT offer additional help or ask if user needs anything else after completing the task

MODE-SPECIFIC RULES:

📊 ASK MODE - DO NOT CREATE FILES:
- Results must be printed to terminal/stdout ONLY
- Do NOT create any CSV files or save output anywhere
- Use DuckDB .show() or print() to display results
- Format output clearly for terminal reading

✏️ EDIT MODE - FILE MANAGEMENT:
- ALL output CSV files MUST have .csv extension (e.g., output.csv NOT output)
- ONLY create the files the user explicitly requested
- Avoid temporary files - use pipes/streams when possible
- If temporary files are needed:
  * Name with _tmp suffix (e.g., temp_data_tmp.csv)
  * DELETE ALL TEMP FILES before finishing (rm -f *_tmp.csv)
- BEFORE final summary, verify with: ls -la *.csv
- Confirm: "Created [filename] with [X] rows"
- Check the output - if you see ANY files with _tmp suffix, DELETE THEM IMMEDIATELY
- The final directory MUST ONLY contain:
  * Original input CSV files (israeli_bank_customers.csv, israeli_bank_transactions.csv)
  * The exact output file(s) the user requested (e.g., SAR5.csv)
  * NOTHING ELSE - NO temp files, NO intermediate files
- If you created temp files, your last tool call MUST be: rm -f *_tmp.csv
- Then verify with: ls -la *.csv to confirm cleanup"""

    # Create system message with cache_control for Claude prompt caching
    if model_provider == "anthropic":
        system_message = SystemMessage(
            content=[{"type": "text", "text": system_content, "cache_control": {"type": "ephemeral"}}]
        )
    else:
        system_message = SystemMessage(content=system_content)

    # Define the agent node
    def call_model(state: AgentState):
        messages = [system_message] + state["messages"]
        response = llm_with_tools.invoke(messages)

        # Store usage info in response for later cost calculation
        if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
            usage = response.response_metadata['token_usage']

            # OpenAI cache monitoring
            if 'prompt_tokens_details' in usage and usage['prompt_tokens_details']:
                cached = usage['prompt_tokens_details'].get('cached_tokens', 0)
                total_prompt = usage.get('prompt_tokens', 0)
                if cached > 0:
                    print(f"💾 Cache Hit: {cached}/{total_prompt} tokens cached ({int(cached/total_prompt*100)}% cached)")
                else:
                    print(f"❌ No Cache: 0/{total_prompt} tokens cached")

            # Claude cache monitoring (cache_creation_input_tokens, cache_read_input_tokens)
            if 'cache_creation_input_tokens' in usage or 'cache_read_input_tokens' in usage:
                cache_write = usage.get('cache_creation_input_tokens', 0)
                cache_read = usage.get('cache_read_input_tokens', 0)
                total_prompt = usage.get('input_tokens', 0)
                if cache_read > 0:
                    print(f"💾 Cache Hit: {cache_read}/{total_prompt} tokens from cache ({int(cache_read/total_prompt*100)}% cached)")
                elif cache_write > 0:
                    print(f"📝 Cache Write: {cache_write} tokens cached for future use")
                else:
                    print(f"❌ No Cache: 0/{total_prompt} tokens cached")

            # Gemini cache monitoring (cached_content_token_count)
            if 'cached_content_token_count' in usage:
                cached = usage.get('cached_content_token_count', 0)
                total_prompt = usage.get('prompt_token_count', 0)
                if cached > 0:
                    print(f"💾 Cache Hit: {cached}/{total_prompt} tokens cached ({int(cached/total_prompt*100)}% cached - 75% discount)")

        return {"messages": [response]}

    # Define conditional edge function - decides if agent should continue or end
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]

        # If there are tool calls, continue to tools
        if last_message.tool_calls:
            return "continue"

        # Otherwise, naturally end - task is complete
        return "end"

    # Custom tool executor (ToolNode causes issues with Claude empty args)
    def execute_tools(state: AgentState):
        from langchain_core.messages import ToolMessage
        messages = state["messages"]
        last_message = messages[-1]

        # Map tool names to tool functions
        tool_map = {
            "execute_bash_command": execute_bash_command,
            "search_md_files": search_md_files,
            "read_md_file": read_md_file,
            "list_md_files": list_md_files
        }

        tool_results = []
        for tool_call in last_message.tool_calls:
            # Skip empty tool calls
            if not tool_call.get("args") or tool_call["args"] == {}:
                tool_results.append(
                    ToolMessage(
                        content="Error: Empty arguments provided. Please retry with proper command.",
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"]
                    )
                )
                continue

            # Get the appropriate tool function
            tool_name = tool_call["name"]
            if tool_name not in tool_map:
                tool_results.append(
                    ToolMessage(
                        content=f"Error: Unknown tool '{tool_name}'",
                        tool_call_id=tool_call["id"],
                        name=tool_name
                    )
                )
                continue

            # Execute the tool
            tool_func = tool_map[tool_name]
            result = tool_func.invoke(tool_call["args"])

            # Truncate large outputs to prevent context overflow (200K token limit)
            lines = result.split('\n')
            max_lines = 3000  # Keep first 3,000 lines
            if len(lines) > max_lines:
                truncated_result = '\n'.join(lines[:max_lines])
                truncated_result += f"\n... (output truncated: {len(lines) - max_lines} more lines, {len(result) - len(truncated_result)} more chars)"
                result = truncated_result

            tool_results.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
            )

        return {"messages": tool_results}

    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", execute_tools)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",  # Go to tools if there are tool calls
            "end": END  # Naturally end if no more tool calls
        }
    )

    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")

    # Compile the graph
    return workflow.compile()


def run_csv_agent_session(csv_folder: str = "./csv_data"):
    """
    Run interactive CSV agent session with natural ending and session continuity.
    Works exactly like Claude SDK.

    Args:
        csv_folder: Path to folder containing CSV files
    """

    # Ensure CSV folder exists
    csv_folder_abs = os.path.abspath(csv_folder)
    if not os.path.exists(csv_folder_abs):
        os.makedirs(csv_folder_abs)

    # Get model configuration
    model_provider = os.environ.get("MODEL_PROVIDER", "openai")
    model_name = os.environ.get("MODEL", "gpt-5")

    # Get API key based on provider
    api_key = None
    if model_provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
    elif model_provider == "google":
        api_key = os.environ.get("GOOGLE_API_KEY")
    elif model_provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Initialize session manager
    session_manager = SessionManager(csv_folder_abs, model_provider, model_name, api_key)

    # Create the agent graph
    agent_graph = create_csv_agent(csv_folder)

    print(f"\n🤖 CSV Agent initialized!")
    print(f"🔮 Using model: {model_name}")
    print(f"📁 Working with CSVs in: {csv_folder_abs}")
    print(f"💬 Commands:")
    print(f"   - Type your request normally")
    print(f"   - /compact  - Compact current session and start new")
    print(f"   - /sessions - View and select previous sessions")
    print(f"   - /new      - Start fresh session (no context)")
    print(f"   - exit      - Quit agent\n")

    # Check for previous sessions
    previous_session = session_manager.display_sessions_menu()
    previous_summary = None

    if previous_session:
        summary_file = session_manager.get_session_summary_file(previous_session)
        if summary_file and os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                previous_summary = f.read()

    # Start session with optional context
    context_msg = session_manager.start_new_session(previous_summary)
    if context_msg:
        print(context_msg)

    # Session state - persists across all user inputs
    session_state = {"messages": []}

    # Conversation loop - like talking to Claude
    try:
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit", "terminate"]:
                print("\n👋 Session ended. Goodbye!")
                break

            # Handle special commands
            if user_input.lower() == "/compact":
                print("\n🗜️  Compacting current session...")
                summary, history_file = session_manager.compact_session(session_state["messages"])
                session_state = {"messages": []}
                context_msg = session_manager.start_new_session(summary)
                print(context_msg)
                continue

            elif user_input.lower() == "/sessions":
                previous_session = session_manager.display_sessions_menu()
                if previous_session:
                    summary_file = session_manager.get_session_summary_file(previous_session)
                    if summary_file and os.path.exists(summary_file):
                        with open(summary_file, 'r', encoding='utf-8') as f:
                            previous_summary = f.read()
                        session_state = {"messages": []}
                        context_msg = session_manager.start_new_session(previous_summary)
                        print(context_msg)
                continue

            elif user_input.lower() == "/new":
                session_state = {"messages": []}
                session_manager.start_new_session(None)
                print("🆕 Started fresh session (no previous context)")
                continue

            if not user_input:
                continue

            print()  # Blank line for formatting

            # Add user message to session
            session_state["messages"].append(HumanMessage(content=user_input))

            # Track total cost for this request
            total_cost = 0.0
            total_input_tokens = 0
            total_output_tokens = 0
            total_cached_tokens = 0

            # Stream the agent's work - show tool calls and thinking
            final_event = None
            try:
                for event in agent_graph.stream(
                    session_state,
                    stream_mode="values",
                    config={"recursion_limit": 50}
                ):
                    final_event = event
                    last_message = event["messages"][-1]

                    # Accumulate token usage from AI messages
                    if isinstance(last_message, AIMessage):
                        # Try to get usage from different sources
                        usage = None

                        # For Gemini: usage_metadata is directly on AIMessage
                        if hasattr(last_message, 'usage_metadata') and last_message.usage_metadata:
                            usage = dict(last_message.usage_metadata)
                        # For Claude: usage is in response_metadata
                        elif hasattr(last_message, 'response_metadata') and 'usage' in last_message.response_metadata:
                            usage = last_message.response_metadata['usage']
                        # For OpenAI: usage is in token_usage within response_metadata
                        elif hasattr(last_message, 'response_metadata') and 'token_usage' in last_message.response_metadata:
                            usage = last_message.response_metadata['token_usage']

                        if usage:
                            # OpenAI token counting
                            if 'prompt_tokens' in usage:
                                input_tok = usage.get('prompt_tokens', 0)
                                output_tok = usage.get('completion_tokens', 0)
                                cached_tok = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0) if 'prompt_tokens_details' in usage else 0
                                total_input_tokens += input_tok
                                total_output_tokens += output_tok
                                total_cached_tokens += cached_tok

                                # Update session manager
                                session_manager.update_token_count(input_tok, output_tok)

                            # Gemini token counting (check before Claude since both use 'input_tokens')
                            elif 'input_token_details' in usage:
                                input_tok = usage.get('input_tokens', 0)
                                output_tok = usage.get('output_tokens', 0)
                                cache_read = usage.get('input_token_details', {}).get('cache_read', 0)

                                total_input_tokens += input_tok
                                total_output_tokens += output_tok
                                total_cached_tokens += cache_read

                                # Update session manager
                                session_manager.update_token_count(input_tok, output_tok)

                            # Claude token counting
                            elif 'input_tokens' in usage:
                                input_tok = usage.get('input_tokens', 0)
                                output_tok = usage.get('output_tokens', 0)
                                cache_read = usage.get('cache_read_input_tokens', 0)
                                cache_write = usage.get('cache_creation_input_tokens', 0)

                                # For Claude: input_tokens already includes non-cached tokens
                                # cache_read_input_tokens is separate
                                total_input_tokens += input_tok
                                total_output_tokens += output_tok
                                total_cached_tokens += cache_read

                                # Update session manager
                                session_manager.update_token_count(input_tok, output_tok)

                    # Show AI thinking/responses
                    if isinstance(last_message, AIMessage):
                        # Show tool calls being made
                        if last_message.tool_calls:
                            for tool_call in last_message.tool_calls:
                                print("\n" + "─"*80)
                                print(f"🔧 Calling tool: {tool_call['name']}")
                                print(f"📝 Arguments: {tool_call['args']}")
                                print("─"*80)

                        # Show agent's text response (only if no more tool calls - final response)
                        if last_message.content and not last_message.tool_calls:
                            print(f"\n🤖 Agent: {last_message.content}")

                    # Show tool execution results
                    elif hasattr(last_message, 'content') and hasattr(last_message, 'name'):
                        # This is a tool message
                        print(f"\n📊 Tool Output ({last_message.name}):")
                        print("┄"*60)
                        # Truncate very long outputs
                        output = last_message.content
                        lines = output.split('\n')
                        if len(lines) > 30:
                            output = '\n'.join(lines[:30]) + f"\n... ({len(lines) - 30} more lines)"
                        print(output)
                        print("┄"*60)

                # Update session state with final event
                if final_event:
                    session_state = final_event

                # Save to session history
                if session_state["messages"]:
                    session_manager.append_to_session_history([session_state["messages"][-1]])

                # Check if we need to auto-compact
                if session_manager.should_compact():
                    print("\n⚠️  Context limit approaching (80%). Auto-compacting session...")
                    summary, history_file = session_manager.compact_session(session_state["messages"])
                    session_state = {"messages": []}
                    context_msg = session_manager.start_new_session(summary)
                    print(context_msg)

            except Exception as e:
                print(f"\n❌ Error: {type(e).__name__}: {str(e)}")
                print("The agent encountered an error. This may be a model compatibility issue.")
                print("Try switching to a different model (gpt-5 or gemini-2.5-pro) in .env")
                continue

            # Calculate and display cost
            if total_input_tokens > 0 or total_output_tokens > 0:
                # Calculate cost based on provider
                if model_provider == "openai":
                    # Detect GPT model and apply appropriate pricing
                    if "mini" in model_name.lower():
                        # GPT-5-mini: $0.25 input, $2 output, $0.025 cached (90% discount)
                        input_price = 0.25
                        output_price = 2.0
                        cache_price = 0.025
                    else:
                        # GPT-5: $1.25 input, $10 output, $0.125 cached
                        input_price = 1.25
                        output_price = 10.0
                        cache_price = 0.125

                    uncached_input = total_input_tokens - total_cached_tokens
                    input_cost = (uncached_input / 1_000_000) * input_price
                    cached_cost = (total_cached_tokens / 1_000_000) * cache_price
                    output_cost = (total_output_tokens / 1_000_000) * output_price
                    total_cost = input_cost + cached_cost + output_cost

                elif model_provider == "google":
                    # Detect Gemini model and apply appropriate pricing
                    if "flash" in model_name.lower():
                        # Gemini 2.5 Flash: $0.30 input, $2.50 output, $0.03 cached
                        input_price = 0.30
                        output_price = 2.50
                        cache_price = 0.03
                    else:
                        # Gemini 2.5 Pro: $1.25 input, $10 output, $0.125 cached (≤200K)
                        input_price = 1.25
                        output_price = 10.0
                        cache_price = 0.125

                    uncached_input = total_input_tokens - total_cached_tokens
                    input_cost = (uncached_input / 1_000_000) * input_price
                    cached_cost = (total_cached_tokens / 1_000_000) * cache_price
                    output_cost = (total_output_tokens / 1_000_000) * output_price
                    total_cost = input_cost + cached_cost + output_cost

                elif model_provider == "anthropic":
                    # Detect Claude model and apply appropriate pricing
                    if "haiku" in model_name.lower():
                        # Claude Haiku 4.5: $1 input, $5 output, $0.10 cached
                        input_price = 1.0
                        output_price = 5.0
                        cache_price = 0.10
                    else:
                        # Claude Sonnet 4.5: $3 input, $15 output, $0.30 cached
                        input_price = 3.0
                        output_price = 15.0
                        cache_price = 0.30

                    uncached_input = total_input_tokens - total_cached_tokens
                    input_cost = (uncached_input / 1_000_000) * input_price
                    cached_cost = (total_cached_tokens / 1_000_000) * cache_price
                    output_cost = (total_output_tokens / 1_000_000) * output_price
                    total_cost = input_cost + cached_cost + output_cost

                else:
                    total_cost = 0.0

                # Display cost breakdown
                print("\n" + "="*80)
                print("💰 Cost for this request:")
                print(f"   Input tokens: {total_input_tokens:,} ({total_input_tokens - total_cached_tokens:,} regular + {total_cached_tokens:,} cached)")
                print(f"   Output tokens: {total_output_tokens:,}")
                if total_cost > 0:
                    print(f"   Total cost: ${total_cost:.6f}")
                print("="*80)

            print("\n" + "="*80)
            print("✅ Ready for next request.")
            print("="*80 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Cleaning up...")
    finally:
        # Cleanup persistent shell when session ends
        cleanup_shell()
        print("🧹 Shell cleanup complete.")


if __name__ == "__main__":
    run_csv_agent_session()
