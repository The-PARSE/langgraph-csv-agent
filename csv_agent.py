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
from bash_tool import execute_bash_command

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
    tools = [execute_bash_command]

    # For Claude models, use specific configuration to prevent empty args bug
    if model_provider == "anthropic":
        from langchain_anthropic import convert_to_anthropic_tool

        # Convert tool and add cache_control for prompt caching
        cached_tool = convert_to_anthropic_tool(execute_bash_command)
        cached_tool["cache_control"] = {"type": "ephemeral"}

        llm_with_tools = llm.bind_tools(
            [cached_tool],
            parallel_tool_calls=False,
            tool_choice="auto"
        )
    else:
        llm_with_tools = llm.bind_tools(tools)

    # System message for the agent (with cache control for Claude)
    system_content = f"""You are an expert CSV data analyst and manipulator with access to a bash execution tool.
You are currently working in the directory: {csv_folder_abs}
All CSV files are in the CURRENT directory (.).

CRITICAL RULES:
- You can ONLY access files in the current directory. Do not try to access parent directories or absolute paths.
- NEVER ask the user questions. You must be fully autonomous and complete tasks without any user clarification.
- If information is unclear, make reasonable assumptions based on the data you find.
- Always complete the task fully - don't stop midway.
- Be proactive and thorough in your analysis.

Your workflow for EVERY request (DO ALL STEPS AUTOMATICALLY):

1. UNDERSTAND THE QUESTION
   - Carefully parse what the user is asking for
   - Extract ALL criteria, conditions, and requirements from their message
   - Identify if they want to: analyze data, edit existing CSV, create new CSV, or perform calculations

2. EXPLORE THE CSVs (use bash tool multiple times from different angles)
   - List all CSV files: ls -la *.csv
   - Check file sizes and basic info
   - View first 10-20 rows: head -20 file.csv
   - View last few rows: tail -10 file.csv
   - Count total rows: wc -l file.csv
   - Check column headers: head -1 file.csv
   - Sample random rows if needed: shuf -n 5 file.csv
   - Check for data patterns, nulls, unique values in key columns
   - Identify column names that match the criteria (dates, amounts, countries, customer IDs, etc.)

3. ANALYZE
   - Based on exploration, understand the data structure
   - Map user's criteria to actual column names in the CSV
   - Identify relevant columns for the user's request
   - Determine the best approach (awk, sed, python script, pandas, etc.)
   - Plan your solution step by step

4. PERFORM THE OPERATION
   - Write and execute scripts/commands to complete the task
   - Can EDIT existing CSV files in place
   - Can CREATE new CSV files as output in current directory
   - Handle all criteria and conditions specified by the user
   - Validate the results (count rows, check samples)
   - Show the user what was done and provide summary statistics

TOOL USAGE GUIDELINES:
- For simple operations: Use awk, sed, grep, cut, sort, uniq with pipes
- For complex multi-step logic: Write Python3 scripts inline using python3 -c "..."
- Python3 has csv module built-in - use it for CSV operations (import csv)
- Pandas is NOT installed - do NOT use pandas or import pandas
- macOS awk does NOT support nested associative arrays - use Python3 for complex data structures instead
- Available: head, tail, cat, wc, awk, sed, cut, sort, uniq, grep, python3

IMPORTANT:
- Always explore CSVs thoroughly before performing operations
- Use only relative paths (like file.csv or ./file.csv)
- DO NOT ask questions - just do the work
- Complete the entire task autonomously
- When task is complete, provide a clear summary and naturally end your response
- DO NOT offer additional help or ask if user needs anything else after completing the task
- ALL output CSV files MUST have .csv extension (e.g., SAR5.csv NOT SAR5 or SAR51)

🚨 CLEAN WORKSPACE RULE - ABSOLUTELY CRITICAL - MUST FOLLOW 🚨
- ONLY create the files the user explicitly requested
- Try to avoid creating temporary/intermediate files - use pipes, streams, or in-memory processing when possible
- If you MUST create temporary files for complex operations:
  * Name them clearly with _tmp suffix (e.g., customer_employer_map_tmp.csv)
  * YOU MUST DELETE ALL TEMPORARY FILES BEFORE FINISHING - THIS IS MANDATORY
  * Use: rm -f *_tmp.csv or rm -f specific_tmp_file.csv
  * NEVER finish a task with temp files still present

MANDATORY CLEANUP VERIFICATION:
- BEFORE providing your final summary, ALWAYS run: ls -la *.csv
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

            result = execute_bash_command.invoke(tool_call["args"])

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

    # Create the agent graph
    agent_graph = create_csv_agent(csv_folder)

    # Get model name for display
    model_name = os.environ.get("MODEL", "gpt-4o")

    print(f"\n🤖 CSV Agent initialized!")
    print(f"🔮 Using model: {model_name}")
    print(f"📁 Working with CSVs in: {csv_folder_abs}")
    print(f"💬 Type your requests below. Agent will complete tasks and wait for your next input.")
    print(f"💬 Type 'exit' to quit.\n")

    # Session state - persists across all user inputs
    session_state = {"messages": []}

    # Get model provider and name for pricing
    model_provider = os.environ.get("MODEL_PROVIDER", "openai")

    # Conversation loop - like talking to Claude
    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit", "terminate"]:
            print("\n👋 Session ended. Goodbye!")
            break

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

                        # Gemini token counting (check before Claude since both use 'input_tokens')
                        elif 'input_token_details' in usage:
                            input_tok = usage.get('input_tokens', 0)
                            output_tok = usage.get('output_tokens', 0)
                            cache_read = usage.get('input_token_details', {}).get('cache_read', 0)

                            total_input_tokens += input_tok
                            total_output_tokens += output_tok
                            total_cached_tokens += cache_read

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


if __name__ == "__main__":
    run_csv_agent_session()
