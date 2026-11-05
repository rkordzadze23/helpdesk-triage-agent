"""
main.py - terminal interface for the Helpdesk Triage Agent.

Commands: 'quit' / 'exit' leave the program, 'reset' clears the conversation.
Run with --debug to also print the raw tool result JSON.
"""

import json
import sys

from agent import MODEL, HelpdeskAgent, OllamaError, check_ollama_ready
from tools import KnowledgeBaseError, load_knowledge_base

DEBUG = "--debug" in sys.argv


def show_tool_call(name: str, arguments, result: dict) -> None:
    """Print a visible trace so you can SEE that the LLM chose to call the tool."""
    print(f"\n  [tool call] {name}({json.dumps(arguments, ensure_ascii=False)})")
    if result.get("status") == "ok":
        ids = ", ".join(f"{r['id']} ({r['title']})" for r in result["results"])
        print(f"  [tool result] {len(result['results'])} entr{'y' if len(result['results']) == 1 else 'ies'}: {ids}")
    else:
        print(f"  [tool result] {result.get('status')}: {result.get('message')}")
    if DEBUG:
        print("  [debug] " + json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    print("=" * 60)
    print(" Helpdesk Triage Agent (local, model: " + MODEL + ")")
    print(" Describe your computer/IT problem. Commands: reset, quit")
    print("=" * 60)

    problem = check_ollama_ready()
    if problem:
        print(f"\n[Warning] {problem}\n(You can fix this and keep the program open; it will retry on each message.)")
    try:
        entries = load_knowledge_base()
        print(f"Knowledge base loaded: {len(entries)} entries.")
    except KnowledgeBaseError as exc:
        print(f"\n[Warning] {exc}\nThe agent will start, but it cannot look up troubleshooting entries.")

    agent = HelpdeskAgent()

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_text:
            print("Please type a description of your problem.")
            continue
        if user_text.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_text.lower() == "reset":
            agent.reset()
            print("Conversation cleared.")
            continue

        print("Thinking...")
        try:
            answer = agent.run_turn(user_text, on_tool_call=show_tool_call)
        except OllamaError as exc:
            print(f"\n[Error] {exc}")
            continue
        except KeyboardInterrupt:
            print("\n[Cancelled]")
            agent.reset()
            continue
        print(f"\nAgent: {answer}")


if __name__ == "__main__":
    main()
