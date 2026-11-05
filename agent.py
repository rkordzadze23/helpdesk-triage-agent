"""
agent.py - the agent loop (LLM <-> tool <-> LLM).

Flow for one user request:
  1. add the user message to the conversation
  2. send conversation + tool definitions to the local LLM (Ollama /api/chat)
  3. the LLM either answers directly OR asks to call a tool
  4. if a tool call is requested: validate -> execute -> add the result to the conversation
  5. ask the LLM again (it now sees the tool result) and repeat, up to MAX_TOOL_CALLS
  6. the LLM's final text is the answer

Python never decides WHICH tool to use or WHEN. It only enforces limits and executes.
"""

import json
import re
from typing import Any, Callable

import requests

from tools import TOOL_DEFINITIONS, execute_tool

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:7b"
MAX_TOOL_CALLS = 3          # safety limit per user request (prevents infinite loops)
REQUEST_TIMEOUT = 300       # seconds; a 7B model on CPU can be slow
MAX_HISTORY_MESSAGES = 40   # keep the context small for the 7B model

SYSTEM_PROMPT = """You are Helpdesk Triage Agent, an assistant that helps users troubleshoot common Windows computer and IT problems.

You have one tool: search_knowledge_base. It searches a small local IT troubleshooting knowledge base.

When to use the tool:
- For any computer, network, printer or Windows problem, call search_knowledge_base FIRST with a short English query containing the key symptoms. Do not rely on memory for troubleshooting steps.
- Do not call the tool for greetings, thanks or messages unrelated to IT problems; just answer briefly and say what you can help with.
- If the first search returns no_results, you may try ONE more search with different wording.

How to answer after you have tool results:
- Base your answer ONLY on the tool results. Do not invent steps, entries or IDs. Mention the entry ID(s) you used.
- If the tool returned no relevant entry, say clearly that the knowledge base has no sufficiently relevant entry. Then give only very general, safe advice (restart, save work, ask IT support) and say it is not from the knowledge base.
- If the tool returned an error, tell the user the knowledge base is unavailable.

Structure the answer with these short sections:
1. Possible causes (these are possibilities, NOT confirmed causes)
2. What to check first
3. Recommended steps
4. When to contact IT support

Safety rules:
- You cannot see or inspect the user's computer. Never claim you checked their PC, and never say a cause is confirmed. Use words like "may", "could", "possible".
- You cannot run commands or change anything. You only give instructions that the user performs themselves.
- Always repeat the warnings from the knowledge base when a step could cause data loss or settings loss.
- Never ask for or accept passwords, product keys or other secrets.
- Keep the answer clear and concise. Answer in the language the user writes in."""


class OllamaError(Exception):
    """Any problem talking to the local Ollama server (shown to the user)."""


# ---------------------------------------------------------------------------
# Talking to Ollama
# ---------------------------------------------------------------------------
def check_ollama_ready() -> str | None:
    """Return None if Ollama is running and MODEL is installed, else an error message."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
    except requests.exceptions.ConnectionError:
        return (f"Cannot connect to Ollama at {OLLAMA_URL}. Start the Ollama app "
                "(or run 'ollama serve') and try again.")
    except (requests.exceptions.RequestException, ValueError) as exc:
        return f"Ollama did not respond correctly: {exc}"

    installed = {m.get("name") for m in models} | {m.get("model") for m in models}
    if MODEL not in installed:
        return f"Model '{MODEL}' is not installed. Run:  ollama pull {MODEL}"
    return None


def call_ollama(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    """One call to POST /api/chat (non-streaming). Returns the assistant 'message' dict."""
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2},  # low temperature = more predictable tool use
    }
    if tools:
        payload["tools"] = tools

    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.ConnectionError:
        raise OllamaError(f"Cannot connect to Ollama at {OLLAMA_URL}. Is Ollama running?")
    except requests.exceptions.Timeout:
        raise OllamaError("The model took too long to answer (timeout). Try again.")
    except requests.exceptions.RequestException as exc:
        raise OllamaError(f"Network error while contacting Ollama: {exc}")

    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", resp.text)
        except ValueError:
            detail = resp.text
        if resp.status_code == 404:
            raise OllamaError(f"Model '{MODEL}' not found. Run:  ollama pull {MODEL}")
        raise OllamaError(f"Ollama returned HTTP {resp.status_code}: {detail}")

    try:
        message = resp.json().get("message")
    except ValueError:
        raise OllamaError("Ollama returned invalid JSON.")
    if not isinstance(message, dict):
        raise OllamaError("Ollama response did not contain a 'message'.")
    return message


# ---------------------------------------------------------------------------
# Reading tool calls from the model's reply
# ---------------------------------------------------------------------------
def _as_dict(arguments: Any) -> Any:
    """Ollama expects tool-call arguments as an object; parse strings when possible."""
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def extract_tool_calls(message: dict[str, Any]) -> list[tuple[str, Any]]:
    """
    Return [(tool_name, arguments), ...] requested by the LLM.

    1. Native format: message["tool_calls"] (what Ollama returns for qwen2.5).
    2. Fallback: small models sometimes print the call as JSON text
       (optionally inside <tool_call> tags). We parse that too.
    Either way, the decision to call the tool was made by the LLM.
    """
    calls: list[tuple[str, Any]] = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        if fn.get("name"):
            calls.append((fn["name"], _as_dict(fn.get("arguments", {}))))
    if calls:
        return calls

    content = (message.get("content") or "").strip()
    content = re.sub(r"^```(?:json)?|```$|</?tool_call>", "", content, flags=re.MULTILINE).strip()
    if content.startswith("{") and content.endswith("}"):
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return []
        if isinstance(data, dict) and isinstance(data.get("name"), str) \
                and ("arguments" in data or "parameters" in data):
            return [(data["name"], _as_dict(data.get("arguments", data.get("parameters"))))]
    return []


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class HelpdeskAgent:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        """Clear the conversation history (keeps only the system prompt)."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _trim_history(self) -> None:
        """Drop the oldest turns (always at a user-message boundary) when history gets long."""
        while len(self.messages) > MAX_HISTORY_MESSAGES:
            next_user = next((i for i in range(2, len(self.messages))
                              if self.messages[i]["role"] == "user"), None)
            if next_user is None:
                break
            del self.messages[1:next_user]

    def run_turn(self, user_text: str,
                 on_tool_call: Callable[[str, Any, dict[str, Any]], None] | None = None) -> str:
        """Handle one user request and return the final answer text."""
        self._trim_history()
        start_len = len(self.messages)
        self.messages.append({"role": "user", "content": user_text})
        tool_calls_used = 0

        try:
            # Upper bound on loop iterations is a second guard against endless loops.
            for _ in range(MAX_TOOL_CALLS + 2):
                allow_tools = tool_calls_used < MAX_TOOL_CALLS
                message = call_ollama(self.messages, TOOL_DEFINITIONS if allow_tools else None)
                calls = extract_tool_calls(message)

                if not calls:  # the LLM gave its final answer
                    answer = (message.get("content") or "").strip()
                    if not answer:
                        raise OllamaError("The model returned an empty answer. Please try again.")
                    self.messages.append({"role": "assistant", "content": answer})
                    return answer

                # Store the assistant's tool request in a normalised form.
                self.messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": n, "arguments": a if isinstance(a, dict) else {}}}
                                   for n, a in calls],
                })
                for name, arguments in calls:
                    if tool_calls_used >= MAX_TOOL_CALLS:
                        result = {"status": "error",
                                  "message": "Tool call limit reached. Answer now using the information you already have."}
                    else:
                        tool_calls_used += 1
                        result = execute_tool(name, arguments)  # validation happens inside
                        if on_tool_call:
                            on_tool_call(name, arguments, result)
                    self.messages.append({"role": "tool", "tool_name": name,
                                          "content": json.dumps(result, ensure_ascii=False)})

            raise OllamaError("The model did not produce a final answer. Please rephrase and try again.")
        except OllamaError:
            del self.messages[start_len:]  # roll back so a failed turn does not corrupt history
            raise
