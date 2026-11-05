"""
tools.py - the tool(s) the agent can call.

Contains:
  * TOOL_DEFINITIONS : the tool description sent to the LLM (JSON schema).
  * search_knowledge_base : the real tool - keyword search over knowledge_base.json.
  * execute_tool : validates arguments from the LLM and runs the tool.

IMPORTANT design point: this file does NOT decide WHEN a tool is used.
That decision is made by the LLM in agent.py. The code here only
validates the arguments and runs the search when the LLM asks for it.
"""

import json
import re
from pathlib import Path
from typing import Any

KB_PATH = Path(__file__).parent / "knowledge_base.json"

DEFAULT_MAX_RESULTS = 3
HARD_MAX_RESULTS = 5
MAX_QUERY_LENGTH = 300
MIN_SCORE = 2.0  # entries scoring below this are treated as "not relevant"

# Words that carry no meaning for matching.
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "with",
    "is", "are", "was", "were", "be", "my", "me", "i", "it", "its", "this",
    "that", "very", "do", "does", "did", "how", "what", "why", "can", "cannot",
    "not", "no", "so", "but", "when", "while", "after", "since", "has", "have",
    "had", "shows", "show", "your", "you", "keeps", "get", "gets", "getting",
}

# ---------------------------------------------------------------------------
# Tool definition sent to the LLM (Ollama / OpenAI-style function schema)
# ---------------------------------------------------------------------------
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the local IT troubleshooting knowledge base (Windows problems: "
                "RAM, CPU, disk space, slow startup, Wi-Fi, DNS, internet, printers, "
                "app crashes, Windows Update, blue screens). Returns the most relevant "
                "entries with possible causes, safe steps and warnings, or an empty "
                "result if nothing relevant exists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Short English search query with the key symptoms, e.g. "
                            "'high ram usage slow computer'."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of entries to return (1-5). Default 3.",
                    },
                },
                "required": ["query"],
            },
        },
    }
]


class KnowledgeBaseError(Exception):
    """Raised when the knowledge base file is missing or invalid."""


# ---------------------------------------------------------------------------
# Loading and text normalisation
# ---------------------------------------------------------------------------
def load_knowledge_base(path: Path = KB_PATH) -> list[dict[str, Any]]:
    """Read and validate the knowledge base JSON file."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise KnowledgeBaseError(f"Knowledge base file not found: {path}")
    except OSError as exc:
        raise KnowledgeBaseError(f"Could not read knowledge base: {exc}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise KnowledgeBaseError(f"Knowledge base is not valid JSON: {exc}")
    if not isinstance(data, list) or not data:
        raise KnowledgeBaseError("Knowledge base must be a non-empty JSON list.")
    for entry in data:
        if not isinstance(entry, dict) or "id" not in entry or "title" not in entry:
            raise KnowledgeBaseError("Every knowledge base entry needs an 'id' and a 'title'.")
    return data


def _stem(word: str) -> str:
    """Very crude stemming so 'crashes'/'crashing'/'crashed' all match 'crash'."""
    for suffix in ("ing", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Lower-case, split into words, drop stopwords, stem."""
    text = text.lower().replace("wi-fi", "wifi")
    words = re.findall(r"[a-z0-9_]+", text)
    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    return [_stem(w) for w in words]


def _contains_phrase(tokens: list[str], phrase: list[str]) -> bool:
    """True if `phrase` appears as a consecutive run inside `tokens`."""
    n = len(phrase)
    if n == 0:
        return False
    return any(tokens[i:i + n] == phrase for i in range(len(tokens) - n + 1))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_entry(query_tokens: list[str], entry: dict[str, Any]) -> float:
    """
    Relevance score of one entry for the query:
      * each keyword found in the query: +2 per word of the keyword
        (so the phrase 'high ram usage' counts more than just 'ram')
      * each query word found in the entry title: +1
    """
    score = 0.0
    for keyword in entry.get("keywords", []):
        kw_tokens = _tokenize(str(keyword))
        if kw_tokens and _contains_phrase(query_tokens, kw_tokens):
            score += 2.0 * len(kw_tokens)
    title_tokens = set(_tokenize(str(entry.get("title", ""))))
    score += sum(1.0 for t in set(query_tokens) if t in title_tokens)
    return score


def search_knowledge_base(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> dict[str, Any]:
    """
    Search the local knowledge base.

    Returns a dict:
      {"status": "ok", "query": ..., "results": [ {id, title, score, causes, steps, warnings}, ... ]}
      {"status": "no_results", "query": ..., "message": ...}
    Raises KnowledgeBaseError if the file cannot be loaded.
    """
    entries = load_knowledge_base()
    query_tokens = _tokenize(query)

    scored = [(score_entry(query_tokens, e), e) for e in entries]
    scored = [(s, e) for s, e in scored if s >= MIN_SCORE]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    if not scored:
        return {
            "status": "no_results",
            "query": query,
            "message": "No sufficiently relevant entry was found in the knowledge base.",
        }

    results = [
        {
            "id": e["id"],
            "title": e["title"],
            "score": round(s, 1),
            "causes": e.get("causes", ""),
            "steps": e.get("steps", []),
            "warnings": e.get("warnings", []),
        }
        for s, e in scored[:max_results]
    ]
    return {"status": "ok", "query": query, "results": results}


# ---------------------------------------------------------------------------
# Argument validation + execution (called by the agent loop)
# ---------------------------------------------------------------------------
def validate_search_arguments(arguments: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Return (clean_arguments, None) or (None, error_message)."""
    if isinstance(arguments, str):  # some models send arguments as a JSON string
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None, "Arguments are not valid JSON."
    if not isinstance(arguments, dict):
        return None, "Arguments must be a JSON object with a 'query' field."

    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return None, "'query' is required and must be a non-empty string."
    query = query.strip()
    if len(query) > MAX_QUERY_LENGTH:
        return None, f"'query' is too long (max {MAX_QUERY_LENGTH} characters)."

    max_results = arguments.get("max_results", DEFAULT_MAX_RESULTS)
    if isinstance(max_results, str) and max_results.isdigit():
        max_results = int(max_results)
    if isinstance(max_results, bool) or not isinstance(max_results, int):
        return None, "'max_results' must be an integer between 1 and 5."
    max_results = max(1, min(max_results, HARD_MAX_RESULTS))

    return {"query": query, "max_results": max_results}, None


def execute_tool(name: str, arguments: Any) -> dict[str, Any]:
    """
    Run a tool requested by the LLM. Never raises: problems are returned as
    {"status": "error", ...} so the LLM can see them and react honestly.
    """
    if name != "search_knowledge_base":
        return {"status": "error", "message": f"Unknown tool '{name}'. Available: search_knowledge_base."}

    clean, error = validate_search_arguments(arguments)
    if error:
        return {"status": "error", "message": f"Invalid arguments: {error}"}

    try:
        return search_knowledge_base(clean["query"], clean["max_results"])
    except KnowledgeBaseError as exc:
        return {"status": "error", "message": f"Knowledge base unavailable: {exc}"}
