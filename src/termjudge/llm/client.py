"""Minimal LLM client for the chat completions protocol, with JSONL cache and retries.

Backends:
- OpenAI API: when ``base_url`` is None, uses ``OPENAI_API_KEY``.
- Any OpenAI-compatible server (vLLM, or another provider exposing the chat
  completions protocol): pass ``base_url=http://host:port/v1`` and an API key
  (``VLLM_API_KEY``, or the dummy key ``"EMPTY"`` for a local server).

OpenAI model families do not accept the same request parameters (gpt-4.x:
temperature + max_tokens; later models: max_completion_tokens, and either
reasoning_effort="none" to keep temperature 0, or no temperature at all). The
client negotiates the request style on the first call (see ``_STYLES``) and
reports it in ``usage_summary()``. An OpenAI-compatible server always gets
the classic style (temperature + max_tokens).

Cache: append-only JSONL keyed by SHA-256 over (model, prompt_name, system,
user, temperature). One entry per line:

    {"key": "<sha256>", "value": "<response>"}

The cache is shared across all callers (reference selection, Q1, Q2 ...)
thanks to the ``prompt_name`` field. Neither the endpoint URL nor the token
cap enters the key.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# JSONL cache (drop-in replacement for diskcache.Cache, with the minimal API)
# ---------------------------------------------------------------------------


class JsonlCache:
    FILE_NAME = "cache.jsonl"

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / self.FILE_NAME
        self._data: Dict[str, Any] = {}
        self._lock = Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "key" not in entry:
                    continue
                self._data[entry["key"]] = entry.get("value")

    def _append(self, key, value) -> None:
        record = {"key": key, "value": value}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __contains__(self, key) -> bool:
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value) -> None:
        with self._lock:
            self._data[key] = value
            self._append(key, value)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)

# Replace anything that is not safe in a POSIX filename with a single dash,
# collapse repeats, trim, and cap length so the saved-prompt filenames stay
# reasonable.
_LABEL_BAD = re.compile(r"[^\w.+=-]+")


def _sanitise_label(label: str, max_len: int = 80) -> str:
    """Make a string safe for use inside a filename."""
    s = _LABEL_BAD.sub("-", label).strip("-_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-_")
    return s or "anon"


def _hash_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# One backslash pair inside the model output. Valid JSON escapes are kept;
# ``\'`` (a habit of some open models, e.g. French « d\'événements ») becomes a
# bare apostrophe; any other invalid escape (``\m``) keeps its backslash as a
# literal one.
_ESCAPE_PAIR_RE = re.compile(r"\\(.)", re.DOTALL)
_JSON_ESCAPES = frozenset('"\\/bfnrtu')


def _repair_escape(m: "re.Match[str]") -> str:
    c = m.group(1)
    if c in _JSON_ESCAPES:
        return m.group(0)
    if c == "'":
        return "'"
    return "\\\\" + c


def extract_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction from a model response.

    Tries: direct ``json.loads`` → markdown ```json``` block → first balanced
    ``{...}`` substring. If all fail, tries them once more after repairing
    invalid backslash escapes (``\\'`` → ``'``, other ``\\x`` → literal
    backslash). Last resort: decode the first complete JSON object (from the
    first ``{``) and ignore whatever follows it, e.g. a stray closing brace
    after a valid object. Both fallbacks only run on otherwise unparseable
    output, so a response that already parsed can never change. Returns
    ``None`` on failure.
    """
    parsed = _extract_json_once(text)
    repaired = None
    if parsed is None and text and "\\" in text:
        repaired = _ESCAPE_PAIR_RE.sub(_repair_escape, text)
        if repaired != text:
            parsed = _extract_json_once(repaired)
        else:
            repaired = None
    if parsed is None and text:
        parsed = _first_json_object(text)
        if parsed is None and repaired is not None:
            parsed = _first_json_object(repaired)
    return parsed


# ``"label": "<value>"`` (or ``"code"``) pairs written as such in the raw output.
# A JSON-escaped pair inside a string value (``\"label\": ...``) does not match.
_LABEL_FIELD_RE = re.compile(r'"(?:label|code)"\s*:\s*"([^"\n]{1,60})"')

# Justification attached to a verdict rebuilt by ``salvage_labels``; also the
# marker the reports count recovered verdicts by.
SALVAGED_NOTE = "Label recovered from malformed JSON output."


def salvage_labels(text: str) -> list:
    """Label values found in a response whose JSON is malformed (missing comma
    or quote) although its label field is intact. Only meant for output that
    ``extract_json`` could not parse; the caller keeps a value only if it is
    unambiguous and belongs to its closed label set."""
    return [v.strip() for v in _LABEL_FIELD_RE.findall(text or "")]


_DECODER = json.JSONDecoder()


def _first_json_object(text: str) -> Optional[dict]:
    """First complete JSON object of ``text``, starting at its first ``{`` and
    ignoring any trailing characters; None if it does not decode to a dict."""
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = _DECODER.raw_decode(text, start)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _extract_json_once(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return None


def _retry_after_seconds(err: Exception) -> Optional[float]:
    """Seconds to wait from a rate-limit response's ``Retry-After`` header, if any.

    OpenAI 429 responses carry ``Retry-After`` (seconds) or ``retry-after-ms``.
    Returns None when the error is not a throttling response with such a header.
    """
    resp = getattr(err, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    try:
        val = headers.get("retry-after")
        if val is not None:
            return float(val)
        ms = headers.get("retry-after-ms")
        if ms is not None:
            return float(ms) / 1000.0
    except (TypeError, ValueError):
        return None
    return None


# ---------------------------------------------------------------------------
# Request styles (OpenAI model families)
# ---------------------------------------------------------------------------

# Tried in this order on the OpenAI API until one is accepted; the first
# accepted style is kept for the rest of the run. Observed on 2026-10-09:
#   classic            gpt-4o, gpt-4.1 (unchanged behaviour for these models)
#   no_reasoning       gpt-5.1 to 5.6, gpt-6-luna, gpt-6-sol
#   default_reasoning  models that refuse reasoning_effort="none" but take a temperature
#   fixed_temperature  gpt-5, gpt-5-mini/nano, o3/o4, gpt-6-astra, gpt-6.1-sol
# A name-based rule does not work: gpt-6-luna takes reasoning_effort="none",
# gpt-6-astra refuses it; gpt-6-luna applies a non-"none" default effort when the
# parameter is omitted, and then rejects temperature 0.
_STYLES = ("classic", "no_reasoning", "default_reasoning", "fixed_temperature")

# Hidden reasoning tokens count against max_completion_tokens: styles where the
# model may reason get at least this budget (a cap, not a charge).
REASONING_TOKEN_FLOOR = 4096


def _request_kwargs(style: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    if style == "classic":
        return {"temperature": temperature, "max_tokens": max_tokens}
    if style == "no_reasoning":
        return {"reasoning_effort": "none", "temperature": temperature,
                "max_completion_tokens": max_tokens}
    if style == "default_reasoning":
        return {"temperature": temperature,
                "max_completion_tokens": max(max_tokens, REASONING_TOKEN_FLOOR)}
    if style == "fixed_temperature":
        return {"max_completion_tokens": max(max_tokens, REASONING_TOKEN_FLOOR)}
    raise ValueError(style)


def _is_unsupported_param_error(err: Exception) -> bool:
    """True for a 400 rejecting a parameter or a value (e.g. ``max_tokens``,
    ``temperature`` 0, ``reasoning_effort`` "none") for this model."""
    if getattr(err, "status_code", None) != 400:
        return False
    msg = str(err)
    return "Unsupported parameter" in msg or "Unsupported value" in msg


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024
    max_retries: int = 5
    retry_base_delay: float = 2.0
    save_prompts_dir: Optional[Path] = None  # if set, dump every call to <dir>/<prompt_name>/<hash[:12]>.md


class LLMClient:
    def __init__(self, config: LLMConfig, cache_dir: Path):
        from openai import OpenAI  # lazy: only required when LLM is used
        self.config = config
        if config.base_url:
            self.client = OpenAI(
                base_url=config.base_url,
                api_key=config.api_key or os.environ.get("VLLM_API_KEY", "EMPTY"),
            )
        else:
            self.client = OpenAI(
                api_key=config.api_key or os.environ.get("OPENAI_API_KEY"),
            )
        self.cache = JsonlCache(cache_dir)
        self._lock = Lock()
        self.save_prompts_dir = Path(config.save_prompts_dir) if config.save_prompts_dir else None
        if self.save_prompts_dir is not None:
            self.save_prompts_dir.mkdir(parents=True, exist_ok=True)
        # Token-usage accounting for ACTUAL API calls (cache hits excluded).
        # Per prompt_name: number of calls, input/output tokens, the largest
        # completion seen (to tune --max-tokens), and how many hit the
        # max_tokens cap (finish_reason == "length" → truncated → JSON likely
        # unparseable → silent default). Guarded by ``_usage_lock``.
        self._usage_lock = Lock()
        self.usage: Dict[str, Dict[str, int]] = {}
        # Request style, negotiated on the first API call (see _STYLES).
        self._styles = ("classic",) if config.base_url else _STYLES
        self._style: Optional[str] = None
        self._style_lock = Lock()

    def _record_usage(self, prompt_name: str, usage, finish_reason: Optional[str]) -> None:
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        with self._usage_lock:
            u = self.usage.setdefault(prompt_name, {
                "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "max_prompt_tokens": 0, "max_completion_tokens": 0, "truncated": 0,
            })
            u["calls"] += 1
            u["prompt_tokens"] += pt
            u["completion_tokens"] += ct
            # Track the largest SINGLE-request input: long-context pricing tiers
            # (e.g. GPT-5.x: >272K input tokens per request) are per-request, so
            # this is the number to compare against a provider threshold.
            u["max_prompt_tokens"] = max(u["max_prompt_tokens"], pt)
            u["max_completion_tokens"] = max(u["max_completion_tokens"], ct)
            if finish_reason == "length":
                u["truncated"] += 1

    def _record_length_retry(self, prompt_name: str, n_len: int, still_truncated: bool) -> None:
        with self._usage_lock:
            u = self.usage.setdefault(prompt_name, {
                "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "max_prompt_tokens": 0, "max_completion_tokens": 0, "truncated": 0,
            })
            u["length_retries"] = u.get("length_retries", 0) + n_len
            if still_truncated:  # still capped after retries → real silent default
                u["hard_truncated"] = u.get("hard_truncated", 0) + 1

    def usage_summary(self) -> Dict[str, object]:
        """Aggregate token usage over ACTUAL API calls (cache hits not counted).

        ``truncated`` counts capped attempts; ``length_retries`` how many were
        retried with a larger budget; ``hard_truncated`` how many were STILL
        capped after retries (the only ones that trigger a silent default).
        """
        with self._usage_lock:
            by_prompt = {k: dict(v) for k, v in self.usage.items()}
        tot = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
               "truncated": 0, "length_retries": 0, "hard_truncated": 0}
        for v in by_prompt.values():
            for k in tot:
                tot[k] += v.get(k, 0)
        return {"model": self.config.model, "max_tokens": self.config.max_tokens,
                "request_style": self._style,
                "by_prompt": by_prompt, "totals": tot}

    def chat(
        self,
        prompt_name: str,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        dump_label: Optional[str] = None,
        dump_name: Optional[str] = None,
    ) -> str:
        """Send a chat completion, returning the raw string content (cached).

        When ``save_prompts_dir`` is set, a markdown trace of the call is
        dumped at ``<dir>/<dump_name or prompt_name>/<filename>``. If
        ``dump_label`` is provided, the filename becomes
        ``<sanitised_label>__<hash[:8]>.md`` for easy navigation; otherwise it
        falls back to ``<hash[:12]>.md``.

        ``dump_name`` routes the SAVED prompt to its own folder WITHOUT changing
        ``prompt_name`` (which is part of the cache key): same cache entry,
        distinct dump folder. Used e.g. by reference validation, which reuses the
        ``hapax_judge`` prompt/cache but is saved under ``validate_reference/``.
        """
        t = self.config.temperature if temperature is None else temperature
        mt = self.config.max_tokens if max_tokens is None else max_tokens
        key = _hash_key({
            "model": self.config.model,
            "prompt_name": prompt_name,
            "system": system,
            "user": user,
            "temperature": t,
        })
        if key in self.cache:
            content = self.cache[key]
        else:
            content, usage, finish_reason = self._call_with_retry(system, user, t, mt)
            if usage is not None:
                self._record_usage(prompt_name, usage, finish_reason)
            # Anti-truncation: a response cut off at the token cap
            # (finish_reason == "length") yields invalid JSON and a silent
            # default verdict. Retry with a doubled budget (up to 4 doublings,
            # e.g. 128 -> 256 -> 512 -> 1024 -> 2048) so it completes. The cache
            # key does not include max_tokens, so the completed content is cached
            # under the same key.
            mt_retry, n_len = mt, 0
            while finish_reason == "length" and n_len < 4:
                mt_retry *= 2
                n_len += 1
                content, usage, finish_reason = self._call_with_retry(system, user, t, mt_retry)
                if usage is not None:
                    self._record_usage(prompt_name, usage, finish_reason)
            if n_len:
                self._record_length_retry(prompt_name, n_len, finish_reason == "length")
            if finish_reason == "length":
                # Still capped after every retry -> invalid JSON -> silent
                # default verdict. Log which call so the affected occurrence can
                # be found (dump_label carries doc/seg/term or ref-form).
                print(f"[TRUNCATED] {dump_name or prompt_name}: still capped at "
                      f"max_tokens={mt_retry} after {n_len} "
                      f"retr{'y' if n_len == 1 else 'ies'} -> silent default. "
                      f"label={dump_label or '(none)'}", file=sys.stderr)
            with self._lock:
                self.cache[key] = content
        if self.save_prompts_dir is not None:
            self._dump_prompt(dump_name or prompt_name, key, system, user, content, t, dump_label)
        return content

    def chat_json(
        self,
        prompt_name: str,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        dump_label: Optional[str] = None,
        dump_name: Optional[str] = None,
    ) -> Optional[dict]:
        raw = self.chat(prompt_name, system, user, temperature, max_tokens, dump_label, dump_name)
        return extract_json(raw)

    def _dump_prompt(
        self,
        prompt_name: str,
        key: str,
        system: str,
        user: str,
        response: str,
        temperature: float,
        dump_label: Optional[str] = None,
    ) -> None:
        """Write a human-readable markdown trace of one LLM call.

        Filename: ``<save_prompts_dir>/<prompt_name>/<sanitised_label>__<hash[:8]>.md``
        when a label is provided (e.g. ``"doc12_seg7_the-list-experiment"``),
        otherwise ``<hash[:12]>.md`` for backward compatibility. Existing files
        are not overwritten (idempotent on re-runs with the same cache).
        """
        sub = self.save_prompts_dir / prompt_name
        sub.mkdir(parents=True, exist_ok=True)
        if dump_label:
            stem = f"{_sanitise_label(dump_label)}__{key[:8]}"
        else:
            stem = key[:12]
        path = sub / f"{stem}.md"
        if path.exists():
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# {prompt_name} - {stem}\n\n")
                f.write(f"- model: `{self.config.model}`\n")
                f.write(f"- temperature: `{temperature}`\n")
                f.write(f"- cache key: `{key}`\n\n")
                f.write("## System\n\n")
                f.write(f"```\n{system}\n```\n\n")
                f.write("## User\n\n")
                f.write(f"```\n{user}\n```\n\n")
                f.write("## Response\n\n")
                f.write(f"```\n{response}\n```\n")
        except Exception:
            pass  # best-effort - dumping must never break the run

    def _create(self, messages, temperature: float, max_tokens: int):
        """One chat completion in the negotiated request style.

        On the first call, tries the styles of ``self._styles`` in order and
        keeps the first one the API accepts; other threads wait meanwhile. Any
        error other than a rejected parameter propagates (retried by the caller).
        """
        if self._style is None:
            with self._style_lock:
                if self._style is None:
                    for style in self._styles:
                        try:
                            resp = self.client.chat.completions.create(
                                model=self.config.model, messages=messages,
                                **_request_kwargs(style, temperature, max_tokens))
                        except Exception as e:  # noqa: BLE001
                            if _is_unsupported_param_error(e) and style != self._styles[-1]:
                                continue
                            raise
                        self._style = style
                        print(f"[INFO] LLM request style for {self.config.model}: {style}",
                              file=sys.stderr)
                        if style == "fixed_temperature":
                            print(f"[WARN] {self.config.model} only accepts its default "
                                  f"temperature: requested temperature {temperature} is "
                                  f"not applied (non-deterministic outputs)", file=sys.stderr)
                        return resp
        return self.client.chat.completions.create(
            model=self.config.model, messages=messages,
            **_request_kwargs(self._style, temperature, max_tokens))

    def _call_with_retry(self, system: str, user: str, temperature: float, max_tokens: int):
        """Return ``(content, usage, finish_reason)``. ``usage``/``finish_reason``
        are None only on total failure (after retries this raises instead)."""
        last_err: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": user})
                resp = self._create(messages, temperature, max_tokens)
                choice = resp.choices[0]
                finish_reason = getattr(choice, "finish_reason", None)
                return (choice.message.content or "", getattr(resp, "usage", None), finish_reason)
            except Exception as e:  # noqa: BLE001
                last_err = e
                # Exponential backoff, but honour the server's Retry-After on 429s.
                delay = self.config.retry_base_delay * (2 ** attempt)
                retry_after = _retry_after_seconds(e)
                if retry_after is not None:
                    delay = max(delay, retry_after)
                time.sleep(delay)
        raise RuntimeError(f"LLM call failed after retries: {last_err}")
