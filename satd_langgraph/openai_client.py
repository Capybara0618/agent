from __future__ import annotations

import ast
import builtins
import copy
import difflib
import json
import re
import os
import textwrap
import time
from datetime import datetime, timezone
from typing import Any

from .bootstrap import bootstrap_vendor

bootstrap_vendor()

from openai import OpenAI

from .local_settings import OPENAI_API_KEY as LOCAL_OPENAI_API_KEY
from .local_settings import OPENAI_BASE_URL as LOCAL_OPENAI_BASE_URL
from .schema import (
    AnalysisResult,
    EditConstraint,
    GraphState,
    MethodInquiryResult,
    RepairAttempt,
    RetrievedMethodContext,
    ReviewResult,
    UncertaintyItem,
    preprocess_python_code,
)

MODEL_ALIASES = {"gpt-4o-mini-global": "gpt-4o-mini"}


class OpenAICompatClient:
    def __init__(self, model: str = "gpt-4o-mini", verbose: bool = False) -> None:
        api_key = os.environ.get("OPENAI_API_KEY") or LOCAL_OPENAI_API_KEY
        if api_key == "PASTE_YOUR_OPENAI_COMPAT_KEY_HERE":
            api_key = None
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        base_url = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or LOCAL_OPENAI_BASE_URL
            or None
        )
        if base_url == "https://your-openai-compatible-host/v1":
            base_url = None
        self.request_timeout = float(os.environ.get("OPENAI_TIMEOUT_SECONDS") or 60)
        self.max_attempts = max(1, int(os.environ.get("OPENAI_MAX_ATTEMPTS") or 3))
        sdk_max_retries = max(0, int(os.environ.get("OPENAI_SDK_MAX_RETRIES") or 0))
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.request_timeout,
            max_retries=sdk_max_retries,
        )
        self.requested_model = model
        self.model = self._normalize_model_name(model)
        self.verbose = bool(verbose)

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        request_label: str = "",
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        sanitized = False
        active_system_prompt = self._coerce_prompt_text(system_prompt)
        active_user_prompt = self._coerce_prompt_text(user_prompt)
        active_model = self.model
        label = request_label or "llm_request"
        for attempt in range(self.max_attempts):
            attempt_started = time.time()
            self._emit_log(
                f"[llm] start label={label} attempt={attempt + 1}/{self.max_attempts} model={active_model} timeout={self.request_timeout:.0f}s"
            )
            try:
                request_kwargs: dict[str, Any] = {
                    "model": active_model,
                    "messages": [
                        {"role": "system", "content": active_system_prompt},
                        {"role": "user", "content": active_user_prompt},
                    ],
                    "temperature": temperature,
                    "response_format": {"type": "json_object"},
                }
                if max_tokens is not None:
                    request_kwargs["max_tokens"] = max_tokens
                response = self.client.chat.completions.create(**request_kwargs)
                content = response.choices[0].message.content or "{}"
                elapsed = time.time() - attempt_started
                self._emit_log(f"[llm] success label={label} attempt={attempt + 1}/{self.max_attempts} elapsed={elapsed:.2f}s")
                return json.loads(content)
            except Exception as exc:
                last_error = exc
                elapsed = time.time() - attempt_started
                self._emit_log(
                    f"[llm] error label={label} attempt={attempt + 1}/{self.max_attempts} "
                    f"elapsed={elapsed:.2f}s type={type(exc).__name__} message={self._short_error(exc)}"
                )
                fallback_model = self._fallback_model_for_error(exc, active_model)
                if fallback_model and fallback_model != active_model:
                    self._emit_log(f"[llm] fallback_model label={label} from={active_model} to={fallback_model}")
                    active_model = fallback_model
                    self.model = fallback_model
                    continue
                if self._is_content_filter_error(exc) and not sanitized:
                    self._emit_log(f"[llm] sanitize_prompts label={label}")
                    active_system_prompt, active_user_prompt = self._sanitize_prompts(system_prompt, user_prompt)
                    sanitized = True
                    continue
                if attempt < self.max_attempts - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("OpenAI-compatible request failed unexpectedly.")

    def generate_tool_call(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        temperature: float = 0.0,
        request_label: str = "",
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        active_system_prompt = self._coerce_prompt_text(system_prompt)
        active_user_prompt = self._coerce_prompt_text(user_prompt)
        active_model = self.model
        label = request_label or "tool_call_request"
        for attempt in range(self.max_attempts):
            attempt_started = time.time()
            self._emit_log(
                f"[llm] start label={label} attempt={attempt + 1}/{self.max_attempts} model={active_model} timeout={self.request_timeout:.0f}s"
            )
            try:
                request_kwargs: dict[str, Any] = {
                    "model": active_model,
                    "messages": [
                        {"role": "system", "content": active_system_prompt},
                        {"role": "user", "content": active_user_prompt},
                    ],
                    "temperature": temperature,
                    "tools": tools,
                    "tool_choice": "required",
                }
                if max_tokens is not None:
                    request_kwargs["max_tokens"] = max_tokens
                response = self.client.chat.completions.create(**request_kwargs)
                message = response.choices[0].message
                tool_calls = message.tool_calls or []
                if not tool_calls:
                    raise RuntimeError("model returned no tool call")
                function = tool_calls[0].function
                elapsed = time.time() - attempt_started
                self._emit_log(f"[llm] success label={label} attempt={attempt + 1}/{self.max_attempts} elapsed={elapsed:.2f}s")
                return {
                    "name": function.name,
                    "arguments": self._loads_json_object(function.arguments),
                }
            except Exception as exc:
                last_error = exc
                elapsed = time.time() - attempt_started
                self._emit_log(
                    f"[llm] error label={label} attempt={attempt + 1}/{self.max_attempts} "
                    f"elapsed={elapsed:.2f}s type={type(exc).__name__} message={self._short_error(exc)}"
                )
                fallback_model = self._fallback_model_for_error(exc, active_model)
                if fallback_model and fallback_model != active_model:
                    self._emit_log(f"[llm] fallback_model label={label} from={active_model} to={fallback_model}")
                    active_model = fallback_model
                    self.model = fallback_model
                    continue
                if attempt < self.max_attempts - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("OpenAI-compatible tool call failed unexpectedly.")

    def _loads_json_object(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        try:
            payload = json.loads(str(value or "{}"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _emit_log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _short_error(self, exc: Exception) -> str:
        message = " ".join(str(exc).split())
        if len(message) > 220:
            return message[:217] + "..."
        return message

    def _normalize_model_name(self, model: str) -> str:
        cleaned = (model or "gpt-4o-mini").strip()
        return MODEL_ALIASES.get(cleaned, cleaned)

    def _fallback_model_for_error(self, exc: Exception, active_model: str) -> str | None:
        message = str(exc).lower()
        normalized = self._normalize_model_name(active_model)
        if "unknown model" in message and normalized != active_model:
            return normalized
        if "unknown model" in message and active_model.endswith("-global"):
            return active_model[: -len("-global")]
        return None

    def _is_content_filter_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "content_filter" in message or "content management policy" in message

    def _sanitize_prompts(self, system_prompt: str, user_prompt: str) -> tuple[str, str]:
        compact_system = self._coerce_prompt_text(system_prompt) + " Prefer concise evidence use and avoid unnecessary raw excerpts."
        compact_user = self._coerce_prompt_text(user_prompt)
        compact_user = re.sub(
            r"(Base \+ repair \+ review context:\n)[\s\S]*",
            r"\1[context compacted for safety; rely on metadata and direct task evidence]",
            compact_user,
        )
        compact_user = re.sub(
            r"(Unified GitHub context:\n)[\s\S]*",
            r"\1[context compacted for safety; rely on metadata and direct task evidence]",
            compact_user,
        )
        compact_user = re.sub(
            r"(Base \+ repair context:\n)[\s\S]*",
            r"\1[context compacted for safety; rely on metadata and direct task evidence]",
            compact_user,
        )
        compact_user = re.sub(
            r"(Base GitHub context:\n)[\s\S]*",
            r"\1[context compacted for safety; rely on metadata and direct task evidence]",
            compact_user,
        )
        compact_user = re.sub(
            r"(Original code block:\n)([\s\S]*?)(\n\n(?:Repaired code block:|Repair plan:|Base \+ repair context:|Base \+ repair \+ review context:))",
            lambda m: m.group(1) + m.group(2)[:1200] + m.group(3),
            compact_user,
        )
        compact_user = re.sub(
            r"(Current code snippet:\n)([\s\S]*?)(\n\nBase GitHub context:)",
            lambda m: m.group(1) + m.group(2)[:1200] + m.group(3),
            compact_user,
        )
        compact_user = re.sub(
            r"(Repaired code block:\n)([\s\S]*?)(\n\nRepair plan:)",
            lambda m: m.group(1) + m.group(2)[:1200] + m.group(3),
            compact_user,
        )
        compact_user = re.sub(r"\n{3,}", "\n\n", compact_user)
        return compact_system, compact_user

    def _coerce_prompt_text(self, prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, (list, tuple)):
            return "".join(self._coerce_prompt_text(item) for item in prompt)
        if prompt is None:
            return ""
        return str(prompt)

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")


