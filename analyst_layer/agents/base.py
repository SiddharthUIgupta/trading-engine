"""Shared base for all Layer-2 sub-agents.

Every agent forces its Claude completion through a single tool call whose
input_schema is the agent's Pydantic output model. This is what makes
"never executes on raw natural language" true at the code level: the
agent literally cannot return free text as its result — `_call_structured`
raises if the model doesn't call the tool, and raises again if the tool
input fails Pydantic validation.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

UsageCallback = Callable[[str, str, Any], None]
"""Called as (agent_name, model, usage) after every successful Claude call.
`usage` is the raw `response.usage` object — base.py deliberately doesn't
know about pricing; that's analyst_layer/pricing.py's job, invoked by
whatever the callback is wired to (execution_layer.runtime.TradingRuntime).
"""


class StructuredOutputError(Exception):
    """Raised when the LLM fails to produce a schema-valid structured result."""


class BaseAgent(ABC):
    name: str

    def __init__(self, client: Anthropic, model: str, usage_callback: UsageCallback | None = None) -> None:
        self._client = client
        self._model = model
        self._usage_callback = usage_callback

    @property
    @abstractmethod
    def system_prompt(self) -> str: ...

    def _call_structured(
        self,
        user_prompt: str,
        output_model: type[ModelT],
        tool_name: str,
        max_tokens: int = 1024,
    ) -> ModelT:
        # system_prompt and the tool's input_schema are identical on every call
        # for a given agent — same ticker loop, same agents, all day. Marking
        # both as cache breakpoints means only the per-call user_prompt (the
        # actual market data) is billed as fresh input tokens after the first
        # call of the day.
        tool = {
            "name": tool_name,
            "description": f"Emit the {output_model.__name__} result. This is the only way to respond.",
            "input_schema": output_model.model_json_schema(),
            "cache_control": {"type": "ephemeral"},
        }
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}],
            tools=[tool],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user_prompt}],
        )

        usage = getattr(response, "usage", None)
        if usage is not None:
            logger.debug(
                "%s token usage: input=%d output=%d cache_creation=%d cache_read=%d",
                self.name,
                usage.input_tokens,
                usage.output_tokens,
                getattr(usage, "cache_creation_input_tokens", 0) or 0,
                getattr(usage, "cache_read_input_tokens", 0) or 0,
            )
            if self._usage_callback is not None:
                self._usage_callback(self.name, self._model, usage)

        tool_use_blocks = [b for b in response.content if b.type == "tool_use" and b.name == tool_name]
        if not tool_use_blocks:
            raise StructuredOutputError(f"{self.name}: model did not call required tool '{tool_name}'")

        raw_input: dict[str, Any] = tool_use_blocks[0].input
        try:
            return output_model.model_validate(raw_input)
        except ValidationError as exc:
            raise StructuredOutputError(
                f"{self.name}: tool input failed schema validation: {exc}\nraw={json.dumps(raw_input)}"
            ) from exc
