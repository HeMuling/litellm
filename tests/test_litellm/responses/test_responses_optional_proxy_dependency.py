"""Responses calls must not require proxy-only dependencies."""

from __future__ import annotations

import builtins
from typing import Any

from litellm.responses.main import _responses_try_dispatch_mcp_gateway


def test_response_without_tools_does_not_import_proxy_mcp_handler(monkeypatch: Any) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "litellm.responses.mcp.litellm_proxy_mcp_handler":
            raise ModuleNotFoundError("proxy dependency is unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    result = _responses_try_dispatch_mcp_gateway(
        tools=None,
        input="hello",
        model="chatgpt/gpt-5.4",
        include=None,
        instructions=None,
        max_output_tokens=None,
        prompt=None,
        metadata=None,
        parallel_tool_calls=None,
        previous_response_id=None,
        reasoning=None,
        store=None,
        background=None,
        stream=True,
        temperature=None,
        text=None,
        tool_choice=None,
        top_p=None,
        truncation=None,
        user=None,
        extra_headers=None,
        extra_query=None,
        extra_body=None,
        timeout=None,
        custom_llm_provider="chatgpt",
        kwargs={},
        _is_async=True,
    )

    assert result is None
