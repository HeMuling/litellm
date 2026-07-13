"""Dependency-light routing predicates for Responses MCP tools."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

LITELLM_PROXY_MCP_SERVER_URL = "litellm_proxy"
LITELLM_PROXY_MCP_SERVER_URL_PREFIX = f"{LITELLM_PROXY_MCP_SERVER_URL}/mcp/"
PROXY_MCP_PATH_RE = re.compile(r"^https?://.+/mcp/([^/]+)$")


def should_use_litellm_mcp_gateway(tools: Iterable[Any] | None) -> bool:
    """Return whether any MCP tool targets LiteLLM's internal gateway."""

    if not tools:
        return False
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "mcp":
            continue
        server_url = tool.get("server_url", "")
        if not isinstance(server_url, str):
            continue
        if server_url.startswith(LITELLM_PROXY_MCP_SERVER_URL):
            return True
        if PROXY_MCP_PATH_RE.match(server_url):
            return True
    return False
