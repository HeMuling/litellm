from unittest.mock import MagicMock, call, patch

import pytest

from litellm.llms.chatgpt.common_utils import GetAccessTokenError
from litellm.llms.chatgpt.request_auth import resolve_chatgpt_request_auth
from litellm.types.router import GenericLiteLLMParams


class TestChatGPTRequestAuth:
    def test_resolve_chatgpt_request_auth_builds_context(self):
        authenticator = MagicMock()
        authenticator.get_access_token.return_value = "access-123"
        authenticator.get_account_id.return_value = "acct-123"
        authenticator.get_api_base.return_value = "https://chatgpt.example.com"

        with patch(
            "litellm.llms.chatgpt.request_auth.get_chatgpt_default_headers",
            return_value={"Authorization": "Bearer access-123"},
        ) as mock_headers:
            context = resolve_chatgpt_request_auth(
                authenticator,
                GenericLiteLLMParams(litellm_session_id="session-123"),
            )

        assert context.api_base == "https://chatgpt.example.com"
        assert context.access_token == "access-123"
        assert context.account_id == "acct-123"
        assert context.session_id == "session-123"
        assert context.default_headers == {"Authorization": "Bearer access-123"}
        assert authenticator.method_calls == [
            call.get_access_token(),
            call.get_account_id(),
            call.get_api_base(),
        ]
        mock_headers.assert_called_once_with("access-123", "acct-123", "session-123")

    def test_resolve_chatgpt_request_auth_omits_account_id_header_when_missing(self):
        authenticator = MagicMock()
        authenticator.get_access_token.return_value = "access-123"
        authenticator.get_account_id.return_value = None
        authenticator.get_api_base.return_value = "https://chatgpt.example.com"

        context = resolve_chatgpt_request_auth(
            authenticator,
            GenericLiteLLMParams(metadata={"session_id": "metadata-session"}),
        )

        assert context.account_id is None
        assert context.session_id == "metadata-session"
        assert context.default_headers["Authorization"] == "Bearer access-123"
        assert context.default_headers["session_id"] == "metadata-session"
        assert "ChatGPT-Account-Id" not in context.default_headers

    def test_resolve_chatgpt_request_auth_does_not_swallow_provider_errors(self):
        authenticator = MagicMock()
        authenticator.get_access_token.side_effect = GetAccessTokenError(
            message="auth failed",
            status_code=401,
        )

        with pytest.raises(GetAccessTokenError, match="auth failed"):
            resolve_chatgpt_request_auth(authenticator, litellm_params=None)
