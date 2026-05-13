import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.exceptions import AuthenticationError
from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig
from litellm.llms.chatgpt.common_utils import GetAccessTokenError
from litellm.llms.chatgpt.request_auth import ChatGPTRequestAuthContext


class TestChatGPTConfig:
    @patch("litellm.llms.chatgpt.chat.transformation.resolve_chatgpt_request_auth")
    def test_get_openai_compatible_provider_info(self, mock_resolve_auth):
        mock_resolve_auth.return_value = ChatGPTRequestAuthContext(
            api_base="https://chatgpt.example.com",
            access_token="access-123",
            account_id="acct-123",
            session_id="session-123",
            default_headers={"Authorization": "Bearer access-123"},
        )
        config = ChatGPTConfig()

        api_base, api_key, custom_llm_provider = (
            config._get_openai_compatible_provider_info(
                model="chatgpt/gpt-5.4",
                api_base=None,
                api_key=None,
                custom_llm_provider="chatgpt",
            )
        )

        assert api_base == "https://chatgpt.example.com"
        assert api_key == "access-123"
        assert custom_llm_provider == "chatgpt"
        mock_resolve_auth.assert_called_once_with(config.authenticator, None)

    @patch("litellm.llms.chatgpt.chat.transformation.resolve_chatgpt_request_auth")
    def test_get_openai_compatible_provider_info_wraps_auth_error(
        self, mock_resolve_auth
    ):
        mock_resolve_auth.side_effect = GetAccessTokenError(
            message="auth failed",
            status_code=401,
        )
        config = ChatGPTConfig()

        with pytest.raises(AuthenticationError, match="auth failed"):
            config._get_openai_compatible_provider_info(
                model="chatgpt/gpt-5.4",
                api_base=None,
                api_key=None,
                custom_llm_provider="chatgpt",
            )

    @patch("litellm.llms.chatgpt.chat.transformation.resolve_chatgpt_request_auth")
    def test_validate_environment_uses_request_auth_context(self, mock_resolve_auth):
        mock_resolve_auth.return_value = ChatGPTRequestAuthContext(
            api_base="https://chatgpt.example.com",
            access_token="access-123",
            account_id="acct-123",
            session_id="session-123",
            default_headers={
                "Authorization": "Bearer access-123",
                "content-type": "application/json",
                "accept": "text/event-stream",
                "originator": "codex",
                "user-agent": "codex/1.0",
                "session_id": "session-123",
                "ChatGPT-Account-Id": "acct-123",
            },
        )
        config = ChatGPTConfig()
        litellm_params = {"litellm_session_id": "session-123"}

        headers = config.validate_environment(
            headers={"custom-header": "custom-value"},
            model="chatgpt/gpt-5.4",
            messages=[],
            optional_params={},
            litellm_params=litellm_params,
            api_key="access-123",
            api_base=None,
        )

        assert headers["Authorization"] == "Bearer access-123"
        assert headers["ChatGPT-Account-Id"] == "acct-123"
        assert headers["session_id"] == "session-123"
        assert headers["custom-header"] == "custom-value"
        mock_resolve_auth.assert_called_once_with(config.authenticator, litellm_params)

    @patch("litellm.llms.chatgpt.chat.transformation.resolve_chatgpt_request_auth")
    def test_validate_environment_wraps_auth_error(self, mock_resolve_auth):
        mock_resolve_auth.side_effect = GetAccessTokenError(
            message="auth failed",
            status_code=401,
        )
        config = ChatGPTConfig()

        with pytest.raises(AuthenticationError, match="auth failed"):
            config.validate_environment(
                headers={},
                model="chatgpt/gpt-5.4",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key="access-123",
                api_base=None,
            )
