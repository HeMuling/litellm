from unittest.mock import patch

from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig
from litellm.llms.chatgpt.request_auth import ChatGPTRequestAuthContext


class TestChatGPTConfig:
    @patch("litellm.llms.chatgpt.chat.transformation.resolve_chatgpt_request_auth")
    def test_get_openai_compatible_provider_info_uses_request_context(
        self,
        mock_resolve_auth,
    ):
        mock_resolve_auth.return_value = ChatGPTRequestAuthContext(
            api_base="https://chatgpt.example.com",
            access_token="access-123",
            account_id="acct-123",
            session_id="session-123",
            default_headers={"Authorization": "Bearer access-123"},
        )
        config = ChatGPTConfig()

        api_base, api_key, provider = config._get_openai_compatible_provider_info(
            model="chatgpt/gpt-5.6-terra",
            api_base=None,
            api_key=None,
            custom_llm_provider="chatgpt",
        )

        assert api_base == "https://chatgpt.example.com"
        assert api_key == "access-123"
        assert provider == "chatgpt"
        mock_resolve_auth.assert_called_once_with(config.authenticator, None)

    @patch("litellm.llms.chatgpt.chat.transformation.resolve_chatgpt_request_auth")
    @patch("litellm.llms.openai.openai.OpenAIConfig.validate_environment")
    def test_validate_environment_uses_same_request_context(
        self,
        mock_openai_validate,
        mock_resolve_auth,
    ):
        mock_openai_validate.return_value = {"custom-header": "custom-value"}
        mock_resolve_auth.return_value = ChatGPTRequestAuthContext(
            api_base="https://chatgpt.example.com",
            access_token="access-123",
            account_id="acct-123",
            session_id="session-123",
            default_headers={
                "Authorization": "Bearer access-123",
                "ChatGPT-Account-Id": "acct-123",
                "session_id": "session-123",
            },
        )
        config = ChatGPTConfig()
        litellm_params = {"litellm_session_id": "session-123"}

        headers = config.validate_environment(
            headers={},
            model="chatgpt/gpt-5.6-terra",
            messages=[],
            optional_params={},
            litellm_params=litellm_params,
            api_key="access-123",
            api_base=None,
        )

        assert headers == {
            "Authorization": "Bearer access-123",
            "ChatGPT-Account-Id": "acct-123",
            "session_id": "session-123",
            "custom-header": "custom-value",
        }
        mock_resolve_auth.assert_called_once_with(config.authenticator, litellm_params)
