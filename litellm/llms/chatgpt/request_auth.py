from dataclasses import dataclass
from typing import Any, Optional

from .authenticator import Authenticator
from .common_utils import ensure_chatgpt_session_id, get_chatgpt_default_headers


@dataclass
class ChatGPTRequestAuthContext:
    api_base: str
    access_token: str
    account_id: Optional[str]
    session_id: str
    default_headers: dict


def resolve_chatgpt_request_auth(
    authenticator: Authenticator, litellm_params: Optional[Any]
) -> ChatGPTRequestAuthContext:
    access_token = authenticator.get_access_token()
    account_id = authenticator.get_account_id()
    api_base = authenticator.get_api_base()
    session_id = ensure_chatgpt_session_id(litellm_params)
    default_headers = get_chatgpt_default_headers(access_token, account_id, session_id)
    return ChatGPTRequestAuthContext(
        api_base=api_base,
        access_token=access_token,
        account_id=account_id,
        session_id=session_id,
        default_headers=default_headers,
    )
