import json
from typing import Any, Optional

from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _safe_convert_created_field,
)
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import (
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import CustomStreamWrapper

from ..authenticator import Authenticator
from ..common_utils import (
    CHATGPT_API_BASE,
    GetAccessTokenError,
    get_chatgpt_default_instructions,
)
from ..request_auth import resolve_chatgpt_request_auth


class ChatGPTResponsesAPIConfig(OpenAIResponsesAPIConfig):
    def __init__(self) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.CHATGPT

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        try:
            auth_context = resolve_chatgpt_request_auth(self.authenticator, litellm_params)
        except GetAccessTokenError as e:
            raise AuthenticationError(
                model=model,
                llm_provider="chatgpt",
                message=str(e),
            )

        return {**auth_context.default_headers, **headers}

    def transform_responses_api_request(
        self,
        model: str,
        input: Any,
        response_api_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        request = super().transform_responses_api_request(
            model,
            input,
            response_api_optional_request_params,
            litellm_params,
            headers,
        )
        base_instructions = get_chatgpt_default_instructions()
        existing_instructions = request.get("instructions")
        if existing_instructions:
            if base_instructions not in existing_instructions:
                request["instructions"] = f"{base_instructions}\n\n{existing_instructions}"
        else:
            request["instructions"] = base_instructions
        request["store"] = False
        request["stream"] = True
        include = list(request.get("include") or [])
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")
        request["include"] = include

        allowed_keys = {
            "model",
            "input",
            "instructions",
            "stream",
            "store",
            "include",
            "text",
            "tools",
            "tool_choice",
            "reasoning",
            "previous_response_id",
            "truncation",
        }

        return {k: v for k, v in request.items() if k in allowed_keys}

    @staticmethod
    def _get_output_item_value(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    @classmethod
    def _has_convertible_output_item(cls, output_items: Any) -> bool:
        if not isinstance(output_items, list):
            return False
        for item in output_items:
            item_type = cls._get_output_item_value(item, "type")
            if item_type == "function_call":
                return True
            if item_type != "message":
                continue
            content_items = cls._get_output_item_value(item, "content", []) or []
            for content_item in content_items:
                if cls._get_output_item_value(content_item, "type") == "output_text":
                    return True
        return False

    @classmethod
    def _merge_completed_output_items(cls, output_items: Any, completed_items: list[dict]) -> list:
        merged = list(output_items) if isinstance(output_items, list) else []
        existing_ids = {
            item_id
            for item in merged
            if isinstance(
                item_id := cls._get_output_item_value(item, "id"),
                str,
            )
            and item_id
        }
        for item in completed_items:
            item_id = cls._get_output_item_value(item, "id")
            if isinstance(item_id, str) and item_id in existing_ids:
                continue
            merged.append(item)
            if isinstance(item_id, str) and item_id:
                existing_ids.add(item_id)
        return merged

    def _build_completed_response(
        self,
        response_payload: Any,
        completed_output_items: dict[int, dict],
    ) -> Optional[ResponsesAPIResponse]:
        if not isinstance(response_payload, dict):
            return None

        response_payload = dict(response_payload)
        if completed_output_items:
            output_items = response_payload.get("output")
            completed_items = [completed_output_items[index] for index in sorted(completed_output_items)]
            if not output_items:
                response_payload["output"] = completed_items
            elif not self._has_convertible_output_item(output_items):
                response_payload["output"] = self._merge_completed_output_items(output_items, completed_items)

        if "created_at" in response_payload:
            response_payload["created_at"] = _safe_convert_created_field(response_payload["created_at"])

        try:
            return ResponsesAPIResponse(**response_payload)
        except Exception:
            return ResponsesAPIResponse.model_construct(**response_payload)

    def transform_response_api_response(
        self,
        model: str,
        raw_response: Any,
        logging_obj: Any,
    ):
        content_type = (raw_response.headers or {}).get("content-type", "")
        body_text = raw_response.text or ""
        if "text/event-stream" not in content_type.lower():
            trimmed_body = body_text.lstrip()
            if not (
                trimmed_body.startswith("event:")
                or trimmed_body.startswith("data:")
                or "\nevent:" in body_text
                or "\ndata:" in body_text
            ):
                return super().transform_response_api_response(
                    model=model,
                    raw_response=raw_response,
                    logging_obj=logging_obj,
                )

        logging_obj.post_call(
            original_response=raw_response.text,
            additional_args={"complete_input_dict": {}},
        )

        completed_response = None
        error_message = None
        completed_output_items: dict[int, dict] = {}
        for chunk in body_text.splitlines():
            stripped_chunk = CustomStreamWrapper._strip_sse_data_from_chunk(chunk)
            if not stripped_chunk:
                continue
            stripped_chunk = stripped_chunk.strip()
            if not stripped_chunk:
                continue
            if stripped_chunk == STREAM_SSE_DONE_STRING:
                break
            try:
                parsed_chunk = json.loads(stripped_chunk)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed_chunk, dict):
                continue
            event_type = parsed_chunk.get("type")
            if event_type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE:
                output_item = parsed_chunk.get("item")
                output_index = parsed_chunk.get("output_index")
                if isinstance(output_item, dict) and isinstance(output_index, int):
                    completed_output_items[output_index] = output_item
                continue
            if event_type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED:
                completed_response = self._build_completed_response(
                    response_payload=parsed_chunk.get("response"),
                    completed_output_items=completed_output_items,
                )
                break
            if event_type in (
                ResponsesAPIStreamEvents.RESPONSE_FAILED,
                ResponsesAPIStreamEvents.ERROR,
            ):
                error_obj = parsed_chunk.get("error") or (parsed_chunk.get("response") or {}).get("error")
                if error_obj is not None:
                    if isinstance(error_obj, dict):
                        error_message = error_obj.get("message") or str(error_obj)
                    else:
                        error_message = str(error_obj)

        if completed_response is None:
            raise OpenAIError(
                message=error_message or raw_response.text,
                status_code=raw_response.status_code,
            )

        raw_headers = dict(raw_response.headers)
        processed_headers = process_response_headers(raw_headers)
        if not hasattr(completed_response, "_hidden_params"):
            setattr(completed_response, "_hidden_params", {})
        completed_response._hidden_params["additional_headers"] = processed_headers
        completed_response._hidden_params["headers"] = raw_headers
        return completed_response

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        api_base = api_base or self.authenticator.get_api_base() or CHATGPT_API_BASE
        api_base = api_base.rstrip("/")
        return f"{api_base}/responses"

    def supports_native_websocket(self) -> bool:
        """ChatGPT does not support native WebSocket for Responses API"""
        return False
