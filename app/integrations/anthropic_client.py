import json
from typing import Any

from anthropic import Anthropic

from app.core.config import get_settings
from app.schemas.email_classification import (
    EmailClassificationRequest,
    EmailClassificationResult,
)


class AnthropicClassificationError(Exception):
    pass


class AnthropicClassificationClient:
    def __init__(self) -> None:
        settings = get_settings()

        if not settings.anthropic_api_key:
            raise AnthropicClassificationError("anthropic_api_key is missing.")

        self.settings = settings
        self.client = Anthropic(api_key=settings.anthropic_api_key)

    def classify_email(
        self,
        payload: EmailClassificationRequest,
    ) -> EmailClassificationResult:
        try:
            response = self.client.messages.create(
                model=self.settings.anthropic_model,
                max_tokens=self.settings.anthropic_max_tokens,
                temperature=0,
                system=self._build_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": self._build_user_prompt(payload),
                    }
                ],
            )
        except Exception as exc:
            raise AnthropicClassificationError(
                f"Anthropic request failed: {exc}"
            ) from exc

        text = self._extract_text_response(response)

        try:
            raw_result = self._parse_json_object(text)
            normalized_result = self._normalize_result(raw_result)
            return EmailClassificationResult(**normalized_result)

        except Exception as exc:
            raise AnthropicClassificationError(
                f"Failed to parse Anthropic classification result: {exc}"
            ) from exc
            

    def _build_system_prompt(self) -> str:
        return """
You are an email classification engine for a job-search mailbox product.

You must classify the email into structured JSON.

Return ONLY valid JSON.
Do not include markdown.
Do not include explanation outside JSON.

Allowed primary_category values:
- APPLICATION_SUBMISSION
- INITIAL_RECRUITER_OUTREACH
- RECRUITER_FOLLOW_UP
- JOB_INTERVIEW
- INTERVIEW_CONFIRMATION
- REJECTION
- APPLICATION_UPDATE
- ASSESSMENT
- OFFER
- FOLLOW_UP_NEEDED
- UNRECOGNIZED
- NOT_JOB_RELATED

Allowed thread_status values:
- NEW_OUTREACH
- WAITING_FOR_USER_REPLY
- WAITING_FOR_RECRUITER_REPLY
- INTERVIEW_SCHEDULING
- INTERVIEW_CONFIRMED
- ASSESSMENT_STAGE
- REJECTED
- OFFER_RECEIVED
- CLOSED
- NEEDS_REVIEW

Allowed review_status values:
- AUTO_CLASSIFIED
- NEEDS_REVIEW

Allowed priority values:
- LOW
- MEDIUME
- HIGH
- URGENT

- APPLICATION_SUBMISSION:
  A confirmation that the candidate successfully applied for a specific
  position or that the application was received.

- APPLICATION_UPDATE:
  A general status change for an application that was already submitted.
  Do not use this for the original application receipt confirmation.

Rules:
1. Choose exactly one primary_category.
2. Choose 0 to 3 secondary_categories.
3. Do not put the primary_category inside secondary_categories.
4. If the email is unclear, use primary_category UNRECOGNIZED.
5. If confidence is below 0.85, use review_status NEEDS_REVIEW.
6. If confidence is 0.85 or higher, use review_status AUTO_CLASSIFIED.
7. Initial recruiter outreach should usually have should_surface=false unless it clearly asks for action.
8. Recruiter follow-up after the user has already replied should usually be important.
9. Do not invent interview dates or deadlines. Use null if not clearly stated.

Return JSON with this exact shape:
{
  "primary_category": "JOB_INTERVIEW",
  "secondary_categories": [],
  "thread_status": "INTERVIEW_SCHEDULING",
  "review_status": "AUTO_CLASSIFIED",
  "priority": "HIGH",
  "confidence": 0.92,
  "company_name": "Acme",
  "role_title": "Backend Engineer",
  "action_needed": true,
  "should_surface": true,
  "interview_at": null,
  "deadline_at": null,
  "reason": "Short reason under 1000 characters."
}
""".strip()

    def _build_user_prompt(self, payload: EmailClassificationRequest) -> str:
        return json.dumps(
            {
                "provider": payload.provider,
                "provider_message_id": payload.provider_message_id,
                "provider_thread_id": payload.provider_thread_id,
                "sender_name": payload.sender_name,
                "sender_email": payload.sender_email,
                "subject": payload.subject,
                "body": payload.body[:6000],
                "received_at": (
                    payload.received_at.isoformat()
                    if payload.received_at
                    else None
                ),
                "thread_context": (
                    payload.thread_context.model_dump(mode="json")
                    if payload.thread_context
                    else None
                ),
            },
            ensure_ascii=False,
        )

    def _extract_text_response(self, response: Any) -> str:
        text_parts: list[str] = []

        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)

        text = "".join(text_parts).strip()

        if not text:
            raise AnthropicClassificationError("Anthropic returned an empty response.")

        return text

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")

            if start == -1 or end == -1 or end <= start:
                raise

            return json.loads(text[start : end + 1])

    def _normalize_result(self, raw_result: dict[str, Any]) -> dict[str, Any]:
        raw_result.setdefault("secondary_categories", [])
        raw_result.setdefault("classification_status", "COMPLETED")
        raw_result.setdefault("error_source", None)
        raw_result.setdefault("error_message", None)

        return raw_result