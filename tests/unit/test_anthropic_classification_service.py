from dataclasses import dataclass
import json
from typing import Literal
from types import SimpleNamespace
from app.enums.classification_error_source import ClassificationErrorSource
from app.enums.classification_status import ClassificationStatus
from app.enums.email_category import EmailCategory
from app.enums.email_priority import EmailPriority
from app.enums.mail_provider import MailProvider
from app.enums.review_status import ReviewStatus
from app.enums.thread_status import ThreadStatus
from app.integrations.anthropic_client import AnthropicClassificationClient, AnthropicClassificationError
from app.schemas.email_classification import (
    EmailClassificationRequest,
    EmailClassificationResult,
)
from app.services import classification_service as classification_service_module
from app.services.classification_service import ClassificationService

@dataclass
class FakeSettings:
    ai_classifier_mode: Literal["fallback", "anthropic"] = "anthropic"

def make_request() -> EmailClassificationRequest:
    return EmailClassificationRequest(
        provider=MailProvider.MANUAL_TEST,
        sender_name="Jane Recruiter",
        sender_email="jane@company.com",
        subject="Interview invitation",
        body="Hi, we would like to schedule a technical interview next week.",
    )
    
    sd
    
def test_anthropic_classification_service(monkeypatch):
    class FakeAnthropicClient:
        def classify_email(self, payload:EmailClassificationRequest):
            return EmailClassificationResult(
                primary_category=EmailCategory.JOB_INTERVIEW,
                secondary_categories=[],
                thread_status=ThreadStatus.INTERVIEW_SCHEDULING,
                review_status=ReviewStatus.AUTO_CLASSIFIED,
                priority=EmailPriority.HIGH,
                classification_status=  ClassificationStatus.COMPLETED,
                confidence=0.94,
                company_name="Acme",
                role_title="Backend Engineer",
                action_needed=True,
                should_surface=True,
                reason="The email asks the candidate to schedule an interview.",
            )
        
    monkeypatch.setattr(
        classification_service_module,
        "get_settings",
        lambda: FakeSettings(ai_classifier_mode="anthropic"),
    )
    
    monkeypatch.setattr(
        classification_service_module,
        "AnthropicClassificationClient",
        lambda: FakeAnthropicClient(),
    )
    
    service = ClassificationService()
    result = service.classify_preview(make_request())
    
    assert result.primary_category == EmailCategory.JOB_INTERVIEW
    assert result.thread_status == ThreadStatus.INTERVIEW_SCHEDULING
    assert result.review_status == ReviewStatus.AUTO_CLASSIFIED
    assert result.priority == EmailPriority.HIGH
    assert result.classification_status == ClassificationStatus.COMPLETED
    assert result.error_source is None
    assert result.action_needed is True
    assert result.should_surface is True
    assert result.company_name == "Acme"
    assert result.role_title == "Backend Engineer"
    
def test_anthropic_mode_returns_failed_result_when_provider_errors(monkeypatch):
    class FakeFailingAnthropicClient:
        def classify_email(self, payload:EmailClassificationRequest):
            raise AnthropicClassificationError("Anthropic request failed: timeout")
        
    monkeypatch.setattr(
        classification_service_module,
        "get_settings",
        lambda: FakeSettings(ai_classifier_mode="anthropic"),
    )
    
    monkeypatch.setattr(
        classification_service_module,
        "AnthropicClassificationClient",
        lambda: FakeFailingAnthropicClient(),
    )
    
    service = ClassificationService()
    result = service.classify_preview(make_request())
    
    assert result.primary_category == EmailCategory.UNRECOGNIZED
    assert result.thread_status == ThreadStatus.NEEDS_REVIEW
    assert result.review_status == ReviewStatus.NEEDS_REVIEW
    assert result.priority == EmailPriority.HIGH
    assert result.classification_status == ClassificationStatus.FAILED
    assert result.error_source == ClassificationErrorSource.ANTHROPIC
    assert "timeout" in result.error_message
    assert result.confidence == 0.0
    assert result.action_needed is True
    assert result.should_surface is True

def test_fallback_mode_does_note_call_anthropic(monkeypatch):
    class FakeAnthropicClientThatShouldNotBeCalled:
        def classify_email(self, payload: EmailClassificationRequest):
            raise AssertionError("Anthropic should not be called in fallback mode.")
        
    monkeypatch.setattr(
        classification_service_module,
        "get_settings",
        lambda: FakeSettings(ai_classifier_mode="fallback"),
    )

    monkeypatch.setattr(
        classification_service_module,
        "AnthropicClassificationClient",
        FakeAnthropicClientThatShouldNotBeCalled,
    )
    
    service = ClassificationService()
    
    payload = EmailClassificationRequest(
        provider=MailProvider.MANUAL_TEST,
        sender_name="Hiring Team",
        sender_email="careers@company.com",
        subject="Application update",
        body="Unfortunately, we are not moving forward with your application.",
    )
    
    result = service.classify_preview(payload)

    assert result.primary_category == EmailCategory.REJECTION
    assert result.classification_status == ClassificationStatus.COMPLETED
    assert result.error_source is None
    
def test_anthropic_classifies_application_submission(monkeypatch):
    anthropic_response = {
        "primary_category": "APPLICATION_SUBMISSION",
        "secondary_categories": [],
        "thread_status": "NEW_OUTREACH",
        "review_status": "AUTO_CLASSIFIED",
        "priority": "MEDIUM",
        "classification_status": "COMPLETED",
        "confidence": 0.96,
        "company_name": "Acme",
        "role_title": "Backend Engineer",
        "action_needed": False,
        "should_surface": True,
        "interview_date": None,
        "reason": (
            "The email confirms that the candidate's application "
            "was successfully received."
        ),
        "error_source": None,
        "error_message": None,
    }

    fake_sdk_response = SimpleNamespace(
        id="msg_test_application_submission",
        content=[
            SimpleNamespace(
                type="text",
                text=json.dumps(anthropic_response),
            )
        ],
        usage=SimpleNamespace(
            input_tokens=300,
            output_tokens=80,
        ),
    )

    client = AnthropicClassificationClient()

    monkeypatch.setattr(
        client.client.messages,
        "create",
        lambda **kwargs: fake_sdk_response,
    )

    request = EmailClassificationRequest(
        provider="MANUAL_TEST",
        provider_message_id="test-message-application-submission",
        provider_thread_id="test-thread-application-submission",
        sender_name="Acme Recruiting",
        sender_email="recruiting@acme.example",
        subject="Thank you for applying to Acme",
        body=(
            "Thank you for applying for the Backend Engineer position. "
            "We have successfully received your application."
        ),
    )

    result = client.classify_email(request)

    assert result.primary_category == EmailCategory.APPLICATION_SUBMISSION
    assert result.secondary_categories == []
    assert result.company_name == "Acme"
    assert result.role_title == "Backend Engineer"
    assert result.confidence == 0.96
    assert result.action_needed is False
    assert result.should_surface is True
    assert result.interview_date is None