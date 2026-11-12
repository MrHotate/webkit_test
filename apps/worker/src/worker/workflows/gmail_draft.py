"""Create a Gmail draft with optional base64 attachments, using the caller's Gmail credentials."""

import pydantic
import mistralai.workflows as workflows
from mistralai.workflows import Depends
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)

gmail = connector("gmail")


class Attachment(pydantic.BaseModel):
    filename: str
    content: str
    mimeType: str = "text/plain"


class GmailDraftInput(pydantic.BaseModel):
    subject: str = ""
    body: str = ""
    attachments: list[Attachment] = []


class GmailDraftOutput(pydantic.BaseModel):
    draft_id: str | None = None
    thread_id: str | None = None
    raw: str = ""


@workflows.activity(name="create-gmail-draft")
async def create_gmail_draft(
    subject: str,
    body: str,
    attachments: list[dict],
    client: ToolCallClient = Depends(gmail),
) -> dict:
    result = await client.call_tool(
        tool_name="create_draft",
        arguments={
            "subject": subject,
            "body": body,
            "attachments": attachments,
        },
    )
    if hasattr(result, "model_dump"):
        dumped = result.model_dump()
    elif isinstance(result, dict):
        dumped = result
    else:
        dumped = {"raw": str(result)}
    return dumped


@workflows.workflow.define(name="gmail-draft", on_behalf_of=True)
@uses_connectors(gmail)
class GmailDraftWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: GmailDraftInput) -> GmailDraftOutput:
        attachments = [a.model_dump() for a in input.attachments]
        dumped = await create_gmail_draft(input.subject, input.body, attachments)
        return GmailDraftOutput(
            draft_id=dumped.get("id"),
            thread_id=dumped.get("threadId"),
            raw=str(dumped),
        )
