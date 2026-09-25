                                        

import json
import os
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from pydantic import BaseModel, Field
from core.llm import get_llm


class MeetingInsights(BaseModel):
    action_items: str = Field(description="Numbered action items, with owner and deadline when known")
    key_decisions: str = Field(description="Numbered list of decisions made in the meeting")
    open_questions: str = Field(description="Numbered unresolved questions and follow-up topics")


def build_chain(system_prompt : str):
    llm = get_llm(temperature=0.2)
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

def extract_action_items(transcript:str)->str:
    chain = build_chain(
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )

    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return chain.invoke(transcript)


def extract_meeting_insights(transcript: str) -> dict[str, str]:
    llm = get_llm(temperature=0.2)
    if os.getenv("LLM_PROVIDER", "mistral").strip().lower() == "groq":
        structured_llm = llm.with_structured_output(
            MeetingInsights,
            method="json_schema",
            strict=True,
        )
        insights = structured_llm.invoke([
            (
                "system",
                "Analyze this meeting transcript. Include only information stated or "
                "clearly implied in the transcript. Use concise numbered lists. "
                "For empty categories, say that none were identified.",
            ),
            ("human", transcript),
        ])
        result = insights.model_dump() if isinstance(insights, MeetingInsights) else insights
        return _normalize_meeting_insights(result)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Analyze the meeting transcript. Return exactly these three headings, "
            "each on its own line, followed by a concise numbered list: "
            "ACTION ITEMS, KEY DECISIONS, OPEN QUESTIONS. Include owners and "
            "deadlines for action items when stated. If a category is empty, say so. "
            "Do not add an introduction or conclusion.",
        ),
        ("human", "{transcript}"),
    ])
    raw_result = (prompt | llm | StrOutputParser()).invoke({"transcript": transcript})

    required_fields = ("action_items", "key_decisions", "open_questions")
    headings = {
        "action items": "action_items",
        "key decisions": "key_decisions",
        "open questions": "open_questions",
    }
    result = {}
    current_field = None
    heading_pattern = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:\d+[.)]\s*)?(?:\*\*)?"
        r"(ACTION ITEMS|KEY DECISIONS|OPEN QUESTIONS)(?:\*\*)?\s*:?[ \t]*(.*)$",
        re.IGNORECASE,
    )

    for line in raw_result.splitlines():
        match = heading_pattern.match(line)
        if match:
            current_field = headings[match.group(1).lower()]
            result[current_field] = [match.group(2)] if match.group(2) else []
        elif current_field:
            result[current_field].append(line)

    if not all(field in result for field in required_fields):
        try:
            result = json.loads(raw_result)
        except json.JSONDecodeError:
            start = raw_result.find("{")
            end = raw_result.rfind("}")
            if start < 0 or end <= start:
                raise ValueError(
                    "The model response did not include the expected meeting insight sections."
                )
            result = json.loads(raw_result[start:end + 1])

    if not isinstance(result, dict) or any(field not in result for field in required_fields):
        raise ValueError("The model response is missing one or more meeting insight sections.")

    return _normalize_meeting_insights(result)


def _normalize_meeting_insights(result: dict) -> dict[str, str]:
    required_fields = ("action_items", "key_decisions", "open_questions")
    if not isinstance(result, dict) or any(field not in result for field in required_fields):
        raise ValueError("The model response is missing one or more meeting insight sections.")

    normalized = {}
    for field in required_fields:
        value = result[field]
        if isinstance(value, list):
            value = "\n".join(str(item) for item in value)
        normalized[field] = str(value).strip()
    return normalized
