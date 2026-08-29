"""Conversational memory management and query contextualization for multi-turn RAG."""

import logging
from typing import Any
from google import genai
from google.genai import types as genai_types

logger = logging.getLogger(__name__)


class ConversationalMemoryManager:
    """Manages chat context, multi-turn query rewriting, and session auto-titling."""

    def __init__(self, genai_client: genai.Client, model_name: str = "gemini-2.5-flash"):
        self.client = genai_client
        self.model_name = model_name

    def condense_query(self, history: list[dict[str, Any]], new_query: str) -> str:
        """
        Rephrase a multi-turn follow-up question into a standalone search query.
        
        If the conversation history is empty, returns the original query.
        """
        if not history:
            return new_query.strip()

        # Format up to last 6 messages into conversation transcript
        transcript_parts = []
        recent_turns = history[-6:]
        for msg in recent_turns:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "").strip()
            # Trim assistant messages to keep rewriting prompt compact
            if role == "Assistant" and len(content) > 300:
                content = content[:300] + "..."
            transcript_parts.append(f"{role}: {content}")

        transcript = "\n".join(transcript_parts)

        prompt = f"""You are an expert search query contextualizer.
Given the previous chat conversation and the user's latest follow-up question, rephrase the follow-up question into a standalone, concise search query for retrieving technical runbooks and documentation.
- Maintain all error codes, service names, configuration parameters, and specific technologies mentioned in previous turns.
- Do NOT answer the question.
- Do NOT add filler phrases or commentary.
- If the question is already fully standalone and self-contained, return it as-is.

Conversation History:
{transcript}

Latest Follow-up Question: {new_query}

Standalone Search Query:"""

        try:
            res = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=100,
                ),
            )
            condensed = res.text.strip()
            if condensed:
                logger.info(f"Condensed multi-turn query '{new_query}' -> '{condensed}'")
                return condensed
        except Exception as e:
            logger.warning(f"Failed to condense query due to error ({e}); using original query.")

        return new_query.strip()

    def generate_chat_title(self, query: str, response: str) -> str:
        """Generate a short (3-5 words) descriptive title for a newly initiated chat session."""
        prompt = f"""Generate a concise, descriptive title (maximum 3 to 5 words) for a technical support chat based on the initial exchange below.
Do not use quotes, punctuation, or generic titles like 'New Chat' or 'Help Request'.

User Request: {query[:300]}
Assistant Response: {response[:300]}

Title:"""

        try:
            res = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=20,
                ),
            )
            title = res.text.strip().replace('"', '').replace("'", "")
            if title and len(title) <= 60:
                return title
        except Exception as e:
            logger.warning(f"Error generating chat title: {e}")

        # Fallback to truncated user query
        words = query.strip().split()
        return " ".join(words[:5]).capitalize() if words else "Technical Discussion"

    def format_generation_contents(
        self,
        history: list[dict[str, Any]],
        current_query: str,
        retrieved_contexts: list[dict[str, Any]],
        attached_file_bytes: bytes | None = None,
        attached_mime_type: str | None = None,
    ) -> list[Any] | str:
        """Construct multimodal prompt contents including retrieved chunks, past turns, and current query."""
        # 1. Format internal runbook context block
        context_parts = []
        for doc in retrieved_contexts:
            metadata = doc.get("metadata") or {}
            url = metadata.get("image_url") or metadata.get("pdf_url") or metadata.get("source_url", "Unknown source")
            context_parts.append(f"[Source: {url}]\n{doc['content']}")
        context_block = "\n\n".join(context_parts) if context_parts else "No internal documentation found for this specific query."

        # 2. Format recent conversation turns (last 6 messages)
        history_formatted = ""
        if history:
            history_lines = []
            for m in history[-6:]:
                r = "User" if m.get("role") == "user" else "Assistant"
                # Strip legacy doc gap tags from assistant history
                c = m.get("content", "")
                if "<!-- DOCUMENTATION_GAPS -->" in c:
                    c = c.split("<!-- DOCUMENTATION_GAPS -->")[0].strip()
                history_lines.append(f"**{r}**: {c}")
            history_formatted = "\n\nRecent Conversation History:\n" + "\n\n".join(history_lines) + "\n\n"

        # 3. Assemble prompt instruction
        if attached_file_bytes and attached_mime_type and attached_mime_type != "application/octet-stream":
            instruction = f"""Internal Runbooks & Knowledge Base Context:
{context_block}
{history_formatted}
User Request: {current_query}

You are an expert technical remediation specialist assisting an engineer.
Analyze the attached report/document alongside the internal runbooks and conversation history provided above.
Strictly structure your response into the following clear sections with these exact Markdown headers:

### 1. 📌 Issue Diagnosis & Summary
Briefly summarize the key findings, error codes, or vulnerabilities identified in the attached report.

### 2. 📚 Internal Runbook Guidance (Verified from Company Documentation)
Provide the explicit operational steps, configuration parameters, and policies directly supported by the internal context above. Every statement or step in this section MUST be grounded in the internal documentation. Inline cite the internal source URLs like `[Source: URL]`.

### 3. 🌐 General Industry Best Practices (General LLM Knowledge)
Provide complementary technical explanations, industry standards, or architectural advice that are NOT explicitly mentioned in the internal documentation. Clearly mark them as general knowledge.

### 4. ✅ Verification & Prevention
Explain how to verify the resolution succeeded and prevent recurrence.

At the very end of your response, record any missing internal documentation strictly inside these tags:
<!-- DOCUMENTATION_GAPS -->
Explain what runbooks, policies, or topics are missing from the internal knowledge base for this request (or write NONE if fully covered).
<!-- END_DOCUMENTATION_GAPS -->"""

            return [
                genai_types.Part.from_bytes(data=attached_file_bytes, mime_type=attached_mime_type),
                instruction,
            ]
        else:
            prompt = f"""Internal Knowledge Base Context:
{context_block}
{history_formatted}
Question: {current_query}

Answer the question clearly and distinguish between internal documentation and general knowledge using these exact headers:

### 📚 Internal Documentation Guidance (Verified from Company Docs)
Directly answer based on the internal context provided above. Only include facts that exist in the internal context, and inline cite the source URLs like `[Source: URL]`.

### 🌐 General Knowledge & Context (General LLM Knowledge)
Provide helpful supplementary technical explanation, industry background, or standard best practices if relevant, clearly marked as general knowledge.

At the very end of your response, record any missing internal documentation strictly inside these tags:
<!-- DOCUMENTATION_GAPS -->
Explain what runbooks, policies, or topics are missing from the internal knowledge base for this question (or write NONE if fully covered).
<!-- END_DOCUMENTATION_GAPS -->"""
            return prompt

