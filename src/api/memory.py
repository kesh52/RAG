import re
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

    def generate_chat_title(self, query: str, response: str = "") -> str:
        """Generate a short (3-5 words) descriptive title for a chat session based on user input."""
        prompt = f"""You are naming a technical chat conversation based on the user's inquiry.
Create a short, specific title (strictly 3 to 5 words).
- Focus on the core topic, service name, error code, or task described in the user request.
- Do NOT use quotes, quotation marks, periods, prefixes like 'Title:', or generic words like 'Question' or 'Help'.
- Use Title Case.

User Request: {query[:300]}
Assistant Context: {response[:200] if response else ''}

Title:"""

        try:
            res = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=25,
                ),
            )
            raw = res.text.strip()
            # Clean prefixes, quotes, markdown, and trailing punctuation
            cleaned = re.sub(r"^(Title\s*:\s*|#+\s*|\*+\s*)", "", raw, flags=re.IGNORECASE).strip()
            cleaned = cleaned.strip('"\'`*').rstrip('.').strip()
            if cleaned and 2 <= len(cleaned.split()) <= 8 and len(cleaned) <= 60:
                return cleaned
        except Exception as e:
            logger.warning(f"Error generating chat title: {e}")

        # Fallback to cleaned first 4-5 words of user query
        words = [w for w in re.sub(r"[^\w\s]", "", query).split() if len(w) > 1]
        fallback = " ".join(words[:4]).title() if words else "Technical Discussion"
        return fallback if fallback else "Technical Discussion"

    def format_generation_contents(
        self,
        history: list[dict[str, Any]],
        current_query: str,
        retrieved_contexts: list[dict[str, Any]],
        attached_file_bytes: bytes | None = None,
        attached_mime_type: str | None = None,
        prompt_template: str | None = None,
    ) -> list[Any] | str:
        """Construct multimodal prompt contents including retrieved chunks, past turns, and current query."""
        from src.utils.config import config
        from src.pipeline.prompts import format_prompt, DEFAULT_STANDARD_PROMPT, DEFAULT_ATTACHED_REPORT_PROMPT

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
            history_formatted = "\n\nRecent Conversation History:\n" + "\n\n".join(history_lines) + "\n"

        full_context = context_block + (f"\n{history_formatted}" if history_formatted else "")

        # 3. Assemble prompt instruction based on template or dynamic default
        if attached_file_bytes and attached_mime_type and attached_mime_type != "application/octet-stream":
            tpl = prompt_template
            if not tpl:
                dynamic_att = config.get_dynamic("pipeline.attached_prompt_template")
                tpl = dynamic_att if dynamic_att else DEFAULT_ATTACHED_REPORT_PROMPT
            
            instruction = format_prompt(tpl, full_context, current_query)
            return [
                genai_types.Part.from_bytes(data=attached_file_bytes, mime_type=attached_mime_type),
                instruction,
            ]
        else:
            tpl = prompt_template
            if not tpl:
                dynamic_tpl = config.get_dynamic("pipeline.prompt_template")
                tpl = dynamic_tpl if dynamic_tpl else DEFAULT_STANDARD_PROMPT
            
            return format_prompt(tpl, full_context, current_query)


