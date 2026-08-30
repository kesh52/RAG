"""Prompt templates and preset definitions for the RAG generation pipeline."""

from typing import Dict

# ---------------------------------------------------------------------------
# Default Prompt Templates
# ---------------------------------------------------------------------------

DEFAULT_STANDARD_PROMPT = """Internal Knowledge Base Context:
{context}

Question: {query}

Answer the question clearly and distinguish between internal documentation and general knowledge using these exact headers:

### 📚 Internal Documentation Guidance (Verified from Company Docs)
Directly answer based on the internal context provided above. Only include facts that exist in the internal context, and inline cite the source URLs like `[Source: URL]`.

### 🌐 General Knowledge & Context (General LLM Knowledge)
Provide helpful supplementary technical explanation, industry background, or standard best practices if relevant, clearly marked as general knowledge.

At the very end of your response, record any missing internal documentation strictly inside these tags:
<!-- DOCUMENTATION_GAPS -->
Explain what runbooks, policies, or topics are missing from the internal knowledge base for this question (or write NONE if fully covered).
<!-- END_DOCUMENTATION_GAPS -->"""


DEFAULT_ATTACHED_REPORT_PROMPT = """Internal Runbooks & Knowledge Base Context:
{context}

User Request: {query}

You are an expert technical remediation specialist.
Analyze the attached report/document alongside the internal runbooks provided above.
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


# ---------------------------------------------------------------------------
# Prompt Presets for Evaluation & Tuning
# ---------------------------------------------------------------------------

PROMPT_PRESETS: Dict[str, str] = {
    "Structured Multi-Section (Default)": DEFAULT_STANDARD_PROMPT,
    
    "Strict Grounding (High Faithfulness)": """Internal Knowledge Base Context:
{context}

Question: {query}

CRITICAL GROUNDING INSTRUCTIONS:
- Answer the question relying ONLY on the facts directly stated in the Internal Knowledge Base Context above.
- Do NOT assume, extrapolate, or bring in outside pre-trained knowledge.
- If the context does not contain enough information to answer the question completely, explicitly state: "The internal documentation does not contain sufficient information to answer this question."
- Inline cite every statement or configuration parameter with its source URL like `[Source: URL]`.

At the very end of your response, record any missing internal documentation strictly inside these tags:
<!-- DOCUMENTATION_GAPS -->
Explain what runbooks, policies, or topics are missing from the internal knowledge base for this question (or write NONE if fully covered).
<!-- END_DOCUMENTATION_GAPS -->""",

    "Concise Technical Q&A (Direct)": """Internal Knowledge Base Context:
{context}

Question: {query}

INSTRUCTIONS:
- Provide a direct, concise, bulleted answer based strictly on the internal context above.
- Skip pleasantries, introductions, and conversational filler.
- Inline cite source URLs as `[Source: URL]`.

<!-- DOCUMENTATION_GAPS -->
List missing documentation topics if any (or NONE).
<!-- END_DOCUMENTATION_GAPS -->""",

    "Detailed SOP Remediation (Operational)": """Internal Knowledge Base Context:
{context}

User Request / Incident: {query}

You are an expert Site Reliability and Operations Engineer.
Provide an actionable, step-by-step remediation guide based on the internal runbooks above:

### 1. Root Cause & Diagnostic Summary
Explain what triggered the issue and the diagnostic signals based on internal SOPs.

### 2. Step-by-Step Remediation Procedure
Provide explicit command lines, configuration blocks, and steps directly verified from internal docs with `[Source: URL]` citations.

### 3. Verification & Rollback
Specify how to confirm resolution and what to do if rollback is required.

<!-- DOCUMENTATION_GAPS -->
List missing runbook instructions if any (or NONE).
<!-- END_DOCUMENTATION_GAPS -->""",
}


def get_prompt_preset(name: str) -> str:
    """Retrieve template text for a named preset, falling back to default."""
    return PROMPT_PRESETS.get(name, DEFAULT_STANDARD_PROMPT)


def list_prompt_presets() -> list[str]:
    """Return available preset names."""
    return list(PROMPT_PRESETS.keys())


def format_prompt(template: str, context: str, query: str) -> str:
    """Format a template with context and query variables, with safe fallback."""
    if not template or not template.strip():
        template = DEFAULT_STANDARD_PROMPT

    # If template contains standard format specifiers
    if "{context}" in template and "{query}" in template:
        try:
            return template.format(context=context, query=query)
        except Exception:
            pass

    # Safe fallback if format fails or user wrote raw template without brackets
    return (
        f"Internal Context:\n{context}\n\n"
        f"User Query: {query}\n\n"
        f"{template}"
    )

