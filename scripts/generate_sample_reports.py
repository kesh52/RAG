"""Script to generate realistic sample incident and diagnostic reports (PDF) for testing RAG remediation workflows."""

import os
import sys

REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "sample_reports",
)


def create_pdf(filepath: str, lines: list[str]):
    """Creates a multi-line valid PDF document."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    stream_parts = ["BT", "/F1 12 Tf", "72 750 Td", "16 TL"]
    for i, line in enumerate(lines):
        # Escape parenthesis in PDF text
        safe_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if i == 0:
            stream_parts.append(f"/F1 16 Tf ({safe_line}) Tj /F1 12 Tf")
        else:
            stream_parts.append(f"T* ({safe_line}) Tj")
    stream_parts.append("ET")
    stream_content = "\n".join(stream_parts) + "\n"

    pdf_data = (
        "%PDF-1.4\n"
        "1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
        "2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
        "3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources <</Font <</F1 <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>>>>> >> endobj\n"
        f"4 0 obj <</Length {len(stream_content)}>> stream\n"
        f"{stream_content}"
        "endstream\nendobj\n"
        "xref\n0 5\n0000000000 65535 f\n"
        "0000000009 00000 n\n"
        "0000000056 00000 n\n"
        "0000000111 00000 n\n"
        "0000000256 00000 n\n"
        "trailer <</Size 5 /Root 1 0 R>>\n"
        "startxref\n350\n%%EOF\n"
    )

    with open(filepath, "wb") as f:
        f.write(pdf_data.encode("ascii", errors="ignore"))
    print(f"Generated sample report: {filepath}")


def generate_all_reports():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Cloud SQL Security & Connection Incident Report
    create_pdf(
        os.path.join(REPORTS_DIR, "incident_report_cloud_sql_auth_failure.pdf"),
        [
            "INCIDENT REPORT: SEC-2026-088 - Database Connection Failure",
            "============================================================",
            "Date: 2026-08-29 | Severity: CRITICAL | Status: UNRESOLVED",
            "Affected Component: Production Database Connector (PostgreSQL)",
            "",
            "INCIDENT SUMMARY:",
            "Production backend services failed to establish connection to Cloud SQL.",
            "Error: Connection refused / timeout on port 5432 over public IP.",
            "",
            "SECURITY AUDIT FINDINGS:",
            "- Service attempted direct TCP connection over public IP without encryption.",
            "- Public IP whitelisting was requested, violating internal GCP security policies.",
            "- Database password exposed in plain text configuration file.",
            "",
            "REQUIRED REMEDIATION:",
            "How do we configure secure connections to Cloud SQL without public IP",
            "whitelisting according to internal architecture standards?",
        ],
    )

    # 2. pgvector Index Degradation Report
    create_pdf(
        os.path.join(REPORTS_DIR, "diagnostic_report_pgvector_index_degradation.pdf"),
        [
            "DIAGNOSTIC REPORT: PERF-2026-104 - Vector Search Latency Spike",
            "============================================================",
            "Date: 2026-08-29 | Severity: HIGH | Status: INVESTIGATING",
            "Affected Component: PostgreSQL pgvector documents table",
            "",
            "PERFORMANCE ISSUE:",
            "Vector cosine similarity query latency spiked from 15ms to 450ms.",
            "Retrieval recall dropped below 60% during high-load vector search.",
            "",
            "DATABASE DIAGNOSTICS:",
            "- Current index is IVFFlat with default lists configuration.",
            "- Table grew to over 50,000 document embedding vectors.",
            "- Full table sequential scans observed during candidate retrieval.",
            "",
            "REQUIRED REMEDIATION:",
            "What index architecture and tradeoffs should we evaluate in pgvector",
            "(HNSW vs IVFFlat) to restore sub-20ms search latency and high recall?",
        ],
    )

    # 3. Spring Batch Chunk Ingestion Failure Report
    create_pdf(
        os.path.join(REPORTS_DIR, "incident_report_spring_batch_oom.pdf"),
        [
            "INCIDENT REPORT: BATCH-2026-042 - Ingestion Job Memory Failure",
            "============================================================",
            "Date: 2026-08-29 | Severity: MEDIUM | Status: FAILED",
            "Affected Component: ETL Confluence Ingestion Pipeline",
            "",
            "INCIDENT SUMMARY:",
            "Spring Batch ingestion job crashed with java.lang.OutOfMemoryError.",
            "Transaction aborted and all parsed chunk records were rolled back.",
            "",
            "ROOT CAUSE ANALYSIS:",
            "- Ingestion step attempted to load all 10,000 items in a single tasklet.",
            "- ItemReader did not use chunk-oriented processing boundaries.",
            "- ThreadPoolTaskExecutor was unconfigured, exhausting heap memory.",
            "",
            "REQUIRED REMEDIATION:",
            "How does Spring Batch manage chunk-based processing across Job, Step,",
            "ItemReader, ItemProcessor, and ItemWriter within transactions?",
        ],
    )

    print("\nAll sample test reports successfully generated in assets/sample_reports/!")


if __name__ == "__main__":
    generate_all_reports()

