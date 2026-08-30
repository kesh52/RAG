# Sample Domain: E-Commerce Payments & Checkout Platform

This folder contains a complete, turnkey sample domain specifically designed for testing **Confluence document ingestion**, **incident/diagnostic report evaluation**, and **RAG-based remediation recommendations**.

---

## 📁 Directory Structure

```text
sample_domain_ecommerce_payments/
├── confluence_pages/
│   ├── 01_payment_gateway_runbook.html             # Payment intent token expiry, error catalog, CLI tools
│   ├── 02_inventory_lock_order_service.html        # Redis Redlock, Postgres 55P03, orphaned lock cleanup
│   └── 03_sop_stuck_transaction_reconciliation.html # Daily settlement audit, lost webhooks, auto-healing
├── sample_reports/
│   ├── incident_report_payment_auth_expiry.md      # P1 incident: 78 orders failed with ERR_PAYMENT_304
│   ├── incident_report_redis_lock_timeout.md       # P2 incident: Flash sale stock lock timeout
│   └── daily_reconciliation_discrepancy_report.md  # Audit discrepancy: $4,850 variance from dropped webhook
├── test_questions_and_remediations.md              # Ready-to-use prompt queries & ground truth answers
├── publish_all.py                                  # One-click script to publish all pages to Confluence
└── README.md
```

---

## 🚀 How to Use

### 1. Publish Pages to Confluence
Ensure your `.env` contains your Confluence credentials (`CONFLUENCE_DOMAIN`, `CONFLUENCE_USERNAME`, `CONFLUENCE_API_TOKEN`), then run:
```bash
python sample_domain_ecommerce_payments/publish_all.py --space-key DEV
```

### 2. Ingest the Confluence Space into your RAG Pipeline
Run your pipeline ingestion to crawl and embed the new Confluence space into pgvector:
```bash
python run_api.py   # Or run your orchestrator crawl trigger
```

### 3. Ask Questions in the Chat UI
Open the Chat UI, attach/paste one of the reports from `sample_reports/`, and ask remediation questions.

See [`test_questions_and_remediations.md`](./test_questions_and_remediations.md) for pre-written prompts and expected answers.

