# Test Questions & Expected RAG Remediations

Use these questions in the Chat UI to test how effectively the RAG engine correlates the sample incident reports against your Confluence documentation.

---

### Scenario 1: Payment Intent Authorization Expiry
* **Attached Report:** `sample_reports/incident_report_payment_auth_expiry.md`
* **Test Prompt:**
  > *"Based on the attached incident report `INC-20260830-PAY-01`, why did the 78 orders fail, what is the root cause from our Confluence runbooks, and what are the exact steps and CLI commands to resolve the issue without double-charging customers?"*
* **Expected RAG Response:**
  1. **Root Cause:** Error `ERR_PAYMENT_304_AUTH_EXPIRED` occurred because worker queue latency (940s) exceeded the 15-minute (900s) TTL of the Payment Intent token before the capture phase could complete.
  2. **Warning Highlight:** Explains that orders must NOT be automatically re-authorized to avoid double charges.
  3. **Actionable Steps & Commands:**
     * Run intent status verification: `python -m services.payments.cli verify-intent --intent-id <INTENT_ID>`
     * Release stuck inventory: `curl -X POST https://api.internal.shop/v1/inventory/unreserve -d '{"order_id": "<ORDER_ID>"}'`
     * Send email notification using template `TPL_PAYMENT_SESSION_EXPIRED`.
     * Scale worker pods: `kubectl scale deployment/payment-worker --replicas=8 -n payments`

---

### Scenario 2: Flash Sale Redis Lock Contention
* **Attached Report:** `sample_reports/incident_report_redis_lock_timeout.md`
* **Test Prompt:**
  > *"We experienced an outage during our flash sale with error `ERR_STOCK_LOCK_TIMEOUT_504`. How do we inspect lock contention in Redis, purge orphaned locks, and tune the deployment according to our documentation?"*
* **Expected RAG Response:**
  1. **Root Cause:** `ERR_STOCK_LOCK_TIMEOUT_504` triggered because worker pods failed to acquire the Redis Redlock within 3000ms due to key contention on `lock:sku:SKU-PRO-4090-X`.
  2. **Inspection Command:**
     `redis-cli -h redis-cluster.internal -p 6379 eval "return #redis.call('keys', 'lock:sku:*')" 0`
  3. **Purge Orphaned Locks:**
     `python -m services.inventory.cli purge-orphaned-locks --older-than-seconds 30`
  4. **Dynamic Jitter Tuning:**
     `kubectl set env deployment/inventory-service REDIS_LOCK_RETRY_JITTER_MS=250 -n fulfillment`

---

### Scenario 3: Daily Financial Reconciliation Mismatch
* **Attached Report:** `sample_reports/daily_reconciliation_discrepancy_report.md`
* **Test Prompt:**
  > *"Review audit report `AUD-20260830-REC-01`. 3 transactions have a `STATUS_DESYNC` totaling $4,850. What SOP governs this, what caused it, and what are the exact commands to auto-heal the ledger and generate invoices?"*
* **Expected RAG Response:**
  1. **Governing SOP:** `SOP-FIN-2026-08 (Financial Reconciliation & Settlement Replay)`.
  2. **Cause:** Dropped webhook caused internal orders to stay in `PENDING_CAPTURE` despite successful Stripe charge.
  3. **Remediation Commands:**
     * Dry run check: `python -m services.reconciliation.sync_tool --mode dry-run --date 2026-08-30`
     * Live sync: `python -m services.reconciliation.sync_tool --mode live --date 2026-08-30 --confirm-ledger-patch`
     * Batch invoice trigger: `curl -X POST https://api.internal.shop/v1/invoices/batch-generate -H "Authorization: Bearer $FIN_SERVICE_TOKEN" -d '{"date": "2026-08-30"}'`

