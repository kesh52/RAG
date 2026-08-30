# INCIDENT REPORT: FLASH SALE INVENTORY LOCK DEGRADATION

**Incident ID:** INC-20260830-INV-02  
**Date & Time:** 2026-08-30 11:45:00 UTC  
**Impacted Service:** Inventory Reservation & Order Lock Engine  
**Severity:** HIGH (P2)  
**Reporter:** Fulfillment SRE Team  

---

## 1. Description
During the launch of limited-edition SKU `SKU-PRO-4090-X`, thousands of concurrent checkout requests triggered high lock contention. 340 customer checkouts failed with HTTP 504 errors stating "Item currently held by another buyer".

---

## 2. Telemetry & Metrics
* **Error Identifier:** `ERR_STOCK_LOCK_TIMEOUT_504`
* **Redis Cluster Metrics:**
  * Active key count: 18,400 keys on `lock:sku:SKU-PRO-4090-X`
  * Redlock acquisition latency p99: 4,820ms (Timeout threshold: 3000ms)
* **Log Sample:**
```
[2026-08-30 11:44:18] [ERROR] [inventory_lock.py:88] Failed to acquire Redis Redlock on key 'lock:sku:SKU-PRO-4090-X' after 3 retries. Error: ERR_STOCK_LOCK_TIMEOUT_504. Aborting transaction for order ORD-119280.
[2026-08-30 11:44:20] [WARN] [inventory_lock.py:102] Detected 84 orphaned lock keys with no active worker heartbeat.
```

---

## 3. Immediate Action Required
Need the exact commands from Confluence documentation to:
1. Inspect lock contention in Redis.
2. Safely purge orphaned keys without affecting active transactions.
3. Configure dynamic backoff jitter in the Kubernetes deployment.

