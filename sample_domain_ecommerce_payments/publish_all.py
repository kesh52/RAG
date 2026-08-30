"""Helper script to publish all sample domain pages to Atlassian Confluence."""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Ensure root is on import path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from scripts.publish_to_confluence import publish_page

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("publish_sample_domain")

PAGES = [
    {
        "title": "Payment Gateway Runbook & Error Catalog",
        "file": os.path.join(ROOT_DIR, "sample_domain_ecommerce_payments", "confluence_pages", "01_payment_gateway_runbook.html")
    },
    {
        "title": "Inventory Reservation & Lock Engine Guide",
        "file": os.path.join(ROOT_DIR, "sample_domain_ecommerce_payments", "confluence_pages", "02_inventory_lock_order_service.html")
    },
    {
        "title": "SOP: Financial Reconciliation & Settlement Replay",
        "file": os.path.join(ROOT_DIR, "sample_domain_ecommerce_payments", "confluence_pages", "03_sop_stuck_transaction_reconciliation.html")
    }
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish all sample e-commerce payment pages to Confluence.")
    parser.add_argument("--space-key", "-s", type=str, default="DEV", help="Target Confluence space key (default: DEV).")
    parser.add_argument("--parent-id", "-p", type=str, default=None, help="Optional parent page ID to nest the pages under.")
    args = parser.parse_args()

    logger.info(f"Publishing {len(PAGES)} sample domain pages to Confluence space '{args.space_key}'...")
    for idx, page in enumerate(PAGES, start=1):
        logger.info(f"[{idx}/{len(PAGES)}] Publishing '{page['title']}'...")
        publish_page(space_key=args.space_key, parent_id=args.parent_id, title=page["title"], xhtml_path=page["file"])

    logger.info("🎉 All sample domain pages successfully published to Confluence!")

