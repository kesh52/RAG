"""Publish Architectural Specification to Atlassian Confluence via REST API.

Usage:
    python scripts/publish_to_confluence.py --space-key DEV --parent-id 123456
"""

import os
import sys
import argparse
import logging
import requests

# Ensure root is on import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("publish_confluence")


def publish_page(space_key: str, parent_id: str | None, title: str, xhtml_path: str):
    domain = os.getenv("CONFLUENCE_DOMAIN", "").strip().replace("https://", "").replace("http://", "")
    username = os.getenv("CONFLUENCE_USERNAME", "").strip()
    api_token = os.getenv("CONFLUENCE_API_TOKEN", "").strip()

    if not domain or not username or not api_token:
        logger.error("Missing CONFLUENCE_DOMAIN, CONFLUENCE_USERNAME, or CONFLUENCE_API_TOKEN environment variables.")
        sys.exit(1)

    if not os.path.exists(xhtml_path):
        logger.error(f"XHTML source file not found at: {xhtml_path}")
        sys.exit(1)

    with open(xhtml_path, "r", encoding="utf-8") as f:
        content_xhtml = f.read()

    base_url = f"https://{domain}"
    api_url = f"{base_url}/wiki/rest/api/content"
    auth = (username, api_token)

    # 1. Check if page with this title already exists in the space
    logger.info(f"Checking if page '{title}' exists in space '{space_key}'...")
    check_url = f"{api_url}?title={requests.utils.quote(title)}&spaceKey={space_key}&expand=version"
    resp = requests.get(check_url, auth=auth, timeout=15)
    resp.raise_for_status()
    results = resp.json().get("results", [])

    if results:
        existing_page = results[0]
        page_id = existing_page["id"]
        version_num = existing_page["version"]["number"] + 1
        logger.info(f"Page found (ID: {page_id}). Updating to version {version_num}...")

        payload = {
            "id": page_id,
            "type": "page",
            "title": title,
            "space": {"key": space_key},
            "body": {
                "storage": {
                    "value": content_xhtml,
                    "representation": "storage"
                }
            },
            "version": {"number": version_num}
        }
        update_url = f"{api_url}/{page_id}"
        put_res = requests.put(update_url, json=payload, auth=auth, timeout=20)
        put_res.raise_for_status()
        page_link = f"{base_url}/wiki/spaces/{space_key}/pages/{page_id}"
        logger.info(f"✅ Successfully updated Confluence page: {page_link}")
    else:
        logger.info(f"Page does not exist. Creating new page '{title}' in space '{space_key}'...")
        payload = {
            "type": "page",
            "title": title,
            "space": {"key": space_key},
            "body": {
                "storage": {
                    "value": content_xhtml,
                    "representation": "storage"
                }
            }
        }
        if parent_id:
            payload["ancestors"] = [{"id": parent_id}]

        post_res = requests.post(api_url, json=payload, auth=auth, timeout=20)
        post_res.raise_for_status()
        new_page = post_res.json()
        new_id = new_page.get("id")
        page_link = f"{base_url}/wiki/spaces/{space_key}/pages/{new_id}"
        logger.info(f"✅ Successfully created new Confluence page: {page_link}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish specification document to Confluence.")
    parser.add_argument("--space-key", "-s", type=str, default="DEV", help="Target Confluence space key (default: DEV).")
    parser.add_argument("--parent-id", "-p", type=str, default=None, help="Optional parent page ID to nest the page under.")
    parser.add_argument("--title", "-t", type=str, default="RAG Platform: Implemented Requirements & Architectural Specification", help="Page title.")
    parser.add_argument("--file", "-f", type=str, default="docs/confluence_storage_format.xhtml", help="Path to Confluence storage format XHTML.")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_file = os.path.join(root_dir, args.file) if not os.path.isabs(args.file) else args.file

    publish_page(space_key=args.space_key, parent_id=args.parent_id, title=args.title, xhtml_path=target_file)

