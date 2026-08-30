import pytest
from src.utils.config import _resolve_env_vars

def test_resolve_env_vars(monkeypatch):
    """Verify that environment variable interpolation in config strings works correctly."""
    # Set dynamic variable
    monkeypatch.setenv("TEST_GCP_PROJECT", "test-project-123")
    
    # 1. Simple replacement
    res = _resolve_env_vars("${TEST_GCP_PROJECT}")
    assert res == "test-project-123"

    # 2. Replacement with default (env var exists)
    res_def_exists = _resolve_env_vars("${TEST_GCP_PROJECT:fallback-project}")
    assert res_def_exists == "test-project-123"

    # 3. Replacement with default (env var absent)
    monkeypatch.delenv("TEST_GCP_PROJECT", raising=False)
    res_def_absent = _resolve_env_vars("${TEST_GCP_PROJECT:fallback-project}")
    assert res_def_absent == "fallback-project"

    # 4. Empty fallback default value
    res_def_empty = _resolve_env_vars("${TEST_GCP_PROJECT:}")
    assert res_def_empty == ""

    # 5. Nested structure resolution (dictionary traversal)
    nested_dict = {
        "gcp": {
            "project": "${TEST_GCP_PROJECT:fallback-nested}",
            "location": "europe-west3"
        },
        "database": {
            "port": 5432
        }
    }
    resolved = _resolve_env_vars(nested_dict)
    assert resolved["gcp"]["project"] == "fallback-nested"
    assert resolved["gcp"]["location"] == "europe-west3"
    assert resolved["database"]["port"] == 5432


def test_settings_store_cache_and_dynamic_config():
    """Verify settings_store in-memory caching, dynamic setting retrieval, and cache invalidation."""
    from unittest.mock import MagicMock, patch
    from src.db import settings_store
    from src.utils.config import config

    settings_store.clear_settings_cache()

    with patch("src.db.settings_store.get_connection") as mock_get_conn:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        # Test set_many_settings updates cache
        settings_store.set_many_settings({
            "pipeline.prompt_preset": "Strict Grounding Only",
            "pipeline.prompt_template": "Strict Template Content",
        })

        # get_setting should return from cache immediately
        val_preset = settings_store.get_setting("pipeline.prompt_preset")
        assert val_preset == "Strict Grounding Only"

        val_tpl = config.get_dynamic("pipeline.prompt_template")
        assert val_tpl == "Strict Template Content"

        # delete_setting updates cache
        settings_store.delete_setting("pipeline.prompt_preset")
        assert "pipeline.prompt_preset" not in settings_store._CACHE

        # clear cache
        settings_store.clear_settings_cache()
        assert len(settings_store._CACHE) == 0

