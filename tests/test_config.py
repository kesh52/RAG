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

