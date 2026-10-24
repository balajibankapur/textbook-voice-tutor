"""Smoke tests: project imports and settings load."""
from __future__ import annotations


def test_settings_load() -> None:
    from config.settings import load_settings

    s = load_settings()
    assert s.embedding_model
    assert s.top_k > 0
    assert s.max_retrieval_attempts > 0


def test_logging_configures_once() -> None:
    import logging

    from config.logging_config import configure_logging

    configure_logging()
    configure_logging()
    assert logging.getLogger().handlers, "root logger should have a handler"


def test_package_imports() -> None:
    import config  # noqa: F401
    import conversation  # noqa: F401
    import decision  # noqa: F401
    import evaluation  # noqa: F401
    import ingestion  # noqa: F401
    import llm  # noqa: F401
    import retrieval  # noqa: F401
    import storage  # noqa: F401
    import voice  # noqa: F401
