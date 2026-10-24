from ingestion.text_normalizer import is_text_usable, normalize


def test_normalize_dehyphenates_linebreaks() -> None:
    assert normalize("photo-\nsynthesis is useful") == "photosynthesis is useful"


def test_normalize_collapses_inline_whitespace() -> None:
    assert normalize("hello    world") == "hello world"


def test_normalize_preserves_paragraph_breaks() -> None:
    out = normalize("para one.\n\npara two.")
    assert "\n\n" in out
    assert "para one." in out and "para two." in out


def test_normalize_collapses_many_newlines() -> None:
    out = normalize("a\n\n\n\n\nb")
    assert out == "a\n\nb"


def test_normalize_strips_control_chars() -> None:
    assert normalize("hello\x01world") == "helloworld"


def test_is_text_usable_true_for_real_text() -> None:
    page = "This is a sentence with plenty of readable alphabetic content on it."
    assert is_text_usable(page)


def test_is_text_usable_false_for_sparse_text() -> None:
    assert not is_text_usable("3", min_alpha_chars=50)
    assert not is_text_usable("", min_alpha_chars=50)
    assert not is_text_usable("Page 1 of 20", min_alpha_chars=50)
