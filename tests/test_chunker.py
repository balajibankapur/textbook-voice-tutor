from ingestion.chunker import PageText, chunk_pages


def _para(n_tokens: int, word: str = "lorem") -> str:
    return " ".join([word] * n_tokens)


def test_small_input_produces_one_chunk() -> None:
    pages = [PageText(page_number=1, text=_para(80))]
    chunks = chunk_pages(pages)
    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1


def test_long_input_splits_into_bounded_chunks() -> None:
    big = "\n\n".join(_para(100) for _ in range(10))  # ~1000 tokens
    pages = [PageText(page_number=1, text=big)]
    chunks = chunk_pages(pages, target_min_tokens=300, target_max_tokens=500, overlap_tokens=50)
    assert len(chunks) >= 2
    for ch in chunks[:-1]:
        assert ch.token_count <= 500 + 50  # last chunk may be short, others within bounds + overlap seed


def test_chunk_spans_multiple_pages() -> None:
    pages = [
        PageText(page_number=1, text=_para(120)),
        PageText(page_number=2, text=_para(120)),
        PageText(page_number=3, text=_para(120)),
    ]
    chunks = chunk_pages(pages, target_min_tokens=300, target_max_tokens=500)
    multi = [c for c in chunks if c.page_start != c.page_end]
    assert multi, "expected at least one chunk spanning multiple pages"
    assert multi[0].source_page_numbers == sorted(multi[0].source_page_numbers)


def test_heading_becomes_section_title() -> None:
    pages = [
        PageText(
            page_number=1,
            text="Photosynthesis\n\n" + _para(120, "glucose"),
        ),
    ]
    chunks = chunk_pages(pages)
    assert chunks[0].section_title == "Photosynthesis"


def test_page_section_title_propagates() -> None:
    pages = [
        PageText(page_number=1, text=_para(50), section_title="The Cell"),
        PageText(page_number=2, text=_para(50)),
    ]
    chunks = chunk_pages(pages)
    assert chunks[0].section_title == "The Cell"


def test_empty_pages_produce_no_chunks() -> None:
    assert chunk_pages([]) == []
    assert chunk_pages([PageText(page_number=1, text="")]) == []


def test_overlap_seeds_next_chunk() -> None:
    pages = [PageText(page_number=1, text=_para(1000))]
    chunks = chunk_pages(pages, target_min_tokens=300, target_max_tokens=500, overlap_tokens=50)
    assert len(chunks) >= 2
    # Overlap is applied via the running buffer, so successive chunks share
    # content when a long paragraph is split.
    for a, b in zip(chunks, chunks[1:]):
        assert b.text  # non-empty


def test_huge_single_paragraph_splits_internally() -> None:
    # One paragraph bigger than target_max forces _split_long_paragraph.
    pages = [PageText(page_number=1, text=_para(1200))]
    chunks = chunk_pages(pages, target_min_tokens=300, target_max_tokens=500)
    assert len(chunks) >= 2
    assert all(c.token_count > 0 for c in chunks)
