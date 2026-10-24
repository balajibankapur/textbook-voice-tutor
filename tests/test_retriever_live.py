"""End-to-end retrieval against the real MiniLM model and a generated PDF.

Marked `live_model` so it runs via `pytest -m live_model`, apart from the
default suite that uses faiss (torch and faiss conflict in-process).
"""
from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

pytestmark = pytest.mark.live_model


_PAGE_TEMPLATE = "{heading}\n\n{body}"

_TOPIC_BODIES = {
    "photosynthesis": (
        "Photosynthesis is the process by which green plants and some other organisms "
        "use sunlight to synthesize foods with the help of chlorophyll pigments. "
        "In this reaction, carbon dioxide from the air enters the leaf through tiny "
        "pores called stomata, and water is drawn up from the roots through the stem. "
        "Chlorophyll inside the chloroplasts absorbs light energy and converts the "
        "water and carbon dioxide into glucose, a sugar that stores chemical energy. "
        "Oxygen is released as a by-product and returned to the atmosphere. "
        "The glucose produced is transported through the plant and used for growth, "
        "stored as starch in roots and seeds, or combined with other molecules to "
        "build cellulose for cell walls. Without photosynthesis there would be no "
        "food chain on land, because almost all living things depend directly or "
        "indirectly on plants for the energy they contain. The overall equation can "
        "be written as six molecules of carbon dioxide plus six molecules of water "
        "yielding one molecule of glucose and six molecules of oxygen. Factors such "
        "as the intensity of light, the temperature of the surroundings, and the "
        "concentration of carbon dioxide all influence how quickly photosynthesis "
        "proceeds in a given plant on a given day."
    ),
    "newton": (
        "Isaac Newton described three fundamental laws of motion that explain how "
        "forces affect the movement of objects. The first law, called the law of "
        "inertia, states that an object at rest will stay at rest and an object in "
        "motion will continue moving in a straight line at a constant speed unless "
        "an unbalanced external force acts upon it. The second law relates the net "
        "force on an object to the acceleration it experiences and to its mass; the "
        "greater the force, the greater the acceleration, and the greater the mass "
        "the smaller the acceleration for the same force. The third law states that "
        "for every action there is an equal and opposite reaction, which is why a "
        "swimmer moves forward by pushing water backward and why a rocket rises by "
        "pushing hot gases downward. Together these three laws of motion form the "
        "foundation of classical mechanics and allow engineers and scientists to "
        "predict how cars, planes, planets, and satellites behave when forces are "
        "applied to them. Newton published these ideas in his Principia Mathematica "
        "in 1687 and they remained the central description of motion until Einstein "
        "extended them with the theory of relativity."
    ),
    "water_cycle": (
        "The water cycle is the continuous movement of water between the oceans, "
        "the atmosphere, and the land. The sun's heat causes water to evaporate "
        "from oceans, lakes, rivers, and even from the leaves of plants in a "
        "process called transpiration. As the warm water vapor rises into the sky "
        "it cools, condenses around tiny particles of dust and salt, and forms "
        "clouds made of droplets or ice crystals. When the droplets in a cloud grow "
        "large and heavy enough they fall to the Earth as precipitation in the "
        "form of rain, snow, sleet, or hail. Some of this precipitation flows "
        "across the surface of the land as runoff, filling rivers and lakes and "
        "eventually returning to the oceans. Another portion soaks into the ground "
        "and becomes groundwater, slowly moving through soil and rock before "
        "emerging again from springs or being drawn up by the roots of plants. The "
        "water cycle has no beginning and no end; the same water molecules move "
        "through its stages again and again, which is why clean fresh water is a "
        "precious resource that must be used carefully."
    ),
}


@pytest.fixture
def topical_pdf(tmp_path: Path) -> Path:
    """A 3-page PDF with three distinct topics, each big enough to form its own chunk."""
    doc = pymupdf.open()
    for heading, body in [
        ("Photosynthesis", _TOPIC_BODIES["photosynthesis"]),
        ("Newton's Laws of Motion", _TOPIC_BODIES["newton"]),
        ("The Water Cycle", _TOPIC_BODIES["water_cycle"]),
    ]:
        page = doc.new_page()
        rect = pymupdf.Rect(72, 72, 540, 740)
        page.insert_textbox(
            rect,
            _PAGE_TEMPLATE.format(heading=heading, body=body),
            fontsize=11,
        )
    out = tmp_path / "topical.pdf"
    doc.save(out)
    doc.close()
    return out


def test_live_retrieval_finds_topic_match(topical_pdf: Path, tmp_path: Path) -> None:
    from ingestion.ingest_chapter import ingest_chapter
    from retrieval.embedding_service import EmbeddingService
    from retrieval.retriever import Retriever
    from storage.metadata_repository import MetadataRepository

    repo = MetadataRepository(tmp_path / "tutor.db")
    service = EmbeddingService()
    index_dir = tmp_path / "indexes"

    ingest_chapter(
        pdf_path=topical_pdf,
        chapter_id="science",
        title="Science Mix",
        subject="General Science",
        repo=repo,
        embedding_service=service,
        vector_store_dir=index_dir,
    )

    retriever = Retriever(
        repo=repo, embedding_service=service, vector_store_dir=index_dir
    )

    cases = [
        ("how do plants make food?", 1, "photosynthesis"),
        ("what did Newton say about motion?", 2, "newton"),
        ("how does rain form from the ocean?", 3, "water"),
    ]
    for question, expected_page, keyword in cases:
        hits = retriever.search("science", question, top_k=3)
        assert hits, f"no hits for: {question}"
        # A chunk spans [page_start, page_end]; the right topic can live in a
        # chunk whose overlap-seeded portion falls on an adjacent page.
        covering = [h for h in hits if h.page_start <= expected_page <= h.page_end]
        assert covering, (
            f"{question!r} did not surface page {expected_page}; "
            f"got ranges {[(h.page_start, h.page_end) for h in hits]}"
        )
        top = hits[0]
        assert top.page_start <= expected_page <= top.page_end, (
            f"{question!r}: expected top hit to cover page {expected_page}, "
            f"got pages {top.page_start}-{top.page_end}"
        )
        assert keyword in top.chunk_text.lower(), (
            f"{question!r}: expected top chunk to mention {keyword!r}"
        )

    repo.close()
