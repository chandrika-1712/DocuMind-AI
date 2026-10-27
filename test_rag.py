
from rag_engine import split_into_chunks


def test_chunking():
    text = " ".join(f"word{i}" for i in range(250))

    chunks = split_into_chunks(
        text,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) == 3
    assert len(chunks[0].split()) == 100
    assert len(chunks[1].split()) == 100


def test_short_document():
    chunks = split_into_chunks("This is a short document.")

    assert len(chunks) == 1
    assert "short document" in chunks[0]


def test_empty_document():
    assert split_into_chunks("") == []
