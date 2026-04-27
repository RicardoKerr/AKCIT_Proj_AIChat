from ingestion.loader import chunk_text


def test_chunk_text_generates_overlap() -> None:
    text = " ".join([f"token{i}" for i in range(1, 1001)])

    chunks = chunk_text(text=text, chunk_size=200, chunk_overlap=50, metadata={"source": "sample"})

    assert len(chunks) > 1
    assert chunks[0]["metadata"]["source"] == "sample"
    assert chunks[0]["metadata"]["word_end"] == 200
    assert chunks[1]["metadata"]["word_start"] == 150


def test_chunk_text_rejects_invalid_overlap() -> None:
    try:
        chunk_text(text="a b c", chunk_size=100, chunk_overlap=100)
    except ValueError as exc:
        assert "chunk_overlap" in str(exc)
    else:
        raise AssertionError("Era esperado ValueError para overlap invalido")
