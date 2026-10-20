"""Shared bounded tokenizer admission, independent of model execution semantics.

Model vocabulary can include rows without tokenizer entries. Admission never
adds entries, reindexes IDs, or removes model rows. Callers retain the original
tokenizer bytes and use the ordinary TokenizersJSONTokenizer for text decoding.
"""
from __future__ import annotations

from contextlib import contextmanager


DECODER_METADATA_MEMORY_BYTES = 384 << 20


@contextmanager
def reserve_decoder_metadata():
    """Shared source-graph, full-vocabulary JSON and reverse-index workspace.

    Charge before source inspection/parsing. Retained source bytes have their
    ordinary caller ownership; this allowance covers transient metadata work.
    """
    from pllm.deployment.resources import reserve_incremental_memory
    from .delegated_protocol import clear_failure_frames

    with reserve_incremental_memory(DECODER_METADATA_MEMORY_BYTES):
        try:
            yield
        except BaseException as error:
            clear_failure_frames(error)
            raise


def validate_document(document, *, vocabulary, max_entries, error, label):
    """Bound IDs and shape before the tokenizer allocates its reverse indexes."""
    if document.get("padding") is not None:
        raise error(f"{label} tokenizer padding is not admitted")
    if document.get("truncation") is not None:
        raise error(f"{label} tokenizer must not silently truncate source input")
    model = document.get("model")
    vocab = model.get("vocab") if isinstance(model, dict) else None
    added = document.get("added_tokens", [])
    if (not isinstance(vocab, (dict, list)) or not 1 <= len(vocab) <= max_entries
            or not isinstance(added, list) or len(added) > max_entries):
        raise error(f"{label} tokenizer vocabulary exceeds its complete-table limit")
    ids = vocab.values() if isinstance(vocab, dict) else range(len(vocab))
    if any(type(i) is not int or not 0 <= i < vocabulary for i in ids):
        raise error(f"{label} tokenizer IDs exceed the complete vocabulary")
    if any(type(row) is not dict or type(row.get("id")) is not int
           or not 0 <= row["id"] < vocabulary for row in added):
        raise error(f"{label} added tokenizer IDs exceed the complete vocabulary")


def validate_vocabulary(inner, *, vocabulary, allow_model_padding, error, label):
    """Require a contiguous, reversible original ID space within the model head."""
    complete = inner.get_vocab(with_added_tokens=True)
    expected_size = int(inner.get_vocab_size()) if allow_model_padding else vocabulary
    if (not 1 <= expected_size <= vocabulary or len(complete) != expected_size
            or set(complete.values()) != set(range(expected_size))
            or any(inner.id_to_token(i) != text for text, i in complete.items())):
        raise error(f"{label} tokenizer must cover the complete table/head vocabulary")
    return expected_size
