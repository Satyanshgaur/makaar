import json
from pathlib import Path

import jsonschema

from makaar_eval.metrics import (edit_distance, flag_quality, has_loop,
                                 normalized_edit_distance, reading_order_distance)

ROOT = Path(__file__).resolve().parents[2]


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3


def test_ned():
    assert normalized_edit_distance("a  b", "a b") == 0.0
    assert normalized_edit_distance("", "") == 0.0
    assert normalized_edit_distance("abcd", "abce") == 0.25


def test_reading_order():
    assert reading_order_distance(["a", "b", "c"], ["a", "c", "b"]) == 2 / 3


def test_flag_quality():
    q = flag_quality([(True, True), (True, False), (False, False), (False, True)])
    assert q == {"wrong_flagged": 0.5, "flag_rate": 0.5}


def test_loop():
    assert has_loop("the cat " * 20)
    assert not has_loop("a normal sentence with no repetition at all")


def test_schema_accepts_sample():
    schema = json.loads((ROOT / "schema/document.schema.json").read_text())
    doc = {"version": 1, "id": "j1", "pages": [{"index": 0, "width": 100, "height": 100, "blocks": [
        {"id": "b0", "type": "formula", "bbox": [0, 0, 10, 10], "order": 0, "text": "x^2", "confidence": 0.9}]}]}
    jsonschema.validate(doc, schema)


def test_schema_rejects_bad_block():
    schema = json.loads((ROOT / "schema/document.schema.json").read_text())
    doc = {"version": 1, "id": "j1", "pages": [{"index": 0, "width": 1, "height": 1, "blocks": [
        {"id": "b0", "type": "bogus", "bbox": [0, 0, 1, 1], "order": 0}]}]}
    try:
        jsonschema.validate(doc, schema)
    except jsonschema.ValidationError:
        return
    raise AssertionError("expected rejection")
