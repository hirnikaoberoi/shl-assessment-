import json
from functools import lru_cache
from typing import List, TypedDict

from app.config import CATALOG_PATH


class TestType(TypedDict):
    code: str
    name: str


class Assessment(TypedDict):
    entity_id: str
    name: str
    url: str
    description: str
    job_levels: List[str]
    test_types: List[TestType]
    duration_minutes: int | None
    remote: bool
    adaptive: bool
    languages: List[str]
    search_text: str


@lru_cache(maxsize=1)
def load_catalog() -> List[Assessment]:
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    if not catalog:
        raise RuntimeError(
            f"Catalog at {CATALOG_PATH} is empty. Run scripts/scrape_catalog.py first."
        )
    return catalog


def primary_test_type_code(assessment: Assessment) -> str:
    test_types = assessment.get("test_types") or []
    if not test_types:
        return ""
    return test_types[0].get("code", "")


@lru_cache(maxsize=1)
def catalog_by_name() -> dict:
    return {a["name"].strip().lower(): a for a in load_catalog()}


def find_by_exact_name(name: str) -> Assessment | None:
    return catalog_by_name().get(name.strip().lower())
