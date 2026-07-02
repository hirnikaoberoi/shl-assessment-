"""Sanity checks on the actual scraped catalog file. These are integration
tests against real scraped data (not fixtures), so they're skipped if the
catalog hasn't been built yet.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "catalog.json"

pytestmark = pytest.mark.skipif(
    not CATALOG_PATH.exists(), reason="catalog.json not built yet -- run scripts/scrape_catalog.py"
)


def _load():
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def test_catalog_is_non_trivial():
    catalog = _load()
    assert len(catalog) > 100, "expected several hundred Individual Test Solutions"


def test_every_item_has_required_fields():
    catalog = _load()
    for item in catalog:
        assert item.get("name"), item
        assert item.get("url", "").startswith("https://www.shl.com/"), item
        assert isinstance(item.get("test_types"), list) and item["test_types"], item


def test_no_duplicate_entity_ids():
    catalog = _load()
    ids = [item["entity_id"] for item in catalog]
    assert len(ids) == len(set(ids))


def test_scraper_scopes_to_individual_test_solutions_table_only():
    """Regression guard for catalog scope: verifies the scraper's actual
    extraction mechanism (matching by table header, not by item name) keeps
    Job Solutions out. Note: some genuine Individual Test Solutions are
    branded "...Solution" (e.g. "Entry Level Sales Solution" -- confirmed by
    checking their listing-table membership directly), so a name-pattern
    check on the final catalog would produce false positives. This test
    checks the mechanism instead of the output."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from scrape_catalog import extract_individual_test_table, parse_listing_page

    html = """
    <table>
        <tr><th class="custom__table-heading__title">Pre-packaged Job Solutions</th></tr>
        <tr data-entity-id="9001">
            <td class="custom__table-heading__title"><a href="/x/view/fake-job-solution/">Fake Job Solution</a></td>
            <td class="custom__table-heading__general"></td>
            <td class="custom__table-heading__general"></td>
            <td class="custom__table-heading__general product-catalogue__keys"></td>
        </tr>
    </table>
    <table>
        <tr>
            <th class="custom__table-heading__title">Individual Test Solutions</th>
            <th class="custom__table-heading__general">Remote Testing</th>
            <th class="custom__table-heading__general">Adaptive/IRT</th>
            <th class="custom__table-heading__general">Test Type</th>
        </tr>
        <tr data-entity-id="9002">
            <td class="custom__table-heading__title"><a href="/x/view/real-individual-test/">Real Individual Test</a></td>
            <td class="custom__table-heading__general"></td>
            <td class="custom__table-heading__general"></td>
            <td class="custom__table-heading__general product-catalogue__keys"></td>
        </tr>
    </table>
    """
    table_html = extract_individual_test_table(html)
    assert "Fake Job Solution" not in table_html
    assert "Real Individual Test" in table_html

    rows = parse_listing_page(html)
    names = [r["name"] for r in rows]
    assert names == ["Real Individual Test"]


def test_test_type_codes_are_known():
    valid_codes = {"A", "B", "C", "D", "E", "K", "P", "S"}
    catalog = _load()
    for item in catalog:
        for t in item["test_types"]:
            assert t["code"] in valid_codes, t
