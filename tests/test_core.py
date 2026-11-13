"""Fast tests: no model downloads. Run with: python -m pytest -q"""
import csv
import json
import random
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_data import make  # noqa: E402
from imgeval import analyze, color, stats, trials  # noqa: E402
from imgeval.manifest import check, load_catalog, load_manifest  # noqa: E402


@pytest.fixture()
def demo(tmp_path):
    return make(tmp_path / "demo")


def test_manifest_loads_and_checks(demo):
    rows = load_manifest(demo["manifest"])
    cat = load_catalog(demo["catalog"])
    assert len(rows) == demo["n"]
    assert check(rows, cat) == []
    assert {r.condition for r in rows} == {"generic", "personalized", "swapped"}


def test_manifest_rejects_bad_condition(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("image_id,image_path,buyer_id,condition,group_id\na,x.png,b,fancy,g\n", "utf-8")
    with pytest.raises(ValueError, match="condition"):
        load_manifest(p)


def test_check_reports_missing_files(demo):
    rows = load_manifest(demo["manifest"])
    rows[0].image_path = Path("nope.png")
    rows[1].items = ["unknown-id"]
    probs = check(rows, load_catalog(demo["catalog"]))
    assert any("image not found" in p for p in probs) and any("unknown catalog id" in p for p in probs)


def test_delta_e_same_vs_different():
    prod = Image.new("RGB", (100, 100), (255, 255, 255))
    prod.paste((150, 100, 60), (20, 20, 80, 80))
    same = Image.new("RGB", (100, 100), (150, 100, 60))
    other = Image.new("RGB", (100, 100), (40, 60, 160))
    assert color.delta_e(prod, same) < 1.0
    assert color.delta_e(prod, other) > 20


def test_wilson_and_rate():
    lo, hi = stats.wilson(8, 10)
    assert 0.44 < lo < 0.5 and 0.94 < hi < 0.97  # textbook values: 0.490, 0.943
    r = stats.rate([True] * 18 + [False] * 6, ["a", "b", "c"] * 8)
    assert r["rate"] == 0.75 and r["p_binom"] < 0.05 and r["boot_lo"] <= 0.75 <= r["boot_hi"]


def test_agreement():
    assert stats.agreement([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])["rho"] == pytest.approx(1.0)
    assert stats.agreement([1, 2], [1, 2])["rho"] is None


def _simulate_answers(study: Path, out: Path, raters, p_win=0.8, seed=1):
    rng = random.Random(seed)
    key = json.loads((study / "key.json").read_text("utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    for r in raters:
        t = json.loads((study / f"trials_{r}.json").read_text("utf-8"))["trials"]
        with open(out / f"answers_{r}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rater", "trial_id", "type", "choice", "quality", "real", "product", "ms"])
            for tr in t:
                k = key[tr["id"]]
                choice = q = real = prod = ""
                if tr["type"] == "ab":
                    win = rng.random() < p_win
                    choice = k["personalized_side"] if win else ("B" if k["personalized_side"] == "A" else "A")
                elif tr["type"] == "rate":
                    q = rng.choice("345") if k["source"] == "real" else rng.choice("234")
                    real = "Real photo" if rng.random() < (0.8 if k["source"] == "real" else 0.3) else "AI-generated"
                else:
                    prod = rng.choice(["Yes, same product", "Similar, but not the same", "No"])
                w.writerow([r, tr["id"], tr["type"], choice, q, real, prod, 1000])


def test_study_build_is_blind_and_analysis_runs(demo, tmp_path):
    rows = load_manifest(demo["manifest"])
    cat = load_catalog(demo["catalog"])
    raters = ["anna", "bo", "cy"]
    study = tmp_path / "study"
    counts = trials.build(rows, cat, raters, study, real_dir=demo["manifest"].parent / "real",
                          rate_per_rater=8, product_per_rater=4)
    # each buyer: 2 groups x 1 personalized x (1 generic + 1 own swapped) = 4 ab trials
    t = json.loads((study / "trials_anna.json").read_text("utf-8"))["trials"]
    assert sum(x["type"] == "ab" for x in t) == 4
    assert all(x["type"] == "ab" for x in t[:4])  # ab first
    blob = (study / "trials_anna.json").read_text("utf-8")
    for leak in ("personalized", "generic", "swapped", "g0-"):
        assert leak not in blob, f"trial file leaks '{leak}'"
    assert counts["anna"] == len(t)
    sides = [json.loads((study / "key.json").read_text("utf-8"))[x["id"]]["personalized_side"] for x in t[:4]]
    assert set(sides) <= {"A", "B"}

    _simulate_answers(study, tmp_path / "answers", raters)
    res = analyze.run(study, tmp_path / "answers", None, tmp_path / "report.md")
    assert res["human"]["n_raters"] == 3
    assert set(res["human"]["ab"]) == {"generic", "swapped"}
    assert "real" in res["human"]["mos"] and "fooling_rate" in res["human"]
    assert "Human study" in (tmp_path / "report.md").read_text("utf-8")
