# imgeval: evaluating the generated rooms

Scores the final images on four things: real **furniture fidelity**, **style fit** to the buyer, **quality**, and
**realism** (including whether the room is still the buyer's actual apartment). It also runs a small blind rating study
with people. The method and hypotheses are in [EVALUATION_PLAN.md](EVALUATION_PLAN.md).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows (bash: source .venv/Scripts/activate) · macOS/Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # or the CPU build
pip install -r requirements.txt
python -m pytest -q               # fast tests, no model downloads
```

The first `score` run downloads about 2.5 GB of model weights from Hugging Face (CLIP ViT-L/14, DINOv2, OWLv2,
Depth Anything V2, MUSIQ, CLIP-IQA). A GPU is recommended; on a CPU it is slow but works.

## Input

**`manifest.csv`**: one row per generated image. Paths are relative to the manifest.

| column | example | meaning |
|---|---|---|
| image_id | `r1-pers-p1` | unique |
| image_path | `gen/r1-pers-p1.png` | the generated image |
| buyer_id | `p1` | who the image was made **for** (= the rater id in the study) |
| condition | `personalized` / `generic` / `swapped` | see the plan |
| group_id | `r1` | same room + brief; A/B pairs are only made inside a group |
| room_image | `rooms/r1.png` | the empty input room (optional, for the geometry check) |
| items | `sofa-001;chair-002` | catalog ids that should be in the image (optional) |

**`catalog.json`**: the same shape as `data/fixtures/furniture.json` in the group repo (`id`, `name`, `category`, `image_path`).

**Taste vectors** (optional): a folder with `<buyer_id>.npy`, a 768-dim vector in CLIP ViT-L/14 space. From the RQ1 model:
`np.save(f"{buyer}.npy", pca.components_.T @ w32)`.

**Real photos** (optional): a folder of real interior photos, used for KID and mixed into the real-or-AI question.

## Run

```bash
python -m imgeval check   --manifest data/manifest.csv --catalog data/catalog.json
python -m imgeval score   --manifest data/manifest.csv --catalog data/catalog.json --tastes data/tastes --real-dir data/real --out results
python -m imgeval study   --manifest data/manifest.csv --catalog data/catalog.json --raters p1,p2,p3,p4,p5 --real-dir data/real --out study
python -m http.server -d study 8000        # each rater opens http://localhost:8000/?rater=<their id>
#   raters download answers_<id>.csv at the end → put them all in answers/
python -m imgeval analyze --study study --answers answers --scores results --out report.md
```

`study/key.json` holds the answer key (which side was personalized). Don't share it with raters. Image files in the study
get random names, so nothing leaks through file names.

## Output

- `results/images.csv`: per image: MUSIQ, CLIP-IQA, depth correlation with the empty room
- `results/items.csv`: per image × product: detected, DINOv2 / CLIP-I similarity, identity top-1, ΔE2000
- `results/style.csv`: per image: score under each buyer's taste, and the identification rank
- `results/set.json`: KID against the real photos
- `report.md` + `report.json`: everything per condition, the human results with CIs and tests, and metric–human agreement

## Try it without real data

```bash
python tests/demo_data.py demo
python -m imgeval score --manifest demo/manifest.csv --catalog demo/catalog.json --tastes demo/tastes --real-dir demo/real --out demo/results
```

The demo images are flat coloured boxes, so the numbers mean nothing. It only shows that the harness runs.
