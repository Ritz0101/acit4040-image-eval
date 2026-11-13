# Evaluation plan: how good are the final images?

We judge each generated room on four questions. Each question gets an automatic metric (cheap, repeatable, runs on every
pipeline change) and a human check (the ground truth). We then test whether the automatic metric agrees with the people.
If it does, we can use it alone during development.

| # | Question | Automatic metric | Human check | Standard / source |
|---|---|---|---|---|
| Q1 | Is the **real catalog product** in the image, unchanged? | Detect the item (OWLv2), crop it, compare it with the product photo: DINOv2 and CLIP-I similarity, top-1 identity among same-category products, colour ΔE2000 | "Is this exact product in the room?" | DreamBench DINO / CLIP-I (Ruiz et al., CVPR 2023); CIEDE2000 (Sharma et al., 2005) |
| Q2 | Does it fit **the buyer's style**? | Score under the buyer's own taste vector; taste identification (does the intended buyer's taste rank it first?) | Blind two-alternative forced choice (2AFC): the buyer picks between their personalised image and a generic or swapped one | 2AFC preference testing; Bradley–Terry (the RQ1 model) |
| Q3 | Is the **image quality** good? | MUSIQ, CLIP-IQA (no-reference IQA) | Quality 1–5 (Absolute Category Rating) → MOS | ITU-T P.910 ACR; Ke et al. ICCV 2021; Wang et al. AAAI 2023 |
| Q4 | Is it **realistic**, and is it still the buyer's actual apartment? | KID against real interior photos; depth-map correlation with the empty input room | "Real photo or AI-generated?" with real photos mixed in → fooling rate | KID (Bińkowski et al., ICLR 2018); HYPE (Zhou et al., NeurIPS 2019) |

## Conditions

For every room and brief (a *group*), we generate:

- **personalized**: made with buyer X's taste vector (the system as intended)
- **generic**: the same room and products, no taste vector (the baseline)
- **swapped**: made for buyer X's room, but with another buyer's taste vector (the control)

The swapped condition is the key control. If X prefers the personalized image over the swapped one, the gain comes from
*X's own* taste, not from personalisation making images look better in general.

Keep everything else fixed across conditions: seed, room, product set, resolution, sampler and steps.

## Hypotheses

- **H1 (style fit):** buyers prefer their personalized image over generic, and over swapped, more than 50% of the time.
- **H2 (fidelity):** personalisation does not cost product fidelity. Detection, identity and ΔE are about the same across conditions.
- **H3 (quality / realism):** personalized images are no worse than generic on MOS, and the fooling rate is not lower.

H2 and H3 are "no harm" checks: we expect no difference, and we report the effect sizes with confidence intervals, not just p-values.

## Procedure (pilot with the 5 group members)

1. Each member does the RQ1 ranking rounds (~2 min) and gets a taste vector.
2. Generate 3–4 rooms × 3 conditions per member. With 5 members that is about 60 images.
3. `python -m imgeval score` runs all automatic metrics.
4. `python -m imgeval study` builds the blind rating page. Each member rates in this order:
   A/B pairs for their own taste, then quality + real-or-AI, then product checks. About 15 minutes each.
5. `python -m imgeval analyze` writes the report.

## Analysis

- **Win rates and fooling rate:** binomial test against 50%, Wilson 95% CI, plus a cluster bootstrap over raters. Each rater
  answers many trials, so trials are not independent, and the bootstrap CI is the honest one. We also report how many of
  the 5 raters were individually above chance.
- **MOS:** the mean with a rater-bootstrap 95% CI, per condition.
- **Metric validity:** Spearman ρ between each automatic metric and the human score per image. ρ ≥ 0.5 → usable as a proxy.

## Threats to validity (write these in the report)

- **Five raters who built the system.** They are not naive and may recognise the conditions. This is a pilot; results are
  indicative, not general. The next step is 15–20 external raters.
- **Circularity:** the taste model both steers and scores the images. The automatic style score only shows the pipeline
  followed the model. H1 rests on the human A/B test.
- **The detector can fail.** A missed sofa can be a detector miss, not a generation miss. Spot-check the crops.
- **Lighting changes colour.** ΔE between a studio product photo and a rendered room is never 0. Compare conditions with
  each other rather than reading the absolute value.
- **KID needs ~50+ images per side;** below that, treat it as a rough number.

## Power, roughly

With 5 raters × 8 A/B trials = 40 trials, we need 26 or more wins to reject 50% (α = 0.05, one-sided
binomial). A true win rate of 75% gives about 95% power, and 70% gives about 81%, if the trials were independent. They are not, so read the per-rater count and the bootstrap CI as well.
