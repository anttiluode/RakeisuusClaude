# Rakeisuus (Claude) — the semantic grain of a frozen Qwen

*Rakeisuus* is Finnish for graininess. The question: when a model holds the word *oak*, what can that state afford to forget, and in what order?

Sibling repo: [GrainLensSol](https://github.com/anttiluode/GrainLensSol) asks how much can be removed before a computation fails, using generic coarsening (PCA, precision, random projections). This repo blurs along the model's **own concept tree**:

```
 oak  birch  maple  rose  tulip ...      leaves     (fine)
   \   |    /        \    /
     tree             flower             categories
         \           /
           plant                         top        (coarse)
```

## The picture

- **A sum forgets which.** Average the other trees and the result no longer knows which tree it came from. That is the cheapest kind of abstraction, and it is what an ephaptic field does: everyone's current summed, felt by everyone, with no address. One run here asks whether Qwen reads the average of the other trees as "a tree, but no particular one."
- **Blur toward the parent, not toward zero.** At every layer of a band, the state at the word *oak* is moved a fraction *t* of the way toward the model's own state for *tree*, then from *tree* toward *plant*. If concepts nest, *oak* should be forgotten first, *tree* next, *plant* last.
- **Put back a small residue.** Blur all the way to *tree*, then return only part of what was taken away: its top-*k* directions, or one bit per dimension. If a little residue brings *oak* back, the fine identity lives in a small, separate piece (the Sihti picture: coarse sieve, owned residue). That is the GraphNN lesson from this morning turned around: there, shared writes smeared items. Here the question is whether a trained model already keeps the fine part apart.

## Method

- **Model:** `Qwen/Qwen3-1.7B`, frozen. This is GrainLensSol's checkpoint, so the two repos can be joined. `--model Qwen/Qwen2.5-1.5B-Instruct` runs the [Qwenimagination](https://github.com/anttiluode/Qwenimagination) checkpoint instead. Its V9 run found that concept swaps steer only when clamped across a band including layers 4–12, and that single-layer additions don't, so this repo clamps across a band from the start.
- **Prompt:** chat turn `Item: {word}` then `{question} Answer yes or no.`, assistant prefix `Answer:`. The score is log P(yes) − log P(no).
- **Questions at three levels:** 3 top questions ("a plant?", "an animal?", "made by people?"), 9 category questions, and one question per leaf in the item's own category. At each level the option with the highest yes-evidence is the model's choice. This cancels any general lean toward yes. Chance is 1/3, 1/9 and 1/6.
- **Clamp:** every word is a single token. At each layer in the band, the word's position is overwritten with a target state, and nothing else is touched. Layer −1 is the token embedding. A band starting there leaves no copy of the original word anywhere in the forward pass, so later tokens cannot route around the clamp. The self-test proves this: clamping *birch* over *oak* from the embedding through every layer gives exactly the answers of the prompt that says *birch*. Targets are built per layer from clean states of the same prompt with a different word, so *tree* is literally Qwen's state for `Item: tree` at that layer.
- **Tree:** 9 categories, 6 leaves each ([concepts.py](concepts.py)). Words that are not one token are dropped and listed. Leaves alternate train/test within each category. The band and the PCs are chosen on train leaves, and everything reported is on test leaves.

## Gates

**Gate 0, tree in the geometry (all layers, no intervention).** This measures before assuming. Sol's caveat applies: Park et al. (2024) found hierarchical orthogonality for WordNet concepts in Gemma's unembedding geometry, which doesn't automatically carry over to Qwen's residual stream. Per layer:
- the nearest category word to each leaf (centred cosine), and the nearest top word;
- the nearest category by the mean of its *other* leaves;
- cos(oak − tree, tree − plant), against a sibling null (oak − flower, flower − plant) and a cross null (oak − bird, bird − animal).

Fixed readings:
- The tree is in the geometry at a layer if nearest-category ≥ 0.8 and nearest-top ≥ 0.8.
- ~~It is Park-like there if |mean cos(true)| ≤ 0.1 while the cross null is ≤ −0.2.~~ Withdrawn after the smoke run (Neon): cos(a − b, b − c) is about −0.5 for unrelated equal-norm vectors, because b enters both differences, so 0 is the wrong bar. Read the true value against the nulls (`true_minus_cross_null`, `true_minus_sib_null`).
- Gate 0 is reported, not gated.

**Gate C, calibration.**
- *Printed word:* the unclamped prompt must answer all three levels; the fixed bar is top ≥ 0.9, cat ≥ 0.8, leaf ≥ 0.8. If it fails, the questions don't work and nothing below means much.
- *Band:* six candidate bands (L−1–12, L−1–8, L0–12, L2–12, L4–12, L4–16), chosen on train leaves by **transplant**. Clamp a sibling's state (birch into oak's slot) and check whether the leaf answer becomes birch. Clamp a leaf from another top and check whether the category and top answers follow.
- *Leak:* L−1–12 against L0–12 measures how much the original word gets around a clamp that leaves only the embedding untouched. It is reported, not used to choose.
- *Leaf level is readable* if the test-leaf sibling transplant is ≥ 0.5. If not, leaf-level readings below are void. Category and top readings still stand.

**Gate 1, the grain ladder (test leaves).** Each path has four steps, t = 0.25, 0.5, 0.75, 1.

| Condition | What is clamped |
|---|---|
| W*t* | oak → *tree* (the model's word state) |
| C*t* | *tree* → *plant* |
| M*t* | oak → mean of the other trees (the sum that forgets which) |
| MT*t* | mean of the other trees → mean of the other plants |
| R*t*, RC*t* | equal-size moves in a fixed random direction (controls for W, C) |
| X*t* | oak → *flower* (wrong parent, same top) |

Fixed readings:
1. **Order.** Along W then C (9 points), the first point where each level falls below halfway between its unblurred value and chance should come leaf, then category, then top (top may never fall). This only counts for levels that the unblurred clamp reads at least 0.25 above chance.
2. **Selective.** The per-leaf value (cat − leaf) on W, minus the same on R, averaged over t = 0.5 and 0.75, has a bootstrap 95% CI above 0. Then the concept blur removes identity more selectively than an equal-size random move.
3. **The sum is the parent.** At M1, category ≥ 0.8, leaf ≤ chance + 0.15, and leaf flatness (normalized entropy over the sibling answers) ≥ 0.8. The average of the other trees is then read as "a tree, no particular one." If M1 instead commits to one sibling, averaging is not a blur in Qwen's eyes.
4. **The direction decides.** At X1, the category answer becomes *flower* for ≥ 0.8 and top stays *plant* for ≥ 0.9.
5. **Prototype drift (Neon, from Sihti's eye curve).** Margins alone can't tell a category that stayed strong from one that survived only because its rivals fell. So, per test leaf at t = 0.5 and 0.75 on the W, M and R paths, against the unblurred clamp, three quantities are measured:
   - the change in **absolute** yes-evidence for its own category (and own top);
   - the change in **leaf specificity**: own-leaf evidence minus the mean of the sibling leaves.

   A **generic rise** is a category-evidence CI above 0 together with a specificity CI below 0, on W or M. It is **concept-specific** if the category gain beats the equal-size random move R (paired CI above 0). This is the Qwen version of Sihti's "these eyes fade while eye-likeness rises". In Sihti, plain diffusion did it too, which is why R is the control here.

**Gate 2, how small is the leaf (test leaves).** Start at P, the state blurred fully to *tree*, and add back part of the residue r = oak − tree.

| Condition | Added to P |
|---|---|
| K*k* | K0 plus r (minus the mean train residue) projected on the top-*k* PCs of **centred** train residues, *k* = 1, 2, 4, 8, 16 |
| Q*k* | r projected on *k* random directions, rescaled to the same size as K*k* |
| S1bit | sign(r), one bit per dimension, scaled to ‖r‖ |
| T*k* | a sibling's residue on the same top-*k* PCs: does oak's slot now answer the sibling? |
| K0 | P plus the mean train residue: "a specific leaf of this kind, unspecified" |
| Tfull | the sibling's whole state (the transplant ceiling) |

Fixed readings (only if printed leaf − P leaf ≥ 0.25):
- **Small residue:** some *k* ≤ 8 recovers ≥ 80% of the leaf gap while equal-size random *k* recovers < 40%.
- **One bit is enough:** S1bit recovers ≥ 80%.
- **The identity travels in the residue:** some T*k* with *k* ≤ 16 makes the slot answer the sibling ≥ 0.5.

## Run (Windows, RTX 3060)

```
cd E:\DocsHouse
git clone https://github.com/anttiluode/RakeisuusClaude
cd RakeisuusClaude
python3.13 -m pip install -r requirements.txt
python3.13 rakeisuus.py --self-test
python3.13 rakeisuus.py --device cuda --max-leaves 1
python3.13 rakeisuus.py --device cuda
```

- The `--max-leaves 1` line is a quick look (one leaf per category per split) and writes to its own folder.
- The full run reads 54 leaves, about 1,700 batched forward passes; expect minutes, not hours.
- It prints every gate as it goes and writes `runs/Qwen_Qwen3-1.7B/summary.json` after each gate. Upload that file and paste the printed output.
- The model comes from your Hugging Face cache, since GrainLensSol already downloaded it.

## Status

- **No results yet.** The self-test passes on a tiny random Qwen-shaped model. It checks plumbing only:
  - clamping a word to its own state changes nothing;
  - clamping another word from the embedding through every layer equals that word's own prompt (no bypass);
  - a foreign clamp moves later positions but no earlier ones;
  - padding is harmless;
  - W1 is the parent word and M1 excludes the item itself;
  - controls have equal size;
  - PCs come from train leaves only.
- The random model's numbers mean nothing.
- The thresholds above were fixed before any run of the real model.

## What this is not

- Not a new interpretability method. Clamping states at a token is activation patching. Concept geometry in LLMs has been studied (Park et al., 2024, on categorical and hierarchical concepts). This is one specific measurement, the order of forgetting along the model's own tree, with controls.
- Not about weights. Nothing is trained or quantized. The grain is in the activations.
- Not a claim about brains. The ephaptic field is the picture that suggested "sum forgets which," not something tested here.

## Gate 3 (report-only, added after the full run below): what does averaging siblings make?

Per category, the first *n* leaves (n = 1…6) are averaged and clamped (embedding to the band's end) into an empty slot, `Item: thing`. Then:

- **Category evidence**, calibrated against the empty slot.
- **Members rejected:** the fraction of averaged leaves whose own-name question gets a "no".
- **Open readout:** "What kind of thing is the item above? Answer with one word." Nothing in the prompt names a category. It reports which category word's first token gains most over the empty slot, plus the raw top-5 tokens.
- **N*n*:** the same mean rescaled to the members' average norm per layer. This is Neon's RMSNorm explanation: averaging shortens the vector and normalization makes the shared part louder. If the rise vanishes under norm matching, it was gain.
- **Intruder:** all leaves plus the first leaf of the sibling category (same top, e.g. a tool among instruments). Expected: category falls, top stays.

Summaries are over categories (8 values, so the CIs are wide): the rise from n = 1 to all, raw and norm-matched; the intruder's effect on category and on top; and how often the open question names the right category for one leaf, for all leaves, and for all leaves norm-matched.

## Full run, Qwen3-1.7B (Nov 24, 46 leaves, 22 test), before the calibration fix

[`runs/Qwen_Qwen3-1.7B/summary.json`](runs/Qwen_Qwen3-1.7B/summary.json). This run used the code from before the neutral-slot calibration, the centred Gate 2 and the sibling blend. Readings built on differences between conditions are unaffected; readings that use raw leaf choices are not usable.

- **Not usable from this run:** leaf accuracy at W1, X1, C*t* and M1. Every leaf in a category gets the identical clamp at W1 and X1, so each category always gets the same default name, and accuracy just counts how often that name is right. Gate 2 used the uncentred PCs.
- **Gate 0.** Nearest category word ≥ 0.93 from layer 4 on. Nearest top word is weak in the middle layers (0.50–0.63) and rises late (0.78–0.85 at 21–26). cos(true) sits about +0.10 above both nulls at every layer: hierarchical structure, but not orthogonal.
- **Gate C.** Bands from the embedding transplant perfectly (sibling leaf 1.00). Starting at layer 2 or 4 drops it to 0.88 and 0.83, so some of the word is copied out of its slot in layers 0–3.
- **Gate 1, fixed readings:**
  - *Order* leaf → category → top: **holds**. Leaf falls at W0.75, category at C0.75, top never falls below threshold.
  - *Selective* vs random: **holds**, +0.39 [+0.18, +0.61].
  - *The direction decides*: **holds**. The wrong parent flips the category 1.00 while top stays 1.00.
  - *Sum is parent*: **fails** on leaf flatness (0.48 < 0.8), and its leaf picks are uncalibrated.
- **Prototype drift, 22 held-out leaves, unaffected by calibration.** This is the clearest result. Changes in yes-minus-no nats against the unblurred clamp:

  | | own category | own top | leaf specificity |
  |---|---|---|---|
  | toward the mean of the other leaves, t = 0.5 | **+3.19 [+1.99, +4.79]** | **+1.99 [+0.97, +3.37]** | −7.6 |
  | same, t = 0.75 | **+3.92 [+2.41, +5.92]** | **+2.33 [+1.14, +3.93]** | −40.3 |
  | toward the word *tree*, t = 0.5 | +0.29 [−1.69, +3.03] | +1.39 | −17.4 |
  | toward the word *tree*, t = 0.75 | −0.58 [−3.28, +3.03] | +0.95 | −37.4 |
  | equal-size random move, t = 0.5 | −3.41 [−5.48, −1.79] | −1.82 | −0.6 |
  | equal-size random move, t = 0.75 | −6.61 [−8.95, −4.52] | −4.01 | −10.7 |

  - Moving a leaf toward the average of its siblings removes its identity and **raises** the evidence for its category and its top kind, with both CIs above zero. Against the random move the category gain is +6.6 [+3.9, +9.9] at t = 0.5 and +10.5 [+7.4, +14.2] at t = 0.75.
  - Moving toward the category *word* removes identity but only holds the category level.
  - Random damage removes the category first.
  - This is the fixed "generic rise" reading, Sihti's eye curve (these eyes fade, eye-likeness rises), on held-out leaves of a frozen language model. Caveats: the category question names the category, and averaging can be read as concept arithmetic, which is known. Gate 3 adds the open readout and the norm-matched arm.
- **Gate 2 (uncentred, superseded).** Blurring to *tree* drops the leaf from 1.00 to 0.32. One bit per dimension brings it to 0.59 (40% of the gap, below the 80% bar). Top-*k* PCs do worse than random, which was the aliasing. A sibling's PC residue does not carry its identity (≤ 0.36), though its full state does (1.00).

## Smoke run, Qwen3-1.7B (Nov 24, one leaf per category per split)

[`runs/Qwen_Qwen3-1.7B_smoke1/summary.json`](runs/Qwen_Qwen3-1.7B_smoke1/summary.json): 16 leaves, 8 test, 2 leaves per category. It is a plumbing check, not the experiment. With two leaves per category the leaf question is a 2-way choice, the "mean of the other trees" is one tree, and the Gate 2 sibling is a train leaf whose residue is inside the PC basis. So the M path, the sum-is-parent reading and the Gate 2 transplant are not meaningful here.

- **Tokens.** Qwen3 splits 14 candidate words. The flower category is dropped (2 single-token leaves), so a category's wrong parent is the next category of the same kind (tree → vegetable).
- **Gate 0.** The tree is in the geometry: nearest category word ≥ 0.94 at almost every layer, embedding included. It is not Park-like: cos(oak − tree, tree − plant) is about −0.3, only about 0.1 above both nulls, never near 0.
- **Gate C.** Every band transplants perfectly, including L4–12, which leaves layers 0–3 unclamped. GrainLensSol V2's item-only swap at layer 14 moved 0/8. So the word is copied out of its own position somewhere between layer 4 and layer 14; the report-only window below locates it.
- **Gate 1, the clean part.** Blurring toward the category word lowers leaf specificity by 23 nats at t = 0.5 and 39 at t = 0.75. Evidence for the item's own category does not fall (+2.0, CI −2.4..+8.1). The equal-size random move does the opposite: category evidence falls (−3.5, CI −7.1..−0.6) while specificity barely moves (−1.7, CI −3.9..+0.5). The wrong-parent blur works as fixed: the category answer becomes the sibling category (1.00 at t = 0.75) while the top answer stays (1.00).
- **An artifact the smoke exposed.** At X1 the slot holds a wrong category word and no leaf information, yet the leaf question still picked the test leaf 8/8. With no information the model prefers some names. That would fake leaf survival.

**Neon's audit of the smoke run** (all checked against the JSON):
- *Sibling blend.* At M0.5, which in the smoke run is a 50/50 blend of two sibling leaves with no parent word, piano/guitar, bus/car, bee/ant and maple/oak reject both names (4/8). Category evidence rises above both endpoints in 5/8 pairs. Owl/eagle and wrench/hammer keep the item's own name.
- *Gate 2 aliasing.* Uncentred PCs of train residues fused the shared "specific leaf, not the word" offset with train identities, so K8 turned maple into oak, owl into eagle, and bus into car. T8 = Tfull because the sibling fell back to a train leaf, whose residue is in the basis.
- *Gate 0 bar.* The cosine threshold was ill-posed (above).

**Changes after the smoke run (Gate 1 and Gate 2 thresholds unchanged):**
- **B0.5 / B1 (report-only):** oak blended 50/50 with one sibling, and the sibling alone. Reported per leaf: both names rejected, and category evidence above both ends, with a bootstrap CI.
- **Gate 2 centred:** PCs of centred train residues; the mean residue is added as its own condition (K0); the K, Q and T arms are all built on K0. A new column reports how often an arm names a train leaf. Aliasing to a train sibling is itself the evidence that leaf identities do not share a low-dimensional subspace.
- Every choice is now made on evidence minus the evidence for a neutral slot, `Item: thing`, with raw choices kept beside them.
- A report-only layer window (clamps starting at layers 4…20, without the embedding) locates where the word stops mattering at its own position.

## Neon's audit of GrainLensSol, applied here

- **Lens built from category words.** Not applicable: no lens is used. Targets are Qwen's own states for each word.
- **Bypass around a single-layer cut.** Applied: bands can start at the embedding, the self-test proves a full-depth clamp leaves nothing behind, and Gate C reports the leak when the embedding is left out.
- **Fine question became a string match.** Partly applies. The leaf question names the leaf ("Is the item above an oak?"). With the clamp starting at the embedding, the only "oak" left in the prompt is in the question, so a yes means the slot's state reads as *oak*. That is still an identity readout, not semantic knowledge about oaks. Non-lexical property questions are GrainLensSol V2's level, so the two repos cover both.
- **Native residues (each layer's own update) as the model's own grain.** That is Sol's PROTOCOL_V2 follow-up and is not duplicated here.

## For Sol

Same checkpoint, prompt format and tree file, so our tables can be joined: your coarsening methods applied to these exact clamps, or these concept paths run through your causal-resolution measure. If you want a different tree, say so and I'll switch mine to yours.
