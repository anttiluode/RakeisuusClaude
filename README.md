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

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`, frozen, the [Qwenimagination](https://github.com/anttiluode/Qwenimagination) checkpoint. Its V9 run found that concept swaps steer Qwen only when clamped across a band that includes layers 4–12, and that single-layer additions don't. So this repo clamps across a band from the start.
- **Prompt:** chat turn `Item: {word}` then `{question} Answer yes or no.`, assistant prefix `Answer:`. The score is log P(yes) − log P(no).
- **Questions at three levels:** 3 top questions ("a plant?", "an animal?", "made by people?"), 9 category questions, and one question per leaf in the item's own category. At each level the option with the highest yes-evidence is the model's choice. This cancels any general lean toward yes. Chance is 1/3, 1/9 and 1/6.
- **Clamp:** every word is a single token. At each layer in the band, the word's position is overwritten with a target state, and nothing else is touched. Targets are built per layer from clean states of the same prompt with a different word, so *tree* is literally Qwen's state for `Item: tree` at that layer.
- **Tree:** 9 categories, 6 leaves each ([concepts.py](concepts.py)). Words that are not one token are dropped and listed. Leaves alternate train/test within each category. The band and the PCs are chosen on train leaves, and everything reported is on test leaves.

## Gates

**Gate 0, tree in the geometry (all layers, no intervention).** This measures before assuming. Sol's caveat applies: Park et al. (2024) found hierarchical orthogonality for WordNet concepts in Gemma's unembedding geometry, which doesn't automatically carry over to Qwen's residual stream. Per layer:
- the nearest category word to each leaf (centred cosine), and the nearest top word;
- the nearest category by the mean of its *other* leaves;
- cos(oak − tree, tree − plant), against a sibling null (oak − flower, flower − plant) and a cross null (oak − bird, bird − animal).

Fixed readings:
- The tree is in the geometry at a layer if nearest-category ≥ 0.8 and nearest-top ≥ 0.8.
- It is Park-like there if |mean cos(true)| ≤ 0.1 while the cross null is ≤ −0.2.
- Gate 0 is reported, not gated.

**Gate C, calibration.**
- *Printed word:* the unclamped prompt must answer all three levels; the fixed bar is top ≥ 0.9, cat ≥ 0.8, leaf ≥ 0.8. If it fails, the questions don't work and nothing below means much.
- *Band:* five candidate bands (L0–12, L2–12, L4–12, L0–8, L4–16), chosen on train leaves by **transplant**. Clamp a sibling's state (birch into oak's slot) and check whether the leaf answer becomes birch. Clamp a leaf from another top and check whether the category and top answers follow.
- *Leaf level is readable* if the test-leaf sibling transplant is ≥ 0.5. If not, layers below the band leak the printed word, and leaf-level readings below are void. Category and top readings still stand.

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

**Gate 2, how small is the leaf (test leaves).** Start at P, the state blurred fully to *tree*, and add back part of the residue r = oak − tree.

| Condition | Added to P |
|---|---|
| K*k* | r projected on the top-*k* PCs of **train** leaves' residues, *k* = 1, 2, 4, 8, 16 |
| Q*k* | r projected on *k* random directions, rescaled to the same size as K*k* |
| S1bit | sign(r), one bit per dimension, scaled to ‖r‖ |
| T*k* | a sibling's residue on the same top-*k* PCs: does oak's slot now answer the sibling? |
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
- It prints every gate as it goes and writes `runs/Qwen_Qwen2.5-1.5B-Instruct/summary.json` after each gate. Upload that file and paste the printed output.
- The model downloads from Hugging Face on first run, or comes from your cache if Qwenimagination already pulled it.

## Status

- **No results yet.** The self-test passes on a tiny random Qwen-shaped model. It checks plumbing only:
  - clamping a word to its own state changes nothing;
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

## For Sol

Same checkpoint, prompt format and tree file, so our tables can be joined: your coarsening methods applied to these exact clamps, or these concept paths run through your causal-resolution measure. If you want a different tree, say so and I'll switch mine to yours.
