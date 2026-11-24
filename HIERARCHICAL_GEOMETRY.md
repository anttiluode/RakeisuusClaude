# When maple fades and tree gets louder

**Research note · 24 November 2026 · Frozen Qwen3-1.7B**

Move the hidden state for *maple* most of the way toward the average of other trees. Will the model simply become uncertain, or will it give a stronger “yes” when asked whether the item is a tree? In our saved run, category evidence rose for **all 22 held-out words** across eight supplied families. Exact identity still won for **3 of 22**, down from 22 of 22. The choice of destination mattered. Moving toward the literal category word gave a small negative mean change in category evidence. [Run data](https://github.com/anttiluode/RakeisuusClaude/blob/d10d73d0f44c75612dd368c9f15186fb089e9e29/runs/Qwen_Qwen3-1.7B/summary.json) · [Intervention and scoring code](https://github.com/anttiluode/RakeisuusClaude/blob/d10d73d0f44c75612dd368c9f15186fb089e9e29/rakeisuus.py)

We encountered Karkada, Korchinski, Nava, Wyart and Bahri’s *Symmetry in language statistics shapes the geometry of model representations* by chance through a YouTube video after running this experiment. The comparison is retrospective. Their question runs from text statistics to learned geometry. Ours starts with an already trained model and asks what happens to its answers when we move one item’s state. [Paper, version 3](https://arxiv.org/abs/2602.15029v3) · [Video through which we found it](https://www.youtube.com/watch?v=5nlR_lofUrk)

## The measured contrast

The vocabulary had 46 single-token items. For each of 22 test items, Gate 1 intervened at the item position at the embedding and through decoder block 12. At each depth the sibling target was the mean of the *other* members of the supplied family. Each row below uses the same test words and questions. Category evidence is the change from the unedited condition in the yes-minus-no logit margin for the correct category. Identity is the highest-scoring name among the tested family’s alternatives after neutral-slot calibration.

| Edit at 75% strength | Mean change in category evidence, nats (95% item bootstrap interval) | Items with increased category evidence | Correct identity |
|---|---:|---:|---:|
| Toward the other siblings’ average | +3.92 (+2.41 to +5.92) | 22/22 | 3/22 |
| Toward the category word | −0.58 (−3.28 to +3.03) | 9/22 | 8/22 |
| In a random direction | −6.61 (−8.95 to −4.52) | 0/22 | 22/22 |

The random displacement was matched in length to the **category-word** displacement at each depth. The sibling-average displacement had its own length. Items in a family share donors, and the intervals resample items. Generalization to new families and checkpoints needs its own test. [Gate 1 implementation](https://github.com/anttiluode/RakeisuusClaude/blob/d10d73d0f44c75612dd368c9f15186fb089e9e29/rakeisuus.py) · [Per-item rows](https://github.com/anttiluode/RakeisuusClaude/blob/d10d73d0f44c75612dd368c9f15186fb089e9e29/runs/Qwen_Qwen3-1.7B/summary.json)

### What averaging already guarantees

There is a useful piece of arithmetic underneath the result. Let a family contain $n$ states with mean $m$, including the current item $h$. Moving a fraction $t$ toward the mean of the *other* members gives a new state $h'$ satisfying

$$
h'-m=\left(1-\frac{nt}{n-1}\right)(h-m).
$$

For the five- and six-member families in this run, $t=0.75$ leaves only **6.25% or 10%** of the original distance to the full-family mean at each patched depth. That alone makes the individual vectors much more alike. The equation says nothing about how Qwen will answer “Is this a tree?” The consistent positive change in that answer is the empirical finding. It also explains why a “very blurry” state alone would be weak evidence for a semantic mechanism.

### Can Qwen supply the name itself?

Gate 3 clamped a family average into a neutral item slot and asked what kind of thing it was, without supplying the category name in the question. For the first listed member alone, the unrestricted highest-ranked next token named the intended category for **2/8** families. With all members averaged, it did so for **7/8**. Eagle alone gave *animal* and the bird average gave *bird*. Guitar gave *object* and the instrument average gave *instrument*. The fish average gave *animal* first and *fish* second. The separate calibrated choice among category labels scored **8/8 both times**. The single-member baseline used a fixed first exemplar per family. A comparison against every member would be stronger. [Gate 3 rankings, including raw top-five tokens](https://github.com/anttiluode/RakeisuusClaude/blob/d10d73d0f44c75612dd368c9f15186fb089e9e29/runs/Qwen_Qwen3-1.7B/summary.json)

## Where the two studies meet

| Connection | What the paper establishes | What the Qwen experiment contributes |
|---|---|---|
| Hierarchies | On p. 9 the authors leave open an account of how hierarchical geometry arises from training statistics. Park et al. had already characterized hierarchical concept geometry. | The supplied hierarchy gives interventions a direction. The Qwen run measures how those interventions change category and member judgments. It does not infer the hierarchy from text or explain its origin. |
| Fourier and Walsh modes | Appendix D, Theorem 5 gives orthogonal seasonal and binary-attribute subspaces for a particular product-space model of pointwise mutual information. | Rakeisuus uses arithmetic means and coordinate signs. A residual’s signs do not identify Walsh modes. Whether Qwen’s families satisfy anything like those assumptions is unmeasured. |
| Helper words | Seasonal helpers can reconstruct month geometry after direct month-pair statistics are removed. Appendix C.1 reports an approximate inverse-square-root trend in reconstruction error for selected helpers. | A family name can be read from sibling activations. No training co-occurrences were removed here. A model answering *bird* cannot establish which corpus words gave rise to that answer. |
| Early layers and ambiguity | Figure 14 follows how context changes *May*’s position relative to the other months in Gemma. | Qwen’s nearest-category-word accuracy is 0.96 at the embedding, falls to 0.67 at block 1, and returns to 0.96 at block 4. CrossWord’s four-block patch is an implementation choice. These observations do not locate a shared disambiguation circuit. |
| Coarse and fine detail | Proposition 4 bounds error when linearly decoding continuous lattice coordinates from a rank-limited embedding under spectral assumptions. | Gate 2 asks whether components fitted on training leaves recover the identities of held-out leaves. Those different targets and train/test designs preclude a general conclusion about continuous versus discrete concepts. |

[Paper, especially Proposition 4, Figure 14 and Theorem 5](https://arxiv.org/pdf/2602.15029v3) · [Park et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/be7430d22a4dae8516894e32f2fcc6db-Abstract-Conference.html) · [CrossWord solver implementation](https://github.com/anttiluode/CrossWord/blob/a8bafcdd28c668c3d692b76974179eae98c27396/engine.py)

**What the residue experiment recovered.** The category-word state alone recovered **5/22** exact names. Adding the signs of the item’s own centered residual, scaled by its true norm at each patched depth, recovered **18/22**. That restores 13 of the 17 lost answers, **76.5%**, below the preset 80% threshold. Adding the tested principal components fitted on other items recovered only **1–3/22**. This shows that a coarse description of an individual residual can be useful here. Establishing orthogonality would require measuring angles, and a storage claim would have to count the per-layer signs, scales and shared states. [Gate 2 data and fixed thresholds](https://github.com/anttiluode/RakeisuusClaude/blob/d10d73d0f44c75612dd368c9f15186fb089e9e29/runs/Qwen_Qwen3-1.7B/summary.json)

## A test that would connect the two questions

Build a small corpus with a known tree. Give siblings shared context words and each leaf a set of private context words. Train matched models on two versions: one with the shared contexts intact, and one with those contexts shuffled across families while preserving word frequencies. Use the same sibling-average intervention, a random move matched to its actual length, and held-out questions about shared properties. The prediction is a **larger sibling-induced category gain when family-specific co-occurrence structure is intact**. Failure of that contrast would count against this particular explanation even if both models still named familiar categories. That experiment would manipulate the proposed origin of the geometry as well as its use.
