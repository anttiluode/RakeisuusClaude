"""The concept tree: 3 top nodes, 9 categories, up to 6 leaves each.

Every word whose vector we read must be ONE token at its place in the prompt,
so the clamp replaces the whole word. Candidates are listed in order of
preference; rakeisuus.py keeps the first single-token ones and reports the rest.
Questions are separate from the words, so a top node can be asked about as
"made by people" while its vector is read from a single word.
"""

# (key, question, word candidates for the vector)
TOPS = [
    ("plant", "Is the item above a plant?", ["plant"]),
    ("animal", "Is the item above an animal?", ["animal"]),
    ("made", "Is the item above made by people?", ["artifact", "object", "machine", "product"]),
]

# (key, top, question, word candidates, leaf candidates)
CATS = [
    ("tree", "plant", "Is the item above a tree?", ["tree"],
     ["oak", "birch", "maple", "pine", "willow", "elm", "cedar", "spruce"]),
    ("flower", "plant", "Is the item above a flower?", ["flower"],
     ["rose", "tulip", "daisy", "lily", "orchid", "lotus", "poppy", "violet"]),
    ("vegetable", "plant", "Is the item above a vegetable?", ["vegetable"],
     ["carrot", "potato", "onion", "cabbage", "lettuce", "cucumber", "garlic", "turnip"]),
    ("bird", "animal", "Is the item above a bird?", ["bird"],
     ["sparrow", "eagle", "owl", "crow", "parrot", "pigeon", "swan", "duck"]),
    ("fish", "animal", "Is the item above a fish?", ["fish"],
     ["salmon", "shark", "trout", "tuna", "cod", "carp", "herring", "eel"]),
    ("insect", "animal", "Is the item above an insect?", ["insect"],
     ["ant", "bee", "beetle", "butterfly", "mosquito", "wasp", "moth", "termite"]),
    ("tool", "made", "Is the item above a tool?", ["tool"],
     ["hammer", "wrench", "drill", "shovel", "axe", "chisel", "rake", "spade"]),
    ("vehicle", "made", "Is the item above a vehicle?", ["vehicle"],
     ["car", "bus", "truck", "bicycle", "train", "tractor", "van", "taxi"]),
    ("instrument", "made", "Is the item above a musical instrument?", ["instrument"],
     ["guitar", "piano", "violin", "drum", "flute", "trumpet", "harp", "cello"]),
]

LEAVES_PER_CAT = 6
MIN_LEAVES = 4


def leaf_question(word):
    art = "an" if word[0] in "aeiou" else "a"
    return f"Is the item above {art} {word}?"


def all_words():
    out = []
    for _, _, ws in TOPS:
        out += ws
    for _, _, _, ws, leaves in CATS:
        out += ws + leaves
    return out


def all_questions():
    qs = [q for _, q, _ in TOPS] + [q for _, _, q, _, _ in CATS]
    qs += [leaf_question(w) for *_, leaves in CATS for w in leaves]
    return qs
