from ufal.udpipe import Model, Pipeline
import networkx as nx

MODEL_PATH = "slovak-snk-ud-2.5-191206.udpipe"

text = """
„Včera som si kúpil nové knihy,“ povedal Martin. Knihy boli zaujímavé, zaujímavejšie než tie, ktoré čítal minulý týždeň.
Čítať, čítal, čítajú — všetky tieto tvary patria k jednému slovu.

Na stole ležali jablká, hrušky a banány; niektoré boli čerstvé, iné pokazené.
„Naozaj to stojí za to?“ spýtala sa.

Programovanie, programoval, programuje — aj tieto slová by mali mať rovnaký základ.
"""

# načítanie modelu
model = Model.load(MODEL_PATH)
if not model:
    raise Exception("Model sa nepodarilo načítať")

pipeline = Pipeline(
    model,
    "tokenize",
    Pipeline.DEFAULT,
    Pipeline.DEFAULT,
    "conllu"
)

# spracovanie textu
processed = pipeline.process(text)

tokens = []

# parsovanie CONLLU
for line in processed.splitlines():
    if not line or line.startswith("#"):
        continue

    cols = line.split("\t")
    if len(cols) != 10:
        continue

    token = cols[1]
    lemma = cols[2]
    upos = cols[3]

    # interpunkcia delimiter
    if upos == "PUNCT":
        tokens.append(token)
        # continue
    else:
        tokens.append(lemma.lower())

print("Tokeny:", tokens)