from ufal.udpipe import Model, Pipeline
import networkx as nx

MODEL_PATH = "german-gsd-ud-2.5-191206.udpipe"

text = """
„Gestern habe ich neue Bücher gekauft“, sagte Martin. Die Bücher waren interessant, interessanter als die, die er letzte Woche gelesen hat.
Lesen, las, gelesen — all diese Formen gehören zu demselben Wort.

Auf dem Tisch lagen Äpfel, Birnen und Bananen; einige waren frisch, andere verdorben.
„Lohnt es sich wirklich?“ fragte sie.

Programmieren, programmierte, programmiert — auch diese Wörter sollten dieselbe Grundform haben.
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