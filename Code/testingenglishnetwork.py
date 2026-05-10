from ufal.udpipe import Model, Pipeline
import networkx as nx

MODEL_PATH = "english-ewt-ud-2.5-191206.udpipe"

text = """
“Yesterday I bought some new books,” Martin said. The books were interesting, more interesting than those he read last week.
Read, reads, reading — all these forms belong to the same word.

On the table lay apples, pears, and bananas; some were fresh, others were spoiled.
“Is it really worth it?” she asked.

Programming, programmed, programs — these words should also share the same base form.
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