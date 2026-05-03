from ufal.udpipe import Model, Pipeline
import networkx as nx

MODEL_PATH = "english-ewt-ud-2.5-191206.udpipe"

text = """
“Do you really think,” he asked quietly, “that everything we see is exactly what it seems to be?”
Martin paused for a moment — not because he did not know the answer, but because he was unsure whether he wanted to say it out loud.

The room was silent. Too silent, unnaturally silent; the kind of silence that appears only when something is about to happen, or when something has already happened and no one wishes to speak of it.
“Perhaps,” he said at last, “reality is nothing more than an interpretation. And interpretation… well, that changes.”

Books lay scattered across the table: old, dusty, filled with notes scribbled in the margins. Some pages were folded, others torn out — as if someone had deliberately removed parts of the story.
Who would do such a thing? And why?

“Look at these sentences,” he continued, tracing the lines of text with his finger. “They repeat themselves. Not exactly, but enough for you to notice. Words, words, words — always the same, and yet somehow different each time.”

Suddenly, there was a sound: knock, knock, knock. Someone stood behind the door. But who? And what did they want?
Martin stood up slowly, cautiously; each step hesitant, as though he doubted the floor beneath him.

“If you open it,” she whispered, “you won’t be able to take it back.”
He said nothing. He knew that some decisions cannot be postponed — only ignored, and even that only for a while.

The door opened. Light from the hallway poured inside, sharp and unforgiving.
And then… nothing. No figure, no voice — only empty space.

“Strange,” he said. “But not unexpected.”

And so they stood there, silently, each absorbed in their own thoughts, overlapping, collapsing, and forming again — like a network that is never entirely stable, yet somehow still holds together."""

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