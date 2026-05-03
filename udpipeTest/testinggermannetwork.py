from ufal.udpipe import Model, Pipeline
import networkx as nx

MODEL_PATH = "german-gsd-ud-2.5-191206.udpipe"

text = """
„Glaubst du wirklich“, fragte er leise, „dass alles, was wir sehen, genau so ist, wie es scheint?“
Martin hielt einen Moment inne — nicht, weil er die Antwort nicht kannte, sondern weil er nicht sicher war, ob er sie laut aussprechen wollte.

Im Raum war es still. Zu still, unnatürlich still; eine Stille, die nur dann entsteht, wenn etwas geschehen wird, oder wenn etwas längst geschehen ist und niemand darüber sprechen möchte.
„Vielleicht“, sagte er schließlich, „ist Wirklichkeit nichts weiter als eine Interpretation. Und Interpretation… nun, sie verändert sich.“

Auf dem Tisch lagen Bücher: alte, staubige, mit Anmerkungen an den Rändern versehene Bücher. Manche Seiten waren umgeknickt, andere herausgerissen — als hätte jemand absichtlich Teile der Geschichte entfernt.
Wer würde so etwas tun? Und warum?

„Sieh dir diese Sätze an“, fuhr er fort, während sein Finger den Zeilen des Textes folgte. „Sie wiederholen sich. Nicht genau, aber deutlich genug, dass man es bemerkt. Wörter, Wörter, Wörter — immer dieselben, und doch jedes Mal ein wenig anders.“

Plötzlich war ein Geräusch zu hören: klopf, klopf, klopf. Jemand stand hinter der Tür. Aber wer? Und was wollte er?
Martin stand langsam auf, vorsichtig; jeder Schritt zögernd, als wäre er sich der Festigkeit des Bodens nicht sicher.

„Wenn du öffnest“, flüsterte sie, „kannst du es nicht mehr rückgängig machen.“
Er antwortete nicht. Er wusste, dass manche Entscheidungen nicht aufgeschoben werden können — nur ignoriert, und selbst das nur für eine Weile.

Die Tür öffnete sich. Licht aus dem Flur fiel hinein, scharf und kompromisslos.
Und dann… nichts. Keine Gestalt, keine Stimme — nur leerer Raum.

„Seltsam“, sagte er. „Aber nicht unerwartet.“

Und so standen sie da, schweigend, jeder in seine eigenen Gedanken vertieft, die sich überlagerten, zerfielen und sich wieder neu zusammensetzten — wie ein Netzwerk, das niemals vollständig stabil ist und dennoch zusammenhält.
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