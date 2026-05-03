from ufal.udpipe import Model, Pipeline
import networkx as nx

MODEL_PATH = "slovak-snk-ud-2.5-191206.udpipe"

text = """
„Naozaj si myslíš,“ spýtal sa potichu, „že všetko, čo vidíme, je presne také, aké sa zdá?“ 
Martin sa na chvíľu zamyslel — nie preto, že by nepoznal odpoveď, ale preto, že si nebol istý, či ju chce vysloviť nahlas.

V miestnosti bolo ticho. Prílišné, neprirodzené ticho; také, ktoré sa objavuje len vtedy, keď sa niečo chystá, alebo keď už niečo dávno prebehlo a nikto o tom nechce hovoriť. 
„Možno,“ povedal napokon, „je realita iba interpretáciou. A interpretácia… no, tá sa mení.“

Na stole ležali knihy: staré, zaprášené, popísané poznámkami na okrajoch. Niektoré stránky boli ohnuté, iné vytrhnuté — akoby niekto zámerne odstraňoval časti príbehu. 
Kto by to robil? A prečo?

„Pozri sa na tieto vety,“ pokračoval, zatiaľ čo prstom sledoval riadky textu. „Opakujú sa. Nie presne, ale dosť na to, aby si si to všimol. Slová, slová, slová — stále tie isté, a predsa vždy trochu iné.“

Zrazu sa ozval zvuk: klop, klop, klop. Niekto bol za dverami. Ale kto? A čo chcel? 
Martin sa postavil, pomaly, opatrne; každý krok bol váhavý, akoby si nebol istý pevnosťou podlahy.

„Ak otvoríš,“ zašepkala, „už to nebudeš môcť vziať späť.“ 
Neodpovedal. Vedel, že niektoré rozhodnutia sa nedajú odložiť — iba ignorovať, a aj to len na chvíľu.

Dvere sa otvorili. Svetlo z chodby preniklo dovnútra, ostré a nekompromisné. 
A potom… nič. Žiadna postava, žiadny hlas, iba prázdny priestor.

„Zvláštne,“ povedal. „Ale nie nečakané.“

A tak tam stáli, mlčky, každý ponorený do vlastných myšlienok, ktoré sa navzájom prekrývali, rozpadali a znovu skladali — ako sieť, ktorá nikdy nie je úplne stabilná, ale napriek tomu drží pohromade.
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