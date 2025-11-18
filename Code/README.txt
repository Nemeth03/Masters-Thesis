Požiadavky:
-----------
- Python 3.7 alebo novší (https://www.python.org/downloads/)
- Knižnice:
   - wxPython
    - matplotlib
    - networkx
    - numpy
    - scipy

- Knižnice je možné nainštalovať pomocou terminálu, príkazom:
    pip install wxPython matplotlib networkx numpy scipy

- Vstupné súbory musia byť vo formáte .txt a kódované v UTF-8.
- Aplikácia podporuje anglické a nemecké texty.


Ako spustiť aplikáciu:
----------------------
1. Otvor terminál alebo príkazový riadok.
2. Prejdi do priečinka 'Code':
    cd Code
3. Spusti aplikáciu:
    python mainApp.py

Alternatíva:
1. Otvor súbor mainApp.py pomocou editora, napr. Visual Studio Code, IDLE.
2. Spusti program.


Ako aplikácia funguje:
----------------------
- Aplikácia poskytuje grafické rozhranie na analýzu vplyvu interpunkčných znamienok pri analýze slovných sieti.
- Vyber textový súbor (.txt) cez tlačidlo 'Select File'.
- Zvoľ jazyk textu cez rozbaľovaciu ponuku (angličtina alebo nemčina).
- Vyber, ktoré interpunkčné znamienka chceš zahrnúť do analýzy.
- Ak je vybraný text a jazyk tak sa sprístupnia tlačidlá, pomocou ktorých môžeš:
    - ('Save Network') Uložiť vytvorenú slovnú sieť do súboru vo formáte GraphML.
    - ('Power-Law Analysis') Spustiť analýzu distribúcie stupňov vrcholov (power-law) a zobraziť výsledky.
      Zobrazí sa neupravená, logaritmicky zhlukovaná distribúcia a taktiež analýza zipfovho zákona.
    - ('Compare Distributions') Porovnať distribúcie stupňov vrcholov sieti bez a so zvolenou interpunkciou.
    - ('Growth Gamma') Analyzovať, ako sa mení exponent mocninového rozdelenia (gamma) pri raste siete.
    - ('Growth Comparison') Porovnať zmenu exponentu mocninového rozdelenia (gamma) pri raste siete bez interpunkcie
      a so zvolenou interpunkciou. 
    - ('Calculate Analysis') Zobraziť grafovú a jazykovú analýzu vytvorenej slovnej siete.

- Výsledky sa zobrazujú v dialógovom okne.
- Grafy sa zobrazujú v samostatných oknách.
