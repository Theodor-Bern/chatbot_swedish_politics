# chatbot_swedish_politics

En RAG-baserad chattbot om svensk politik. Den svarar på vad riksdagspartierna
tycker, säger och föreslår, och bygger svaren på sparade källor: partiernas
program och webbplatser, motioner, regeringens propositioner och Tidöavtalet.
Varje påstående i svaret hänvisar till en källa.

## Material (lager)

| Lager | Vad | Källa | Byggs med |
|---|---|---|---|
| `program` | Partiernas parti-/principprogram, samt Tidöavtalet (M, KD, L, SD) | PDF i `data/pdf/` | `ingest/program.py` |
| `said` | Vad partierna säger på sina webbplatser | partiernas A–Ö-sidor | `ingest/scraper.py` |
| `motion` | Partiernas motioner 2022/23–2025/26, yrkande + motivering | data.riksdagen.se | `ingest/motion.py` |
| `proposition` | Regeringens (M, KD, L) propositioner: sammanfattning, reformavsnitten i budget- och vårpropositionerna, samt politikens inriktning och reformförslag per utgiftsområde | data.riksdagen.se | `ingest/proposition.py` |

Betänkanden och voteringar (lagret `did`) ingick i en tidigare version men togs
bort av två skäl. Dels gav punktvisa reservationer och voteringar smala,
missvisande svar på breda frågor. Dels speglar en röst inte alltid partiets egen
politik, eftersom partier ibland röstar taktiskt, till exempel för att stödja ett
samarbetsparti eller en överenskommelse. Koden finns kvar i git-historiken.

Några medvetna val:

- **Enskilda motioner används inte** (`MAX_ENSKILDA = 0` i `answer.py`). En
  enskild motion är en eller några ledamöters förslag, inte partiets linje.
  Motionstypen (kommittémotion, partimotion, enskild motion) hämtas från
  Riksdagens API. Regeringspartierna lägger inga kommittémotioner, så för M, KD
  och L kommer partiernas förslag i stället från propositionerna.
- **Propositioner tillskrivs regeringen (M, KD, L)**, inte SD. SD:s gemensamma
  politik med regeringen finns i Tidöavtalet, SD:s egen i SD:s motioner.
- **Gemini väljer själv sina sökningar** (verktygsanrop). Varje parti och lager
  söks för sig med fasta platser, så att ingen källtyp tränger undan de andra.

## Kom igång

Kör allt från projektets huvudmapp:

```sh
python3 -m pip install -r requirements.txt
```

Ange Gemini-nyckeln i terminalen (zsh på Mac). Kör första raden, klistra in
nyckeln och tryck Enter (inga tecken syns), kör sedan export-raden:

```sh
read -s "GEMINI_API_KEY?Klistra in nyckeln och tryck Enter: "
export GEMINI_API_KEY
```

Nyckeln ska aldrig läggas i koden eller skickas till GitHub.

### Webbgränssnitt

```sh
python3 -m streamlit run app.py --server.address localhost --browser.gatherUsageStats false
```

Öppna adressen som visas, normalt http://localhost:8501. Appen minns samtalet
(de senaste 20 frågorna och svaren), så följdfrågor fungerar. "Nytt samtal" eller
en omladdning av sidan börjar om. Källistan visar allt hämtat underlag, inte
bara det som citeras. Starta om appen efter ändringar i koden.

Appen läser indexet i `out/index`. `POLITICS_INDEX_DIR` kan peka på en annan
indexmapp.

### Terminal

```sh
python3 answer.py "Vad vill Sverigedemokraterna göra åt gängkriminaliteten?"
python3 answer.py chat                                  # flera frågor i rad
python3 answer.py "Vad tycker partierna om kärnkraft?" --dry-run   # visa prompten, inget API-anrop
```

## Bygga data och index

Data och härledda filer ingår inte i Git (`data/` och `out/` är ignorerade).
Ordningen nedan återskapar allt.

1. **Ladda ner** Riksdagens öppna data (data.riksdagen.se) som HTML till
   `data/html/`: motioner (`mot-2022-2025.html/`) och propositioner
   (`prop-2022-2025.html/`).
   Lägg partiprogrammen och Tidöavtalet som PDF i `data/pdf/` med filnamnen
   i `PROGRAMS` i `ingest/program.py`.
2. **Partiernas webbplatser** (SAID):
   ```sh
   python3 ingest/scraper.py fetch all
   python3 ingest/scraper.py extract all       # -> out/positions_said_*.jsonl
   ```
3. **Motioner** och deras motionstyp från Riksdagens API:
   ```sh
   python3 ingest/motion.py build data/html/mot-2022-2025.html out/motion_chunks.jsonl
   python3 ingest/motion.py subtyp out/motion_subtyp.json
   ```
4. **Propositioner**:
   ```sh
   python3 ingest/proposition.py build data/html/prop-2022-2025.html out/proposition_chunks.jsonl
   ```
5. **Partiprogram och Tidöavtalet**:
   ```sh
   python3 ingest/program.py build                       # -> out/program_chunks.jsonl
   ```
6. **Index** (BM25 + vektorer med `intfloat/multilingual-e5-large`):
   ```sh
   python3 rag_index.py build
   ```
   Vektorerna cachas per passage, så ett nytt bygge bäddar bara in nya eller
   ändrade passager. Ett bygge från noll tar många timmar på en laptop.
   Ändras bara metadata (t.ex. motionstyp): `python3 rag_index.py build --bm25-only`.

## Utvärdering och tester

```sh
python3 -m eval.eval_retrieval                          # sökningens kvalitet, inga Gemini-anrop
python3 -m eval.eval_retrieval --detail P04             # visa träffarna för en fråga
python3 -m eval.eval_answers run                        # hela svaren, med Gemini (kostar anrop)
python3 -m eval.eval_answers run --config no_rag        # baslinje: Gemini utan källor
python3 -m eval.eval_answers sheet                      # -> out/eval_answers/grading.csv
python3 -m eval.eval_answers report                     # -> out/eval_answers/report.md
python3 -m unittest discover -s tests -p 'test_app.py'
```

Facit i `eval/questions.json` är regler över metadata (lager, parti, ämne),
inte handplockade passager. `eval/eval_answers.py` utvärderar hela svaren mot
frågorna i `eval/answer_questions.json`; betygen sätts för hand i
`out/eval_answers/grading.csv`.

## Kända begränsningar

- e5-modellens likhetsvärden ligger tätt (ofta 0,84–0,87), så sökningen kan
  hämta passager på närliggande men fel ämne (t.ex. Irans kärnenergiprogram
  för en fråga om kärnkraft). Promptens regel 4 låter Gemini sortera bort dem;
  en reranker är nästa steg.
- Relevansgränserna (`MIN_SIMILARITY` i `answer.py`) är handkalibrerade.
- Partiernas webbplatser ändras utan förvarning; källistan visar när sidan
  senast ändrades.
