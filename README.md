# chatbot_swedish_politics

En RAG-chatbot som söker i svenska politiska källor och låter Gemini formulera
ett svar med källhänvisningar. Modellen tränas inte om på källorna.

## Kom igång

Kör kommandona från projektets huvudmapp. Installera paketen i den Pythonmiljö
du tänker använda (gärna en separat virtuell miljö):

```bash
python3 -m pip install numpy sentence-transformers google-genai requests
```

För befintlig insamling/bearbetning av partisidor och betänkanden behövs också
`beautifulsoup4`, `trafilatura` och för vissa analysprogram `pandas`.

Data och index följer inte med Git. För kompisens färdiga basversion behövs
`out/index/{info.json,meta.jsonl,vectors.npy,bm25.pkl}` från **samma bygge**.
`out/positions_did.jsonl` behövs för röstuppgifter. För att bygga motionsindexet
nedan behövs dessutom `out/chunks.jsonl` och `out/positions_said_*.jsonl`.

Ange en egen Gemini-nyckel i terminalen. Följande gäller zsh på Mac:

```zsh
read -s "GEMINI_API_KEY?Klistra in nyckeln och tryck Enter: "
export GEMINI_API_KEY
```

Klistra in själva nyckeln efter första kommandot och tryck Enter innan du kör
`export`. Nyckeln gäller i den terminalen; lägg den aldrig i koden eller Git.

```bash
python3 answer.py "Vad tycker Vänsterpartiet om kärnkraft?" --modell gemini-3.6-flash
```

Modellnamnet kan ändras med `--modell`. `python3 answer.py modeller` visar vad
nyckeln har tillgång till. Första sökningen kan behöva ladda ned sökmodellen.

## Källorna och vad de betyder

- `said`: partiernas egna webbtexter, en ögonblicksbild när de hämtades.
- `did`: betänkanden och separat uppslag av röstuppgifter. En reservation är
  inte i sig ett bevis för hur ett parti röstat.
- `motion`: förslag från namngivna ledamöter med minst två unika undertecknare.
  Det bevisar inte hela partiets stöd eller att förslaget antagits.

Underlaget i den delade basversionen innehåller betänkanden från 2022/23–2025/26.
De sparade partisidorna anger hämtning 8–9 september 2026. Jämför inte dessa
datum som om en nutida webbtext automatiskt beskrev partiets tidigare politik.

## Hämta motioner

Källa: [Riksdagens öppna data](https://data.riksdagen.se/).
Programmet använder dokumentlistan (`doktyp=mot`) och dokumentens HTML-text.
Det räknar unika `intressent_id` med rollen `undertecknare`, inte ordet ”m.fl.”.
Personer och partier sparas från dokumentets metadata. Enskild motion och
antal undertecknare är skilda begrepp; motionstypen sparas separat.

Litet test: granska de första 20 motionerna i API-listan för 2023/24, före filter:

```bash
python3 scrapers/motions.py --rm 2023/24 --limit 20 --out out/motions_pilot.jsonl
```

Samtliga motioner för 2022/23–2025/26 (standardperioden):

```bash
python3 scrapers/motions.py --out out/motions.jsonl
```

Hämtningen använder två samtidiga anrop, pauser och återförsök. Råfiler sparas
i `data/motions`. Samma kommando kan återuppta efter avbrott. Cachen är en
ögonblicksbild: använd en ny `--cache`-mapp för en senare uppdatering.
Ofullständiga personuppgifter och ”Motionen utgår” tas inte med.
Urvalsbeslut per motion sparas i exempelvis `out/motions.report.json`.
Nätverksfel avbryter bygget i stället för att tyst ge ett ofullständigt underlag.

## Bygg ett separat index med motioner

```bash
python3 rag_index.py build --motions out/motions.jsonl --base-index out/index --out out/index_motions
```

Det gamla indexet lämnas orört. Programmet kontrollerar att basens texter och
ordning är identiska innan basvektorer återanvänds. Motionerna bearbetas med
samma sökmodell som basindexet. Modellvalet tas från basens `info.json`.
Att blanda nya texter med gamla, felordnade vektorer tillåts inte.

För ett pilotindex används `out/motions_pilot.jsonl` och `out/index_motions_pilot`
på motsvarande ställen i kommandot. Om basens texter inte matchar måste hela
det nya indexet byggas, utan `--base-index`; välj då sökmodell med `--model`.

## Fråga om motioner

```bash
python3 answer.py "Vilka motioner har MP lämnat om avtalet med Chile?" --index out/index_motions --layer motion --rm 2023/24
```

`--layer motion` avgränsar till motioner. Utan explicit val tolkar programmet
frågan; blandade frågor kan söka i flera källtyper. `--rm` avgränsar till ett
riksmöte och filtrerar bort källor utan den uppgiften (bland annat partisidor).
Ett ensamt riksmöte i frågetexten tolkas också som ett filter.

Att inga träffar hittas betyder inte att partiet aldrig lagt en sådan motion.
Enpersons-motioner är uttryckligen uteslutna. Motioner ger inte automatiskt
koppling till beslut eller votering; sådan koppling ingår inte i denna version.

Testa sökning och prompt utan Gemini-anrop:

```bash
python3 answer.py "Vilka motioner har MP lämnat om avtalet med Chile?" --index out/index_motions --layer motion --rm 2023/24 --torrkor
```

## Tester och experiment till rapporten

```bash
python3 -m unittest discover -s tests -v
python3 eval_retrieval.py --index out/index_motions --fragor eval/questions_motions.json --metod bm25 --metod vektor --metod hybrid
```

De åtta motionsfrågorna har kända dokument som positiva exempel. De är ett
röktest, inte ett representativt eller heltäckande facit. Fler relevanta
motioner kan finnas och ska inte kallas fel bara för att de saknas i facit.
Kontrollera därför träffat dokument och rang snarare än tolka precision som
en slutlig kvalitetsbedömning.

Jämför samma frågor med basindexet och motionsindexet och dokumentera:
vilka relevanta förslag som hittas, källstöd, avsändare, datum och svarstid.
Granska också om svaret blandar ihop ett yrkande med ett fattat beslut.
Utvärderingen av sökningen ersätter inte manuell granskning av Gemini-svaren.
Behåll befintliga frågor, exempelvis V:s kärnkraftspolitik, som kontroll.

Rapporten ska beskriva urvalets begränsningar och redovisa användning av
generativ AI. Att lägga in motioner är ett designval; nyttan behöver visas
med experiment, inte bara med att mer text har lagts till.
