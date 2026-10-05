# chatbot_swedish_politics

A RAG-based chatbot for Swedish Poltics. Based on real data from Riksdagen, like betänkanden, motions and votes.

## Lokalt webbgränssnitt

Kör från projektets huvudmapp, i samma Pythonmiljö som terminalversionen:

```sh
python3 -m pip install -r requirements.txt
```

Ange Gemini-nyckeln i terminalen (zsh på Mac). Kör första raden, klistra in
själva nyckeln och tryck Enter. Inga tecken syns. Kör sedan export-raden:

```sh
read -s "GEMINI_API_KEY?Klistra in nyckeln och tryck Enter: "
export GEMINI_API_KEY
```

Starta webbgränssnittet i samma terminal:

```sh
python3 -m streamlit run app.py --server.address localhost --browser.gatherUsageStats false
```

Öppna adressen som visas, normalt http://localhost:8501. Terminalen behöver
vara igång; Ctrl+C stoppar appen. Vid en ny API-nyckel behöver appen startas om.
Nyckeln ska inte läggas i koden eller skickas till GitHub.

Appen använder `out/index_motions` som standard. Detta index innehåller även
partisidor och riksdagsmaterial. Modellen väljs från standardvärdet i `answer.py`.
Varje index behöver `info.json`, `meta.jsonl`, `vectors.npy` och `bm25.pkl` från
samma bygge. Röstningsfunktionen använder dessutom `out/positions_did.jsonl`
och betänkandetexterna i `out/chunks.jsonl`. Datafilerna ingår inte i Git.
`POLITICS_INDEX_DIR` kan användas som miljövariabel för en annan indexmapp.

Varje fråga anropar samma verktygsstyrda sökning som terminalversionen och kan
innebära flera Gemini-anrop. Indexet återanvänds mellan frågor. Den första
frågan kan ta längre tid när sökmodellen laddas. Appen visar det senaste svaret;
tidigare frågor skickas inte med som samtalsminne. Indexvalet avgör vilket
material som faktiskt kan hittas, även om koden stöder fler källtyper.

Källistan visar det hämtade underlaget, inte enbart citerade källor. Länkar till
original visas där metadata innehåller en webbadress. Gränssnittet ändrar inte
söklogik, prompt eller underliggande data och garanterar inte korrekta svar.

Gränssnittstester utan Gemini-anrop:

```sh
python3 -m unittest discover -s tests -p 'test_app.py'
```
