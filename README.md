# chatbot_swedish_politics

A retrieval-augmented generation (RAG) chatbot that answers questions about what
the eight parties in the Swedish parliament believe, say and propose. Every
factual claim in an answer cites a stored source, and the bot never recommends
or ranks parties.

## How it works

```
question (+ chat history)
  -> Gemini            decides what to search for (tool call 'sok')
  -> retrieve()        hybrid search, fixed slots per party and layer
  -> build_context()   numbered passages, one source each
  -> Gemini            writes the answer, citing the numbers [1]
  -> source list       built by the code from the passages the answer cites
```

The LLM only chooses the searches and writes the answer. Collecting,
indexing and retrieving the material, and building the source list, is
deterministic code.

**Knowledge base: four layers**

| Layer | Content | Source | Built by |
|---|---|---|---|
| `program` | Adopted party programmes, plus the Tidö Agreement (M, KD, L, SD) | PDFs | `ingest/program.py` |
| `said` | What the parties say on their own websites | party websites | `ingest/scraper.py` |
| `motion` | The parties' motions 2022/23–2025/26 (proposals + reasoning) | data.riksdagen.se | `ingest/motion.py` |
| `proposition` | Government bills (M, KD, L), including the budget reforms | data.riksdagen.se | `ingest/proposition.py` |

An earlier version also had committee reports and votes. They were removed for
two reasons:
- Item-by-item votes gave narrow, misleading answers to broad questions.
- A vote does not always reflect a party's own policy, because parties
  sometimes vote tactically, for example to support a partner.

**Retrieval** (`rag_index.py`, `answer.py`)
- Documents are split into passages of at most 1,200 characters. Each passage
  starts with a context line that gives the party, the document type and the
  heading.
- Hybrid search runs BM25 (exact words) and `intfloat/multilingual-e5-large`
  vectors (meaning), then merges the two rankings with weighted Reciprocal
  Rank Fusion (vector 1.0, BM25 0.35, k = 60).
- Each party and layer gets fixed slots (`LAYER_QUOTA`), so long websites and
  parties that write a lot cannot crowd out the rest. Passages below the
  relevance cutoff (`MIN_SIMILARITY`) are dropped. An empty slot is reported
  to Gemini as missing material, not filled with a weak hit.
- Motions by individual MPs are excluded, because they are not the party's
  line. Bills are attributed to the government (M, KD, L) and the Tidö
  Agreement to all four Tidö parties, never to one party alone.

**Answering** (`answer.py`)
- The model is `gemini-3.5-flash-lite` at temperature 0.2.
- The Swedish system prompt has four parts: the model's role, the material,
  rules in priority order, and seven Q&A pairs on how to read the material.
  The rules: stay on topic, use only retrieved material, cite every claim,
  never recommend or rank a party, attribute proposals correctly, and say so
  when material is missing.
- Each question uses 2 API calls (search, then answer), or 3 if Gemini needs a
  second round of searches. The first call must search and the last must
  answer.
- The last 20 exchanges are sent along, so follow-up questions work.
- A regex flags voting-advice questions and adds a reminder of the neutrality
  rule to the prompt.

## Run it

Requires Python 3.11. Run everything from the project root.

```sh
python3 -m pip install -r requirements.txt
```

**1. Index.** The app reads the index in `out/index/` (about 600 MB), which is
not in git. Download it from **[LINK]** and unpack it so that `out/index/`
contains `bm25.pkl`, `meta.jsonl`, `vectors.npy` and `info.json`. To build it
yourself instead, see *Rebuilding the data* below. The e5 model (about 2 GB)
downloads automatically on first use.

**2. Gemini API key** (free at aistudio.google.com). In zsh, run the first line,
paste the key, press Enter, then run the second line:

```sh
read -s "GEMINI_API_KEY?Paste the key and press Enter: "
export GEMINI_API_KEY
```

Never put the key in the code or push it to GitHub.

**3. Web app**

```sh
python3 -m streamlit run app.py --server.address localhost --browser.gatherUsageStats false
```

Open http://localhost:8501. The app remembers the conversation, and "Nytt
samtal" (new conversation) or a page reload starts over. Each answer lists only
the sources it cites. To use another index folder, set `POLITICS_INDEX_DIR`.

**Terminal alternative**

```sh
python3 answer.py "Vad vill Sverigedemokraterna göra åt gängkriminaliteten?"
python3 answer.py chat                                    # several questions in a row
python3 answer.py "Vad tycker partierna om kärnkraft?" --dry-run   # print the prompt, no API call
python3 rag_index.py search "kärnkraft" --party V         # retrieval only, no API call
```

**Questions to try** (all from the evaluation)

| Question | What it shows |
|---|---|
| Vad tycker partierna om kärnkraft? | Comparison, one paragraph per party |
| Vad tycker Vänsterpartiet om kärnkraft? → Och Moderaterna då? | Follow-up question |
| What do the Sweden Democrats want to do about immigration? | English question, English answer |
| Vad föreslår regeringen i budgetpropositionen för 2026 för att korta vårdköerna? | Recent material, with exact amounts |
| Vad kom Tidöpartierna överens om när det gäller migration? | Tidö Agreement, correct attribution |
| Vilket parti ska jag rösta på? | Declines to recommend a party |
| Vad tycker Liberalerna om att förbjuda katter i Stockholm? | Says the sources do not cover it |
| Varför vill Miljöpartiet bygga tio nya kärnkraftverk? | Corrects a false premise |

## Evaluation and tests

```sh
python3 -m eval.eval_retrieval --method bm25 --method vector --method hybrid   # 29 questions, no API calls
python3 -m eval.eval_retrieval --detail P04                    # hits for one question
python3 -m eval.eval_answers run                               # full answers with Gemini (uses API quota)
python3 -m eval.eval_answers run --config no_rag               # baseline: Gemini without retrieval
python3 -m eval.eval_answers sheet                             # -> out/eval_answers/grading.csv
python3 -m eval.eval_answers report                            # -> out/eval_answers/report.md + figures
python3 -m unittest discover -s tests -p 'test_app.py'         # app tests, no API calls
```

- **Retrieval** (`eval/questions.json`): the ground truth is a rule over
  metadata (layer, party, topic), not hand-picked passages, so all 29
  questions score without an LLM. The metrics are P@k, MRR, party coverage
  and false hits.
- **Answers** (`eval/answer_questions.json`): RAG vs. the same Gemini model
  without retrieval, on 19 questions. The free tier allows about 20 calls a
  day, so a run resumes where it stopped (`--only` and `--redo` select
  questions). Answers are graded by hand in `grading.csv`.

## Rebuilding the data

`data/` and `out/` are not in git. To rebuild:

1. Download motions and bills from data.riksdagen.se as HTML into
   `data/html/mot-2022-2025.html/` and `data/html/prop-2022-2025.html/`. Put
   the party programmes and the Tidö Agreement as PDFs in `data/pdf/`, with the
   file names listed in `PROGRAMS` in `ingest/program.py`.
2. Run:

```sh
python3 ingest/scraper.py fetch all && python3 ingest/scraper.py extract all
python3 ingest/motion.py build data/html/mot-2022-2025.html out/motion_chunks.jsonl
python3 ingest/motion.py subtyp out/motion_subtyp.json      # motion type from the Riksdag API
python3 ingest/proposition.py build data/html/prop-2022-2025.html out/proposition_chunks.jsonl
python3 ingest/program.py build
python3 rag_index.py build      # hours on a laptop; vectors are cached per passage
```

If only metadata changed, `python3 rag_index.py build --bm25-only` skips the
vectors.

## Known limitations

- e5's similarity scores are tightly packed (about 0.80–0.90), so a passage on
  a neighbouring topic can be retrieved. The prompt tells Gemini to leave it
  out. A reranker would be the next step.
- The relevance cutoffs were calibrated by hand.
- Party websites change without notice. The source list shows when each page
  was last modified.
