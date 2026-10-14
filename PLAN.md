# Plan

Goal: when someone asks what the parties think, the answer should rest on what
they *did* (votes) and *proposed* (motions), with their website (SAID) as
context, not the other way round. Broad questions should find broad evidence.

## Where things stand (2026-09-28)

- The full rebuild finished on 2026-09-26 (took ~23 h). The index is consistent:
  96,910 passages, 96,910 vectors.
- SAID labels are now in the context line (`said_context`), re-embedded.
- Uncommitted work on `motions-theo`: motion layer, `yrkande_kärna`, answer.py
  changes. Commit it once a question works end to end.

## First: baseline

```
python answer.py "Vad tycker partierna om klimatet?" --dry-run
```
Read the prompt Gemini would get. Note which passages are off-topic (e.g. the
crypto hit): which party, which layer, what score. That's the baseline.
Also run `python eval_retrieval.py` and save the numbers.

## Steps, in order

### 1. Embedding cache (`rag_index.py` `build`) — DONE 2026-09-28
Cache vectors keyed by a hash of `passage_text(p)`. A rebuild only embeds new or
changed passages: minutes instead of ~23 hours.
Every later step changes the index, so this comes first.
**Done when:** a second `build` with no data changes takes minutes.

### 1b. Embed the motion argument, not just the yrkande (`rag_index.py` `passage_text`)
Today a motion's vector covers the context line + the yrkande (median ~190
chars). The argument (`sektion_text`, median ~2,200 chars) is only in BM25,
not in the vector. Add heading + the first ~500 chars of `sektion_text` to what
gets embedded for motion passages. 23% of motions have no matched section;
those stay as they are.
Needs step 1, or it's another ~23 h rebuild.
**Done when:** "vad vill SD med biståndet?" finds SD's aid motions by vector,
not only by BM25.

### 2. Layer quotas + relevance cutoff (`answer.py` `answer()`/`retrieve()`) — DONE 2026-09-28
- Always search per layer, also for questions that name a party (today those
  are one mixed search, which SAID's polished prose wins).
- Weight the slots: e.g. MOTION 3, DID 3, SAID 1 per party, as constants.
  This applies to "vad tycker SD om X?" too: motions are the party's concrete
  policy, so opinion questions should lead with MOTION, not the website.
- Minimum score: an empty slot stays empty and the prompt says "no material
  found for party X in layer Y" instead of filling it with noise.
- Bug: `DID_ORD` matches "motion", so motion questions route to all layers
  (`answer.py:73`). Remove it from `DID_ORD`.
**Done when:** the climate question has no off-topic passages and every party
gets motion/DID evidence where it exists.

### 2b. Votes that explain themselves (`answer.py` `vote_facts()`) — PARTLY DONE 2026-09-28
Done: each vote block shows the point's title, the committee's proposal, what
each reservation wanted (its "Ställningstagande"), and the vote date. Parties
that abstained only because they had their own reservation are dropped. The
prompt says a no against a reservation is not a no to the idea, and a party's
own reservation is its position.
**Open (parked with the betänkande work): the majority's reasons.** "M röstade
med utskottets majoritet mot reservation 1 (S)" can mean "we don't want this"
or "an inquiry is already running". The reason is in the committee's
reasoning (`deliberation`), but one deliberation covers many sub-questions
(SfU15 2022/23: 104 sentences), and the rejection sentence ("Utskottet
avstyrker med det ovan anförda motionerna …") often sits at the end of a long
section. Matching on motion numbers or keywords ("pågår", "avvakta",
"föregripa") picked sentences about other sub-questions in tests. Needs the
deliberation split by its own sub-headings in `parser_riksdagen/betankande.py`,
then linking each sub-section to the reservations/motions it rejects.
Also: 52 votes with an unidentified reservation still produce an empty
sentence; drop them like the abstentions.
**Done when:** a vote block for a government party says *why* it voted no,
e.g. "utskottet hänvisade till en pågående utredning (dir. 2022:79)".

### 3. Shared topic labels across all three layers
- Define ~25 broad topics (klimat & miljö, energi, migration, rättsväsende,
  vård, skola, försvar, …).
- Map the 855 SAID `sakfraga` slugs onto them (one mapping table).
- Label each betänkande **point** (not whole reports, since budget reports span
  many subjects) with 1–3 topics: one Gemini call per betänkande (title + all
  point headings), ~1,500 calls, saved to JSONL so it runs once.
- Label motions the same way (one call per motion: title + yrkanden +
  headings; ~12,900 motions, so batch several motions per call).
- Spot-check 30–50 labels by hand.
- `search()` gets a topic filter. Metadata only, no re-embedding.
Note: the committee (`utskott`) is not enough on its own; climate spans MJU,
NU, TU, SkU, FiU.
**Done when:** "klimat" filters SAID, MOTION and DID to the same subject.

### 4. Per-report, per-party summary passages
One new passage per betänkande per party, built from `positions_did.jsonl` +
chunks:
```
MJU12 2023/24 "Klimatpolitik" — Centerpartiet: reserverade sig på 5 av 18 punkter
  - p.3 Vätgasstrategi: reservation 2 (C, L) -> röstade nej
  - övriga 13 punkter: röstade med utskottets majoritet
```
Broad questions match these; narrow ones ("AU10") still hit the point chunks.
Needs step 1, or it costs another full rebuild.
**Done when:** a broad question returns a summary instead of 2–3 random points.

### 5. Maybe: reranker
Cross-encoder (`BAAI/bge-reranker-v2-m3`, multilingual) over the candidates
before the cutoff in step 2. Only if steps 2–4 still let off-topic passages
through. ~2 GB model, a few seconds per question, no re-embedding.

## After each step
`python eval_retrieval.py` and the `--dry-run` climate question, compared with
tomorrow's baseline. If a step doesn't improve the numbers, don't keep it.
