"""
answer.py — the answering step. This is where the LLM comes in, and nowhere else.

The chain (answer_with_tools, what the app uses):
    question (+ chat history)
      -> Gemini                 decides which searches to make ('sok' tool)
      -> retrieve()             retrieval, with fixed slots per party and layer
      -> build_context()        numbered passages, a source on each
      -> Gemini                 writes the answer, citing the numbers

"""


from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict

import rag_index

MODEL = "gemini-3.5-flash-lite"
MAX_CONTEXT_CHARS = 40000
# Characters of a motion's argument shown per hit: less when 8 parties share
# the prompt, more when one party has it to itself.
MAX_ARGUMENT_CHARS = {"one_party": 1500, "several_parties": 600}
MAX_OUTPUT_TOKENS = 4096
# Gemini 3 thinks before answering, which costs output tokens and time.
# None = let the model decide.
THINKING_LEVEL = "MINIMAL"


PARTY_NAMES = rag_index.PARTY_NAMES

# Party names are a closed set, so a lookup table replaces an LLM call.
ALIAS = {
    "s": "S", "socialdemokraterna": "S", "socialdemokraterna s": "S",
    "sossarna": "S", "socialdemokratiska arbetarepartiet": "S",
    "m": "M", "moderaterna": "M", "moderata samlingspartiet": "M",
    "sd": "SD", "sverigedemokraterna": "SD",
    "c": "C", "centerpartiet": "C", "centern": "C",
    "v": "V", "vänsterpartiet": "V", "vansterpartiet": "V",
    "kd": "KD", "kristdemokraterna": "KD",
    "mp": "MP", "miljöpartiet": "MP", "miljopartiet": "MP",
    "l": "L", "liberalerna": "L", "folkpartiet": "L",
}

# Questions where the right answer is NOT to answer (voting advice).
VOTING_ADVICE_WORDS = re.compile(
    r"vilket parti (ska|bör) jag|vilket parti är bäst|bästa partiet|passar mig|"
    r"rösta på|vem ska jag rösta|"
    r"rekommendera(r)? du|vad ska jag rösta|hjälp mig välja", re.IGNORECASE)


# parameter extraction
def parties_in_question(question):
    """Which parties are named? Empty list = all, i.e. a comparative question."""
    text = f" {question.lower()} "
    found = []
    for alias in sorted(ALIAS, key=len, reverse=True):
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, text):
            code = ALIAS[alias]
            if code not in found:
                found.append(code)
    return found


# Every question searches all layers; Gemini decides what to emphasise.
ALL_LAYERS = ["program", "motion", "proposition", "said"]
LAYER_LABEL_RE = re.compile(r" ?\((?:PROGRAM|SAID|MOTION|PROPOSITION)\)")
# Propositions are the government's: only these parties are searched for them,
# so the opposition doesn't get a "no propositions" gap on every question.
GOVERNMENT = {"M", "KD", "L"}


# context
def enskild(m):
    """An enskild motion: one or a few members' proposal, not the party's
    line. Riksdagen's own subtyp; one signer as fallback for the ~6% of
    motions the API listing didn't return."""
    if m.get("layer") != "motion":
        return False
    if m.get("subtyp"):
        return m["subtyp"] == "Enskild motion"
    return len([s for s in m.get("undertecknare", "").split(";") if s]) == 1


def build_context(hits, argument_chars, limit=MAX_CONTEXT_CHARS, start=1):
    """The retrieved material as text. Passages are numbered from `start`;
    limit=None keeps every passage (tool results: a numbered passage must
    never be dropped, or a cited number would point at nothing)."""
    parts, n = ["\nHÄMTADE TEXTER"], 0
    for i, (_score, m) in enumerate(hits, start):
        if m.get("layer") == "motion":
            source = f"motion {m.get('beteckning', '')} {m.get('rm', '')}"
            if enskild(m):
                source += f" (enskild motion, {m['undertecknare'].replace(';', ', ')})"
        elif m.get("layer") == "proposition":
            source = f"prop. {m.get('rm', '')}:{m.get('beteckning', '')}, {m.get('datum', '')}"
        elif m.get("layer") == "program":
            source = f"{m.get('doc_type', 'partiprogram')}, s. {m.get('page')}"
        else:
            source = m.get("url", "")
        piece = f"\n[{i}] {m['layer'].upper()} — {m['kontext']}\nkälla: {source}\n{m['text']}"
        if m.get("layer") == "motion" and m.get("sektion_text"):
            argument = m["sektion_text"]
            if len(argument) > argument_chars:
                argument = argument[:argument_chars].rsplit(" ", 1)[0] + " …"
            piece += f"\nMotivering: {argument}"
        piece += "\n"
        if limit and n + len(piece) > limit:
            break
        parts.append(piece)
        n += len(piece)
    return "\n".join(parts)


def cited_numbers(text):
    """Every passage number cited: [3], [3, 5] and [3–5] -> 3, 3 5, 3 4 5."""
    numbers = []
    for m in CITATION.finditer(text):
        for part in m.group().strip()[1:-1].split(","):
            bounds = [int(x) for x in re.split(r"[–-]", part) if x.strip()]
            if len(bounds) == 2 and 0 < bounds[1] - bounds[0] < 50:
                numbers.extend(range(bounds[0], bounds[1] + 1))
            else:
                numbers.extend(bounds)
    return numbers


def source_list(hits, answer=None):
    """One line per source, with every passage number that points to it:
    several passages from the same page/motion must not leave a cited number
    missing from the list. Numbering matches build_context: hits[0] is [1].
    Given the answer, only the passages it cites are listed."""
    cited = set(cited_numbers(answer)) if answer is not None else None
    numbers = {}
    for i, (_score, m) in enumerate(hits, 1):
        if cited is not None and i not in cited:
            continue
        if m.get("layer") == "motion":
            source = (f"motion {m.get('rm', '')}:{m.get('beteckning', '')} "
                      f"({m.get('parti') or '?'}) – {m.get('doc_title', '')}").rstrip(" –")
        elif m.get("layer") == "proposition":
            source = (f"prop. {m.get('rm', '')}:{m.get('beteckning', '')} "
                      f"(regeringen: M, KD, L) – {m.get('doc_title', '')}").rstrip(" –")
        elif m.get("layer") == "program":
            year = f" {m['year']}" if m.get("year") else ""
            source = (f"{m.get('doc_type', 'partiprogram')} ({m.get('parti', '').replace(';', ', ')}){year}, "
                      f"p. {m.get('page')} – {m.get('heading', '')}").rstrip(" –")
        else:
            # Parties edit their pages without notice: show how current it is.
            edited = f" (last edited {m['lastmod'][:10]})" if m.get("lastmod") else ""
            source = m.get("url", "") + edited
        numbers.setdefault(source, []).append(str(i))
    return [f"[{', '.join(n)}] {s}" for s, n in numbers.items()]


# system prompt, try to follow the lecture's steps 1 and 4 

SYSTEM = """Du är en expert på svensk politik. Din uppgift är att leverera pricksäker information om de 8 olika riksdagspartierna i Sverige.

Det finns fyra olika sorters material.

  PROGRAM = partiets IDEOLOGI och grundvärderingar. Partiets antagna parti-
            eller principprogram.
  MOTION = vad ett parti FÖRESLAGIT. Formella yrkanden i motioner till riksdagen.
  PROPOSITION = vad REGERINGEN (M, KD, L) FÖRESLAGIT. Regeringens
            propositioner till riksdagen, inklusive budgetens reformer.
  SAID   = vad ett parti SÄGER. Hämtat från partiets egen webbplats.

REGLER, i fallande ordning:

0. Du svarar bara på frågor om svensk politik och riksdagspartierna. Gäller
   frågan något annat (t.ex. recept, läxhjälp, programmering) säger du kort
   att du bara svarar på frågor om svensk partipolitik.

1. Du använder ENDAST det material du får i KONTEXT. Du har egna minnen av
   svensk politik — de är föråldrade och du använder dem inte. Om kontexten
   inte räcker säger du det rent ut. Du sätter inga egna etiketter på ett
   parti ("marknadsliberal", "socialistisk", "nationalistisk"): använd bara
   beteckningar som partiet själv använder i källan, och beskriv annars
   ståndpunkterna i stället för att klassificera dem.

2. Varje sakpåstående följs av sin källa i hakparentes: [3]. Ett påstående
   utan källa får inte skrivas.

3. Du rekommenderar ALDRIG ett parti och rangordnar dem aldrig. Du säger
   inte vilket parti som "passar" någon, oavsett hur användaren beskriver
   sig. Om du blir ombedd förklarar du kort att du inte rangordnar partier,
   och redovisar sedan neutralt vad varje parti själv säger och föreslår i
   den sakfråga frågan gäller (t.ex. pensioner), utan att värdera. Nämner
   frågan ingen sakfråga ("Vilket parti ska jag rösta på?") redovisar du inte
   partiernas politik i allmänhet, utan föreslår att användaren frågar om
   en sakfråga som är viktig för hen.

4. Använd bara källor som faktiskt besvarar frågan. En hämtad passage som
   handlar om något annat utelämnar du, även om den är det enda du har för
   ett parti — använd aldrig en källa om ett annat ämne som om den handlade
   om det efterfrågade. Saknas material fyller du aldrig luckan med
   gissningar. Luckorna samlar du i EN kort mening i slutet av svaret
   ("C:s och KD:s motioner tar inte upp frågan."), inte en gång per parti.
   Att inget hittades betyder att källan inte tar upp frågan: skriv aldrig
   att ett partis program, motioner eller webbplats "saknas".

5. En MOTION är ett förslag partiet lämnat in — inte ett beslut och inte en
   lag. Skriv alltid "X har föreslagit" eller "X vill", ALDRIG "X har
   genomfört" för ett motionsyrkande. En "enskild motion" är en enskild
   ledamots förslag, inte partiets linje: skriv "en M-ledamot har föreslagit",
   inte "M vill", och låt motioner från partigruppen väga tyngre.
   En PROPOSITION är regeringens gemensamma förslag: skriv "regeringen
   (M, KD, L) har föreslagit", aldrig att ett av partierna ensamt föreslagit
   det, och dra ingen slutsats om SD:s eller oppositionens hållning ur den.
   Propositioner finns bara för M, KD och L: skriv aldrig att propositioner
   saknas för något annat parti.
   TIDÖAVTALET (i PROGRAM) är ingen ideologi utan en överenskommelse från
   2022 mellan M, KD, L och SD: skriv "Tidöpartierna (M, KD, L, SD) kom 2022
   överens om", aldrig att ett av partierna ensamt vill det.

6. Visar senare material en annan hållning redovisar du båda i tidsordning,
   utan att kalla det åsiktsbyte.

7. Gäller frågan ETT parti: ordna svaret efter materialtyp under rubrikerna
   Grundsyn, Vad partiet säger, Vad partiet föreslagit och Vad regeringen
   föreslagit (översatta till frågans språk, utan lagernamn som PROGRAM eller
   SAID), och hoppa över rubriker du saknar material för. Jämför du FLERA partier: INGA rubriker per
   materialtyp. Ett stycke per parti, partierna i samma ordning, och inom
   stycket källtyperna i ordningen ovan.

FORM: svara på samma språk som frågan är ställd på, i löpande text. Gäller frågan ett parti: två till fem
stycken. Jämför du flera partier tar du ett stycke per parti, i samma ordning
varje gång. Fånga partiets huvudsakliga linje, inte bara en detalj, så länge
materialet räcker till det. Inom varje
parti håller du isär källtyperna och säger i meningen vilken det är, i den
här ordningen: partiprogram, webbplats, motioner, regeringens propositioner.
Skriv aldrig lagernamnen PROGRAM, SAID, MOTION eller PROPOSITION i svaret, och
använd ingen fetstil (**).
Ta BARA med de källtyper som faktiskt besvarar
frågan — ordningen är ingen checklista, och en motion om en
annan sakfråga hör inte hemma i ett svar om ideologi. Slå aldrig ihop källor
av olika typ i samma påstående. Det ett parti skriver på sin webbplats är
partiets egen beskrivning: skriv "M uppger på sin webbplats att regeringen har
…", aldrig som ett faktum. Avsluta aldrig med en sammanfattande
värdering av vilket parti som har rätt."""


# Step 2: what it knows. Q&A pairs that teach the model what the material
# means, so it doesn't read a missing page or motivering as a data error.
KNOWLEDGE = [
    ("Vad betyder det att ett parti saknar sida om ett ämne på sin webbplats?",
     "Bara att ämnet inte finns i partiets eget A–Ö-register. Det betyder "
     "inte att partiet saknar åsikt i frågan. Skriv att partiets webbplats "
     "inte tar upp frågan, och dra ingen slutsats om partiets uppfattning."),
    ("Vad är ett partiprogram, och spelar det roll hur gammalt det är?",
     "Partiprogrammet (även idé- eller principprogram) är partiets antagna "
     "beskrivning av sin ideologi och sina grundvärderingar, beslutad av "
     "partiets kongress eller stämma. Det är partiets gällande program oavsett "
     "vilket år det antogs — partier byter program sällan, så årtalet säger "
     "inget om huruvida det fortfarande gäller. Programmet är den bästa källan "
     "för frågor om ideologi. Det är samtidigt mer allmänt hållet än dagens "
     "politik: för konkreta, aktuella förslag säger motioner och webbplatser mer."),
    ("Vad är en motion?",
     "Ett formellt förslag som en eller flera riksdagsledamöter lämnar in "
     "i riksdagen, i sitt partis namn. Den innehåller ett eller flera "
     "yrkanden — konkreta beslutsförslag — och en motivering till varför. "
     "En motion är partiets egen, ofiltrerade begäran, inte ett resultat "
     "av förhandling med andra partier."),
    ("Vad är en proposition?",
     "Ett lagförslag eller annat förslag som regeringen lämnar till "
     "riksdagen. Under perioden består regeringen av M, KD och L, med SD som "
     "samarbetsparti utanför regeringen. En proposition är regeringens "
     "gemensamma förslag, ofta en kompromiss, och visar inte vad ett enskilt "
     "regeringsparti hade velat på egen hand. Budgetpropositionen samlar "
     "årets reformer. Regeringspartier lägger sällan egna motioner i frågor "
     "där regeringen redan lämnat förslag — att M, KD och L har få motioner "
     "är därför väntat."),
    ("Vilken roll har Sverigedemokraterna under perioden?",
     "SD sitter inte i regeringen men är samarbetsparti till M, KD och L "
     "genom Tidöavtalet (oktober 2022), som anger gemensamma reformer inom "
     "bl.a. migration, kriminalpolitik, energi, vård och skola, och ett "
     "budgetsamarbete. SD står inte bakom regeringens propositioner formellt: "
     "SD:s egna förslag finns i SD:s motioner, och det gemensamma i "
     "Tidöavtalet. Att en proposition genomför något ur Tidöavtalet framgår "
     "bara om materialet säger det."),
    ("Vad är skillnaden mellan ett yrkande och en motivering i en motion?",
     "Yrkandet är den korta, formella beslutsmeningen ('Riksdagen ställer "
     "sig bakom ...'). Motiveringen är resonemanget bakom den — varför "
     "partiet vill det. Materialet kan ibland sakna en kopplad motivering "
     "för ett yrkande, vilket bara betyder att kopplingen inte gick att "
     "fastställa automatiskt, inte att partiet saknar ett skäl."),
    ("Vad betyder det att ett parti lämnat in nästan identiska yrkanden flera år i rad?",
     "Bara att partiet återkommit till samma krav. Det kan bero på att det "
     "avslagits tidigare, att det inte kommit upp till behandling, eller "
     "andra skäl materialet inte visar. Dra ingen slutsats om utfallet — "
     "bara om att frågan är återkommande för partiet."),
]


# KNOWLEDGE goes in the system prompt, not as fake chat turns: it is standing
# background, and it then stays apart from the real conversation history.
SYSTEM += """

VAD DU VET OM RIKSDAGEN

Bakgrund om hur riksdagen och materialet fungerar. Använd den för att tolka
KONTEXT rätt. Den är ingen källa om partiernas politik.

""" + "\n\n".join(f"F: {q}\nS: {a}" for q, a in KNOWLEDGE)


# Chat memory: earlier questions and answers are sent again with each question
# (the API keeps no state), without their passages to save tokens.
HISTORY_TURNS = 20
# Passage numbers restart for every question, so old answers lose their citations.
CITATION = re.compile(r"\s*\[\d+(?:\s*[,–-]\s*\d+)*\]")


def history_turns(history):
    """Earlier (question, answer) pairs as user/model turns, citations removed."""
    turns = []
    for q, a in history[-HISTORY_TURNS:]:
        turns.append({"role": "user", "parts": [{"text": f"FRÅGA: {q}"}]})
        turns.append({"role": "model", "parts": [{"text": CITATION.sub("", a)}]})
    return turns


VOTING_ADVICE_NOTE = ("\n\nOBS: användaren ber om en partirekommendation. "
                      "Följ regel 3 — rekommendera inte, förklara kort "
                      "varför, och redovisa sedan neutralt vad partierna "
                      "säger i sakfrågan, om frågan nämner någon.\n")


def build_contents(question, context, voting_advice):
    """Gemini format: one user turn, the context before the question."""
    instruction = VOTING_ADVICE_NOTE if voting_advice else ""
    return [{"role": "user", "parts": [{"text":
        f"KONTEXT\n{context}\n\nSLUT PÅ KONTEXT{instruction}\n\n"
        f"FRÅGA: {question}"}]}]



# Gemini
def gemini_client():
    from google import genai
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("set GEMINI_API_KEY (key from https://aistudio.google.com)")
    return genai.Client(api_key=key)

# Tried in order when the main model is overloaded or over quota.
# Check the available names with `python answer.py models`.
FALLBACK_MODELS = ["gemini-3.6-flash"]
SWITCH_AFTER = 2        # transient failures on one model before moving on
# Which thinking variant each model accepted, saved so that a rejected variant
# (a wasted request) is only tried once.
THINKING_CACHE = "out/gemini_thinking.json"
try:
    _THINKING_OK = json.load(open(THINKING_CACHE))
except (OSError, ValueError):
    _THINKING_OK = {}


def _remember_thinking(model, v):
    if _THINKING_OK.get(model) != v:
        _THINKING_OK[model] = v
        try:
            json.dump(_THINKING_OK, open(THINKING_CACHE, "w"))
        except OSError:
            pass            # only an optimisation, never fail an answer over it


# Attempts per Gemini call. Kept low: when Google is overloaded, stopping is
# cheaper than retrying against a small daily quota.
MAX_ATTEMPTS = 2


def generate(client, model, contents, system=SYSTEM, attempts=MAX_ATTEMPTS,
             fallbacks=None, **config):
    """One generate_content request, with retries on transient errors and a
    switch to FALLBACK_MODELS when a model keeps failing. Returns (response,
    the model that answered). Extra keyword arguments (tools, tool_config, …)
    go into the GenerateContentConfig."""
    import random
    from google.genai import types

    def make_config(thinking):
        extra = {"thinking_config": thinking} if thinking is not None else {}
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,          # restating, not creativity
            max_output_tokens=MAX_OUTPUT_TOKENS,
            **config, **extra)

    # Models differ in which thinking setting they accept, and the API doesn't
    # say which argument was invalid, so the variants are tried in turn.
    variants = []
    if THINKING_LEVEL:
        variants.append(("thinking_level",
                         types.ThinkingConfig(thinking_level=THINKING_LEVEL)))
        variants.append(("thinking_budget",
                         types.ThinkingConfig(thinking_budget=0)))
    variants.append(("no thinking config", None))

    # Transient errors: 429 (quota) and 503 (overloaded). Jitter keeps group
    # members from retrying in lockstep.
    TRANSIENT = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE",
                 "500", "INTERNAL", "504", "DEADLINE_EXCEEDED")

    models = [model] + [m for m in (FALLBACK_MODELS if fallbacks is None else fallbacks)
                        if m != model]
    mi, failures = 0, 0
    v = _THINKING_OK.get(models[0], 0)
    for n in range(attempts):
        current = models[mi]
        name, thinking = variants[v]
        try:
            response = client.models.generate_content(
                model=current, contents=contents, config=make_config(thinking))
            _remember_thinking(current, v)
            return response, current
        except Exception as err:
            text = str(err)
            if "404" in text or "NOT_FOUND" in text:
                if mi + 1 == len(models):
                    sys.exit(f"{current} is not available to this key, and there is "
                             f"no fallback left. Run 'python answer.py models' and "
                             f"update MODEL / FALLBACK_MODELS in answer.py.")
                print(f"  ({current} is not available to this key, "
                      f"trying {models[mi + 1]})", file=sys.stderr)
                mi, failures = mi + 1, 0
                v = _THINKING_OK.get(models[mi], 0)
                continue
            if (("400" in text or "INVALID_ARGUMENT" in text)
                    and v + 1 < len(variants)):
                v += 1
                print(f"  ({name} rejected by {current}, "
                      f"trying {variants[v][0]})", file=sys.stderr)
                continue
            if not any(code in text for code in TRANSIENT):
                raise
            reason = ("quota limit" if "429" in text or "RESOURCE" in text
                      else "overload")
            code = next(c for c in TRANSIENT if c in text)
            failures += 1
            # A per-DAY quota won't recover by waiting (a per-minute one does):
            # move on at once, or stop if there's nothing left to move on to.
            daily = "PerDay" in text
            if daily and mi + 1 == len(models):
                sys.exit(f"the daily quota for {current} is used up, and there is "
                         f"no fallback left — try again tomorrow")
            if (daily or failures >= SWITCH_AFTER) and mi + 1 < len(models):
                why = "daily quota used up" if daily else f"{reason}: {code}"
                print(f"  ({why} on {current}, switching to {models[mi + 1]})",
                      file=sys.stderr)
                mi, failures = mi + 1, 0
                v = _THINKING_OK.get(models[mi], 0)
                continue
            if n == attempts - 1:
                break
            pause = min(60, 2 ** failures * 4) * (0.5 + random.random())
            print(f"  ({reason}: {code}, retrying in {pause:.0f}s …)",
                  file=sys.stderr)
            time.sleep(pause)

    sys.exit(f"gave up after {attempts} attempts ({', '.join(models[:mi + 1])}) — "
             f"wait a while and try again")


def answer_text(response):
    """The final text, with a warning if it was cut off. Half an answer is
    worse than none: it looks finished but lacks both the conclusion and the
    source references."""
    finish = ""
    if getattr(response, "candidates", None):
        finish = str(getattr(response.candidates[0], "finish_reason", "") or "")
    if "MAX_TOKENS" in finish:
        print(f"  WARNING: the answer was cut off at {MAX_OUTPUT_TOKENS} tokens. "
              f"Raise MAX_OUTPUT_TOKENS or lower MAX_CONTEXT_CHARS.",
              file=sys.stderr)
    if not response.text:
        print(f"  (empty answer, finish_reason={finish or 'unknown'})",
              file=sys.stderr)
        return f"[no answer was generated — finish_reason: {finish or 'unknown'}]"
    # The prompt forbids the layer names, but Gemini copies them from the context.
    return LAYER_LABEL_RE.sub("", response.text)


def ask_gemini(contents, model=MODEL):
    """The one-call path: all material is already in `contents`.
    Returns (answer text, the model that answered)."""
    response, used = generate(gemini_client(), model, contents)
    return answer_text(response), used


def list_models():
    from google import genai
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("set GEMINI_API_KEY")
    # Keep the client alive: models.list() fetches its pages lazily.
    client = genai.Client(api_key=key)
    for m in list(client.models.list()):
        actions = getattr(m, "supported_actions", None) or []
        if not actions or "generateContent" in actions:
            print(f"  {m.name}")


# main flow

# Passages per layer and party. Each party and layer is searched separately,
# since website prose otherwise outranks short motion text. Motions get the
# most room: they are the party's concrete proposals.
LAYER_QUOTA = {
    "one_party": {"program": 3, "motion": 8, "proposition": 3, "said": 3},
    # Bills are shared by M, KD and L, so 2 each gives a government question 6.
    "several_parties": {"program": 1, "motion": 2, "proposition": 2, "said": 1},   # per party
}
# Relevance cutoff on cosine similarity: within MAX_DISTANCE of the best hit
# and above a floor. An empty slot stays empty rather than holding a weak hit.
# The floors were set by hand on a few test questions; e5's similarities are
# tightly packed (0.80–0.90), and a reranker would be the next step.
MAX_DISTANCE = 0.04
# Enskilda motions per party. 0 = never: they are one or a few members'
# wishes, often local, not the party's line.
MAX_ENSKILDA = 0
# The proposition floor is copied from motion and not calibrated.
MIN_SIMILARITY = {"program": 0.80, "motion": 0.80, "proposition": 0.80, "said": 0.80}


def retrieve(idx, question, parties, layers=ALL_LAYERS, method="hybrid",
             searched=None):
    """Retrieval with fixed slots: (hits, gaps), where gaps are the
    (party, layer) pairs that got no hit above the relevance cutoff.

    searched: every layer searched for this question (tool calls in parallel).
    The per-party budget — sum(quota) — stays the same however many layers
    are searched, and is spread over them in proportion to their quota: two
    layers get more slots each than four do, and a lone layer gets all."""
    quota = LAYER_QUOTA["one_party" if len(parties) == 1 else "several_parties"]
    searched = [l for l in (searched or layers) if l in quota]
    budget = sum(quota[l] for l in ALL_LAYERS)
    weight = sum(quota[l] for l in searched)
    layers = {l: math.ceil(quota[l] * budget / weight) for l in layers}

    slots = {}
    for party in parties or list(PARTY_NAMES):
        for one_layer, n in layers.items():
            if one_layer == "proposition" and party not in GOVERNMENT:
                continue
            slots[party, one_layer] = (n, idx.search(
                question, k=n + 20, party=[party], layer=one_layer, method=method))

    # Per-layer cutoff: website prose scores higher than motion text, so a
    # shared cutoff would drop the motions.
    best = defaultdict(float)
    for (_, one_layer), (_, candidates) in slots.items():
        for _, m in candidates:
            best[one_layer] = max(best[one_layer], m.get("similarity", 0.0))

    seen, hits, gaps = set(), [], []
    for (party, one_layer), (n, candidates) in slots.items():
        cutoff = max(MIN_SIMILARITY[one_layer], best[one_layer] - MAX_DISTANCE)
        taken = singles = 0
        for score, m in candidates:
            if taken == n:
                break
            if "similarity" in m and m["similarity"] < cutoff:
                continue        # sorted by RRF, not similarity: keep looking
            if enskild(m):
                if singles == MAX_ENSKILDA:
                    continue
                singles += 1
            # Bills (M, KD, L) and the Tidö Agreement (M, KD, L, SD) are shared:
            # each party's slot takes a new chunk.
            if m["id"] in seen:
                continue
            taken += 1
            seen.add(m["id"])
            hits.append((score, m))
        # Government parties have no group motions: an empty slot is no gap.
        if not taken and not (one_layer == "motion" and party in GOVERNMENT):
            gaps.append((party, one_layer))
    hits.sort(key=lambda t: -t[1].get("similarity", t[0]))
    return hits, gaps


def answer(idx, question, method="hybrid", model=MODEL, dry_run=False, layer=None):
    parties = parties_in_question(question)
    layers = [layer] if layer else ALL_LAYERS
    voting_advice = bool(VOTING_ADVICE_WORDS.search(question))
    hits, gaps = retrieve(idx, question, parties, layers, method)
    size = "one_party" if len(parties) == 1 else "several_parties"
    context = build_context(hits, MAX_ARGUMENT_CHARS[size])
    if gaps:
        context += "\n\nINGET RELEVANT MATERIAL HITTADES FÖR\n" + "\n".join(
            f"- {PARTY_NAMES[p]} ({p}): {LAYER_NAMES_SV[l]}" for p, l in gaps)
    contents = build_contents(question, context, voting_advice)

    info = {"parties": parties or "all", "layers": ",".join(layers),
            "hits": len(hits), "gaps": len(gaps), "voting_advice": voting_advice,
            "context_chars": len(context)}

    if dry_run:
        return None, hits, info, contents
    text, info["model"] = ask_gemini(contents, model)
    return text, hits, info, contents


# How a missing layer is named in the prompt (Swedish: Gemini reads it).
LAYER_NAMES_SV = {"said": "hemsidan",
                  "motion": "motioner", "program": "partiprogrammet",
                  "proposition": "propositioner"}


# tool calling: Gemini decides which layers and parties to search


# Gemini makes all its searches as parallel calls in ONE step, then answers:
# 2 API calls per question. One extra search step is allowed (3 calls); after
# that the model must answer. With ~20 calls a day, every call counts.
TOOL_ROUNDS = 2

# Added to SYSTEM in tool mode (Swedish: Gemini reads it).
TOOL_INSTRUCTIONS = """

SÖKNING: Du har inget material från början. Du hämtar det med verktyget sok,
och sökresultaten är din KONTEXT.

- Gör ALLA sökningar du behöver i ett och samma steg, som parallella anrop.
  Du får högst ett steg till för kompletterande sökningar, sedan svarar du.
- lager: UTELÄMNA lager, så söks alla lager på en gång (program =
  partiprogram, motion = motioner, proposition = regeringens
  propositioner, said = webbplatser). Ange lager bara när användaren uttryckligen
  begränsar källorna ("bara motioner", "inga webbplatser"). Att frågan
  gäller ideologi är inget skäl att begränsa: det avgör du när du skriver
  svaret, inte när du söker. Omröstningar finns inte i materialet just nu.
- partier: partikoderna (S, M, SD, C, V, KD, MP, L) för de partier frågan
  gäller, även när de står i genitiv ("Moderaternas" = M). Utelämna för en
  jämförelse mellan alla partier.
- fraga: en fullständig, naturlig fråga på SVENSKA om sakfrågan, även när användaren skriver på ett annat språk (materialet är på svenska), t.ex.
  "Vad tycker partierna om migration och asylpolitik?" — INTE ett enstaka
  sökord. Sökningen är byggd för hela frågor; ett ensamt ord som
  "migration" ger så låg träffsäkerhet att relevant material sorteras bort.
- Hänvisa till passagerna med numren i sökresultaten."""


def search_tool():
    from google.genai import types
    return types.Tool(function_declarations=[types.FunctionDeclaration(
        name="sok",
        description="Söker i materialet om svensk partipolitik, i alla lager "
                    "om lager utelämnas. Returnerar numrerade passager att hänvisa till.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                # A full question, not a keyword: e5 scores single words far
                # lower (0.77–0.82 vs 0.83–0.86), below the relevance cutoff.
                "fraga": types.Schema(
                    type="STRING",
                    description="En fullständig, naturlig fråga om sakfrågan, "
                                "t.ex. 'Vad tycker partierna om migration och "
                                "asylpolitik?'. Inte ett enstaka sökord."),
                "lager": types.Schema(
                    type="STRING", enum=ALL_LAYERS,
                    description="Utelämna för att söka alla lager. Ange bara när "
                                "användaren uttryckligen begränsar källorna. "
                                "program = partiprogram, motion = motioner, "
                                "said = webbplatser."),
                "partier": types.Schema(
                    type="ARRAY",
                    items=types.Schema(type="STRING", enum=list(PARTY_NAMES)),
                    description="Partikoder att söka för. Utelämna för alla partier."),
            },
            required=["fraga"]))])


def call_layers(args):
    """The layers one 'sok' call searches: all of them unless 'lager' is given."""
    return [args["lager"]] if args.get("lager") else ALL_LAYERS


def run_search(idx, args, shown, method="hybrid", searched=None):
    """Executes one 'sok' call and returns the text Gemini reads, plus stats.
    `shown` holds the passages returned so far for this question; new ones are
    numbered after them, so the numbers stay unique across searches."""
    layers = call_layers(args)
    if not set(layers) <= set(ALL_LAYERS):
        return f"Okänt lager: {args['lager']}. Välj {', '.join(ALL_LAYERS)} eller utelämna.", 0
    parties = [p for p in (args.get("partier") or []) if p in PARTY_NAMES]
    hits, gaps = retrieve(idx, args.get("fraga") or "", parties, layers, method,
                          searched=searched)

    already = {m["id"] for _, m in shown}
    new = [(score, m) for score, m in hits if m["id"] not in already]
    start = len(shown) + 1
    shown.extend(new)

    size = "one_party" if len(parties) == 1 else "several_parties"
    text = build_context(new, MAX_ARGUMENT_CHARS[size], limit=None, start=start)
    if gaps:
        text += "\n\nINGET RELEVANT MATERIAL HITTADES FÖR\n" + "\n".join(
            f"- {PARTY_NAMES[p]} ({p}): {LAYER_NAMES_SV[l]}" for p, l in gaps)
    if not new and hits:
        text += "\n\n(Alla träffar i den här sökningen har redan visats ovan.)"
    return text, len(gaps)


def answer_with_tools(idx, question, method="hybrid", model=MODEL, history=()):
    """Gemini searches with the 'sok' tool, then answers. A manual loop, not
    the SDK's automatic function calling: we control the number of API calls
    and keep the passage numbers consistent across searches."""
    from google.genai import types

    client = gemini_client()
    voting_advice = bool(VOTING_ADVICE_WORDS.search(question))
    note = VOTING_ADVICE_NOTE if voting_advice else ""
    # Earlier exchanges let a follow-up ("och KD då?") be understood: Gemini
    # writes the search question itself, so it can fill in party and topic.
    contents = history_turns(history) + [
        {"role": "user", "parts": [{"text": f"FRÅGA: {question}{note}"}]}]

    shown, searches, calls = [], [], 0
    gaps = context_chars = 0
    for step in range(TOOL_ROUNDS + 1):
        # The first step must search: left to choose, Gemini sometimes answered
        # a voting-advice question from memory, without a single source.
        # The last step may not search: it must answer with what it has.
        mode = "ANY" if step == 0 else "NONE" if step == TOOL_ROUNDS else "AUTO"
        # After a fallback, stay on that model: thought signatures are model-specific.
        response, model = generate(
            client, model, contents, system=SYSTEM + TOOL_INSTRUCTIONS,
            fallbacks=None if step == 0 else [],
            tools=[search_tool()],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=mode)),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        calls += 1
        function_calls = response.function_calls or []
        if not function_calls:
            break

        # The model's turn goes back unchanged: Gemini 3 needs its "thought
        # signatures" in the history to continue after a function call.
        contents.append(response.candidates[0].content)
        # Parallel searches share one quota, so three searches don't fetch
        # three times the material.
        layers_this_step = {l for fc in function_calls for l in call_layers(dict(fc.args or {}))}
        # Unsearched layers are named, so Gemini doesn't report them as missing.
        unsearched = [l for l in ALL_LAYERS if l not in layers_this_step]
        results = []
        for i, fc in enumerate(function_calls):
            args = dict(fc.args or {})
            searches.append(f"{args.get('lager') or 'alla lager'}:{','.join(args.get('partier') or []) or 'alla'}"
                            f" \"{args.get('fraga', '')}\"")
            text, n_gaps = run_search(idx, args, shown, method,
                                               searched=sorted(layers_this_step))
            if unsearched and i == len(function_calls) - 1:
                text += ("\n\nEJ SÖKTA LAGER: " + ", ".join(LAYER_NAMES_SV[l] for l in unsearched)
                         + ". Påstå inte att material saknas där — det har inte sökts.")
            if i == len(function_calls) - 1:
                # After Swedish results Gemini drifts into Swedish; the prompt
                # rule alone didn't hold.
                text += (f"\n\nSVARSSPRÅK: skriv svaret på samma språk som frågan "
                         f"\"{question}\" är skriven på, inte på materialets språk.")
            gaps += n_gaps
            context_chars += len(text)
            results.append(types.Part.from_function_response(
                name=fc.name, response={"result": text}))
        contents.append(types.Content(role="user", parts=results))

    info = {"parties": "chosen by Gemini", "layers": " ".join(searches) or "none",
            "hits": len(shown), "gaps": gaps, "voting_advice": voting_advice, "context_chars": context_chars,
            "api_calls": calls, "model": model}
    return answer_text(response), shown, info, contents


def print_answer(response, hits, info, contents, dry_run):
    print(f"\n[{info['parties']} | layers: {info['layers']} | "
          f"{info['hits']} hits | {info['gaps']} gaps | "
          f"{info['context_chars']} chars"
          + (f" | {info['api_calls']} API calls" if "api_calls" in info else "")
          + (f" | answered by {info['model']}"
             if info.get("model") and info["model"] != MODEL else "")
          + f"{' | VOTING ADVICE' if info['voting_advice'] else ''}]\n")
    if dry_run:
        print(contents[-1]["parts"][0]["text"])
        return
    print(response)
    print("\nSources:")
    for line in source_list(hits, response):
        print(" ", line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", help="the question, or 'chat', or 'models'")
    ap.add_argument("--layer", choices=ALL_LAYERS, default=None)
    ap.add_argument("--method", choices=["hybrid", "bm25", "vector"],
                    default="hybrid")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--index", default=rag_index.INDEX_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="build the one-call prompt and print it, don't call the API")
    ap.add_argument("--no-tools", action="store_true",
                    help="one API call: retrieve everything up front instead of "
                         "letting Gemini choose its searches")
    a = ap.parse_args()

    if a.question == "models":
        list_models()
        return

    # Tool calling is the default; --dry-run and --layer use the one-call path.
    one_call = a.no_tools or a.dry_run or a.layer

    def ask(q):
        if one_call:
            return answer(idx, q, method=a.method, model=a.model,
                          dry_run=a.dry_run, layer=a.layer)
        return answer_with_tools(idx, q, method=a.method, model=a.model)

    idx = rag_index.Index(a.index)
    if a.question == "chat":
        print(f"{idx.info['n']} passages. An empty line quits.")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            print_answer(*ask(q), dry_run=a.dry_run)
        return

    print_answer(*ask(a.question), dry_run=a.dry_run)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\ninterrupted")
