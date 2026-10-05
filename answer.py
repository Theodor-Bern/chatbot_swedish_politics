"""
answer.py — the answering step. This is where the LLM comes in, and nowhere else.

The chain:
    question
      -> parties_in_question()  parameter extraction: "moderaterna" -> M
      -> retrieve()             retrieval, with fixed slots per party and layer
      -> vote_facts()           looks up the votes for the retrieved punkter
      -> build_context()        all of the above as text, with a source id on each piece
      -> Gemini                 phrases the answer. The model may NEVER invent facts.

The system prompt follows the lecture's four steps: who it is, what it knows
(the Q&A pairs that teach it the Riksdag's semantics), which data to use, and
how to answer. Everything Gemini reads is Swedish on purpose — the bot
answers in Swedish.

Needs a key from Google AI Studio:
    export GEMINI_API_KEY=...        # NEVER put the key in the repo

Run:
    python answer.py models                         # what the key has access to
    python answer.py "Vad tycker V om kärnkraft?"
    python answer.py "Vad tycker partierna om klimatet?"
    python answer.py "Har M drivit sin kärnkraftslinje?" --dry-run
    python answer.py chat
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from functools import lru_cache
from collections import defaultdict

import rag_index

try:
    import party_positions
except ImportError:
    party_positions = None

MODEL = "gemini-3.5-flash-lite"
MAX_CONTEXT_CHARS = 40000
# A motion's argument can be several thousand characters. With 8 parties in the
# prompt, the start is enough to show what the yrkande is about, and more would
# crowd out other parties. A single-party question has room for more of the
# reasoning — which is what "vad innebär de i praktiken?" needs.
MAX_ARGUMENT_CHARS = {"one_party": 1500, "several_parties": 600}
MAX_OUTPUT_TOKENS = 4096
# Gemini 3 "thinks" before answering, and the thinking counts against
# max_output_tokens. We want the given text restated, not reasoning — that
# saves both quota and waiting time. None = let the model decide.
THINKING_LEVEL = "MINIMAL"
POSITIONS_DID = "out/positions_did.jsonl"
DID_CHUNKS = rag_index.DID_CHUNKS
MAX_VOTE_POINTS = 6           # betänkande punkter in the vote block
MAX_RESERVATION_CHARS = 350   # characters from each reservation's ställningstagande


PARTY_NAMES = rag_index.PARTY_NAMES

# Parameter extraction: free text -> party code. A deterministic table instead
# of an LLM call — party names are a closed set and need no model.
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
    r"vilket parti (ska|bör) jag|passar mig|rösta på|vem ska jag rösta|"
    r"rekommendera(r)? du|vad ska jag rösta|hjälp mig välja", re.IGNORECASE)


# --------------------------------------------------------------------------
# parameter extraction
# --------------------------------------------------------------------------

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


# Every question searches all layers. There's deliberately no keyword routing:
# guessing layers from words misfired ("använd inga betänkanden" routed TO
# betänkanden), and a question that mentions votes still deserves the party's
# motions and website for context. Gemini reads the question and decides what
# to emphasise; --layer on the CLI restricts retrieval by hand.
# ponytail: "did" (votes) switched off for now; add it back here to re-enable.
ALL_LAYERS = ["program", "motion", "said"]


# --------------------------------------------------------------------------
# vote facts for the punkter that retrieval hit
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def punkt_texts():
    """What makes a vote understandable, per betänkande punkt: its heading,
    the committee's proposal in brief, and what each reservation wanted (its
    ställningstagande). From the same chunks.jsonl as the DID layer."""
    heading, summary, reservation = {}, {}, {}
    for line in open(DID_CHUNKS, encoding="utf-8"):
        d = json.loads(line)
        doc = (d["rm"], d["beteckning"])
        punkter = [p for p in str(d.get("punkter") or "").split(";") if p]
        if d["section"] == "decision" and punkter:
            heading[doc + (punkter[0],)] = d["heading"]
        elif d["section"] == "summary":
            for p in punkter:
                summary.setdefault(doc + (p,), d["text"])
        elif d["section"] == "reservation" and d.get("number"):
            text = d["text"]
            i = text.find("Ställningstagande")
            if i != -1:
                text = text[i + len("Ställningstagande"):].strip()
            reservation[doc + (d["number"],)] = text
    return heading, summary, reservation


def shorten(text, n):
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def vote_facts(hits, parties):
    """One block per betänkande punkt that retrieval hit: what the punkt was
    about, what the reservations wanted, and how each party voted.

    A bare line like "L röstade mot reservation 1 (S)" says nothing unless you
    know what reservation 1 proposed — so the reservations' content sits in
    the same block as the votes. The block text is Swedish: Gemini reads it.

    Without an explicit party filter ALL parties are looked up, not just the
    ones that happen to be tagged on the hit's reservation. Government parties
    practically never file reservations (see KNOWLEDGE), so without this their
    "voted with the majority" stance would never show.
    """
    if party_positions is None or not os.path.exists(POSITIONS_DID):
        return []
    positions = party_positions.load_positions(POSITIONS_DID)
    heading, summary, reservation = punkt_texts()
    seen, blocks = set(), []
    for _score, m in hits:
        if m["layer"] != "did" or not m.get("punkt"):
            continue
        key = f"{m['rm']}|{m['beteckning']}|{m['punkt']}"
        row = positions.get(key)
        if row is None or key in seen:
            continue
        seen.add(key)
        punkt = (m["rm"], m["beteckning"], m["punkt"])

        dates = sorted({v["datum"] for v in row["voteringar"] if v.get("datum")})
        lines = [f"[{m['beteckning']} {m['rm']} punkt {m['punkt']}] "
                 f"{m.get('doc_title', '')} — {heading.get(punkt, '')}".rstrip(" —")
                 + (f" (votering {', '.join(dates)})" if dates else "")]
        if punkt in summary:
            lines.append(f"Utskottets förslag: {summary[punkt]}")
        for r in row["reservationer"]:
            wanted = reservation.get(punkt[:2] + (r["number"],))
            if wanted:
                lines.append(f"Reservation {r['number']} ({', '.join(r['partier'])}) "
                             f"ville: {shorten(wanted, MAX_RESERVATION_CHARS)}")
        if row["status"] == "no_vote":
            lines.append("Ingen votering hölls; punkten avgjordes med acklamation, "
                         "så inget parti tog ställning i en omröstning. Reservationerna "
                         "ovan visar vad partierna bakom dem ville.")
        else:
            lines.append("Så agerade partierna:")
            for p in (parties or PARTY_NAMES):
                described = party_positions.describe(row, p)
                # A party with its own reservation abstains when a different
                # reservation is being voted on — pure procedure, it says nothing
                # about the issue. The party's position is its own reservation,
                # which is already listed above.
                stances = {v["partier"][p]["stance"] for v in row["voteringar"]
                           if p in v["partier"]}
                if described["own_reservations"] and stances == {"Avstår"}:
                    continue
                lines.append(f"- {described['text']}")
        blocks.append("\n".join(lines))
        if len(blocks) >= MAX_VOTE_POINTS:
            break
    return blocks


# --------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------

def build_context(hits, votes, argument_chars, limit=MAX_CONTEXT_CHARS, start=1):
    """The retrieved material as text. Passages are numbered from `start`;
    limit=None keeps every passage (tool results: a numbered passage must
    never be dropped, or a cited number would point at nothing)."""
    parts, n = [], 0
    if votes:
        block = "OMRÖSTNINGAR I RIKSDAGEN\n\n" + "\n\n".join(votes)
        parts.append(block)
        n += len(block)

    parts.append("\nHÄMTADE TEXTER")
    for i, (_score, m) in enumerate(hits, start):
        if m.get("layer") == "motion":
            source = f"motion {m.get('beteckning', '')} {m.get('rm', '')}"
        elif m.get("layer") == "program":
            source = f"{m.get('doc_type', 'partiprogram')}, s. {m.get('page')}"
        else:
            source = m.get("url") or f"{m.get('beteckning', '')} {m.get('rm', '')}"
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


def source_list(hits):
    """One line per source, with every passage number that points to it:
    several passages from the same page/motion must not leave a cited number
    missing from the list. Numbering matches build_context: hits[0] is [1]."""
    numbers = {}
    for i, (_score, m) in enumerate(hits, 1):
        if m.get("layer") == "motion":
            source = (f"motion {m.get('rm', '')}:{m.get('beteckning', '')} "
                      f"({m.get('parti') or '?'}) – {m.get('doc_title', '')}").rstrip(" –")
        elif m.get("layer") == "program":
            year = f" {m['year']}" if m.get("year") else ""
            source = (f"{m.get('doc_type', 'partiprogram')} ({m.get('parti')}){year}, "
                      f"p. {m.get('page')} – {m.get('heading', '')}").rstrip(" –")
        elif m.get("url"):
            # Parties edit their pages without notice: show how current it is.
            edited = f" (last edited {m['lastmod'][:10]})" if m.get("lastmod") else ""
            source = m["url"] + edited
        else:
            punkt = f" punkt {m['punkt']}" if m.get("punkt") else ""
            source = f"{m.get('beteckning', '')} {m.get('rm', '')}{punkt}"
        numbers.setdefault(source, []).append(str(i))
    return [f"[{', '.join(n)}] {s}" for s, n in numbers.items()]


# --------------------------------------------------------------------------
# system prompt — the lecture's steps 1 and 4 (Swedish: Gemini reads it)
# --------------------------------------------------------------------------

SYSTEM = """Du är en expert på svensk politik. Din uppgift är att leverera pricksäker information om de 8 olika riksdagspartierna i Sverige.

Det finns fyra olika sorters material. 

  PROGRAM = partiets IDEOLOGI och grundvärderingar. Partiets antagna parti-
            eller principprogram.
  MOTION = vad ett parti FÖRESLAGIT. Formella yrkanden i motioner till riksdagen.
  SAID   = vad ett parti SÄGER. Hämtat från partiets egen webbplats.
  DID    = vad som FAKTISKT HÄNT i riksdagen. Betänkanden och omröstningar.

REGLER, i fallande ordning:

1. Du använder ENDAST det material du får i KONTEXT. Du har egna minnen av
   svensk politik — de är föråldrade och du använder dem inte. Om kontexten
   inte räcker säger du det rent ut. Du sätter inga egna etiketter på ett
   parti ("marknadsliberal", "socialistisk", "nationalistisk"): använd bara
   beteckningar som partiet själv använder i källan, och beskriv annars
   ståndpunkterna i stället för att klassificera dem.

2. Varje sakpåstående följs av sin källa i hakparentes: [3], eller
   [AU10 2022/23 punkt 1]. Ett påstående utan källa får inte skrivas.

3. Du rekommenderar ALDRIG ett parti och rangordnar dem aldrig. Du säger
   inte vilket parti som "passar" någon, oavsett hur användaren beskriver
   sig. Om du blir ombedd förklarar du kort att du redovisar vad partierna
   säger och gör, och erbjuder en jämförelse i en sakfråga användaren väljer.

4. Använd bara källor som faktiskt besvarar frågan. En hämtad passage som
   handlar om något annat utelämnar du, även om den är det enda du har för
   ett parti — använd aldrig en källa om ett annat ämne som om den handlade
   om det efterfrågade. Saknas material fyller du aldrig luckan med
   gissningar. Luckorna samlar du i EN kort mening i slutet av svaret
   ("I materialet saknas motioner från C och KD och omröstningar om frågan
   för samtliga partier."), inte en gång per parti. Ett parti som står med
   under "Så agerade partierna" i omröstningsblocket HAR material om den
   omröstningen — skriv aldrig att omröstningar saknas för det partiet.


5. En MOTION är ett förslag partiet lämnat in — inte ett beslut och inte en
   lag. Skriv alltid "X har föreslagit" eller "X vill", ALDRIG "X har
   genomfört" för ett motionsyrkande.

6. En omröstning redovisar du alltid med datum och vad som prövades, hämtat
   ur omröstningsblocket: "L röstade i april 2023 mot S:s reservation, som
   bl.a. ville att regeringen skulle ta fram ett lagförslag om kunskapskrav
   för permanent uppehållstillstånd [SfU15 2022/23 punkt 1]". ALDRIG bara
   "L röstade mot reservation 1 (S)", och ALDRIG "L är emot kunskapskrav"
   utifrån en sådan röst — en nej-röst mot en reservation är inte ett nej
   till sakfrågan (se vad du vet om omröstningar). Framgår det inte vad som
   prövades, utelämnar du omröstningen. Ett partis egen reservation ÄR dess
   ställning på punkten: redovisa vad den ville, inte hur partiet röstade om
   andra partiers reservationer. Visar senare material en annan
   hållning redovisar du båda i tidsordning, utan att kalla det åsiktsbyte.

 7. Använd primärt PROGRAM, SAID och MOTION. Använd endast DID om användaren frågar efter voteringar eller reservationer,
     
8. Gäller frågan ETT parti: ordna svaret efter materialtyp under rubrikerna
   Grundsyn (PROGRAM), Vad partiet säger (SAID), Vad partiet föreslagit
   (MOTION), Vad som hänt i riksdagen (DID), och hoppa över rubriker du
   saknar material för. Jämför du FLERA partier: INGA rubriker per
   materialtyp. Ett stycke per parti, partierna i samma ordning, och inom
   stycket källtyperna i ordningen ovan.

FORM: svar på svenska, i löpande text. Gäller frågan ett parti: två till fem
stycken. Jämför du flera partier tar du ett stycke per parti, i samma ordning
varje gång fånga partiets huvudsakliga
linje, inte bara en detalj, så länge materialet räcker till det. Inom varje
parti håller du isär källtyperna och säger i meningen vilken det är, i den
här ordningen: partiprogram (PROGRAM), webbplats (SAID), motioner (MOTION),
agerande i riksdagen (DID). Ta BARA med de källtyper som faktiskt besvarar
frågan — ordningen är ingen checklista, och en motion eller omröstning om en
annan sakfråga hör inte hemma i ett svar om ideologi. Slå aldrig ihop källor
av olika typ i samma påstående. Avsluta aldrig med en sammanfattande
värdering av vilket parti som har rätt."""


# Step 2: what it knows. Q&A pairs that teach the model the Riksdag's
# semantics — without them it reads "avstod" and "acklamation" as data errors.
KNOWLEDGE = [
    ("Vad är en reservation i ett betänkande?",
     "En reservation är ett avvikande förslag från en minoritet i utskottet. "
     "Den talar om vad de partier som står bakom den ville i stället för "
     "utskottets förslag. Reservationer är därför den tydligaste källan till "
     "ett partis position i ett enskilt ärende."),
    ("Vad betyder det att en punkt avgjordes med acklamation?",
     "Att ingen omröstning begärdes. Riksdagen sa ja utan votering. Det är "
     "helt normalt — omkring två tredjedelar av alla punkter avgörs så — och "
     "det betyder att det inte finns några röstsiffror att redovisa. Det är "
     "inte en lucka i materialet."),
    ("Vad ställs mot vad i en votering?",
     "Alltid utskottets förslag mot EN reservation. Ja betyder stöd för "
     "utskottets förslag, nej betyder stöd för reservationen. Ett rått "
     "\"nej\" säger därför ingenting förrän man vet vilken reservation som "
     "prövades."),
    ("Varför har Moderaterna, Kristdemokraterna och Liberalerna så få "
     "reservationer i materialet?",
     "De är regeringspartier under perioden. Regeringspartier reserverar sig "
     "i praktiken aldrig mot sitt eget utskotts förslag. Deras position syns "
     "i utskottets förslag i stället, inte i reservationer. Att ett "
     "regeringsparti saknar reservationer är alltså väntat och säger "
     "ingenting om partiets aktivitet."),
    ("Betyder en röst mot en reservation att partiet är emot det "
     "reservationen föreslår?",
     "Nej, inte i sig. En reservation begär ofta ett tillkännagivande: att "
     "regeringen ska göra något. Majoriteten röstar ofta nej för att arbetet "
     "redan pågår (en utredning, en kommande proposition), för att frågan "
     "hanteras i ett annat ärende, eller för att reservationen buntar ihop "
     "flera krav. Regeringspartier röstar i praktiken alltid nej till "
     "oppositionens reservationer. En nej-röst visar bara att partiet inte "
     "ville fatta just det beslutet vid den tidpunkten. Att partiet är emot "
     "själva sakfrågan kan bara skrivas om annat material säger det."),
    ("Vad är ett särskilt yttrande?",
     "En kommentar som en ledamot fogar till betänkandet utan att yrka på "
     "något annat beslut. Särskilda yttranden röstas aldrig om."),
    ("Vad betyder det att ett parti saknar sida om ett ämne på sin webbplats?",
     "Bara att ämnet inte finns i partiets eget A–Ö-register. Det betyder "
     "inte att partiet saknar åsikt i frågan. Skillnaden ska alltid skrivas "
     "ut: materialet saknas, slutsatsen om partiets uppfattning uteblir."),
    ("Vad betyder det att ett parti avstod i en omröstning?",
     "Att partiet varken stödde utskottets förslag eller reservationen som "
     "prövades. Det är ofta ett taktiskt val när partiet har en egen "
     "reservation som inte var uppe i just den omröstningen."),
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
     "av förhandling med andra partier (till skillnad från ett betänkande)."),
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


def knowledge_turns():
    """KNOWLEDGE as alternating user/model turns (Gemini's chat format)."""
    turns = []
    for q, a in KNOWLEDGE:
        turns.append({"role": "user", "parts": [{"text": q}]})
        turns.append({"role": "model", "parts": [{"text": a}]})
    return turns


VOTING_ADVICE_NOTE = ("\n\nOBS: användaren ber om en partirekommendation. "
                      "Följ regel 3 — rekommendera inte, förklara kort "
                      "varför, och erbjud en sakfrågejämförelse.\n")


def build_contents(question, context, voting_advice):
    """Gemini format: alternating user/model turns, the context last."""
    instruction = VOTING_ADVICE_NOTE if voting_advice else ""
    return knowledge_turns() + [{"role": "user", "parts": [{"text":
        f"KONTEXT\n{context}\n\nSLUT PÅ KONTEXT{instruction}\n\n"
        f"FRÅGA: {question}"}]}]


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------

def gemini_client():
    from google import genai
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("set GEMINI_API_KEY (key from https://aistudio.google.com)")
    return genai.Client(api_key=key)


# Tried in order when the main model is overloaded (or over quota — the free
# tier counts quota per model). A model this key can't use (404 NOT_FOUND) is
# skipped at once. Check the names with `python answer.py models`.
# gemini-2.5-flash is closed to new keys (404); 3.6-flash was the main model before 3.8.
FALLBACK_MODELS = ["gemini-3.6-flash"]
SWITCH_AFTER = 2        # transient failures on one model before moving on
# Which thinking variant each model accepted (index into `variants` below).
# Saved to disk: a rejected variant is a request that counts against the
# free tier's daily quota, so it must be discovered once, not on every run.
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
            pass            # only an optimisation — never fail an answer over it


# Attempts per Gemini call, across all models. Kept low on purpose: it's not
# documented whether failed (503) attempts count against the free tier's ~20
# requests a day, so when Google is overloaded it's cheaper to stop and try
# again later than to keep retrying.
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

    # Gemini 3 controls thinking with thinking_level, older models with
    # thinking_budget, and some allow neither. The API only answers "invalid
    # argument" without saying which argument was wrong, so we try the
    # variants in turn instead of parsing the text.
    variants = []
    if THINKING_LEVEL:
        variants.append(("thinking_level",
                         types.ThinkingConfig(thinking_level=THINKING_LEVEL)))
        variants.append(("thinking_budget",
                         types.ThinkingConfig(thinking_budget=0)))
    variants.append(("no thinking config", None))

    # Transient errors, both common on the free tier:
    #   429 RESOURCE_EXHAUSTED  we've hit the quota
    #   503 UNAVAILABLE         Google is overloaded right now
    # Jitter, so three group members running at the same time don't retry
    # in lockstep.
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
    return response.text


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
    # The client must stay alive while we iterate: models.list() is a lazy
    # pager that fetches pages only on iteration, and a temporary client gets
    # closed before then ("client has been closed").
    client = genai.Client(api_key=key)
    for m in list(client.models.list()):
        actions = getattr(m, "supported_actions", None) or []
        if not actions or "generateContent" in actions:
            print(f"  {m.name}")


# --------------------------------------------------------------------------
# main flow
# --------------------------------------------------------------------------

# How many passages each layer gets per party. Every party and layer is
# searched separately: in a shared pool the websites' polished prose almost
# always beats short motion yrkanden on pure vector similarity. Motions weigh
# most — they are the party's concrete, dated proposals; the website is
# self-description.
# A single party has room for more: the prompt for one party used ~5,500 of
# 40,000 characters, and "vilka motioner har M gjort…" got only 4 motions. The
# relevance cutoff still drops weak hits, so a bigger quota adds room, not noise.
LAYER_QUOTA = {
    "one_party": {"program": 3, "motion": 8, "did": 4, "said": 3},
    "several_parties": {"program": 1, "motion": 2, "did": 1, "said": 1},   # per party
}
# Relevance cutoff on cosine similarity. e5's values are tightly packed (often
# 0.80–0.90), so the cutoff is relative to the question's best hit, plus an
# absolute floor for questions no party has material on. An empty slot is left
# empty and stated in the prompt, instead of being filled with the least bad hit.
# The floors were set on 7 test questions (2026-09-28): motions at 0.82 were
# relevant, DID hits below 0.83 were not (e.g. water management on a nuclear question).
# ponytail: hand-calibrated, replace with a reranker (PLAN step 5) if the floors don't hold.
MAX_DISTANCE = 0.04
MIN_SIMILARITY = {"program": 0.80, "motion": 0.80, "said": 0.80, "did": 0.83}


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
            slots[party, one_layer] = (n, idx.search(
                question, k=n + 5, party=[party], layer=one_layer, method=method))

    # The cutoff is computed per layer: website prose systematically gets
    # higher similarity than motion yrkanden, so a shared cutoff would cut
    # exactly the motions the quotas are meant to make room for.
    best = defaultdict(float)
    for (_, one_layer), (_, candidates) in slots.items():
        for _, m in candidates:
            best[one_layer] = max(best[one_layer], m.get("similarity", 0.0))

    seen, hits, gaps = set(), [], []
    for (party, one_layer), (n, candidates) in slots.items():
        cutoff = max(MIN_SIMILARITY[one_layer], best[one_layer] - MAX_DISTANCE)
        taken = 0
        for score, m in candidates:
            if taken == n:
                break
            if "similarity" in m and m["similarity"] < cutoff:
                continue        # sorted by RRF, not similarity: keep looking
            taken += 1          # a shared reservation counts for each party,
            if m["id"] not in seen:      # but is shown only once
                seen.add(m["id"])
                hits.append((score, m))
        if not taken:
            gaps.append((party, one_layer))
    hits.sort(key=lambda t: -t[1].get("similarity", t[0]))
    return hits, gaps


def answer(idx, question, method="hybrid", model=MODEL, dry_run=False, layer=None):
    parties = parties_in_question(question)
    layers = [layer] if layer else ALL_LAYERS
    voting_advice = bool(VOTING_ADVICE_WORDS.search(question))
    hits, gaps = retrieve(idx, question, parties, layers, method)
    votes = vote_facts(hits, parties)
    size = "one_party" if len(parties) == 1 else "several_parties"
    context = build_context(hits, votes, MAX_ARGUMENT_CHARS[size])
    if gaps:
        context += "\n\nINGET RELEVANT MATERIAL HITTADES FÖR\n" + "\n".join(
            f"- {PARTY_NAMES[p]} ({p}): {LAYER_NAMES_SV[l]}" for p, l in gaps)
    contents = build_contents(question, context, voting_advice)

    info = {"parties": parties or "all", "layers": ",".join(layers),
            "hits": len(hits), "gaps": len(gaps),
            "vote_points": len(votes), "voting_advice": voting_advice,
            "context_chars": len(context)}

    if dry_run:
        return None, hits, info, contents
    text, info["model"] = ask_gemini(contents, model)
    return text, hits, info, contents


# How a missing layer is named in the prompt (Swedish: Gemini reads it).
LAYER_NAMES_SV = {"said": "hemsidan", "did": "riksdagsbeslut/voteringar",
                  "motion": "motioner", "program": "partiprogrammet"}


# --------------------------------------------------------------------------
# tool calling: Gemini decides which layers and parties to search
# --------------------------------------------------------------------------

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
  partiprogram, motion = motioner, said = webbplatser). Ange lager bara när användaren uttryckligen
  begränsar källorna ("bara motioner", "inga webbplatser"). Att frågan
  gäller ideologi är inget skäl att begränsa: det avgör du när du skriver
  svaret, inte när du söker. Omröstningar finns inte i materialet just nu.
- partier: partikoderna (S, M, SD, C, V, KD, MP, L) för de partier frågan
  gäller, även när de står i genitiv ("Moderaternas" = M). Utelämna för en
  jämförelse mellan alla partier.
- fraga: en fullständig, naturlig fråga på svenska om sakfrågan, t.ex.
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
                # A full question, not a keyword: e5 scores one-word queries
                # much lower (0.77–0.82 vs 0.83–0.86 for "migration"), below
                # MIN_SIMILARITY, which was calibrated on natural questions —
                # a keyword search dropped 5 of 8 parties' websites as gaps.
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
    """Executes one 'sok' call: the same retrieval as the one-call path, for a
    layer, or all layers if none is given. Returns the text Gemini reads, plus stats. `shown` holds the
    passages returned so far this question; new ones are numbered after them
    and appended, so the numbers stay unique across searches."""
    layers = call_layers(args)
    if not set(layers) <= set(ALL_LAYERS):
        return f"Okänt lager: {args['lager']}. Välj {', '.join(ALL_LAYERS)} eller utelämna.", 0, 0
    parties = [p for p in (args.get("partier") or []) if p in PARTY_NAMES]
    hits, gaps = retrieve(idx, args.get("fraga") or "", parties, layers, method,
                          searched=searched)

    already = {m["id"] for _, m in shown}
    new = [(score, m) for score, m in hits if m["id"] not in already]
    start = len(shown) + 1
    shown.extend(new)

    votes = vote_facts(new, parties) if "did" in layers else []
    size = "one_party" if len(parties) == 1 else "several_parties"
    text = build_context(new, votes, MAX_ARGUMENT_CHARS[size], limit=None, start=start)
    if gaps:
        text += "\n\nINGET RELEVANT MATERIAL HITTADES FÖR\n" + "\n".join(
            f"- {PARTY_NAMES[p]} ({p}): {LAYER_NAMES_SV[l]}" for p, l in gaps)
    if not new and hits:
        text += "\n\n(Alla träffar i den här sökningen har redan visats ovan.)"
    return text, len(gaps), len(votes)


def answer_with_tools(idx, question, method="hybrid", model=MODEL):
    """Gemini searches with the 'sok' tool, then answers. A manual loop, not
    the SDK's automatic function calling: we control the number of API calls
    and keep the passage numbers consistent across searches."""
    from google.genai import types

    client = gemini_client()
    voting_advice = bool(VOTING_ADVICE_WORDS.search(question))
    note = VOTING_ADVICE_NOTE if voting_advice else ""
    contents = knowledge_turns() + [
        {"role": "user", "parts": [{"text": f"FRÅGA: {question}{note}"}]}]

    shown, searches, calls = [], [], 0
    gaps = vote_points = context_chars = 0
    for step in range(TOOL_ROUNDS + 1):
        # The last step may not search: it must answer with what it has.
        mode = "NONE" if step == TOOL_ROUNDS else "AUTO"
        # After a fallback, the rest of the question stays on that model:
        # Gemini 3's thought signatures in the history belong to the model
        # that wrote them.
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
        # Several layers searched in the same step share the quota the way the
        # one-call path does (e.g. 2 motions + 1 vote + 1 website per party);
        # only a lone search ("bara motioner") gets all the slots. Without
        # this, three parallel searches fetched three times the material.
        layers_this_step = {l for fc in function_calls for l in call_layers(dict(fc.args or {}))}
        # Layers nobody searched aren't gaps: without saying so, Gemini wrote
        # "I materialet saknas motioner…" after a program-only search.
        unsearched = [l for l in ALL_LAYERS if l not in layers_this_step]
        results = []
        for i, fc in enumerate(function_calls):
            args = dict(fc.args or {})
            searches.append(f"{args.get('lager') or 'alla lager'}:{','.join(args.get('partier') or []) or 'alla'}"
                            f" \"{args.get('fraga', '')}\"")
            text, n_gaps, n_votes = run_search(idx, args, shown, method,
                                               searched=sorted(layers_this_step))
            if unsearched and i == len(function_calls) - 1:
                text += ("\n\nEJ SÖKTA LAGER: " + ", ".join(LAYER_NAMES_SV[l] for l in unsearched)
                         + ". Påstå inte att material saknas där — det har inte sökts.")
            gaps += n_gaps
            vote_points += n_votes
            context_chars += len(text)
            results.append(types.Part.from_function_response(
                name=fc.name, response={"result": text}))
        contents.append(types.Content(role="user", parts=results))

    info = {"parties": "chosen by Gemini", "layers": " ".join(searches) or "none",
            "hits": len(shown), "gaps": gaps, "vote_points": vote_points,
            "voting_advice": voting_advice, "context_chars": context_chars,
            "api_calls": calls, "model": model}
    return answer_text(response), shown, info, contents


def print_answer(response, hits, info, contents, dry_run):
    print(f"\n[{info['parties']} | layers: {info['layers']} | "
          f"{info['hits']} hits | {info['gaps']} gaps | "
          f"{info['vote_points']} vote points | "
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
    for line in source_list(hits):
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

    # Tool calling is the default. --dry-run can't use it (every tool step is
    # an API call), and --layer is a manual restriction for the one-call path.
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
