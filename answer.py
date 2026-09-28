"""
answer.py — the answering step. This is where the LLM comes in, and nowhere else.

The chain:
    question
      -> parties_in_question()  parameter extraction: "moderaterna" -> M
      -> choose_layer()         is it about what they SAY, PROPOSE or DID?
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

MODEL = "gemini-3.6-flash"
MAX_CONTEXT_CHARS = 40000
# A motion's argument can be several thousand characters; the start is enough
# to show what the yrkande is about, and the rest would crowd out other parties.
MAX_ARGUMENT_CHARS = 600
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

# The question is specifically about a vote / a Riksdag decision. "drivit" and
# "gjort" were removed from here: they're too vague to mean "only votes" and
# used to push the motion layer out of answers to "vad har partierna gjort för X".
DID_WORDS = re.compile(
    r"\brösta|\broste|votering|reservation|riksdag|utskott|betänkand|"
    r"genomfört|beslut|proposition", re.IGNORECASE)
# The question is specifically about motions / yrkanden.
MOTION_WORDS = re.compile(
    r"\bmotioner?|\byrkand|\bföreslagit|\bföresla[gr]\b",
    re.IGNORECASE)
# Questions where the right answer is NOT to answer (voting advice).
VOTING_ADVICE_WORDS = re.compile(
    r"vilket parti (ska|bör) jag|passar mig|rösta på|vem ska jag rösta|"
    r"rekommendera(r)? du|vad ska jag rösta|hjälp mig välja", re.IGNORECASE)


# --------------------------------------------------------------------------
# parameter extraction and routing
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


def choose_layer(question):
    """'did', 'motion', or None (= all layers).

    Routes to a specific layer only on clear keywords. General party-position
    questions ("vad tycker SD om …") search all layers, with a fixed quota per
    layer — see LAYER_QUOTA.
    """
    did = bool(DID_WORDS.search(question))
    motion = bool(MOTION_WORDS.search(question))
    if did and not motion:
        return "did"
    if motion and not did:
        return "motion"
    return None


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

def build_context(hits, votes, limit=MAX_CONTEXT_CHARS):
    parts, n = [], 0
    if votes:
        block = "OMRÖSTNINGAR I RIKSDAGEN\n\n" + "\n\n".join(votes)
        parts.append(block)
        n += len(block)

    parts.append("\nHÄMTADE TEXTER")
    for i, (_score, m) in enumerate(hits, 1):
        if m.get("layer") == "motion":
            source = f"motion {m.get('beteckning', '')} {m.get('rm', '')}"
        else:
            source = m.get("url") or f"{m.get('beteckning', '')} {m.get('rm', '')}"
        piece = f"\n[{i}] {m['layer'].upper()} — {m['kontext']}\nkälla: {source}\n{m['text']}"
        if m.get("layer") == "motion" and m.get("sektion_text"):
            argument = m["sektion_text"]
            if len(argument) > MAX_ARGUMENT_CHARS:
                argument = argument[:MAX_ARGUMENT_CHARS].rsplit(" ", 1)[0] + " …"
            piece += f"\nMotivering: {argument}"
        piece += "\n"
        if n + len(piece) > limit:
            break
        parts.append(piece)
        n += len(piece)
    return "\n".join(parts)


def source_list(hits):
    """One line per source, with every passage number that points to it:
    several passages from the same page/motion must not leave a cited number
    missing from the list."""
    numbers = {}
    for i, (_score, m) in enumerate(hits, 1):
        if m.get("layer") == "motion":
            source = f"motion {m.get('beteckning', '')} {m.get('rm', '')}"
        else:
            punkt = f" punkt {m['punkt']}" if m.get("punkt") else ""
            source = m.get("url") or f"{m.get('beteckning', '')} {m.get('rm', '')}{punkt}"
        numbers.setdefault(source, []).append(str(i))
    return [f"[{', '.join(n)}] {s}" for s, n in numbers.items()]


# --------------------------------------------------------------------------
# system prompt — the lecture's steps 1 and 4 (Swedish: Gemini reads it)
# --------------------------------------------------------------------------

SYSTEM = """Du är en granskare av svensk partipolitik. Du arbetar med tre
sorters material och blandar dem aldrig ihop:

  MOTION = vad ett parti FÖRESLAGIT. Formella yrkanden i motioner till riksdagen.
  SAID   = vad ett parti SÄGER. Hämtat från partiets egen webbplats.
  DID    = vad som FAKTISKT HÄNT i riksdagen. Betänkanden och omröstningar.

REGLER, i fallande ordning:

1. Du använder ENDAST det material du får i KONTEXT. Du har egna minnen av
   svensk politik — de är föråldrade och du använder dem inte. Om kontexten
   inte räcker säger du det rent ut.

2. Varje sakpåstående följs av sin källa i hakparentes: [3], eller
   [AU10 2022/23 punkt 1]. Ett påstående utan källa får inte skrivas.

3. Du rekommenderar ALDRIG ett parti och rangordnar dem aldrig. Du säger
   inte vilket parti som "passar" någon, oavsett hur användaren beskriver
   sig. Om du blir ombedd förklarar du kort att du redovisar vad partierna
   säger och gör, och erbjuder en jämförelse i en sakfråga användaren väljer.

4. Saknas material för ett parti skriver du det som ett eget konstaterande:
   "I materialet finns ingen motion från SD om detta." Du fyller aldrig
   luckan med gissningar och du använder aldrig en källa om ett annat ämne
   som om den handlade om det efterfrågade.

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

7. Skiljer sig MOTION/SAID från DID är det den intressanta iakttagelsen och
   du lyfter fram den. Men du kallar det inte löftesbrott — du redovisar
   båda och låter läsaren dra slutsatsen.

FORM: svar på svenska, i löpande text. Två till fem stycken. Jämför du flera
partier tar du ett stycke per parti i samma ordning varje gång. Avsluta
aldrig med en sammanfattande värdering av vilket parti som har rätt."""


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


def build_contents(question, context, voting_advice):
    """Gemini format: alternating user/model turns, the context last."""
    contents = []
    for q, a in KNOWLEDGE:
        contents.append({"role": "user", "parts": [{"text": q}]})
        contents.append({"role": "model", "parts": [{"text": a}]})

    instruction = ""
    if voting_advice:
        instruction = ("\n\nOBS: användaren ber om en partirekommendation. "
                       "Följ regel 3 — rekommendera inte, förklara kort "
                       "varför, och erbjud en sakfrågejämförelse.\n")

    contents.append({"role": "user", "parts": [{"text":
        f"KONTEXT\n{context}\n\nSLUT PÅ KONTEXT{instruction}\n\n"
        f"FRÅGA: {question}"}]})
    return contents


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------
def ask_gemini(contents, model=MODEL, attempts=8):
    import random
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("set GEMINI_API_KEY (key from https://aistudio.google.com)")
    client = genai.Client(api_key=key)

    def make_config(thinking):
        extra = {"thinking_config": thinking} if thinking is not None else {}
        return types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0.2,          # restating, not creativity
            max_output_tokens=MAX_OUTPUT_TOKENS,
            **extra)

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

    v = 0
    for n in range(attempts):
        name, thinking = variants[v]
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=make_config(thinking))
        except Exception as err:
            text = str(err)
            if (("400" in text or "INVALID_ARGUMENT" in text)
                    and v + 1 < len(variants)):
                v += 1
                print(f"  ({name} rejected by {model}, "
                      f"trying {variants[v][0]})", file=sys.stderr)
                continue
            if not any(code in text for code in TRANSIENT):
                raise
            if n == attempts - 1:
                break
            pause = min(60, 2 ** n * 4) * (0.5 + random.random())
            reason = ("quota limit" if "429" in text or "RESOURCE" in text
                      else "overload")
            print(f"  ({reason}, retrying in {pause:.0f}s …)",
                  file=sys.stderr)
            time.sleep(pause)
            continue

        # Was the answer cut off? Half an answer is worse than none: it looks
        # finished but lacks both the conclusion and the source references.
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

    sys.exit(f"gave up after {attempts} attempts against {model} — "
             f"try another model or wait a while")



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
LAYER_QUOTA = {
    "one_party": {"motion": 4, "did": 3, "said": 2},
    "several_parties": {"motion": 2, "did": 1, "said": 1},   # per party
}
# Relevance cutoff on cosine similarity. e5's values are tightly packed (often
# 0.80–0.90), so the cutoff is relative to the question's best hit, plus an
# absolute floor for questions no party has material on. An empty slot is left
# empty and stated in the prompt, instead of being filled with the least bad hit.
# The floors were set on 7 test questions (2026-09-28): motions at 0.82 were
# relevant, DID hits below 0.83 were not (e.g. water management on a nuclear question).
# ponytail: hand-calibrated, replace with a reranker (PLAN step 5) if the floors don't hold.
MAX_DISTANCE = 0.04
MIN_SIMILARITY = {"motion": 0.80, "said": 0.80, "did": 0.83}


def retrieve(idx, question, parties, layer=None, method="hybrid"):
    """Retrieval with fixed slots: (hits, gaps), where gaps are the
    (party, layer) pairs that got no hit above the relevance cutoff."""
    quota = LAYER_QUOTA["one_party" if len(parties) == 1 else "several_parties"]
    layers = {layer: sum(quota.values())} if layer else quota

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
    if layer is None:
        layer = choose_layer(question)
    voting_advice = bool(VOTING_ADVICE_WORDS.search(question))
    hits, gaps = retrieve(idx, question, parties, layer, method)
    votes = vote_facts(hits, parties)
    context = build_context(hits, votes)
    if gaps:
        context += "\n\nINGET RELEVANT MATERIAL HITTADES FÖR\n" + "\n".join(
            f"- {PARTY_NAMES[p]} ({p}): {LAYER_NAMES_SV[l]}" for p, l in gaps)
    contents = build_contents(question, context, voting_advice)

    info = {"parties": parties or "all", "layer": layer or "all",
            "hits": len(hits), "gaps": len(gaps),
            "vote_points": len(votes), "voting_advice": voting_advice,
            "context_chars": len(context)}

    if dry_run:
        return None, hits, info, contents
    return ask_gemini(contents, model), hits, info, contents


# How a missing layer is named in the prompt (Swedish: Gemini reads it).
LAYER_NAMES_SV = {"said": "hemsidan", "did": "riksdagsbeslut/voteringar",
                  "motion": "motioner"}


def print_answer(response, hits, info, contents, dry_run):
    print(f"\n[{info['parties']} | layer: {info['layer']} | "
          f"{info['hits']} hits | {info['gaps']} gaps | "
          f"{info['vote_points']} vote points | "
          f"{info['context_chars']} chars"
          f"{' | VOTING ADVICE' if info['voting_advice'] else ''}]\n")
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
    ap.add_argument("--layer", choices=["said", "did", "motion"], default=None)
    ap.add_argument("--method", choices=["hybrid", "bm25", "vector"],
                    default="hybrid")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--index", default=rag_index.INDEX_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="build the prompt and print it, don't call the API")
    a = ap.parse_args()

    if a.question == "models":
        list_models()
        return

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
            print_answer(*answer(idx, q, method=a.method,
                                 model=a.model, dry_run=a.dry_run, layer=a.layer),
                         dry_run=a.dry_run)
        return

    print_answer(*answer(idx, a.question, method=a.method,
                         model=a.model, dry_run=a.dry_run, layer=a.layer),
                 dry_run=a.dry_run)


if __name__ == "__main__":
    main()
