"""
answer.py — svarssteget. Här kommer LLM:en in, och ingen annanstans.

Kedjan:
    fråga
      -> parti_i_fragan()      parameterutvinning: "moderaterna" -> M
      -> valj_lager()          gäller frågan vad de SÄGER eller vad de GJORT?
      -> rag_index.search()    hämtning, kvoterad per parti
      -> rostfakta()           slår upp voteringarna för de träffade punkterna
      -> bygg_kontext()        allt ovan som text, med käll-id på varje bit
      -> Gemini                formulerar. Modellen får ALDRIG hitta på fakta.

Systemprompten följer föreläsningens fyra steg: vem den är, vad den vet
(Q&A-paren som lär den riksdagens semantik), vilken data den ska använda,
och hur den ska svara.

Kräver en nyckel från Google AI Studio:
    export GEMINI_API_KEY=...        # lägg ALDRIG nyckeln i repot

Kör:
    python answer.py modeller                       # vad nyckeln kommer åt
    python answer.py "Vad tycker V om kärnkraft?"
    python answer.py "Vad tycker partierna om klimatet?" --per-parti 2
    python answer.py "Har M drivit sin kärnkraftslinje?" --torrkor
    python answer.py chatt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import rag_index

try:
    import party_positions
except ImportError:
    party_positions = None

MODELL = "gemini-3.6-flash"
MAX_KONTEXT_TECKEN = 24000
MAX_UT_TOKENS = 4096
# Gemini 3 "tänker" före svaret, och tankestegen räknas mot max_output_tokens.
# 0 stänger av det. Vi vill ha återgivning av given text, inte resonemang —
# det sparar både kvot och väntetid. None = låt modellen bestämma själv.
TANKENIVA = "MINIMAL" 
POSITIONS_DID = "out/positions_did.jsonl"


PARTY_NAMES = rag_index.PARTY_NAMES

# Parameterutvinning: fritext -> partikod. Deterministisk tabell i stället för
# ett LLM-anrop — partinamn är ett slutet fält och behöver ingen modell.
ALIAS = {
    "s": "S", "socialdemokraterna": "S", "socialdemokraterna s": "S",
    "sossarna": "S", "socialdemokratiska arbetarepartiet": "S",
    "m": "M", "moderaterna": "M", "moderata samlingspartiet": "M",
    "sd": "SD", "sverigedemokraterna": "SD",
    "c": "C", "centerpartiet": "C", "centern": "C",
    "v": "V", "vänsterpartiet": "V", "vansterpartiet": "V",
    "kd": "KD", "kristdemokraterna": "KD",
    "mp": "MP", "miljöpartiet": "MP", "miljopartiet": "MP",
    "de gröna": "MP", "l": "L", "liberalerna": "L", "folkpartiet": "L",
}

# Frågan gäller vad partiet GJORT (riksdagsmaterialet) …
DID_ORD = re.compile(
    r"\brösta|\broste|votering|reservation|riksdag|utskott|betänkand|"
    r"drivit|genomfört|gjort|beslut|proposition|motion", re.IGNORECASE)
# … eller vad det SÄGER (hemsidorna).
SAID_ORD = re.compile(
    r"\btycker|\bvill|\banser|\bsäger|\bstår för|politik|åsikt|linje|"
    r"ståndpunkt|löfte", re.IGNORECASE)
# Frågor där rätt svar är att INTE svara.
ROSTRAD_ORD = re.compile(
    r"vilket parti (ska|bör) jag|passar mig|rösta på|vem ska jag rösta|"
    r"rekommendera(r)? du|vad ska jag rösta|hjälp mig välja", re.IGNORECASE)


# --------------------------------------------------------------------------
# parameterutvinning och routing
# --------------------------------------------------------------------------

def parti_i_fragan(fråga):
    """Vilka partier nämns? Tom lista = alla, dvs en jämförande fråga."""
    text = f" {fråga.lower()} "
    funna = []
    for alias in sorted(ALIAS, key=len, reverse=True):
        möns = r"\b" + re.escape(alias) + r"\b"
        if re.search(möns, text):
            kod = ALIAS[alias]
            if kod not in funna:
                funna.append(kod)
    return funna


def valj_lager(fråga):
    """'said', 'did' eller None (= båda)."""
    did = bool(DID_ORD.search(fråga))
    said = bool(SAID_ORD.search(fråga))
    if did and not said:
        return "did"
    if said and not did:
        return "said"
    return None


# --------------------------------------------------------------------------
# röstfakta för de punkter hämtningen träffade
# --------------------------------------------------------------------------

def rostfakta(träffar, partier, max_rader=12):
    """Wordalisering av voteringarna för de betänkandepunkter vi hämtat.

    Det här är kopplingen mellan rag_index och party_positions: hämtningen
    ger oss TEXT, den här funktionen ger oss vad partiet FAKTISKT GJORDE på
    samma punkt — i klartext, så att modellen aldrig ser en rå röstsiffra.
    """
    if party_positions is None or not os.path.exists(POSITIONS_DID):
        return []
    positions = party_positions.load_positions(POSITIONS_DID)
    sedda, rader = set(), []
    for _poäng, m in träffar:
        if m["layer"] != "did" or not m.get("punkt"):
            continue
        nyckel = f"{m['rm']}|{m['beteckning']}|{m['punkt']}"
        rad = positions.get(nyckel)
        if rad is None:
            continue
        for p in (partier or [q for q in m.get("parti", "").split(";") if q]):
            if (nyckel, p) in sedda:
                continue
            sedda.add((nyckel, p))
            svar = party_positions.describe(rad, p)
            rader.append(f"[{m['beteckning']} {m['rm']} punkt {m['punkt']}] "
                         f"{svar['text']}")
            if len(rader) >= max_rader:
                return rader
    return rader


# --------------------------------------------------------------------------
# kontext
# --------------------------------------------------------------------------

def bygg_kontext(träffar, röster, tak=MAX_KONTEXT_TECKEN):
    delar, n = [], 0
    if röster:
        block = "OMRÖSTNINGAR I RIKSDAGEN\n" + "\n".join(röster)
        delar.append(block)
        n += len(block)

    delar.append("\nHÄMTADE TEXTER")
    for i, (_poäng, m) in enumerate(träffar, 1):
        källa = m.get("url") or f"{m.get('beteckning', '')} {m.get('rm', '')}"
        bit = (f"\n[{i}] {m['layer'].upper()} — {m['kontext']}\n"
               f"källa: {källa}\n{m['text']}\n")
        if n + len(bit) > tak:
            break
        delar.append(bit)
        n += len(bit)
    return "\n".join(delar)


def kallista(träffar):
    ut, sedda = [], set()
    for i, (_p, m) in enumerate(träffar, 1):
        källa = m.get("url") or (f"{m.get('beteckning', '')} {m.get('rm', '')}"
                                 f" punkt {m.get('punkt', '')}").strip()
        if källa in sedda:
            continue
        sedda.add(källa)
        ut.append(f"[{i}] {källa}")
    return ut


# --------------------------------------------------------------------------
# systemprompt — föreläsningens steg 1 och 4
# --------------------------------------------------------------------------

SYSTEM = """Du är en granskare av svensk partipolitik. Du arbetar med två
sorters material och blandar dem aldrig ihop:

  SAID = vad ett parti SÄGER. Hämtat från partiets egen webbplats.
  DID  = vad som FAKTISKT HÄNT i riksdagen. Betänkanden och omröstningar.

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
   "I materialet finns ingen sida från SD om klimat." Du fyller aldrig
   luckan med gissningar och du använder aldrig en sida om ett annat ämne
   som om den handlade om det efterfrågade.

5. Skiljer sig SAID från DID är det den intressanta iakttagelsen och du
   lyfter fram den. Men du kallar det inte löftesbrott — du redovisar båda
   och låter läsaren dra slutsatsen.

FORM: svar på svenska, i löpande text. Två till fem stycken. Jämför du flera
partier tar du ett stycke per parti i samma ordning varje gång. Avsluta
aldrig med en sammanfattande värdering av vilket parti som har rätt."""


# Steg 2: vad den vet. Q&A-par som lär modellen riksdagens semantik — utan
# dem tolkar den "avstod" och "acklamation" som fel i datan.
KUNSKAP = [
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
]


def bygg_innehall(fråga, kontext, avrader):
    """Gemini-format: växlande user/model-turer, kontexten sist."""
    innehåll = []
    for f, s in KUNSKAP:
        innehåll.append({"role": "user", "parts": [{"text": f}]})
        innehåll.append({"role": "model", "parts": [{"text": s}]})

    instruktion = ""
    if avrader:
        instruktion = ("\n\nOBS: användaren ber om en partirekommendation. "
                       "Följ regel 3 — rekommendera inte, förklara kort "
                       "varför, och erbjud en sakfrågejämförelse.\n")

    innehåll.append({"role": "user", "parts": [{"text":
        f"KONTEXT\n{kontext}\n\nSLUT PÅ KONTEXT{instruktion}\n\n"
        f"FRÅGA: {fråga}"}]})
    return innehåll


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------
def fraga_gemini(innehåll, modell=MODELL, forsok=8):
    import random
    from google import genai
    from google.genai import types

    nyckel = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not nyckel:
        sys.exit("sätt GEMINI_API_KEY (nyckel från https://aistudio.google.com)")
    klient = genai.Client(api_key=nyckel)

    def gor_config(tanke):
        extra = {"thinking_config": tanke} if tanke is not None else {}
        return types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0.2,          # återgivning, inte kreativitet
            max_output_tokens=MAX_UT_TOKENS,
            **extra)

    # Gemini 3 styr tänkandet med thinking_level, äldre modeller med
    # thinking_budget, och vissa tillåter ingetdera. API:t svarar bara
    # "invalid argument" utan att säga vilket argument som var fel, så vi
    # provar varianterna i tur och ordning i stället för att tolka texten.
    varianter = []
    if TANKENIVA:
        varianter.append(("thinking_level",
                          types.ThinkingConfig(thinking_level=TANKENIVA)))
        varianter.append(("thinking_budget",
                          types.ThinkingConfig(thinking_budget=0)))
    varianter.append(("utan tankeinställning", None))

    # Övergående fel, båda vanliga på free tier:
    #   429 RESOURCE_EXHAUSTED  vi har slagit i kvoten
    #   503 UNAVAILABLE         Google är överbelastat just nu
    # Jitter så att tre gruppmedlemmar som kör samtidigt inte gör om
    # sina försök i takt.
    OVERGAENDE = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE",
                  "500", "INTERNAL", "504", "DEADLINE_EXCEEDED")

    v = 0
    for n in range(forsok):
        namn, tanke = varianter[v]
        try:
            svar = klient.models.generate_content(
                model=modell, contents=innehåll, config=gor_config(tanke))
        except Exception as fel:
            text = str(fel)
            if (("400" in text or "INVALID_ARGUMENT" in text)
                    and v + 1 < len(varianter)):
                v += 1
                print(f"  ({namn} avvisades av {modell}, "
                      f"provar {varianter[v][0]})", file=sys.stderr)
                continue
            if not any(kod in text for kod in OVERGAENDE):
                raise
            if n == forsok - 1:
                break
            paus = min(60, 2 ** n * 4) * (0.5 + random.random())
            orsak = ("kvotgräns" if "429" in text or "RESOURCE" in text
                     else "överbelastning")
            print(f"  ({orsak}, försöker igen om {paus:.0f}s …)",
                  file=sys.stderr)
            time.sleep(paus)
            continue

        # Klipptes svaret? Ett halvt svar är värre än inget: det ser färdigt
        # ut men saknar både slutsats och källhänvisning.
        skal = ""
        if getattr(svar, "candidates", None):
            skal = str(getattr(svar.candidates[0], "finish_reason", "") or "")
        if "MAX_TOKENS" in skal:
            print(f"  VARNING: svaret klipptes vid {MAX_UT_TOKENS} tokens. "
                  f"Höj MAX_UT_TOKENS eller sänk MAX_KONTEXT_TECKEN.",
                  file=sys.stderr)
        if not svar.text:
            print(f"  (tomt svar, finish_reason={skal or 'okänt'})",
                  file=sys.stderr)
            return f"[inget svar genererades — finish_reason: {skal or 'okänt'}]"
        return svar.text

    sys.exit(f"gav upp efter {forsok} försök mot {modell} — "
             f"prova en annan modell eller vänta en stund")



def lista_modeller():
    from google import genai
    nyckel = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not nyckel:
        sys.exit("sätt GEMINI_API_KEY")
    # Klienten måste leva så länge vi itererar: models.list() är en lat pager
    # som hämtar sidorna först vid iteration, och en temporär klient hinner
    # stängas dessförinnan ("client has been closed").
    klient = genai.Client(api_key=nyckel)
    for m in list(klient.models.list()):
        handlingar = getattr(m, "supported_actions", None) or []
        if not handlingar or "generateContent" in handlingar:
            print(f"  {m.name}")


# --------------------------------------------------------------------------
# huvudflöde
# --------------------------------------------------------------------------

def svara(idx, fråga, k=10, per_parti=None, metod="hybrid", modell=MODELL,
          torrkor=False, lager=None):
    partier = parti_i_fragan(fråga)
    if lager is None:
        lager = valj_lager(fråga)
    avrader = bool(ROSTRAD_ORD.search(fråga))
    if per_parti is None and not partier:
        per_parti = 2            # jämförande fråga: alla partier ska rymmas

    träffar = idx.search(fråga, k=k, parti=partier or None, layer=lager,
                         per_parti=per_parti, metod=metod)
    röster = rostfakta(träffar, partier)
    kontext = bygg_kontext(träffar, röster)
    innehåll = bygg_innehall(fråga, kontext, avrader)

    info = {"partier": partier or "alla", "lager": lager or "båda",
            "träffar": len(träffar), "röstrader": len(röster),
            "avrådan": avrader, "kontexttecken": len(kontext)}

    if torrkor:
        return None, träffar, info, innehåll
    return fraga_gemini(innehåll, modell), träffar, info, innehåll


def skriv(svar, träffar, info, innehåll, torrkor):
    print(f"\n[{info['partier']} | lager: {info['lager']} | "
          f"{info['träffar']} träffar | {info['röstrader']} röstrader | "
          f"{info['kontexttecken']} tecken"
          f"{' | AVRÅDAN' if info['avrådan'] else ''}]\n")
    if torrkor:
        print(innehåll[-1]["parts"][0]["text"])
        return
    print(svar)
    print("\nKällor:")
    for rad in kallista(träffar):
        print(" ", rad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fraga", help="frågan, eller 'chatt', eller 'modeller'")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--per-parti", type=int, default=None)
    ap.add_argument("--layer", choices=["said", "did"], default=None)
    ap.add_argument("--metod", choices=["hybrid", "bm25", "vektor"],
                    default="hybrid")
    ap.add_argument("--modell", default=MODELL)
    ap.add_argument("--index", default=rag_index.INDEX_DIR)
    ap.add_argument("--torrkor", action="store_true",
                    help="bygg prompten och skriv ut den, anropa inte API:t")
    a = ap.parse_args()

    if a.fraga == "modeller":
        lista_modeller()
        return

    idx = rag_index.Index(a.index)
    if a.fraga == "chatt":
        print(f"{idx.info['n']} passager. Tom rad avslutar.")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            skriv(*svara(idx, q, k=a.k, per_parti=a.per_parti, metod=a.metod,
                         modell=a.modell, torrkor=a.torrkor, lager=a.layer),
                  torrkor=a.torrkor)
        return

    skriv(*svara(idx, a.fraga, k=a.k, per_parti=a.per_parti, metod=a.metod,
                 modell=a.modell, torrkor=a.torrkor, lager=a.layer),
          torrkor=a.torrkor)


if __name__ == "__main__":
    main()