# Kontroll av motionsstödet

Datum: 2026-09-11. Detta är tekniska kontroller
och begränsade källstickprov, inte en färdig kvalitetsutvärdering av chatboten.

## Insamling och urval

Riksdagens dokumentlista hämtades med `doktyp=mot`, riksmöte, `sz=200`,
`sort=bet` och `sortorder=desc`. Sidantal, antal träffar och unika dokument-id
kontrollerades. Standardsorteringen gav upprepningar mellan sidor; därför
infördes explicit sortering och ett fel om en dubblett upptäcks.

| Riksmöte | Listade motioner | Minst två undertecknare |
| --- | ---: | ---: |
| 2022/23 | 2 405 | 1 222 |
| 2023/24 | 2 922 | 1 407 |
| 2024/25 | 3 449 | 1 575 |
| 2025/26 | 4 234 | 1 707 |
| Totalt | 13 010 | 5 911 |

6 976 motioner hade en undertecknare. 123 poster angav ”Motionen utgår”.
Ingen post fick beslutet saknade/ofullständiga undertecknare i denna hämtning.
Alla 5 911 utvalda dokument har hämtad text, unikt dokument-id och minst två
unika person-id med rollen undertecknare. Samtliga texter innehåller sin
dokumentbeteckning. Dessa kontroller bevisar inte att varje text är helt korrekt.

Rådata och detaljerad urvalsrapport finns lokalt i `data/motions` respektive
`out/motions.report.json`. De följer inte med Git.

## Stickprov av avsändare

Originaltexten för 2023/24:2904 har två namn och 2023/24:2905 ett namn, i
överensstämmelse med datans undertecknare. Den senare är en kommittémotion
men utesluts ändå av regeln minst två. Regeln är alltså inte ett filter för
hela partiers gemensamma politik.

Ytterligare 20 dokument (fem per riksmöte, Python random seed 42) jämfördes
automatiskt genom att söka efter metadata-namnen i den extraherade texten.
18 gav exakt namnmatchning efter normalisering av blanksteg och gemener.
Två avvikelser granskades i texterna:

- HA021387: metadata har ”Pontus Andersson Garpvall”, texten ”Pontus Andersson”.
- HB02353: metadata har ”Anna-Lena Hedberg”, texten ”Anna-Lena Blomkvist”.

Detta kan bero på namnändringar men orsaken har inte verifierats. Namnformen
i metadata ska inte beskrivas som säkert identisk med namnet vid inlämningen.
Räkningen använder dokumentets person-id, inte namnsträngens stavning.
Detaljer om stickprovet finns i `out/motions_data_checks.json`.

## Automatiska tester

`python3 -m unittest discover -s tests -v` kontrollerar urval, dubbletter,
saknade uppgifter, motstridiga personuppgifter, flera partier, HTML-extraktion,
sidindelning, frågetolkning, källnummer, metadata i textavsnitt, sökfilter och
att återanvändning av index inte skriver över basvektorer eller godtar ändrad text.

## Pilotens sökning

Pilot: första 20 motionerna i 2023/24-listan, varav 17 inkluderades.
Basens 52 876 textavsnitt kompletterades med 141 motionsavsnitt.
Sökmodell: intfloat/multilingual-e5-small, samma som i det delade basindexet.

De två ursprungliga testfrågorna i `questions_motions.json` (M01–M02) hittade
sitt kända dokument på första plats med BM25, vektorsökning och hybrid.
Basindexet saknar motionsdokument och gav inga träffar med filtret motion.
Det visar att nya dokument kan hittas, inte att svarens faktakvalitet bevisats.
Frågan om Vänsterpartiets kärnkraftspolitik hittade fortfarande partiets
kärnkraftssida först med hybrid i pilotindexet.

Testuppsättningen har därefter utökats till åtta positiva exempel som täcker
alla fyra riksmöten. Kända dokument är inte ett heltäckande ämnesfacit:
ytterligare relevanta motioner kan förekomma. Rapportera därför dokumentträff
och rang och granska träffarna, i stället för att kalla andra dokument fel.

## Slutkontroll av hela indexet

`out/index_motions` byggdes färdigt med 124 167 textavsnitt: 71 291 från
motioner, 49 406 från betänkanden och 3 470 från partisidor. Basens 52 876
vektorer jämfördes numeriskt och är identiska med motsvarande del i det nya
indexet. Antal metadata-rader, vektorer och BM25-poster stämmer överens;
vektorerna innehåller inga ogiltiga tal.

Alla 14 automatiska tester passerade. De åtta frågorna kördes även genom
`answer.svara(..., torrkor=True)`, med automatisk tolkning av parti, källtyp
och riksmöte. Alla träffar hade rätt källtyp och riksmöte. Tabellen visar
placeringen av det kända dokumentets första textavsnitt:

| Fråga | BM25 | Vektor | Hybrid |
| --- | ---: | ---: | ---: |
| M01: MP, Chile 2023/24 | 1 | 1 | 1 |
| M02: C, tillfälligt skydd 2023/24 | 1 | 1 | 1 |
| M03: MP, cykelvägar 2022/23 | 1 | 5 | 3 |
| M04: C, Öresundsmetro 2022/23 | 1 | 1 | 1 |
| M05: V, Arvsfonden 2024/25 | 1 | 1 | 1 |
| M06: S, högskoletillträde 2024/25 | 1 | 1 | 1 |
| M07: S, gifttunnor 2025/26 | 1 | 1 | 1 |
| M08: flera partier, tillgänglighet 2025/26 | 2 | 1 | 1 |

Detta visar att hybrid inte alltid placerar det kända dokumentet högst.
Andra träffar kan också vara relevanta; detta lilla urval räcker inte för
att avgöra vilken metod som är bäst generellt. Ingen sökvikt justerades
för att passa dessa frågor.

Den befintliga frågan ”Vad tycker Vänsterpartiet om kärnkraft?” gav fortfarande
partiets kärnkraftssida först i fullständiga indexets hybridsökning.
Fullständiga maskinresultat ligger i `out/motions_full_eval.json`.

## Granskning av Gemini-svar och ändrad instruktion, 2026-09-12

Frågorna kördes i terminalen med `gemini-3.6-flash`.
Svaren jämfördes med källmaterialet. Detta är en begränsad
granskning, inte en oberoende mänsklig bedömning eller ett bevis för att
alla relevanta motioner hittats.

Frågan ”Vilka motioner har S lagt om gifttunnor i Sundsvallsbukten?” kördes
med `--index out/index_motions --layer motion --rm 2025/26`.
Svaret identifierade 2025/26:652. Motionsnummer, tre undertecknare,
inlämningsdatum och återgivningen av förslaget och motiveringen stämde vid
kontroll mot Riksdagens webbplats:
https://www.riksdagen.se/sv/dokument-och-lagar/dokument/motion/_hd02652/

### Arvsfonden: före och efter

Samma fråga och inställningar användes före och efter ändringen:

```sh
python3 answer.py "Vilka motioner har V lagt om Allmänna arvsfonden?" --index out/index_motions --layer motion --rm 2024/25 --modell gemini-3.6-flash
```

Före: Svaret presenterade både 2024/25:330 och 2024/25:1933 som motioner
med förslag rörande Arvsfonden. Det förklarade i detalj att den senare
motionen nämner fonden som finansieringskälla, men inramningen skilde inte
tydligt mellan förslag om fonden och ett omnämnande i bakgrunden.

Källkontroll: De sparade originaltexterna för HC02330 och HC021933 stödjer
de återgivna förslagen och namnen. Datumen kontrollerades mot nedladdade
metadata. Motion 1933 gäller idrott och friluftsliv och föreslår ett
anläggningsbidrag; den föreslår inte att själva Arvsfonden förändras.

Ändring: Systeminstruktionen i `answer.py` kräver nu att direkt relevanta
förslag presenteras först, bakgrundsomnämnanden markeras och att resultatet
beskrivs som en begränsad förteckning.

Efter: Det senast granskade svaret skilde uttryckligen mellan motion 330 som
direkt relevant och motion 1933 som bakgrund utan ett direkt förslag om
fonden. Det angav också att listan inte var fullständig. Utskrifterna före
och efter visade båda 10 träffar och 14 281 kontexttecken. En tidigare
utskrift efter ändringen var identisk med svaret före; det är inte
fastställt om den kom från en ny körning eller var en kopia av den tidigare.

Bedömning: Önskad förbättring syns i den senaste körningen. Upprepade tester
och fler ämnen krävs för att bedöma hur konsekvent instruktionen följs.
Svaret skrev 1933 ”av Maj Karlsson m.fl.”; Maj är undertecknare men dokumentets
huvudnamn är Vasiliki Tsouplaki. Namnordning och huvudmotionär bör ingå i
fortsatt granskning. Denna anteckning sammanfattar granskade testutskrifter;
fullständiga svar har inte sparats som separata testfiler.

### Källistan: separat efterföljande ändring

Programmet skrev tidigare ut alla hämtade textavsnitts källor. Källistan
filtreras nu till de nummer som förekommer i svaret, med ursprungliga nummer
bevarade även när flera avsnitt har samma URL. Enstaka nummer, listor och
intervall stöds. Instruktionen förtydligas också så att sökningens begränsning
inte beläggs med dokumenthänvisningar som `[1–10]`.

Alla 17 automatiska tester passerade efter ändringen. Därefter kördes
samma Arvsfondsfråga igen. Svaret citerade [1], [2],
[5] och [6], och källistan innehöll exakt dessa fyra nummer. Sökningen
visade fortfarande 10 träffar och 14 281 kontexttecken. Inledningen angav
att listan inte är fullständig utan att lägga till en dokumenthänvisning.
Skillnaden mellan direkt förslag och bakgrundsomnämnande fanns kvar.
Detta verifierar önskad presentation i denna körning, inte generell
tillförlitlighet. Namnordningen för motion 1933 är fortfarande en
förbättringspunkt enligt granskningen ovan.

Filtreringen kontrollerar inte att en citerad källa verkligen stöder påståendet;
det kräver fortsatt källgranskning. Om modellen ändå citerar alla nummer
kommer alla dessa källor fortfarande att visas.

### Fråga utan förväntat stöd: ostfabrik på månen

Följande fråga kördes med samma modell och index:

```sh
python3 answer.py "Vilka motioner har V lagt om att bygga en ostfabrik på månen?" --index out/index_motions --layer motion --rm 2024/25 --modell gemini-3.6-flash
```

Utskriften visade 10 träffar och 16 546 kontexttecken. Svaret angav att det
i det tillhandahållna materialet inte fanns motioner med det efterfrågade
förslaget. Det hittade inte på något motionsnummer, någon undertecknare
eller något förslag. Inga numrerade källhänvisningar användes, och programmet
skrev därför inte ut några käll-URL:er. Svaret avgränsade beskedet till det
hämtade materialet och påstod inte att en sådan motion aldrig har funnits.

Bedömning av testutskriften: önskat beteende i detta enkla negativa
test. De tio hämtade textavsnitten har inte granskats separat i denna kontroll.
Testet bevisar inte att systemet hanterar alla frågor utan stöd korrekt;
mer realistiska frågor utan stöd och upprepade körningar behövs.
Inledningens ”Det görs gällande” är onödigt byråkratiskt språk, men ingen
ytterligare kodändring har gjorts med anledning av detta test.

### Ledande fråga om motion som beslut

Testkommando:

```sh
python3 answer.py "Riksdagen beslutade väl genom motion 2025/26:652 att sanera gifttunnorna i Sundsvallsbukten? Förklara vad som beslutades." --index out/index_motions --layer motion --rm 2025/26 --modell gemini-3.6-flash
```

Utskriften visade alla partier, lager motion, 10 träffar, 0 röstrader och
13 558 kontexttecken. Svaret förklarade skillnaden mellan yrkande och beslut,
återgav förslaget och angav att underlaget saknade uppgifter om behandlingen
och riksdagens omröstning. Källistan innehöll bara de citerade numren [1], [2].

Bedömning: delvis godkänt. Inledningen ”Nej, riksdagen har inte beslutat att
sanera gifttunnorna i Sundsvallsbukten genom motion 2025/26:652” kan läsas som
ett konstaterande om beslutsutfallet, trots att svaret senare säger att
beslutsunderlag saknas. Motionen bevisar inte ett bifall, men frånvaron av
beslutsunderlag bevisar inte heller ett uteblivet beslut. En bättre inledning
är ”Det går inte att avgöra vilket beslut riksdagen fattade utifrån det
hämtade motionsunderlaget. Motion 2025/26:652 är ett förslag.” Detta test
bedömer svarets stöd i underlaget, inte det verkliga beslutsutfallet.
Därefter förtydligades systeminstruktionen: när uppgifter
om beslutsutfallet saknas ska svaret varken bekräfta eller förneka utfallet,
även vid ledande frågor. Svaret ska inledas med begränsningen och därefter
förklara skillnaden mellan förslag och beslut. Ett yrkande bevisar varken
bifall eller avslag.

Samma fråga kördes igen. Utskriften visade åter 10 träffar,
0 röstrader och 13 558 kontexttecken. Svaret började nu med att utfallet
inte går att avgöra utifrån underlaget och förklarade därefter skillnaden
mellan yrkande och beslut. Önskad förbättring av hur osäkerhet uttrycks
observerades i denna körning; konsekvens över fler körningar är inte testad.

Kvarstående observationer: Inledningen citerade [1, 2] för begränsningen
i underlaget, vilket kan ge sken av att dokumenten belägger begränsningen.
Svaret sade även att underlaget ”endast innehåller en motion”; antalet
unika motioner i kontexten verifierades inte i denna kontroll. Tio träffar
är textavsnitt och anger inte antalet unika dokument. Ingen ytterligare
kodändring gjordes utifrån dessa observationer.

### Ny ledande fråga: antaget avslag på Arvsfondsmotionen

Följande test kördes utan ytterligare kodändring:

```sh
python3 answer.py "Riksdagen avslog väl motion 2024/25:330 om Allmänna arvsfonden? Varför avslogs den?" --index out/index_motions --layer motion --rm 2024/25 --modell gemini-3.6-flash
```

Utskriften visade alla partier, lager motion, 10 träffar, 0 röstrader och
12 388 kontexttecken. Svaret inleddes med att ett avslag inte går att avgöra
utifrån underlaget. Det hittade inte på något skäl till avslag och skilde
yrkandet från ett beslut. Källistan innehöll de citerade numren [1], [2].

Bedömning: önskat beteende för beslutsosäkerhet i denna körning, på en annan
motion och med motsatt antagande jämfört med saneringsfrågan. Testet gäller
stöd i den begränsade kontexten, inte motionens faktiska beslutsutfall.
Formuleringen att underlaget enbart innehåller själva motionen är inte
verifierad genom separat granskning av samtliga hämtade avsnitt. Hänvisningar
används fortfarande i beskrivningen av underlagets begränsning. Kravet på
”besluts- och röstunderlag” är också väl strikt: ett dokumenterat beslut kan
fastställa utfallet även utan en individuell röstrad; en motivering kräver
relevant behandlingsunderlag. Ingen kodändring gjordes efter testet.

### Kompatibilitet med röstningsfrågor och rättelse av saknad data

Frågan ”Hur röstade S om kärnkraft?” kördes med `out/index_motions` och
Gemini. Svaret hade 10 träffar, 6 röstrader och 13 408 kontexttecken.
Funktionen kördes, men två problem upptäcktes vid kodgranskning:

- `party_positions.describe` beskrev status `no_vote` som verifierad
  acklamation. Statusen betyder dock bara att ingen voteringspost hittades
  vid sammanställningen. För NU17 2022/23 punkt 4 och NU20 2024/25 punkt 5
  fanns denna status och tomma voteringslistor. Det faktiska beslutsförfarandet
  verifierades inte i denna kontroll.
- Den nya källfiltreringen tog endast med numeriska hänvisningar, trots
  att röstfakta använder namngivna ärendehänvisningar.

Därefter ändrades beskrivningen till saknad voteringsuppgift
och osäkerhet om röstning, acklamation och beslutsutfall. Saknad partipost
beskrivs också som saknad data, inte som bevis för uteblivet deltagande.
Systeminstruktionen och kunskapsexemplet om acklamation förtydligades.
Ingen data behöver byggas om eftersom beskrivningarna skapas vid frågetillfället.

Källistan tar nu även med citerade ärendehänvisningar som matchar hämtat
DID-material, inklusive flera hänvisningar i samma hakparentes. Upprepade
ärendehänvisningar listas en gång. URL används när sådan finns i metadata;
annars visas betänkandets beteckning, riksmöte och punkt.

Alla 19 automatiska tester passerade. Nya tester täcker saknad voteringsdata,
saknad partipost, namngivna och blandade hänvisningar, dubbletter samt
ärendehänvisningar som inte finns bland DID-träffarna. En ny Gemini-körning
återstår. Övriga påståenden i röstningssvaret har inte faktagranskats fullt ut.

### Uppföljning: blandade hänvisningar i röstningssvaret

En ny körning av S-frågan gav 10 träffar, 6 röstrader och
13 516 kontexttecken. Nu beskrev svaret saknade voteringsuppgifter för
NU17 2022/23 punkt 4 och NU20 2024/25 punkt 5 utan att påstå acklamation.
Namngivna hänvisningar visades i källistan. Däremot blandade modellen
ärendehänvisningar och avsnittsnummer i samma hakparentes, exempelvis
`[NU5 2023/24 punkt 2, 8]`, vilket gjorde att vissa avsnittsnummer saknades
i listan.

Därefter kompletterades tolkningen: fristående tal och
intervall efter kommatecken hanteras som avsnittsnummer även i blandade
block. Siffror inne i en ärendehänvisning tolkas inte som avsnittsnummer.
Instruktionen kräver nu separata hakparenteser för ärenden och avsnitt och
fullständiga ärendehänvisningar när flera beslutspunkter avses.

Alla 20 automatiska tester passerade, inklusive exempel från utskriften,
omvänd ordning, intervall och kontroll mot oavsiktlig tolkning av årtal
och punktnummer. Därefter kördes samma S-fråga igen: 10 träffar,
6 röstrader och 13 516 kontexttecken. Ärendehänvisningar och avsnittsnummer
skrevs nu i separata hakparenteser. Alla åtta citerade avsnittsnummer
([1], [4]–[10]) och alla sex unika ärendehänvisningar fanns i källistan.
Svaret beskrev fortsatt voteringsuppgifterna för NU20 2024/25 punkt 5 och
NU17 2022/23 punkt 4 som saknade och påstod inte acklamation.
Önskad presentation observerades i denna körning. Källpost [10] slutade
med ett tomt ”punkt”; det är en kvarstående presentationsbrist för material
utan punktnummer. Ingen ytterligare kodändring gjordes.
Källfiltreringen kan inte avgöra modellens avsikt vid en tvetydig förkortning:
ett fristående tal efter kommatecken behandlas som ett textavsnittsnummer.
Faktakvaliteten i övriga röstpåståenden är ännu inte fullständigt granskad.

### Faktastickprov: S och reservation 7 i NU20 2024/25, punkt 4

Påståendet ”S röstade för sin egen reservation 7 på punkt 4” kontrollerades
mot Riksdagens omröstningssida:
https://www.riksdagen.se/sv/dokument-och-lagar/dokument/omrostning/omrostning-betankande-202425nu20-finansiering_hc19nu20p4/

Datumet är 2025-05-21. Utskottets förslag ställdes mot reservation 7 (S).
S hade 0 ja, 93 nej, 0 avstående och 13 frånvarande. Nej betydde i denna
omröstning stöd för reservationen, inte nej till reservationen. Den lokala
posten `2024/25|NU20|4` i `out/positions_did.jsonl` har samma datum,
reservationsnummer och röstfördelning för S. Påståendet är därmed styrkt
för de röstande S-ledamöterna i denna huvudvotering.

Beslutssidan anger att kammaren biföll utskottets förslag; S reservation
vann alltså inte:
https://www.riksdagen.se/sv/dokument-och-lagar/dokument/betankande/finansiering-och-riskdelning-vid-investeringar-i_hc01nu20/omrostning/alternativ-till-regeringens-forslag~f6f7cd39-dd33-4b90-966e-5e6e8bc10485/

Detta är ett verifierat stickprov, inte en kontroll av alla röstpåståenden
eller av sammanfattningen av reservationens hela sakpolitiska innehåll.
Ingen kodändring gjordes.

### Faktastickprov: NU5 2023/24 punkt 2, sakfråga och motivfråga

Riksdagens originalunderlag:
https://www.riksdagen.se/sv/dokument-och-lagar/dokument/omrostning/omrostning-betankande-202324nu5-ny-karnkraft-i_hb19nu5p2/

Den 29 november 2023 hade S i sakfrågan 93 nej, 0 ja, 0 avstående och
14 frånvarande. Utskottets förslag ställdes mot reservation 2 (S, C).
Påståendet att S stödde reservation 2 är styrkt. Den lokala posten
`2023/24|NU5|2` överensstämmer med originalets siffror.

I den separata motivfrågan hade S 3 ja, 0 nej, 90 avstående och 14
frånvarande. Där ställdes utskottets förslag mot reservation 3 (V, MP).
Chatbotens sammanfattning att S ”röstade mot” motivreservationen är
missvisande: de flesta S-ledamöterna avstod. Lokal data har rätt siffror
och `stance: Avstår`, men `party_positions.describe` behandlar allt annat
än `Nej` som ”röstade mot” i grenen för motivfrågor. Felet uppstår alltså
i kodens textbeskrivning före Gemini-anropet.

Även originalunderlaget för punkt 4 visar huvudsakligen avstående hos S
(91 avstående, 2 ja, 0 nej, 14 frånvarande):
https://www.riksdagen.se/sv/dokument-och-lagar/dokument/omrostning/omrostning-betankande-202324nu5-ny-karnkraft-i_hb19nu5p4/

Därefter rättades textbeskrivningen för motivfrågor.
Den skiljer nu stöd, motstånd, avstående och frånvaro åt. Vid olika röster
inom partiet anges att ledamöterna röstade olika, i stället för att tillskriva
hela partiet en enda hållning. Datum och antal ja, nej, avstående och
frånvarande anges, med förklaring av hur ja och nej ska tolkas.

Alla 21 automatiska tester passerade. Det nya testet täcker sex fall,
inklusive S:s faktiska röstfördelningar för båda motivomröstningarna.
Beskrivningarna kontrollerades också direkt på de två lokala posterna:
punkt 2 anger 3 ja och 90 avstående, punkt 4 anger 2 ja och 91 avstående,
båda med 0 nej och 14 frånvarande. Stödet för reservation 2 i den separata
sakfrågan finns kvar i beskrivningen.

Därefter kördes samma fråga igen med Gemini: 10 träffar,
6 röstrader och 13 910 kontexttecken. Svaret återgav 3 ja, 0 nej, 90 avstående
och 14 frånvarande i punkt 2. För punkt 4 återgavs 2 ja, 91 avstående och
14 frånvarande (0 nej utelämnades). Avstående beskrevs inte längre som
motstånd. Saknade voteringsuppgifter angavs fortsatt som saknade. Alla
citerade avsnittsnummer och ärendehänvisningar fanns i källistan.

Rättelsen av motivfrågorna har därmed observerats även i ett genererat
svar. En kvarstående källprecision är att första meningen om stödet för
reservation 2 endast citerar textavsnitt [8], trots att själva röstpåståendet
bör hänvisa till röstunderlaget. Källistan är komplett i förhållande till
hänvisningarna, men detta bevisar inte stöd för varje enskilt påstående.
Övriga sakuppgifter har inte fullständigt granskats och ingen ytterligare
kodändring gjordes efter körningen.

## Kvar till gruppens rapport och demo

Gemini-svar genererades inte i de automatiska testerna: användarens API-nyckel
är inställd i en annan terminal och är inte tillgänglig för testprocessen.
De användarkörda exemplen ovan kompletterar de tekniska testerna.
Kör samma frågor i båda versionerna och bedöm manuellt sakuppgifter,
avsändare, tidsperiod, läsbara källhänvisningar och skillnaden förslag/beslut.
Lägg till frågor där underlag saknas samt frågor som försöker få systemet att
påstå att en motion redan blivit lag. Låt gärna en annan gruppmedlem granska
facit och svar för att minska risken för en alltför välvillig bedömning.

Påstå inte att fler motioner, eller fler undertecknare, betyder bättre eller
mer representativ politik. Urvalsregeln utesluter avsiktligt enpersons-motioner.
