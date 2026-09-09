"""Collect what each party says about each issue, from their own websites.

Two phases, deliberately separate — the same lesson the betänkande parser
taught: never couple fetching to parsing.

    python3 scrapers/scraper.py discover SD    # dry run: what would be fetched
    python3 scrapers/scraper.py fetch V        # network, polite, resumable
    python3 scrapers/scraper.py extract V      # offline, repeatable

    python3 scrapers/scraper.py fetch all      # every party in PARTIES
    python3 scrapers/scraper.py extract all

Paths default to <repo>/data/parties and <repo>/out and are resolved against
the repository root, so the commands work from any working directory. Pass
them explicitly to override.

Riksdagen's data is a record of what parties DID. This is what they SAY.
Both are needed, and the bot must never present one as the other: a party's
own website is self-description, not an independent record.

Dependencies: requests, trafilatura, beautifulsoup4
    pip install requests trafilatura beautifulsoup4
"""

import gzip
import hashlib
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import requests
import trafilatura
from bs4 import BeautifulSoup

# This file lives in <repo>/scrapers/, so the repo root is one level up.
# Anchoring here means the script does not care where it is run from.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "parties"
DEFAULT_OUT_DIR = REPO_ROOT / "out"

# Neutral and non-deceptive: says what the client is, identifies no one, and
# does not pretend to be a browser. Set at all because a few sites reject
# the default python-requests agent outright.
USER_AGENT = "Mr Robot"

# Default one request per second. A party whose robots.txt asks for slower
# gets its own crawl_delay below.
DELAY_SECONDS = 1.0

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# Shorter than this and trafilatura found navigation, not an article.
MIN_TEXT_CHARS = 150


@dataclass
class PartySite:
    """Everything party-specific lives here, so adding a party is config.

    topic_pattern MUST be anchored at the party's own domain. Several of
    these sites host municipal and regional branches under the same domain
    (mp.se/vellinge/politik/klimat/), and an unanchored pattern pulls in
    hundreds of pages of local politics that are not the party's national
    position.

    index_only decides whether the party's A-Ö page defines membership. It
    is False everywhere, because no party's index turned out to be a
    complete register of its own policy pages: MP's A-Ö omits abort,
    pensioner and vattenkraft, C's lists only the 29 top-level topics and
    not their sub-pages, and S's and SD's have no link list at all. So the
    path pattern defines membership and the index contributes alias labels.
    What the index does list is recorded per page as sources/in_index, so a
    curated view is one filter away — decide at retrieval time, not by
    throwing pages away at collection time.

    crawl_delay is per party because robots.txt is. SD asks for ten seconds
    between requests and that is a directive, not advice.

    robots_checked records the date someone actually read the party's
    robots.txt and confirmed the topic pages are allowed. Keep it honest —
    it goes in the report.
    """
    code: str
    name: str
    topic_pattern: re.Pattern
    robots_checked: str
    sitemap: str = ""
    index_urls: list = field(default_factory=list)
    index_only: bool = False
    crawl_delay: float = DELAY_SECONDS


PARTIES = {
    "V": PartySite(
        code="V",
        name="Vänsterpartiet",
        sitemap="https://www.vansterpartiet.se/page-sitemap.xml",
        index_urls=["https://www.vansterpartiet.se/var-politik/politik-a-o/"],
        topic_pattern=re.compile(
            r"^https://www\.vansterpartiet\.se/var-politik/politik-a-o/"
            r"[^/]+/?$"),
        robots_checked="2026-09-08",   # only /wp-admin/ disallowed
    ),
    "MP": PartySite(
        code="MP",
        name="Miljöpartiet de gröna",
        sitemap="https://www.mp.se/sitemap.xml",
        index_urls=["https://www.mp.se/politik/"],
        topic_pattern=re.compile(r"^https://www\.mp\.se/politik/[^/]+/?$"),
        robots_checked="2026-09-08",   # only /wp-content/uploads/ir_cache/
    ),
    "S": PartySite(
        code="S",
        name="Socialdemokraterna",
        sitemap="https://www.socialdemokraterna.se/rest-api/sitemapXml",
        # No index_urls: their A-Ö page is a JavaScript search box, not a
        # link list, so there is nothing for BeautifulSoup to find.
        topic_pattern=re.compile(
            r"^https://www\.socialdemokraterna\.se/var-politik/a-till-o/"
            r"[^/]+/?$"),
        robots_checked="2026-09-08",   # nothing under /var-politik/ blocked
    ),
    "C": PartySite(
        code="C",
        name="Centerpartiet",
        sitemap="https://www.centerpartiet.se/sitemapindex.xml",
        index_urls=["https://www.centerpartiet.se/centerpartiets-politik/"
                    "centerpartiets-politik-a-o"],
        # Two levels: 29 topics, each with sub-pages such as
        # …/digitalisering/artificiell-intelligens-ai. The sub-pages hold the
        # specific answers and are the granularity people ask about.
        topic_pattern=re.compile(
            r"^https://www\.centerpartiet\.se/centerpartiets-politik/"
            r"centerpartiets-politik-a-o/[^/]+(/[^/]+)?/?$"),
        robots_checked="2026-09-08",   # same Sitevision ruleset as S
    ),
    "SD": PartySite(
        code="SD",
        name="Sverigedemokraterna",
        sitemap="https://www.sd.se/sitemap_index.xml",
        # They publish a dedicated a-o-matters sitemap: 368 pages under
        # /a-till-o/, their register in full. Their /var-politik/ page is a
        # thematic summary, not a link list, so there is no index to crawl.
        topic_pattern=re.compile(r"^https://www\.sd\.se/a-till-o/[^/]+/?$"),
        crawl_delay=10.0,              # their robots.txt asks for it
        robots_checked="2026-09-08",
    ),
        "M": PartySite(
        code="M",
        name="Moderaterna",
        sitemap="https://moderaterna.se/sitemap_index.xml",
        index_urls=["https://moderaterna.se/var-politik/"],
        # No "www." — moderaterna.se serves without it, unlike every other
        # party so far. And the anchor matters more here than anywhere else:
        # they run 26 regional multisites under the same domain
        # (moderaterna.se/skane/var-politik/…), each with its own sitemap
        # listed in robots.txt.
        topic_pattern=re.compile(
            r"^https://moderaterna\.se/var-politik/[^/]+/?$"),
        robots_checked="2026-09-09",
    ),
        "KD": PartySite(
        code="KD",
        name="Kristdemokraterna",
        # No sitemap at all: robots.txt declares none, and /sitemap.xml,
        # /sitemapindex.xml and /rest-api/sitemapXml all 404 or 400. Their
        # A-Ö page lists every topic as a plain HTML link, so the index is
        # the only source — and therefore the only membership rule.
        index_urls=["https://kristdemokraterna.se/var-politik/"
                    "politik-a-till-o"],
        topic_pattern=re.compile(
            r"^https://kristdemokraterna\.se/var-politik/politik-a-till-o/"
            r"[^/]+/?$"),
        robots_checked="2026-09-09",   # same Sitevision ruleset as S and C
    ),
        "L": PartySite(
        code="L",
        name="Liberalerna",
        sitemap="https://www.liberalerna.se/sitemap_index.xml",
        index_urls=["https://www.liberalerna.se/politik/"],
        topic_pattern=re.compile(
            r"^https://www\.liberalerna\.se/politik/[^/]+/?$"),
        robots_checked="2026-09-09",   # Yoast block, allt tillåtet
    ),
    # M, KD, L go here — one entry each, no new code.
}


def slug_of(url):
    """Filename-safe id from the last path segment."""
    return url.rstrip("/").rsplit("/", 1)[-1] or "index"


def party_codes(argument):
    """"all" expands to every configured party; anything else must exist."""
    if argument == "all":
        return list(PARTIES)
    if argument not in PARTIES:
        raise SystemExit(
            f"unknown party {argument!r}. known: {', '.join(PARTIES)} (or 'all')")
    return [argument]


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def xml_bytes(response):
    """Sitemap bytes, decompressed and cleaned up enough to parse.

    Two things get in the way in practice:

    Centerpartiet serves its child sitemap as sitemap1.xml.gz, and requests
    only decompresses automatically when the server sets Content-Encoding —
    a .gz file normally arrives as application/x-gzip without it, so check
    the magic bytes rather than trusting the headers.

    Sverigedemokraterna's Yoast sitemap emits a blank line before the <?xml?>
    declaration, which ElementTree rejects outright ("XML or text
    declaration not at start of entity"). A BOM does the same. Neither is
    valid XML, and neither is worth failing a whole party over.
    """
    data = response.content
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    if data[:3] == b"\xef\xbb\xbf":              # UTF-8 BOM
        data = data[3:]
    data = data.lstrip()
    if not data.startswith(b"<?xml"):
        start = data.find(b"<?xml")              # stray output before it
        if start > 0:
            data = data[start:]
    return data


def locs(xml_data, tag):
    """[(url, lastmod)] for every <loc> under <tag>, namespace or not.

    The tag scoping matters: asked for "sitemap", a plain page sitemap must
    return nothing. A fallback that returned every <loc> in the document
    would make a page sitemap look like a sitemap index, and we would then
    fetch all ~100 pages as if each were a child sitemap.

    lastmod matters because parties publish campaign material from years ago
    beside current policy, and undated retrieval reports the two alike.
    """
    root = ET.fromstring(xml_data)

    def pair(element):
        url = lastmod = None
        for child in element:
            name = child.tag.rsplit("}", 1)[-1]
            if name == "loc" and child.text:
                url = child.text.strip()
            elif name == "lastmod" and child.text:
                lastmod = child.text.strip()[:10]      # date part only
        return url, lastmod

    out = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != tag:
            continue
        url, lastmod = pair(element)
        if url:
            out.append((url, lastmod))
    return out


def from_sitemap(site, session):
    """{url: lastmod} from the sitemap, following one level of index."""
    if not site.sitemap:
        return {}
    try:
        r = session.get(site.sitemap, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"  sitemap unavailable ({e})")
        return {}

    children = locs(xml_bytes(r), "sitemap")
    if children:
        found = {}
        for child_url, _ in children:
            time.sleep(site.crawl_delay)
            try:
                cr = session.get(child_url, timeout=30)
                cr.raise_for_status()
                found.update(dict(locs(xml_bytes(cr), "url")))
            except Exception as e:
                print(f"  child sitemap {child_url} failed ({e})")
        return found
    return dict(locs(xml_bytes(r), "url"))


def from_index(site, session):
    """{url: {link labels}} from the party's own A-Ö index page(s).

    The label matters as much as the URL. C lists "Artificiell intelligens,
    AI" pointing at /digitalisering, and MP lists "Elbilar" pointing at
    /bilar-och-bransle — the party's own vocabulary for the topic, which a
    reader is far more likely to type than the slug.
    """
    found = {}
    for index_url in site.index_urls:
        time.sleep(site.crawl_delay)
        try:
            r = session.get(index_url, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"  index {index_url} failed ({e})")
            continue
        # r.content, not r.text: mp.se sends no charset in its Content-Type,
        # so requests falls back to ISO-8859-1 per the old HTTP spec and
        # every å ä ö arrives as mojibake ("Ãldreomsorg"). Given bytes,
        # BeautifulSoup reads the <meta charset> instead.
        soup = BeautifulSoup(r.content, "html.parser")
        for a in soup.find_all("a", href=True):
            url = urljoin(index_url, a["href"].split("#")[0])
            label = " ".join(a.get_text().split())
            found.setdefault(url, set())
            if label:
                found[url].add(label)
    return found


def discover(site, session, verbose=True):
    """[{url, lastmod, sources, labels}] — the pages to collect, where each
    came from, and what the party calls it."""
    sitemap = {u: lm for u, lm in from_sitemap(site, session).items()
               if site.topic_pattern.search(u)}
    index_labels = {u: labels
                    for u, labels in from_index(site, session).items()
                    if site.topic_pattern.search(u)}
    index = set(index_labels)

    if site.index_urls and site.index_only:
        selected = index                      # the A-Ö register decides
    else:
        selected = set(sitemap) | index

    pages = []
    for url in sorted(selected):
        sources = []
        if url in sitemap:
            sources.append("sitemap")
        if url in index:
            sources.append("index")
        pages.append({"url": url,
                      "lastmod": sitemap.get(url),
                      "sources": sources,
                      "labels": sorted(index_labels.get(url, ()))})

    if verbose:
        dated = sum(1 for p in pages if p["lastmod"])
        print(f"  sitemap: {len(sitemap)}   index page: {len(index)}"
              f"   selected: {len(pages)}   with a date: {dated}")
        n_labels = sum(len(p["labels"]) for p in pages)
        if index and n_labels > len(index):
            print(f"  ({n_labels} index entries for {len(index)} indexed "
                  f"pages — the rest are aliases)")
        if site.index_urls and site.index_only:
            dropped = len(set(sitemap) - index)
            if dropped:
                print(f"  ({dropped} sitemap pages not on the A-Ö index, "
                      f"skipped)")
        elif index:
            only_index = len(index - set(sitemap))
            if only_index:
                print(f"  ({only_index} not listed in the sitemap)")

    # A JavaScript-driven A-Ö page has no links for BeautifulSoup to find,
    # so index_only would silently select nothing while the sitemap is full
    # of perfectly good pages. Several party sites are built this way.
    if not pages and sitemap:
        print(f"  WARNING: {len(sitemap)} pages matched the pattern but none "
              f"were selected. If this party's A-Ö page is JavaScript-driven "
              f"there are no links to scrape — clear index_urls or leave "
              f"index_only False and let the path pattern decide.")

    return pages


# --------------------------------------------------------------------------
# Phase 1: fetch
# --------------------------------------------------------------------------

def session_with_agent():
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def show_discovery(party_code):
    """Dry run — check a new party's pattern before fetching anything."""
    site = PARTIES[party_code]
    print(f"{site.name}:")
    for page in discover(site, session_with_agent()):
        marks = "+".join(page["sources"])
        labels = f"  [{'; '.join(page['labels'])}]" if page["labels"] else ""
        print(f"    {page['lastmod'] or '----------'}  {marks:14s} "
              f"{page['url']}{labels}")


def fetch(party_code, data_dir=DEFAULT_DATA_DIR):
    """Save the raw HTML of every topic page. Resumable: a page already on
    disk is skipped, so an interrupted run costs nothing to repeat."""
    site = PARTIES[party_code]
    out_dir = Path(data_dir) / party_code
    out_dir.mkdir(parents=True, exist_ok=True)

    session = session_with_agent()
    print(f"{site.name}:")
    pages = discover(site, session)

    index = []
    used = {}
    for n, page in enumerate(pages, 1):
        url = page["url"]
        slug = slug_of(url)
        if used.get(slug, url) != url:
            # Nested topics can share a final segment (…/klimat/energi and
            # …/energi). Without this the second page silently overwrites
            # the first one's HTML file.
            parent = url.rstrip("/").rsplit("/", 2)[-2]
            slug = f"{parent}__{slug}"
        used[slug] = url

        path = out_dir / f"{slug}.html"
        if path.exists():
            print(f"  [{n}/{len(pages)}] {slug} (cached)")
        else:
            time.sleep(site.crawl_delay)
            try:
                r = session.get(url, timeout=30)
                r.raise_for_status()
            except Exception as e:
                print(f"  [{n}/{len(pages)}] {slug} FAILED: {e}")
                continue
            # Bytes, not r.text. When a server declares no charset, requests
            # decodes as ISO-8859-1 and write_text then re-encodes to UTF-8:
            # two wrongs that do not cancel, and the file on disk is
            # permanently double-encoded. Raw data on disk stays raw.
            path.write_bytes(r.content)
            print(f"  [{n}/{len(pages)}] {slug} ({len(r.content)} bytes)")

        index.append({"slug": slug, "url": url, "file": path.name,
                      "lastmod": page["lastmod"], "sources": page["sources"],
                      "labels": page["labels"]})

    (out_dir / "index.json").write_text(
        json.dumps({"party": party_code, "name": site.name,
                    "fetched": date.today().isoformat(),
                    "robots_checked": site.robots_checked,
                    "crawl_delay": site.crawl_delay,
                    "index_only": bool(site.index_urls and site.index_only),
                    "pages": index}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"saved {len(index)} pages to {out_dir}")


# --------------------------------------------------------------------------
# Phase 2: extract
# --------------------------------------------------------------------------

def extract(party_code, data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR):
    """Pull the article text out of the saved HTML.

    One file per party, overwritten on every run. An append-mode shared file
    silently doubles its contents when you re-extract, which is exactly the
    kind of corruption that surfaces three weeks later as an unexplainable
    retrieval result. Concatenate at index time instead.

    trafilatura rather than per-party CSS selectors: eight parties means
    eight site redesigns to survive, and a readability extractor degrades
    gracefully where a hand-written selector simply breaks. Markdown output
    keeps the "Miljöpartiet vill…" bullet lists, which are the most concrete
    statements on the page and must not be flattened away.
    """
    site = PARTIES[party_code]
    in_dir = Path(data_dir) / party_code
    index_path = in_dir / "index.json"
    if not index_path.exists():
        raise SystemExit(
            f"no index.json in {in_dir} — run 'fetch {party_code}' first")

    meta = json.loads(index_path.read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"positions_said_{party_code}.jsonl"

    lengths, skipped = [], []
    with open(out_path, "w", encoding="utf-8") as fh:      # not "a"
        for page in meta["pages"]:
            # Bytes: trafilatura reads the document's own <meta charset>,
            # which is more reliable than assuming UTF-8 across five CMSes.
            raw = (in_dir / page["file"]).read_bytes()
            text = trafilatura.extract(
                raw,
                output_format="markdown",
                include_links=False,
                include_comments=False,
                include_tables=True,
            )
            if not text or len(text) < MIN_TEXT_CHARS:
                skipped.append(page["slug"])
                continue

            metadata = trafilatura.extract_metadata(raw)
            heading = (metadata.title if metadata and metadata.title
                       else page["slug"].replace("-", " ").capitalize())
            sources = page.get("sources", [])
            labels = page.get("labels", [])

            # The aliases go into the embedded text, not only the metadata,
            # so that "vad tycker C om AI?" can reach a page whose slug and
            # heading both say "digitalisering".
            alias_line = f"({'; '.join(labels)})\n" if labels else ""

            fh.write(json.dumps({
                "chunk_id": f"{party_code}:said:{page['slug']}",
                "layer": "said",          # never mix with the "did" layer
                "parti": party_code,
                "parti_namn": site.name,
                "sakfraga": page["slug"],
                "heading": heading,
                "labels": labels,
                "text": text,
                "text_for_embedding": f"{heading}\n{alias_line}{text}",
                "url": page["url"],
                "hamtad": meta["fetched"],
                # When the party last touched the page. Weight retrieval by
                # this: a 2021 campaign page is not a current position.
                "lastmod": page.get("lastmod"),
                "sources": sources,
                "in_index": "index" in sources,
                # Lets a later run detect that a party changed its position
                # without diffing prose by hand.
                "content_hash": hashlib.sha256(
                    text.encode("utf-8")).hexdigest()[:16],
            }, ensure_ascii=False) + "\n")
            lengths.append(len(text))

    print(f"{site.name}: wrote {len(lengths)} positions to {out_path}")
    if lengths:
        lengths.sort()
        print(f"  text length: median {lengths[len(lengths) // 2]}, "
              f"p90 {lengths[int(len(lengths) * 0.9)]}, max {lengths[-1]}")
    if skipped:
        print(f"  too short / no content ({len(skipped)}): "
              f"{', '.join(skipped[:10])}")


# --------------------------------------------------------------------------

USAGE = """usage:
    python3 scrapers/scraper.py discover <party|all>
    python3 scrapers/scraper.py fetch    <party|all> [data_dir]
    python3 scrapers/scraper.py extract  <party|all> [data_dir] [out_dir]
"""

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    command, target = sys.argv[1], sys.argv[2]
    if command == "discover":
        for code in party_codes(target):
            show_discovery(code)
    elif command == "fetch":
        data_dir = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_DATA_DIR
        for code in party_codes(target):
            fetch(code, data_dir)
    elif command == "extract":
        data_dir = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_DATA_DIR
        out_dir = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_OUT_DIR
        for code in party_codes(target):
            extract(code, data_dir, out_dir)
    else:
        print(USAGE)
        sys.exit(1)