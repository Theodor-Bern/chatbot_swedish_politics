"""Collect what each party says about each issue, from their own websites.

Two phases, deliberately separate — the same lesson the betänkande parser
taught: never couple fetching to parsing.

    python3 scrapers/scraper.py discover MP    # dry run: what would be fetched
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

# One request per second. None of these sites are large and there is no
# reason to hurry.
DELAY_SECONDS = 1.0

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# Shorter than this and trafilatura found navigation, not an article.
MIN_TEXT_CHARS = 150


@dataclass
class PartySite:
    """Everything party-specific lives here, so adding a party is config.

    Two discovery sources, unioned, because neither is reliable alone: V's
    page sitemap is complete, while MP's sitemap lists 8 of ~55 topic pages.
    A party is only missed if BOTH its sitemap and its A-Ö index break.

    index_urls defaults to empty, so a party configured before this existed
    keeps exactly the behaviour it had.

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


PARTIES = {
    # V is left exactly as it was when it collected 102 pages successfully.
    # Its page sitemap is complete, so it needs no index crawl.
    "V": PartySite(
        code="V",
        name="Vänsterpartiet",
        sitemap="https://www.vansterpartiet.se/page-sitemap.xml",
        topic_pattern=re.compile(r"/var-politik/politik-a-o/[^/]+/?$"),
        robots_checked="2026-09-08",   # only /wp-admin/ disallowed
    ),
    # MP's sitemap lists 8 topic pages; its A-Ö index lists ~55. Without the
    # index crawl we would silently collect a tenth of their platform.
    "MP": PartySite(
        code="MP",
        name="Miljöpartiet de gröna",
        sitemap="https://www.mp.se/sitemap.xml",
        index_urls=["https://www.mp.se/politik/"],
        topic_pattern=re.compile(r"/politik/[^/]+/?$"),
        robots_checked="2026-09-08",   # only /wp-content/uploads/ir_cache/
    ),
    # S, SD, M, C, KD, L go here — one entry each, no new code.
}


def slug_of(url):
    """Last path segment: .../politik/energi/ -> "energi"."""
    return url.rstrip("/").rsplit("/", 1)[-1]


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

def locs(xml_bytes, tag):
    """<loc> values that are children of <tag>, namespace or not.

    The tag matters: asked for "sitemap", a plain page sitemap must return
    nothing. A fallback that returns every <loc> in the document would make
    a page sitemap look like a sitemap index, and we would then try to fetch
    all ~100 pages as if each were a child sitemap.
    """
    root = ET.fromstring(xml_bytes)
    found = [e.text.strip()
             for e in root.findall(f".//sm:{tag}/sm:loc", SITEMAP_NS)
             if e.text]
    if found:
        return found

    # Some generators omit the namespace entirely; walk it by local name.
    out = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != tag:
            continue
        for child in element:
            if child.tag.rsplit("}", 1)[-1] == "loc" and child.text:
                out.append(child.text.strip())
    return out


def from_sitemap(site, session):
    """URLs from the sitemap, following one level of sitemap index."""
    if not site.sitemap:
        return set()
    try:
        r = session.get(site.sitemap, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"  sitemap unavailable ({e})")
        return set()

    children = locs(r.content, "sitemap")
    if children:
        urls = set()
        for child in children:
            time.sleep(DELAY_SECONDS)
            try:
                cr = session.get(child, timeout=30)
                cr.raise_for_status()
                urls.update(locs(cr.content, "url"))
            except Exception as e:
                print(f"  child sitemap {child} failed ({e})")
        return urls
    return set(locs(r.content, "url"))


def from_index(site, session):
    """URLs linked from the party's own A-Ö index page(s)."""
    urls = set()
    for index_url in site.index_urls:
        time.sleep(DELAY_SECONDS)
        try:
            r = session.get(index_url, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"  index {index_url} failed ({e})")
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            urls.add(urljoin(index_url, a["href"].split("#")[0]))
    return urls


def discover(site, session, verbose=True):
    """Topic page URLs, from sitemap and index page together."""
    sitemap_urls = {u for u in from_sitemap(site, session)
                    if site.topic_pattern.search(u)}
    index_urls = {u for u in from_index(site, session)
                  if site.topic_pattern.search(u)}
    both = sorted(sitemap_urls | index_urls)
    if verbose:
        print(f"  sitemap: {len(sitemap_urls)}   index page: {len(index_urls)}"
              f"   union: {len(both)}")
        only_index = len(index_urls - sitemap_urls)
        if only_index:
            print(f"  ({only_index} topics the sitemap does not list)")
    return both


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
    for url in discover(site, session_with_agent()):
        print(f"    {url}")


def fetch(party_code, data_dir=DEFAULT_DATA_DIR):
    """Save the raw HTML of every topic page. Resumable: a page already on
    disk is skipped, so an interrupted run costs nothing to repeat."""
    site = PARTIES[party_code]
    out_dir = Path(data_dir) / party_code
    out_dir.mkdir(parents=True, exist_ok=True)

    session = session_with_agent()
    print(f"{site.name}:")
    urls = discover(site, session)

    index = []
    for n, url in enumerate(urls, 1):
        slug = slug_of(url)
        path = out_dir / f"{slug}.html"
        if path.exists():
            print(f"  [{n}/{len(urls)}] {slug} (cached)")
        else:
            time.sleep(DELAY_SECONDS)
            try:
                r = session.get(url, timeout=30)
                r.raise_for_status()
            except Exception as e:
                print(f"  [{n}/{len(urls)}] {slug} FAILED: {e}")
                continue
            path.write_text(r.text, encoding="utf-8")
            print(f"  [{n}/{len(urls)}] {slug} ({len(r.text)} bytes)")
        index.append({"slug": slug, "url": url, "file": path.name})

    (out_dir / "index.json").write_text(
        json.dumps({"party": party_code, "name": site.name,
                    "fetched": date.today().isoformat(),
                    "robots_checked": site.robots_checked,
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
    with open(out_path, "w", encoding="utf-8") as fh:     # not "a"
        for page in meta["pages"]:
            raw = (in_dir / page["file"]).read_text(encoding="utf-8")
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

            fh.write(json.dumps({
                "chunk_id": f"{party_code}:said:{page['slug']}",
                "layer": "said",          # never mix with the "did" layer
                "parti": party_code,
                "parti_namn": site.name,
                "sakfraga": page["slug"],
                "heading": heading,
                "text": text,
                "url": page["url"],
                "hamtad": meta["fetched"],
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