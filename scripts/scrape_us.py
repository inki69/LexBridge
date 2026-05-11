import requests
from bs4 import BeautifulSoup
import os
import time
import json
from datetime import datetime

# ============================================================
# All URLs verified working (HTTP 200) as of 2026-05-04
# DOL main site and BLS block scrapers (403/Cloudflare).
# PDFs also blocked. We use EEOC, Cornell LII, OSHA instead.
# ============================================================

HTML_SOURCES = {
    # -- Cornell LII WEX (additional topics replacing JS-blocked EEOC pages) --
    "cornell_harassment": {
        "url": "https://www.law.cornell.edu/wex/sexual_harassment",
        "source": "Cornell Law",
        "topic": "sexual harassment"
    },
    "cornell_hostile_workplace": {
        "url": "https://www.law.cornell.edu/wex/hostile_work_environment",
        "source": "Cornell Law",
        "topic": "hostile work environment"
    },
    "cornell_workers_comp": {
        "url": "https://www.law.cornell.edu/wex/workers_compensation",
        "source": "Cornell Law",
        "topic": "workers compensation"
    },
    "cornell_unemployment": {
        "url": "https://www.law.cornell.edu/wex/unemployment_compensation",
        "source": "Cornell Law",
        "topic": "unemployment"
    },

    # -- Cornell LII (WEX legal encyclopedia) --
    "cornell_at_will": {
        "url": "https://www.law.cornell.edu/wex/at-will_employment",
        "source": "Cornell Law",
        "topic": "at-will employment"
    },
    "cornell_wrongful_termination": {
        "url": "https://www.law.cornell.edu/wex/wrongful_termination",
        "source": "Cornell Law",
        "topic": "wrongful termination"
    },
    "cornell_discrimination": {
        "url": "https://www.law.cornell.edu/wex/employment_discrimination",
        "source": "Cornell Law",
        "topic": "discrimination"
    },
    "cornell_flsa": {
        "url": "https://www.law.cornell.edu/wex/fair_labor_standards_act",
        "source": "Cornell Law",
        "topic": "FLSA"
    },
    "cornell_minimum_wage": {
        "url": "https://www.law.cornell.edu/wex/minimum_wage",
        "source": "Cornell Law",
        "topic": "minimum wage"
    },
    "cornell_fmla": {
        "url": "https://www.law.cornell.edu/wex/fmla",
        "source": "Cornell Law",
        "topic": "FMLA"
    },

    # -- Cornell LII (US Code statutory text: raw, unstructured legal content) --
    "cornell_flsa_statute": {
        "url": "https://www.law.cornell.edu/uscode/text/29/207",
        "source": "Cornell Law",
        "topic": "overtime statute"
    },
    "cornell_fmla_statute": {
        "url": "https://www.law.cornell.edu/uscode/text/29/2601",
        "source": "Cornell Law",
        "topic": "FMLA statute"
    },

    # -- OSHA --
    "osha_workers": {
        "url": "https://www.osha.gov/workers",
        "source": "OSHA",
        "topic": "worker rights"
    },
    "whistleblowers": {
        "url": "https://www.whistleblowers.gov/",
        "source": "OSHA",
        "topic": "whistleblower protection"
    },
}

# ============================================================
# Setup
# ============================================================
HTML_DIR = "data/raw/us/html"
METADATA_DIR = "data/raw/us"

os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
})

metadata = []
total = len(HTML_SOURCES)

print("=" * 60)
print("LexBridge Scraper — U.S. Employment Law")
print(f"Sources: {total} (EEOC, Cornell LII, OSHA)")
print("=" * 60)

for doc_id, (key, info) in enumerate(HTML_SOURCES.items()):
    url = info["url"]
    print(f"[{doc_id + 1}/{total}] {key}")

    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()

        html_path = os.path.join(HTML_DIR, f"{key}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(r.text)

        soup = BeautifulSoup(r.text, "lxml")
        title_tag = soup.find("title")
        title_text = title_tag.string.strip() if title_tag else info["topic"].title()

        metadata.append({
            "doc_id": doc_id,
            "key": key,
            "url": url,
            "title": title_text,
            "country": "USA",
            "language": "en",
            "source": info["source"],
            "topic": info["topic"],
            "format": "html",
            "file": f"{key}.html",
            "file_dir": "html",
            "scraped_at": datetime.now().isoformat(),
        })
        print(f"  [OK] Saved ({len(r.text):,} bytes)")

    except Exception as e:
        print(f"  [FAIL] {e}")

    time.sleep(2)

# ============================================================
# Save metadata
# ============================================================
metadata_path = os.path.join(METADATA_DIR, "scrape_metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print()
print("=" * 60)
print(f"Done! {len(metadata)}/{total} documents saved")
print(f"  HTML files : {HTML_DIR}")
print(f"  Metadata   : {metadata_path}")
print("=" * 60)
