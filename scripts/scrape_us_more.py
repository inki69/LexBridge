"""
Additional U.S. employment law sources — extends the existing corpus.
Reads existing scrape_metadata.json, scrapes new URLs, merges metadata.
Run: python scripts/scrape_us_more.py
"""
import requests
from bs4 import BeautifulSoup
import os
import time
import json
from datetime import datetime

# ============================================================
# Additional sources (beyond the original 14)
# Focus: civil rights statutes, labor law, benefits, specific protections
# ============================================================

ADDITIONAL_SOURCES = {
    # ── Cornell LII WEX — civil rights & discrimination ──────────────────────
    "cornell_title_vii": {
        "url": "https://www.law.cornell.edu/wex/title_vii",
        "source": "Cornell Law", "topic": "Title VII Civil Rights Act"
    },
    "cornell_ada": {
        "url": "https://www.law.cornell.edu/wex/americans_with_disabilities_act_(ada)",
        "source": "Cornell Law", "topic": "Americans with Disabilities Act"
    },
    "cornell_adea": {
        "url": "https://www.law.cornell.edu/wex/age_discrimination_in_employment_act_of_1967_(adea)",
        "source": "Cornell Law", "topic": "Age Discrimination in Employment Act"
    },
    "cornell_pregnancy_discrimination": {
        "url": "https://www.law.cornell.edu/wex/pregnancy_discrimination_act",
        "source": "Cornell Law", "topic": "pregnancy discrimination"
    },
    "cornell_equal_pay_act": {
        "url": "https://www.law.cornell.edu/wex/equal_pay_act_of_1963",
        "source": "Cornell Law", "topic": "Equal Pay Act"
    },
    "cornell_retaliation": {
        "url": "https://www.law.cornell.edu/wex/retaliation",
        "source": "Cornell Law", "topic": "workplace retaliation"
    },
    "cornell_eeoc": {
        "url": "https://www.law.cornell.edu/wex/equal_employment_opportunity_commission",
        "source": "Cornell Law", "topic": "EEOC"
    },

    # ── Cornell LII WEX — labor & unions ─────────────────────────────────────
    "cornell_nlra": {
        "url": "https://www.law.cornell.edu/wex/national_labor_relations_act",
        "source": "Cornell Law", "topic": "National Labor Relations Act"
    },
    "cornell_collective_bargaining": {
        "url": "https://www.law.cornell.edu/wex/collective_bargaining",
        "source": "Cornell Law", "topic": "collective bargaining"
    },
    "cornell_right_to_work": {
        "url": "https://www.law.cornell.edu/wex/right-to-work_laws",
        "source": "Cornell Law", "topic": "right-to-work laws"
    },
    "cornell_taft_hartley": {
        "url": "https://www.law.cornell.edu/wex/taft-hartley_act",
        "source": "Cornell Law", "topic": "Taft-Hartley Act"
    },

    # ── Cornell LII WEX — contracts, benefits, classifications ───────────────
    "cornell_severance": {
        "url": "https://www.law.cornell.edu/wex/severance_pay",
        "source": "Cornell Law", "topic": "severance pay"
    },
    "cornell_non_compete": {
        "url": "https://www.law.cornell.edu/wex/non-compete_clause",
        "source": "Cornell Law", "topic": "non-compete agreements"
    },
    "cornell_erisa": {
        "url": "https://www.law.cornell.edu/wex/employee_retirement_income_security_act_(erisa)",
        "source": "Cornell Law", "topic": "ERISA"
    },
    "cornell_independent_contractor": {
        "url": "https://www.law.cornell.edu/wex/independent_contractor",
        "source": "Cornell Law", "topic": "independent contractor"
    },
    "cornell_employee": {
        "url": "https://www.law.cornell.edu/wex/employee",
        "source": "Cornell Law", "topic": "employee classification"
    },
    "cornell_overtime": {
        "url": "https://www.law.cornell.edu/wex/overtime",
        "source": "Cornell Law", "topic": "overtime pay"
    },

    # ── Cornell LII US Code — statutory text ─────────────────────────────────
    "cornell_title_vii_statute": {
        "url": "https://www.law.cornell.edu/uscode/text/42/2000e-2",
        "source": "Cornell Law", "topic": "Title VII unlawful employment practices statute"
    },
    "cornell_ada_statute": {
        "url": "https://www.law.cornell.edu/uscode/text/42/12112",
        "source": "Cornell Law", "topic": "ADA employment discrimination statute"
    },
    "cornell_adea_statute": {
        "url": "https://www.law.cornell.edu/uscode/text/29/623",
        "source": "Cornell Law", "topic": "ADEA prohibited practices statute"
    },
    "cornell_equal_pay_statute": {
        "url": "https://www.law.cornell.edu/uscode/text/29/206",
        "source": "Cornell Law", "topic": "Equal Pay statute (29 USC 206)"
    },
    "cornell_nlra_statute": {
        "url": "https://www.law.cornell.edu/uscode/text/29/158",
        "source": "Cornell Law", "topic": "NLRA unfair labor practices statute"
    },
    "cornell_osha_statute": {
        "url": "https://www.law.cornell.edu/uscode/text/29/654",
        "source": "Cornell Law", "topic": "OSHA employer duties statute"
    },
    "cornell_whistleblower_statute": {
        "url": "https://www.law.cornell.edu/uscode/text/29/660",
        "source": "Cornell Law", "topic": "OSHA whistleblower statute"
    },
}

# ============================================================
HTML_DIR = "data/raw/us/html"
METADATA_DIR = "data/raw/us"
METADATA_PATH = os.path.join(METADATA_DIR, "scrape_metadata.json")

os.makedirs(HTML_DIR, exist_ok=True)

# Load existing metadata
existing_metadata = []
existing_keys = set()
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, encoding="utf-8") as f:
        existing_metadata = json.load(f)
        existing_keys = {entry["key"] for entry in existing_metadata}
    print(f"[INFO] Loaded {len(existing_metadata)} existing entries")

# Filter out already-scraped sources
new_sources = {k: v for k, v in ADDITIONAL_SOURCES.items() if k not in existing_keys}
print(f"[INFO] {len(new_sources)} new sources to scrape, {len(ADDITIONAL_SOURCES) - len(new_sources)} already present")

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

print("=" * 60)
print("LexBridge Scraper — Extended U.S. Employment Law")
print(f"Adding {len(new_sources)} new sources")
print("=" * 60)

# Compute next doc_id
next_id = max((e["doc_id"] for e in existing_metadata), default=-1) + 1
new_entries = []
total = len(new_sources)

for i, (key, info) in enumerate(new_sources.items()):
    url = info["url"]
    print(f"[{i + 1}/{total}] {key}")

    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()

        html_path = os.path.join(HTML_DIR, f"{key}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(r.text)

        soup = BeautifulSoup(r.text, "lxml")
        title_tag = soup.find("title")
        title_text = title_tag.string.strip() if title_tag and title_tag.string else info["topic"].title()

        new_entries.append({
            "doc_id": next_id,
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
        next_id += 1
        print(f"  [OK] Saved ({len(r.text):,} bytes)")

    except Exception as e:
        print(f"  [FAIL] {str(e)[:100]}")

    time.sleep(2)

# Merge & save
merged = existing_metadata + new_entries
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print()
print("=" * 60)
print(f"Done! Added {len(new_entries)}/{total} new documents")
print(f"Total corpus: {len(merged)} documents")
print(f"Metadata: {METADATA_PATH}")
print("=" * 60)
print()
print("Next: re-run the ingestion pipeline:")
print("  python scripts/pipeline.py")
