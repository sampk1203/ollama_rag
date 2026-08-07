import json
import httpx
from datetime import datetime
from pathlib import Path
from config import DOWNLOAD_DIR
from websearch import search_papers


def get_downloaded_urls():
    meta_file = Path(DOWNLOAD_DIR) / "downloaded.json"
    if meta_file.exists():
        try:
            return {e["url"] for e in json.loads(meta_file.read_text())}
        except:
            pass
    return set()

def save_download_meta(entry):
    meta_file = Path(DOWNLOAD_DIR) / "downloaded.json"
    data = []
    if meta_file.exists():
        try:
            data = json.loads(meta_file.read_text())
        except:
            pass
    data.append(entry)
    meta_file.write_text(json.dumps(data, indent=2))

def download_paper(url, title="paper"):
    if url in get_downloaded_urls():
        print(f"  ⚠ Already downloaded: {url}")
        return None

    # Arxiv: convert abs → pdf
    if "arxiv.org/abs/" in url:
        url = url.replace("arxiv.org/abs/", "arxiv.org/pdf/") + ".pdf"

    try:
        print(f"  ⬇ Downloading: {url}", flush=True)
        response = httpx.get(url, follow_redirects=True, timeout=60)
        if response.status_code != 200:
            print(f"  ✗ Failed ({response.status_code})")
            return None
        if "application/pdf" not in response.headers.get("content-type", ""):
            print(f"  ✗ Not a PDF")
            return None

        date_folder = Path(DOWNLOAD_DIR) / datetime.now().strftime("%Y-%m-%d")
        date_folder.mkdir(parents=True, exist_ok=True)

        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:80]
        filepath = date_folder / f"{safe_title}.pdf"
        counter = 1
        while filepath.exists():
            filepath = date_folder / f"{safe_title}_{counter}.pdf"
            counter += 1

        filepath.write_bytes(response.content)
        print(f"  ✓ Saved: {filepath}")
        save_download_meta({
            "url": url, "title": title,
            "file": str(filepath),
            "date": datetime.now().isoformat()
        })
        return str(filepath)

    except Exception as e:
        print(f"  ✗ Download error: {e}")
        return None

def interactive_paper_download(results):
    if not results:
        print("  No results to download from.")
        return []

    print("\n  📄 Search results:")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r.get('title', 'No title')}")
        print(f"       {r.get('href', '')}")

    print("\n  Enter numbers (e.g. 1,3), 'all', or Enter to skip:")
    choice = input("  > ").strip().lower()
    if not choice:
        return []

    indices = list(range(len(results))) if choice == "all" else [
        int(x.strip()) - 1 for x in choice.split(",")
        if x.strip().isdigit()
    ]

    downloaded = []
    for idx in indices:
        if 0 <= idx < len(results):
            r = results[idx]
            path = download_paper(r.get("href", ""), title=r.get("title", "paper"))
            if path:
                downloaded.append(path)
    return downloaded
