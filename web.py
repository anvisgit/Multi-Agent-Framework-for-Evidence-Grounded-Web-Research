from ddgs import DDGS
import requests
from bs4 import BeautifulSoup


def search(query, n=5):
    try:
        return list(DDGS().text(query, max_results=n))
    except Exception as e:
        print("Search failed:", e)
        return []


def scrape(url):
    try:
        html = requests.get(
            url, timeout=8, headers={"User-Agent": "Mozilla/5.0"}
        ).text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return " ".join(soup.get_text().split())[:1500]
    except Exception:
        return ""


def get_sources(query, n=5):
    pieces = []
    for r in search(query, n):
        text = scrape(r["href"]) or r["body"]
        pieces.append(f"Title: {r['title']}\nURL: {r['href']}\n{text}")
    return "\n\n---\n\n".join(pieces)