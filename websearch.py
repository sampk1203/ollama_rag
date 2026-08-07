from ddgs import DDGS


def web_search(query, max_results=5):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        text = "\n\n".join(
            f"[Web: {r['href']}]\n{r['title']}\n{r['body']}"
            for r in results
        )
        return text, results
    except Exception as e:
        print(f"  ⚠ Web search failed: {e}")
        return "", []

def search_papers(query, max_results=5):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                f"{query} filetype:pdf OR site:arxiv.org OR site:semanticscholar.org",
                max_results=max_results
            ))
        return results
    except Exception as e:
        print(f"  ⚠ Paper search failed: {e}")
        return []
