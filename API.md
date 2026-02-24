# Search API Reference

The Flask app (`app.py`) exposes a single search endpoint. All embedding and reranking logic lives in `scripts/search_embeddings.py`; the CLI wrapper is `scripts/search_cli.py`.

---

## `GET /search`

Searches your bookmarked tweets and media using Voyage AI embeddings and cosine similarity.

### Query Parameters

| Parameter  | Type    | Default      | Description |
|------------|---------|--------------|-------------|
| `q`        | string  | *(required)* | The natural-language search query. |
| `mode`     | string  | `weighted`   | Search mode. `weighted` returns three separate ranked lists (tweets, media_combined, media_image_only). `traditional` pools and ranks everything into one list. |
| `expanded` | boolean | `false`      | When `true`, returns up to 10 results per list instead of 5. |

> **Note:** `type`, `limit`, and `rerank` are CLI-only flags (see [`scripts/search_cli.py`](scripts/search_cli.py)). They are not accepted by the HTTP endpoint.

---

## Response Schemas

### Weighted mode (`mode=weighted`, default)

Three separate ranked lists. Each list is independently scored against the query.

```json
{
  "tweets": [
    {
      "id": "1234567890",
      "text": "Tweet text content",
      "created_at": "2024-01-15T12:00:00",
      "tweet_url": "https://twitter.com/x/status/1234567890",
      "media_url": "https://pbs.twimg.com/media/...",
      "image_desc": "AI-generated image description or null",
      "extr_text": "OCR / extracted text from image or null",
      "similarity": 0.921,
      "embed_html": "<blockquote class=\"twitter-tweet\" ...>",
      "quoted_tweet": {
        "id": "9876543210",
        "text": "The quoted tweet's text",
        "url": "https://twitter.com/x/status/9876543210",
        "user_name": "handle"
      }
    }
  ],
  "media_combined": [
    {
      "tweet_id": "1234567890",
      "text": "Tweet text that accompanies the image",
      "tweet_url": "https://twitter.com/x/status/1234567890",
      "media_url": "https://pbs.twimg.com/media/...",
      "image_desc": "AI-generated image description",
      "extr_text": "OCR text or null",
      "created_at": "2024-01-15T12:00:00",
      "similarity": 0.874,
      "joint_similarity": 0.891,
      "image_similarity": 0.857,
      "embed_html": "<blockquote class=\"twitter-tweet\" ...>",
      "quoted_tweet": null
    }
  ],
  "media_image_only": [
    {
      "tweet_id": "1234567890",
      "text": "Tweet text",
      "tweet_url": "https://twitter.com/x/status/1234567890",
      "media_url": "https://pbs.twimg.com/media/...",
      "image_desc": "AI-generated image description",
      "extr_text": "OCR text or null",
      "created_at": "2024-01-15T12:00:00",
      "similarity": 0.843,
      "embed_html": "<blockquote class=\"twitter-tweet\" ...>",
      "quoted_tweet": null
    }
  ],
  "has_more": true
}
```

**Field notes:**
- `media_combined.similarity` = `0.5 * joint_similarity + 0.5 * image_similarity`
- `media_image_only.similarity` is scored purely against the image-only embedding (ignores tweet text)
- `quoted_tweet` is `null` when the tweet does not quote another tweet
- `has_more` is `true` when any list hit the result cap (5 or 10)

### Traditional mode (`mode=traditional`)

One flat list pooling tweets and both media embedding types, ranked purely by cosine similarity.

```json
{
  "combined_results": [
    {
      "type": "tweet",
      "id": "1234567890",
      "text": "Tweet text content",
      "created_at": "2024-01-15T12:00:00",
      "tweet_url": "https://twitter.com/x/status/1234567890",
      "media_url": "https://pbs.twimg.com/media/... or null",
      "image_desc": "AI-generated description or null",
      "extr_text": "OCR text or null",
      "similarity": 0.934,
      "embed_html": "<blockquote class=\"twitter-tweet\" ...>",
      "quoted_tweet": null
    },
    {
      "type": "media_joint",
      "tweet_id": "9876543210",
      "text": "Tweet text",
      "tweet_url": "https://twitter.com/x/status/9876543210",
      "media_url": "https://pbs.twimg.com/media/...",
      "image_desc": "AI-generated description",
      "extr_text": null,
      "created_at": "2024-01-10T08:30:00",
      "similarity": 0.912,
      "embed_html": "<blockquote class=\"twitter-tweet\" ...>",
      "quoted_tweet": null
    },
    {
      "type": "media_image",
      "tweet_id": "9876543210",
      "text": "Tweet text",
      "tweet_url": "https://twitter.com/x/status/9876543210",
      "media_url": "https://pbs.twimg.com/media/...",
      "image_desc": "AI-generated description",
      "extr_text": null,
      "created_at": "2024-01-10T08:30:00",
      "similarity": 0.898,
      "embed_html": "<blockquote class=\"twitter-tweet\" ...>",
      "quoted_tweet": null
    }
  ],
  "has_more": false
}
```

**`type` values:**
- `tweet` — matched via tweet text embedding
- `media_joint` — matched via joint (text + image) embedding
- `media_image` — matched via image-only embedding

> A single media item can appear twice (as both `media_joint` and `media_image`) if both embeddings rank highly against the query.

---

## Error Responses

The endpoint does not return HTTP error codes for bad queries — it returns empty lists instead.

### Missing or empty `q`

```json
{
  "tweets": [],
  "media_combined": [],
  "media_image_only": [],
  "combined_results": []
}
```

### Server error

Flask returns a standard 500 HTML error page if an unhandled exception occurs (e.g. database unreachable). Run with `FLASK_DEBUG=1` to get a JSON traceback during development.

---

## curl Examples

### Basic weighted search

```bash
curl "http://localhost:5000/search?q=machine+learning"
```

### Expanded results (10 per list instead of 5)

```bash
curl "http://localhost:5000/search?q=machine+learning&expanded=true"
```

### Traditional mode (single ranked list)

```bash
curl "http://localhost:5000/search?q=neural+networks&mode=traditional"
```

### Pretty-print JSON with jq

```bash
curl -s "http://localhost:5000/search?q=rust+programming" | jq '.tweets[] | {id, similarity, text: .text[:80]}'
```

### Extract just tweet URLs from weighted results

```bash
curl -s "http://localhost:5000/search?q=distributed+systems&expanded=true" \
  | jq -r '.tweets[].tweet_url'
```

### Agent integration: get top result URL and similarity

```bash
QUERY="vector databases"
RESULT=$(curl -s "http://localhost:5000/search?q=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$QUERY'))")&mode=traditional")
echo "$RESULT" | jq -r '.combined_results[0] | "\(.similarity | . * 1000 | round / 1000)\t\(.tweet_url)"'
```

---

## Python Example (requests)

```python
import requests

BASE_URL = "http://localhost:5000"

def search(query: str, mode: str = "weighted", expanded: bool = False) -> dict:
    resp = requests.get(
        f"{BASE_URL}/search",
        params={"q": query, "mode": mode, "expanded": str(expanded).lower()},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# --- Weighted mode (default) ---
results = search("transformer architecture")

for tweet in results["tweets"]:
    print(f"[{tweet['similarity']:.3f}] {tweet['tweet_url']}")
    print(f"  {tweet['text'][:120]}")

for media in results["media_combined"]:
    print(f"[{media['similarity']:.3f}] {media['media_url']}")
    print(f"  joint={media['joint_similarity']:.3f}  image={media['image_similarity']:.3f}")


# --- Traditional mode (single ranked list) ---
results = search("attention mechanism", mode="traditional", expanded=True)

for item in results["combined_results"]:
    label = item["type"]
    url = item.get("tweet_url") or item.get("media_url", "")
    print(f"[{item['similarity']:.3f}] ({label}) {url}")
```

---

## CLI Reference (`scripts/search_cli.py`)

The CLI supports additional flags not available via the HTTP API.

```
python -m scripts.search_cli "query text" [OPTIONS]

Options:
  --type {all,tweets,media,media_image}   Content type to search (weighted mode only; default: all)
  --limit INT                             Number of results per list (default: 5)
  --mode {weighted,traditional}           Search mode (default: weighted)
  --rerank                                Rerank results with Voyage rerank-2.5
  --force-cpu                             No-op (kept for compatibility with local embedding path)
```

### CLI examples

```bash
# Search all content types, weighted mode
python -m scripts.search_cli "attention is all you need"

# Search only tweets, return 10 results
python -m scripts.search_cli "RLHF" --type tweets --limit 10

# Traditional mode with Voyage reranker
python -m scripts.search_cli "language model alignment" --mode traditional --rerank

# Image-only search (good for finding screenshots, diagrams)
python -m scripts.search_cli "architecture diagram" --type media_image --limit 5
```
