# TODO — untitled-project-1

last updated: 2026-02-24 01:21 PST  
owner: albus (coordination), gene (approval)

---

## Active Work

### [IN PROGRESS] UI Redesign — subagent `ui-redesign`
- **branch:** `hosted` (direct commits)
- **files touched:** `templates/search.html`, `static/` (css/js if created)
- **status:** running
- **do not touch:** `app.py` API routes (frontend only)

### [IN PROGRESS] Embedding Encoding — background job on bluephoria
- **pid:** 67989
- **status:** tweets done (2764/2764), media in progress (~21/140 at last check)
- **do not touch:** nothing to conflict with, runs against live db

---

## Queued (not started)

### [DONE] Quote Tweet Pipeline — branch `fix/quote-tweets`
- [x] `bookmarks/scraper.py` — extract expanded_url, display_url from URL entities
- [x] `pipelines/ingest.py` — insert OG tweets as independent rows with dedup
- [x] `scripts/encode_embeddings.py` — composite embedding `[QUOTE]...[ORIGINAL]...`
- [x] `scripts/backfill_og_tweets.py` — parse JSON backups, insert missing OG tweets + expanded URLs
- [x] `scripts/resolve_urls.py` — resolve remaining t.co → expanded + fetch og metadata
- [x] `db/schema.py` — added title, description, content_snippet columns to urls table
- **ready to merge** — needs gene approval, then run backfill on bluephoria

### Link Content Extraction
- **branch:** `feat/link-content` (create from hosted, after quote-tweets merges)
- **files to modify:**
  - `db/schema.py` — add title, description, content_snippet columns to urls table
- **files to create:**
  - `scripts/fetch_link_metadata.py` — fetch og:title, og:description, content snippet
- **depends on:** quote tweet pipeline (needs expanded_url populated first)

### Code Cleanup (from audit)
- **branch:** various, already merged
- [x] schema consolidation (merged)
- [x] import path dedup (merged)
- [x] pyproject entry point fix (merged)
- [x] embedding dimension migration (merged)
- [x] voyage-4 upgrade (merged)
- [x] batch embedding support (merged)
- [ ] max retry depth on 429 in scraper.py (deferred, low priority)
- [ ] grab_headers: attach to existing browser profile instead of launching chromium (deferred)

---

## File Ownership (conflict prevention)

When multiple agents/branches are active, respect these boundaries:

| file/dir | owner | notes |
|----------|-------|-------|
| `templates/`, `static/` | ui-redesign subagent | frontend only |
| `app.py` | shared (coordinate) | API routes — don't change shape without syncing |
| `bookmarks/scraper.py` | quote-tweet branch | extraction logic |
| `pipelines/ingest.py` | quote-tweet branch | ingestion logic |
| `scripts/encode_embeddings.py` | quote-tweet branch | embedding composition |
| `db/schema.py` | coordinate | schema changes need migration plan |
| `utils/voyage.py` | stable (merged) | don't touch unless API changes |
| `.env` | gene only | never committed, never modified by agents |

---

## Rules

1. **No pushing without gene's explicit ok**
2. **Create branch before starting work** — don't commit directly to hosted unless it's a hotfix
3. **Atomic commits** — one logical change per commit
4. **Check TODO.md before starting** — make sure nobody else owns the files you're touching
5. **Update TODO.md when starting/finishing work**
