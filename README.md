# Untitled Project 1

a scrappy social-media content search engine with an ai twist. yes, this is a mild resurrection of briend, but with a complete rewrite. you can pull in tweets + images, vector-search them, auto-resize/encode with one command, and spin up a little flask ui if you want. i have Plans, none that'll be implemented quickly but this is more just to have yall along for the ride

## what’s inside

- **all-in-one wrapper**  
  just run `python run_pipeline.py` to:  
  1. grab/refresh creds & headers  
  2. pull new content  
  3. generate embeddings + image descriptions  
  4. resize & preprocess media  
  5. index everything for search  

  > on first run you might need to manually kick off the auth grab:  
  > ```bash
  > python -m src.bookmarks.grab_headers
  > ```

- **vector search**  
  cosine sims over stored embeddings (no pgvector needed).  
  peek at `src/scripts/search_embeddings.py` for full deets.

- **web ui (optional)**  
  tiny flask app to poke around your data in the browser.

## quick start

1. clone + cd in  
   ```bash
   git clone https://github.com/yerbymatey/untitled-project-1.git
   cd untitled-project-1
   ```

2. install deps  
   ```bash
   pip install -e .
   ```

3. tweak your API creds in `src/utils/config.py` (or export as env vars)

4. fire up the pipeline  
   ```bash
   python run_pipeline.py
   ```

5. (optional) search right from the CLI:  
   ```bash
   python -m src.scripts.search_cli "rf hacking" --limit 5
   ```

6. (optional) launch the web ui:  
   ```bash
   python app.py
   # open http://localhost:5000
   ```

## repo layout

```
.
├── run_pipeline.py          # main wrapper for everything
├── app.py                   # flask web server
└── src/
    ├── pipelines/           # data pipelines (scrape → encode → index)
    ├── scripts/
    │   ├── run_scraper.py       # import tweets & media
    │   ├── encode_embeddings.py # embed content
    │   ├── search_cli.py        # CLI search interface
    │   └── search_embeddings.py # search logic
    ├── utils/
    │   ├── config.py            # db & api settings
    │   ├── embedding_utils.py   # embedding helpers
    │   └── process_images.py    # resize + preprocess images
    └── db/
        └── schema.py            # schema & session setup
```

## notes & caveats

- **token refresh** / headers grab is built into `run_pipeline.py` but you can still run `python -m src.bookmarks.grab_headers` if something goes sideways.  
- **advanced usage** (custom scrapes, one-off scripts, special image tricks) is all wrapped up in `run_pipeline.py` now—no need to juggle dozens of commands.
- **model usage** i'm using deepseek-vl-7b for first pass semantic descriptions for images but you could just use another one. nomic-embed-vision-v1.5 for image embeddings but actually moving away from that and using interleaved img desc text + raw img tensor smoothie as a placeholder
- **extra notes** some feats might still be artifacts of other runs i've tried--like analyzing tweets with gpt4o along with the text and image produces an explainer-source text paired doc you can embed for a bit more nuance in yr searches but by and large it's fine without it so it's not been in usage lately

## contributing

feel free to PR, riff on it, or just grab what you need. MIT license 👍
