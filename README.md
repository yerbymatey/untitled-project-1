# Untitled Project 1

A social media content search engine with AI-powered analysis capabilities. This project enables searching tweets and media using vector embeddings, generating image descriptions with vision-language models, and analyzing content with semantic search.

## Features

- **Vector Search**: Search tweets and images using semantic similarity
- **Image Understanding**: Automatically generate descriptions for images using Gemini 2.5 Flash
- **Web Interface**: Simple Flask-based UI for searching content
- **Command Line Tools**: Scripts for batch processing and content analysis
- **GPU Acceleration**: Optimized image processing with Metal GPU support on macOS

## Installation

### Prerequisites

- Python 3.8+ (Python 3.11 recommended)
- PostgreSQL with pgvector extension
- Gemini API key (for image descriptions)
- Metal-compatible GPU (for accelerated image processing on macOS)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yerbymatey/untitled-project-1.git
   cd untitled-project-1
   ```

2. Install the base package:
   ```bash
   pip install -e .
   ```

3. Set environment variables in `.env` (copy from `.env.example`):
   - `POSTGRES_*` for database
   - `GEMINI_API_KEY` for image descriptions
   - `VOYAGE_API_KEY` for embeddings and rerank
   - `EMBEDDING_DIM=1024` (default for Voyage)

4. Configure the database:
   ```bash
   # Update database connection in utils/config.py if necessary
   python -m db.schema  # Sets up the database schema
   ```

## Usage

### Importing Social Media Content

1. Configure API credentials in `utils/config.py`
2. Run the scraper:
   ```bash
   python -m scripts.run_scraper
   ```

### Pipeline (Scrape → Describe → Embed)

```bash
# Generate embeddings for imported content
python run_pipeline.py
```

### Searching Content

#### Command Line Search

```bash
# Search for tweets and media containing "climate change"
python -m scripts.search_cli "climate change"

# Search only for tweets
python -m scripts.search_cli "climate change" --type tweets

# Search only for media
python -m scripts.search_cli "climate change" --type media

# Limit results
python -m scripts.search_cli "climate change" --limit 10
```

#### Web Interface

```bash
# Start the web server (after running the pipeline at least once)
python -m app

# Then open http://localhost:5000 in your browser
```

## Advanced Usage

### Image Processing (Optional)

The project includes a simple image resize utility for macOS:

```bash
python -m utils.process_images input.jpg output.jpg --width 512 --height 512
```

## Project Structure

- `app.py` - Flask web application
- `bookmarks/` - Social media scraping and parsing
- `db/` - Database schema and session management
- `pipelines/` - Data processing pipelines
- `scripts/` - Command-line utilities
  - `search_cli.py` - Command-line search interface
  - `image_descriptions_gemini.py` - Gemini image description generator
  - `encode_embeddings.py` - Generate embeddings (Voyage APIs)
- `utils/` - Utility functions
  - `gemini.py` - Gemini HTTP client
  - `voyage.py` - Voyage HTTP client (multimodal/contextualized/rerank)
  - `embedding_utils.py` - Legacy local embedding helpers (still used by some tools)
  - `process_images.py` - Image resize helper

## Technical Considerations

### Vector Search

This project uses the pgvector extension for PostgreSQL to enable efficient similarity searches. Embeddings are generated using Voyage AI:

- Text/query embeddings: `voyage-context-3` (contextualized embeddings)
- Media embeddings: `voyage-multimodal-3` (image-only and joint text+image)

### Image Processing

On macOS, the system uses Metal for GPU-accelerated image processing. The implementation includes multiple fallback methods:

1. Metal Performance Shaders (primary method)
2. Core Image (fallback)
3. NumPy/SciPy (CPU fallback)

### Hosted Models

Image descriptions are generated via Gemini 2.5 Flash. Embeddings and reranking are provided by Voyage APIs. No local model downloads are required.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Caveats and Limitations

- **API Rate Limits**: Social media APIs may impose rate limits that affect data collection
- **API Keys**: Gemini and Voyage credentials are required
- **PostgreSQL Requirements**: Requires pgvector extension for vector similarity search
- **Image Processing**: The resize helper is basic; heavy OCR is not included
