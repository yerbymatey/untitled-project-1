# Untitled Project 1

A social media content search engine with AI-powered analysis capabilities. This project enables searching tweets and media using vector embeddings, generating image descriptions with vision-language models, and analyzing content with semantic search.

## Features

- **Vector Search**: Search tweets and images using semantic similarity
- **Image Understanding**: Automatically generate descriptions for images using DeepSeek VL
- **Web Interface**: Simple Flask-based UI for searching content
- **Command Line Tools**: Scripts for batch processing and content analysis
- **GPU Acceleration**: Optimized image processing with Metal GPU support on macOS

## Installation

### Prerequisites

- Python 3.8+ (Python 3.11 recommended)
- PostgreSQL with pgvector extension
- DeepSeek VL model (optional, for image description)
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

3. Install DeepSeek VL dependencies (optional, for image understanding):
   ```bash
   pip install -r requirements-dsvl.txt
   ```

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

### Generating Image Embeddings and Descriptions

```bash
# Generate embeddings for imported content
python -m scripts.encode_embeddings

# Generate image descriptions with DeepSeek VL (if installed)
python -m scripts.ds_vl
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
# Start the web server
python -m app

# Then open http://localhost:5000 in your browser
```

## Advanced Usage

### Image Processing

The project includes optimized image processing for macOS using Metal:

```bash
# Resize an image using GPU acceleration
python -m utils.process_images input.jpg output.jpg --width 512 --height 512
```

### Analyze Tweets with GPT-4o

```bash
# Analyze recent tweets
python -m scripts.analyze_tweets_gpt4o --limit 10
```

## Project Structure

- `app.py` - Flask web application
- `bookmarks/` - Social media scraping and parsing
- `db/` - Database schema and session management
- `pipelines/` - Data processing pipelines
- `scripts/` - Command-line utilities
  - `search_cli.py` - Command-line search interface
  - `ds_vl.py` - DeepSeek VL image description generator
  - `encode_embeddings.py` - Generate embeddings for content
- `utils/` - Utility functions
  - `embedding_utils.py` - Vector embedding generation
  - `process_images.py` - GPU-accelerated image processing
  - `vl_utils.py` - Vision-language model utilities

## Technical Considerations

### Vector Search

This project uses the pgvector extension for PostgreSQL to enable efficient similarity searches. Embeddings are generated using the Nomic Embed models for both text and images:

- Text embeddings: `nomic-ai/nomic-embed-text-v1.5`
- Vision embeddings: `nomic-ai/nomic-embed-vision-v1.5`

### Image Processing

On macOS, the system uses Metal for GPU-accelerated image processing. The implementation includes multiple fallback methods:

1. Metal Performance Shaders (primary method)
2. Core Image (fallback)
3. NumPy/SciPy (CPU fallback)

### DeepSeek VL Integration

The DeepSeek Vision-Language model requires specific dependencies and GPU resources. The model is used to generate semantic descriptions of images, enhancing the search capabilities.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Caveats and Limitations

- **API Rate Limits**: Social media APIs may impose rate limits that affect data collection
- **GPU Requirements**: DeepSeek VL model works best with GPU acceleration
- **PostgreSQL Requirements**: Requires pgvector extension for vector similarity search
- **Image Processing**: Metal acceleration only works on macOS with compatible GPUs
- **Model Size**: DeepSeek VL model requires significant RAM (16GB+) for optimal performance