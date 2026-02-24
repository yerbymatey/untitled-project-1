import os

# Central place to read embedding dimension from environment.
# Defaults to 1024 to match Voyage context-3 default.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

__all__ = ["EMBEDDING_DIM"]

