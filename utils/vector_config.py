import os

# Central place to read embedding dimension from environment.
# Defaults to 1024 to match Voyage 4 default output size.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

__all__ = ["EMBEDDING_DIM"]
