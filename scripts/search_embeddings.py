import torch
import numpy as np
import logging
from typing import List, Dict, Tuple

from db.session import get_db_session
from utils.vector_config import EMBEDDING_DIM
from utils.voyage import voyage_embeddings, voyage_contextualized_embeddings, voyage_rerank


def get_query_embedding(query: str) -> torch.Tensor:
    """Generate a Voyage embedding for the search query."""
    # Use input_type='query' so Voyage applies its retrieval prompt.
    vecs = voyage_embeddings(texts=[query], input_type='query')
    vec = vecs[0] if vecs else []
    if not vec:
        raise RuntimeError("Voyage returned empty embedding for query")
    return torch.tensor(vec, dtype=torch.float32)


logger = logging.getLogger(__name__)


def _enrich_with_quoted_tweets(session, results: List[Dict], id_key: str = 'id') -> List[Dict]:
    """Batch-fetch quoted tweet data for search results that have a quoted_tweet_id.

    Adds a 'quoted_tweet' dict to each result that quotes another tweet,
    containing the quoted tweet's id, text, url, and user name.
    """
    # Collect tweet IDs from results to look up their quoted_tweet_id
    result_ids = [r[id_key] for r in results if r.get(id_key)]
    if not result_ids:
        return results

    placeholders = ','.join(['%s'] * len(result_ids))
    session.execute(f"""
        SELECT t.id, t.quoted_tweet_id
        FROM tweets t
        WHERE t.id IN ({placeholders}) AND t.quoted_tweet_id IS NOT NULL
    """, tuple(result_ids))
    quote_map = {row['id']: row['quoted_tweet_id'] for row in session.fetchall()}

    if not quote_map:
        return results

    # Fetch the quoted tweets themselves
    quoted_ids = list(set(quote_map.values()))
    placeholders = ','.join(['%s'] * len(quoted_ids))
    session.execute(f"""
        SELECT t.id, t.text, t.url, u.name as user_name
        FROM tweets t
        LEFT JOIN users u ON t.user_id = u.id
        WHERE t.id IN ({placeholders})
    """, tuple(quoted_ids))
    quoted_tweets = {row['id']: dict(row) for row in session.fetchall()}

    # Attach to results
    for r in results:
        rid = r.get(id_key)
        if rid and rid in quote_map:
            qt_id = quote_map[rid]
            if qt_id in quoted_tweets:
                r['quoted_tweet'] = quoted_tweets[qt_id]

    return results


def rerank_text_results(query: str, items: List[Dict], text_key: str) -> List[Dict]:
    """Use Voyage rerank-2.5 to rerank items by relevance to query.

    Args:
        query: The query string
        items: List of result dicts
        text_key: Key in item dict to use as rerank document (e.g., 'text' or 'image_desc')
    Returns:
        New list of items ordered by reranker relevance.
    """
    if not items:
        return items
    documents = [str(it.get(text_key) or "") for it in items]
    try:
        reranked = voyage_rerank(query=query, documents=documents, model="rerank-2.5", top_k=len(documents))
        order = [r["index"] for r in reranked]
        # Map by original index
        reord = [items[i] for i in order if 0 <= i < len(items)]
        return reord
    except Exception as e:
        logger.warning(f"Rerank failed, returning original order: {e}")
        return items

def compute_similarities(query_embedding: torch.Tensor, stored_embeddings: List[List[float]]) -> torch.Tensor:
    """Compute cosine similarities between query and stored embeddings"""
    if not stored_embeddings:
        return torch.tensor([])
    
    # Convert stored embeddings to CPU tensor (no GPU use in hosted path)
    stored_tensor = torch.tensor(stored_embeddings, dtype=torch.float32)
    
    # Compute cosine similarities
    # Ensure matching dtype/device
    if query_embedding.dtype != stored_tensor.dtype:
        query_vec = query_embedding.to(dtype=stored_tensor.dtype)
    else:
        query_vec = query_embedding
    similarities = torch.matmul(stored_tensor, query_vec)
    return similarities

def debug_vector_stats(vector_str: str):
    """Print statistics about a vector stored in the database"""
    try:
        # Convert string representation to numpy array
        vector = np.array(eval(vector_str))
        print(f"\nStored vector shape: {vector.shape}")
        print(f"Stored vector norm: {np.linalg.norm(vector):.4f}")
        print(f"Stored vector mean: {np.mean(vector):.4f}")
        print(f"Stored vector std: {np.std(vector):.4f}")
    except Exception as e:
        print(f"Error analyzing vector: {e}")

def search_tweets(query: str, limit: int = 10) -> List[Dict]:
    """
    Search tweets using tensor operations for similarity computation
    
    Args:
        query (str): Search query text
        limit (int): Maximum number of results to return
        
    Returns:
        list: List of dictionaries containing tweet data and similarity scores
    """
    try:
        # Generate query embedding
        query_embedding = get_query_embedding(query)
        
        with get_db_session() as session:
            # Get all tweet embeddings - cast vector to text then parse
            # Also LEFT JOIN to get associated media info (picking one media item per tweet)
            # Using ROW_NUMBER() for compatibility and clarity
            sql = """
            WITH RankedTweets AS (
                SELECT
                    t.id,
                    t.text,
                    t.created_at,
                    t.url as tweet_url,
                    t.embedding::text as embedding_array,
                    m.media_url,
                    m.image_desc,
                    m.extr_text,
                    ROW_NUMBER() OVER(PARTITION BY t.id ORDER BY m.media_url NULLS LAST) as rn
                FROM tweets t
                LEFT JOIN media m ON t.id = m.tweet_id AND m.type = 'photo'
                WHERE t.embedding IS NOT NULL
            )
            SELECT
                id,
                text,
                created_at,
                tweet_url,
                embedding_array,
                media_url,
                image_desc,
                extr_text
            FROM RankedTweets
            WHERE rn = 1;
            """
            session.execute(sql)
            results = session.fetchall()

            if not results:
                return []

            # Parse text representation back into list
            stored_embeddings = []
            tweet_data = []
            for r in results:
                try:
                    # Remove brackets and split into floats
                    vector_str = r['embedding_array'].strip('[]')
                    vector = [float(x) for x in vector_str.split(',')]
                    stored_embeddings.append(vector)
                    tweet_data.append({
                        'id': r['id'],
                        'text': r['text'],
                        'created_at': r['created_at'],
                        'tweet_url': r['tweet_url'],
                        'media_url': r['media_url'],
                        'image_desc': r['image_desc'],
                        'extr_text': r['extr_text']
                    })
                except Exception as e:
                    print(f"Error parsing vector for tweet {r['id']}: {e}")
                    continue

            if not stored_embeddings:
                return []

            # Compute similarities
            similarities = compute_similarities(query_embedding, stored_embeddings)

            # Sort by similarity and get top results
            if len(similarities) > 0:
                top_indices = torch.argsort(similarities, descending=True)[:limit]

                # Build final results
                search_results = []
                for idx in top_indices:
                    i = idx.item()
                    search_results.append({
                        **tweet_data[i],
                        'similarity': similarities[i].item()
                    })

                # Enrich with quoted tweet data
                search_results = _enrich_with_quoted_tweets(session, search_results, id_key='id')

                return search_results

            return []
            
    except Exception as e:
        print(f"Error searching tweets: {e}")
        return []

def search_media(query: str, limit: int = 10, weight_joint: float = 0.5, weight_image: float = 0.5) -> List[Dict]:
    """
    Search media using tensor operations for similarity computation.
    Compares the text query against both the joint (text+image) embedding 
    and the image-only embedding, then combines the scores.
    
    Args:
        query (str): Search query text
        limit (int): Maximum number of results to return
        weight_joint (float): Weight for the joint embedding similarity score.
        weight_image (float): Weight for the image embedding similarity score.
        
    Returns:
        list: List of dictionaries containing media data and combined similarity scores
    """
    # Ensure weights sum approximately to 1 if needed, or handle as desired.
    # For simplicity, we assume they are provided appropriately.
    
    try:
        # Generate query embedding
        query_embedding = get_query_embedding(query)
        
        with get_db_session() as session:
            # Get all media embeddings (joint and image) - cast vector to text then parse
            session.execute("""
                SELECT
                    m.tweet_id,
                    t.text,
                    t.url as tweet_url,
                    m.media_url,
                    m.image_desc,
                    m.extr_text,
                    t.created_at,
                    m.joint_embedding::text as joint_embedding_array,
                    m.image_embedding::text as image_embedding_array
                FROM media m
                JOIN tweets t ON t.id = m.tweet_id
                WHERE m.joint_embedding IS NOT NULL
                  AND m.image_embedding IS NOT NULL
                  AND m.type = 'photo'
            """)
            results = session.fetchall()
            
            if not results:
                return []
            
            # Parse text representations back into lists
            stored_joint_embeddings = []
            stored_image_embeddings = []
            media_data = []
            valid_indices = [] # Keep track of indices with valid embeddings
            for idx, r in enumerate(results):
                try:
                    # Parse joint embedding
                    joint_vector_str = r['joint_embedding_array'].strip('[]')
                    joint_vector = [float(x) for x in joint_vector_str.split(',')]
                    
                    # Parse image embedding
                    image_vector_str = r['image_embedding_array'].strip('[]')
                    image_vector = [float(x) for x in image_vector_str.split(',')]
                    
                    # Check embedding dimensions match expected (optional but good practice)
                    if len(joint_vector) != EMBEDDING_DIM or len(image_vector) != EMBEDDING_DIM:
                        logger.warning(f"Skipping media {r['tweet_id']}/{r['media_url']} due to unexpected embedding dimension.")
                        continue

                    stored_joint_embeddings.append(joint_vector)
                    stored_image_embeddings.append(image_vector)
                    media_data.append({
                        'tweet_id': r['tweet_id'],
                        'text': r['text'],
                        'tweet_url': r['tweet_url'],
                        'media_url': r['media_url'],
                        'image_desc': r['image_desc'],
                        'extr_text': r['extr_text'],
                        'created_at': r['created_at']
                    })
                    valid_indices.append(idx)

                except Exception as e:
                    print(f"Error parsing vectors for media {r['tweet_id']}/{r['media_url']}: {e}")
                    continue

            if not media_data:
                return []

            # Compute similarities for both embedding types
            joint_similarities = compute_similarities(query_embedding, stored_joint_embeddings)
            image_similarities = compute_similarities(query_embedding, stored_image_embeddings)

            if joint_similarities.shape != image_similarities.shape:
                 raise ValueError("Shape mismatch between joint and image similarities")

            combined_scores = (weight_joint * joint_similarities) + (weight_image * image_similarities)

            if len(combined_scores) > 0:
                top_indices_in_filtered = torch.argsort(combined_scores, descending=True)[:limit]

                search_results = []
                for idx_in_filtered in top_indices_in_filtered:
                    original_data_index = idx_in_filtered.item()
                    search_results.append({
                        **media_data[original_data_index],
                        'similarity': combined_scores[original_data_index].item(),
                        'joint_similarity': joint_similarities[original_data_index].item(),
                        'image_similarity': image_similarities[original_data_index].item()
                    })

                # Enrich with quoted tweet data
                search_results = _enrich_with_quoted_tweets(session, search_results, id_key='tweet_id')

                return search_results
            
            return []
            
    except Exception as e:
        print(f"Error searching media: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return []

def search_media_image_only(query: str, limit: int = 10) -> List[Dict]:
    """
    Search media based *only* on the similarity between the text query 
    and the image-only embedding.
    
    Args:
        query (str): Search query text
        limit (int): Maximum number of results to return
        
    Returns:
        list: List of dictionaries containing media data and image similarity scores
    """
    try:
        # Generate query embedding
        query_embedding = get_query_embedding(query)
        
        with get_db_session() as session:
            # Get only the image embeddings - cast vector to text then parse
            session.execute("""
                SELECT
                    m.tweet_id,
                    t.text,
                    t.url as tweet_url,
                    m.media_url,
                    m.image_desc,
                    m.extr_text,
                    t.created_at,
                    m.image_embedding::text as image_embedding_array
                FROM media m
                JOIN tweets t ON t.id = m.tweet_id
                WHERE m.image_embedding IS NOT NULL
                  AND m.type = 'photo'
            """)
            results = session.fetchall()
            
            if not results:
                return []
            
            # Parse text representation back into list
            stored_image_embeddings = []
            media_data = []
            for r in results:
                try:
                    # Parse image embedding
                    image_vector_str = r['image_embedding_array'].strip('[]')
                    image_vector = [float(x) for x in image_vector_str.split(',')]
                    
                    if len(image_vector) != EMBEDDING_DIM:
                        logger.warning(f"Skipping media {r['tweet_id']}/{r['media_url']} due to unexpected embedding dimension.")
                        continue
                        
                    stored_image_embeddings.append(image_vector)
                    media_data.append({
                        'tweet_id': r['tweet_id'],
                        'text': r['text'],
                        'tweet_url': r['tweet_url'],
                        'media_url': r['media_url'],
                        'image_desc': r['image_desc'],
                        'extr_text': r['extr_text'],
                        'created_at': r['created_at']
                    })
                except Exception as e:
                    print(f"Error parsing image vector for media {r['tweet_id']}/{r['media_url']}: {e}")
                    continue

            if not media_data:
                return []

            # Compute similarities against image embeddings
            image_similarities = compute_similarities(query_embedding, stored_image_embeddings)

            if len(image_similarities) > 0:
                top_indices = torch.argsort(image_similarities, descending=True)[:limit]

                search_results = []
                for idx in top_indices:
                    i = idx.item()
                    search_results.append({
                        **media_data[i],
                        'similarity': image_similarities[i].item()
                    })

                # Enrich with quoted tweet data
                search_results = _enrich_with_quoted_tweets(session, search_results, id_key='tweet_id')

                return search_results
            
            return []
            
    except Exception as e:
        print(f"Error searching media (image only): {e}")
        import traceback
        traceback.print_exc()
        return []

def search_traditional_combined(query: str, limit: int = 10) -> List[Dict]:
    """
    Performs a combined search across tweets and media, ranking all items together.

    Compares the text query individually against:
    1. Tweet text embeddings
    2. Media joint embeddings
    3. Media image embeddings

    All results are pooled and ranked purely by their calculated cosine similarity
    to the query embedding. A single media item might appear twice if both its
    joint and image embeddings rank highly.

    Args:
        query (str): Search query text
        limit (int): Maximum number of results to return from the combined list.

    Returns:
        list: A single list of dictionaries, containing tweet or media data,
              ranked by similarity score. Each item includes a 'type' field
              ('tweet', 'media_joint', 'media_image') and 'similarity'.
    """
    all_results = []
    try:
        query_embedding = get_query_embedding(query)

        with get_db_session() as session:
            # 1. Fetch Tweet Embeddings
            session.execute("""
                SELECT
                    t.id,
                    t.text,
                    t.created_at,
                    t.url as tweet_url,
                    t.embedding::text as embedding_array,
                    m.media_url,
                    m.image_desc,
                    m.extr_text
                FROM tweets t
                LEFT JOIN (
                    SELECT DISTINCT ON (tweet_id) tweet_id, media_url, image_desc, extr_text
                    FROM media WHERE type = 'photo' ORDER BY tweet_id, media_url
                ) m ON t.id = m.tweet_id
                WHERE t.embedding IS NOT NULL
            """)
            tweet_db_results = session.fetchall()

            if tweet_db_results:
                tweet_embeddings = []
                tweet_data_map = {} # Store data by index
                for i, r in enumerate(tweet_db_results):
                    try:
                        vector_str = r['embedding_array'].strip('[]')
                        vector = [float(x) for x in vector_str.split(',')]
                        if len(vector) == EMBEDDING_DIM:
                           tweet_embeddings.append(vector)
                           tweet_data_map[len(tweet_embeddings) - 1] = {
                               'id': r['id'],
                               'text': r['text'],
                               'created_at': r['created_at'],
                               'tweet_url': r['tweet_url'],
                               'media_url': r['media_url'],
                               'image_desc': r['image_desc'],
                               'extr_text': r['extr_text']
                           }
                        else:
                            logger.warning(f"Skipping tweet {r['id']} due to unexpected embedding dimension.")
                    except Exception as e:
                        logger.error(f"Error parsing vector for tweet {r['id']}: {e}")

                if tweet_embeddings:
                    tweet_similarities = compute_similarities(query_embedding, tweet_embeddings)
                    for i, score in enumerate(tweet_similarities):
                        if i in tweet_data_map: # Check if data was stored for this index
                            all_results.append({
                                'type': 'tweet',
                                **tweet_data_map[i],
                                'similarity': score.item()
                            })


            # 2. Fetch Media Embeddings (Joint and Image)
            session.execute("""
                SELECT
                    m.tweet_id,
                    t.text,
                    t.url as tweet_url,
                    m.media_url,
                    m.image_desc,
                    m.extr_text,
                    t.created_at,
                    m.joint_embedding::text as joint_embedding_array,
                    m.image_embedding::text as image_embedding_array
                FROM media m
                JOIN tweets t ON t.id = m.tweet_id
                WHERE m.joint_embedding IS NOT NULL
                  AND m.image_embedding IS NOT NULL
                  AND m.type = 'photo'
            """)
            media_db_results = session.fetchall()

            if media_db_results:
                media_joint_embeddings = []
                media_image_embeddings = []
                media_data_map = {} # Store data by index (applies to both lists)

                for i, r in enumerate(media_db_results):
                    try:
                        joint_vector_str = r['joint_embedding_array'].strip('[]')
                        joint_vector = [float(x) for x in joint_vector_str.split(',')]

                        image_vector_str = r['image_embedding_array'].strip('[]')
                        image_vector = [float(x) for x in image_vector_str.split(',')]

                        if len(joint_vector) == EMBEDDING_DIM and len(image_vector) == EMBEDDING_DIM:
                           media_joint_embeddings.append(joint_vector)
                           media_image_embeddings.append(image_vector)
                           idx = len(media_joint_embeddings) - 1 # Current index
                           media_data_map[idx] = {
                               'tweet_id': r['tweet_id'],
                               'text': r['text'],
                               'tweet_url': r['tweet_url'],
                               'media_url': r['media_url'],
                               'image_desc': r['image_desc'],
                               'extr_text': r['extr_text'],
                               'created_at': r['created_at']
                           }
                        else:
                             logger.warning(f"Skipping media {r['tweet_id']}/{r['media_url']} due to unexpected embedding dimension.")

                    except Exception as e:
                        logger.error(f"Error parsing vectors for media {r['tweet_id']}/{r['media_url']}: {e}")

                # Compute similarities for joint embeddings
                if media_joint_embeddings:
                    joint_similarities = compute_similarities(query_embedding, media_joint_embeddings)
                    for i, score in enumerate(joint_similarities):
                         if i in media_data_map:
                            all_results.append({
                                'type': 'media_joint',
                                **media_data_map[i],
                                'similarity': score.item()
                            })

                # Compute similarities for image embeddings
                if media_image_embeddings:
                    image_similarities = compute_similarities(query_embedding, media_image_embeddings)
                    for i, score in enumerate(image_similarities):
                        if i in media_data_map:
                           all_results.append({
                               'type': 'media_image',
                               **media_data_map[i],
                               'similarity': score.item()
                           })

        # 3. Sort all collected results by similarity
        all_results.sort(key=lambda x: x['similarity'], reverse=True)

        # 4. Take top N and enrich with quoted tweets
        top_results = all_results[:limit]
        if top_results:
            with get_db_session() as enrich_session:
                # _enrich_with_quoted_tweets mutates dicts in-place (shared refs with top_results)
                tweet_type = [r for r in top_results if r.get('type') == 'tweet']
                media_type = [r for r in top_results if r.get('type') != 'tweet']
                if tweet_type:
                    _enrich_with_quoted_tweets(enrich_session, tweet_type, id_key='id')
                if media_type:
                    _enrich_with_quoted_tweets(enrich_session, media_type, id_key='tweet_id')

        return top_results

    except Exception as e:
        print(f"Error during traditional combined search: {e}")
        import traceback
        traceback.print_exc()
        return []

def search_all(query: str, limit: int = 10) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Search tweets (text vs text), media (text vs combined), and media (text vs image only) 
    with a single query.
    
    Args:
        query (str): Search query text
        limit (int): Maximum number of results to return for each type
        
    Returns:
        tuple: (tweet_results, media_combined_results, media_image_only_results)
    """
    tweet_results = search_tweets(query, limit)
    media_combined_results = search_media(query, limit) # Uses default 0.5/0.5 weights
    media_image_only_results = search_media_image_only(query, limit)
    
    return tweet_results, media_combined_results, media_image_only_results 
