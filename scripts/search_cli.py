#!/usr/bin/env python3
import argparse
import os
# Import all necessary search functions
from scripts.search_embeddings import (
    search_tweets,
    search_media,
    search_media_image_only,
    search_all,
    search_traditional_combined,
    rerank_text_results,
)

def _format_quoted_tweet(qt):
    """Format an inline quoted tweet block."""
    if not qt:
        return ""
    user = qt.get('user_name', '?')
    text = (qt.get('text') or '')[:140]
    url = qt.get('url', '')
    return f"  >> Quoting @{user}: {text}\n     {url}"

def format_tweet_result(tweet):
    parts = [
        f"\nScore: {tweet['similarity']:.3f}",
        f"ID: {tweet.get('id', 'N/A')}",
        f"URL: {tweet.get('tweet_url', 'N/A')}",
        f"Text: {tweet['text']}",
        f"Created at: {tweet['created_at']}",
        f"Media URL: {tweet.get('media_url', 'None')}",
    ]
    if tweet.get('extr_text'):
        # Truncate long extracted text for terminal display
        extr_preview = tweet['extr_text'][:200]
        if len(tweet['extr_text']) > 200:
            extr_preview += "..."
        parts.append(f"Extracted text: {extr_preview}")
    qt_block = _format_quoted_tweet(tweet.get('quoted_tweet'))
    if qt_block:
        parts.append(qt_block)
    return "\n".join(parts) + "\n"

def format_media_result(media, result_type="media_combined"):
    score = media['similarity']
    type_info = f" (Type: {result_type})"

    joint_sim = media.get('joint_similarity')
    image_sim = media.get('image_similarity')
    score_detail = ""
    if joint_sim is not None and image_sim is not None:
        score_detail = f" (Joint: {joint_sim:.3f}, Image: {image_sim:.3f})"

    parts = [
        f"\nScore: {score:.3f}{score_detail}{type_info}",
        f"Tweet ID: {media['tweet_id']}",
        f"URL: {media.get('tweet_url', 'N/A')}",
        f"Tweet text: {media['text']}",
        f"Image description: {media.get('image_desc', 'N/A')}",
        f"Media URL: {media['media_url']}",
        f"Created at: {media['created_at']}",
    ]
    if media.get('extr_text'):
        extr_preview = media['extr_text'][:200]
        if len(media['extr_text']) > 200:
            extr_preview += "..."
        parts.append(f"Extracted text: {extr_preview}")
    qt_block = _format_quoted_tweet(media.get('quoted_tweet'))
    if qt_block:
        parts.append(qt_block)
    return "\n".join(parts) + "\n"

def format_combined_result(item):
    """Formats a result from the traditional combined list."""
    item_type = item['type']
    if item_type == 'tweet':
        # Adapt formatting if needed, maybe add type explicitly
        return f"Type: Tweet\n{format_tweet_result(item)}"
    elif item_type == 'media_joint' or item_type == 'media_image':
        # Adapt formatting, explicitly state which embedding match this is
        return f"Type: {item_type.replace('_', ' ').capitalize()}\n{format_media_result(item, result_type=item_type)}"
    else:
        return f"Unknown item type: {item}"

def main():
    parser = argparse.ArgumentParser(description='Search tweets and media using embeddings')
    parser.add_argument('query', type=str, help='Search query text')
    parser.add_argument('--type', choices=['all', 'tweets', 'media', 'media_image'], default='all',
                      help='Type of content to search (default: all). `all` uses the selected mode.')
    parser.add_argument('--limit', type=int, default=5,
                      help='Number of results to return (default: 5)')
    parser.add_argument('--mode', choices=['weighted', 'traditional'], default='weighted',
                      help='Search mode: `weighted` uses separate lists (tweet, media_combined, media_image); `traditional` uses one combined list ranked purely by similarity. (default: weighted)')
    parser.add_argument('--rerank', action='store_true',
                      help='Rerank results with Voyage rerank-2.5 (tweets/media lists).')
    parser.add_argument('--force-cpu', action='store_true',
                      help='No-op for hosted embeddings (kept for compatibility)')
    # Add weights for search_media if needed in weighted mode?
    # parser.add_argument('--weight-joint', type=float, default=0.5)
    # parser.add_argument('--weight-image', type=float, default=0.5)

    args = parser.parse_args()
    
    # --- Device Setup --- 
    if args.force_cpu:
        print("force-cpu flag ignored (hosted embeddings)")
    print(f"Search Mode: {args.mode}")
    print(f"Searching for: \"{args.query}\"")
    
    try:
        if args.mode == 'weighted':
            # --- Weighted Mode (Separate Lists) --- 
            if args.type == 'all':
                tweet_results, media_combined_results, media_image_only_results = search_all(args.query, args.limit)
                
                print("\n--- Weighted Mode Results ---")
                if tweet_results:
                    if args.rerank:
                        tweet_results = rerank_text_results(args.query, tweet_results, text_key='text')
                    print(f"\nTop {len(tweet_results)} Tweets (Text Similarity):")
                    for tweet in tweet_results:
                        print(format_tweet_result(tweet))
                else:
                    print("\nNo tweets found")
                    
                if media_combined_results:
                    if args.rerank:
                        # Build documents from image_desc + text for reranking
                        from utils.voyage import voyage_rerank
                        docs = [f"{m.get('image_desc','')}\n\n{m.get('text','')}" for m in media_combined_results]
                        try:
                            rr = voyage_rerank(args.query, docs, model='rerank-2.5', top_k=len(docs))
                            order = [r['index'] for r in rr]
                            media_combined_results = [media_combined_results[i] for i in order]
                        except Exception as e:
                            print(f"Rerank failed (media combined): {e}")
                    print(f"\nTop {len(media_combined_results)} Media (Combined 0.5*Joint + 0.5*Image Similarity):")
                    for media in media_combined_results:
                        print(format_media_result(media, result_type="media_combined")) # Use combined formatter
                else:
                    print("\nNo media found (combined score)")

                if media_image_only_results:
                    if args.rerank:
                        from utils.voyage import voyage_rerank
                        docs = [f"{m.get('image_desc','')}\n\n{m.get('text','')}" for m in media_image_only_results]
                        try:
                            rr = voyage_rerank(args.query, docs, model='rerank-2.5', top_k=len(docs))
                            order = [r['index'] for r in rr]
                            media_image_only_results = [media_image_only_results[i] for i in order]
                        except Exception as e:
                            print(f"Rerank failed (media image-only): {e}")
                    print(f"\nTop {len(media_image_only_results)} Media (Image-Only Similarity):")
                    for media in media_image_only_results:
                         # Use the basic media formatter, score is direct image similarity
                        print(format_media_result(media, result_type="media_image_only"))
                else:
                    print("\nNo media found (image-only score)")

            elif args.type == 'tweets':
                results = search_tweets(args.query, args.limit)
                if args.rerank and results:
                    results = rerank_text_results(args.query, results, text_key='text')
                if results:
                    print(f"\nTop {len(results)} tweets:")
                    for tweet in results:
                        print(format_tweet_result(tweet))
                else:
                    print("\nNo tweets found")
            elif args.type == 'media': # In weighted mode, 'media' means combined score
                results = search_media(args.query, args.limit)
                if args.rerank and results:
                    from utils.voyage import voyage_rerank
                    docs = [f"{m.get('image_desc','')}\n\n{m.get('text','')}" for m in results]
                    try:
                        rr = voyage_rerank(args.query, docs, model='rerank-2.5', top_k=len(docs))
                        order = [r['index'] for r in rr]
                        results = [results[i] for i in order]
                    except Exception as e:
                        print(f"Rerank failed (media combined): {e}")
                if results:
                    print(f"\nTop {len(results)} media (combined score):")
                    for media in results:
                        print(format_media_result(media, result_type="media_combined"))
                else:
                    print("\nNo media found (combined score)")
            elif args.type == 'media_image': # Added specific type for image-only
                results = search_media_image_only(args.query, args.limit)
                if results:
                    print(f"\nTop {len(results)} media (image-only score):")
                    for media in results:
                        print(format_media_result(media, result_type="media_image_only"))
                else:
                    print("\nNo media found (image-only score)")
        
        elif args.mode == 'traditional':
            # --- Traditional Mode (Single Combined List) --- 
            if args.type != 'all':
                 print(f"Warning: --type argument is ignored in 'traditional' mode. Searching all types.")
                 
            results = search_traditional_combined(args.query, args.limit)
            print("\n--- Traditional Mode Results (Combined Ranking) ---")
            if results:
                print(f"\nTop {len(results)} results (ranked purely by similarity):")
                for item in results:
                    print(format_combined_result(item)) # Use the combined formatter
            else:
                print("\nNo results found in combined search.")
                
    except Exception as e:
        print(f"\nError processing query: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 
