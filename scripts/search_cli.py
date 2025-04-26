#!/usr/bin/env python3
import argparse
import os
# Import all necessary search functions
from search_embeddings import (
    search_tweets, 
    search_media, 
    search_media_image_only, 
    search_all, 
    search_traditional_combined, 
    DEVICE
)

def format_tweet_result(tweet):
    return f"""
Score: {tweet['similarity']:.3f}
ID: {tweet.get('id', 'N/A')} 
Text: {tweet['text']}
Created at: {tweet['created_at']}
Media URL: {tweet.get('media_url', 'None')}
"""

def format_media_result(media, result_type="media_combined"):
    # Determine score field based on type if needed, though 'similarity' should be standard
    score = media['similarity']
    type_info = f" (Type: {result_type})" # Clarify origin in combined mode
    
    # Optionally include individual scores if available
    joint_sim = media.get('joint_similarity')
    image_sim = media.get('image_similarity')
    score_detail = ""
    if joint_sim is not None and image_sim is not None:
        score_detail = f" (Joint: {joint_sim:.3f}, Image: {image_sim:.3f})"
        
    return f"""
Score: {score:.3f}{score_detail}{type_info}
Tweet ID: {media['tweet_id']}
Tweet text: {media['text']}
Image description: {media.get('image_desc', 'N/A')}
Media URL: {media['media_url']}
Created at: {media['created_at']}
"""

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
    parser.add_argument('--force-cpu', action='store_true',
                      help='Force using CPU even if GPU is available')
    # Add weights for search_media if needed in weighted mode?
    # parser.add_argument('--weight-joint', type=float, default=0.5)
    # parser.add_argument('--weight-image', type=float, default=0.5)

    args = parser.parse_args()
    
    # --- Device Setup --- 
    if args.force_cpu:
        # Ensure environment variable is set *before* torch potentially initializes CUDA/MPS
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Update DEVICE variable if search_embeddings uses it dynamically (it seems to define it at import time)
        # Ideally, DEVICE logic would be centralized or passed around.
        # For now, assume DEVICE reflects env correctly if set early.
        print("Attempting to force CPU usage... NOTE: May require script restart if torch already initialized GPU/MPS.")
    
    print(f"Using device: {DEVICE}") # Note: This might show GPU/MPS if already initialized
    print(f"Search Mode: {args.mode}")
    print(f"Searching for: \"{args.query}\"")
    
    try:
        if args.mode == 'weighted':
            # --- Weighted Mode (Separate Lists) --- 
            if args.type == 'all':
                tweet_results, media_combined_results, media_image_only_results = search_all(args.query, args.limit)
                
                print("\n--- Weighted Mode Results ---")
                if tweet_results:
                    print(f"\nTop {len(tweet_results)} Tweets (Text Similarity):")
                    for tweet in tweet_results:
                        print(format_tweet_result(tweet))
                else:
                    print("\nNo tweets found")
                    
                if media_combined_results:
                    print(f"\nTop {len(media_combined_results)} Media (Combined 0.5*Joint + 0.5*Image Similarity):")
                    for media in media_combined_results:
                        print(format_media_result(media, result_type="media_combined")) # Use combined formatter
                else:
                    print("\nNo media found (combined score)")

                if media_image_only_results:
                    print(f"\nTop {len(media_image_only_results)} Media (Image-Only Similarity):")
                    for media in media_image_only_results:
                         # Use the basic media formatter, score is direct image similarity
                        print(format_media_result(media, result_type="media_image_only"))
                else:
                    print("\nNo media found (image-only score)")

            elif args.type == 'tweets':
                results = search_tweets(args.query, args.limit)
                if results:
                    print(f"\nTop {len(results)} tweets:")
                    for tweet in results:
                        print(format_tweet_result(tweet))
                else:
                    print("\nNo tweets found")
            elif args.type == 'media': # In weighted mode, 'media' means combined score
                results = search_media(args.query, args.limit) # Add weights here if desired args.weight_joint, args.weight_image)
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