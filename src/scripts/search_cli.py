#!/usr/bin/env python3
import argparse
import os
from src.scripts.search_embeddings import search_tweets, search_media, search_all, DEVICE

def format_tweet_result(tweet):
    return f"""
Score: {tweet['similarity']:.3f}
Text: {tweet['text']}
Created at: {tweet['created_at']}
"""

def format_media_result(media):
    return f"""
Score: {media['similarity']:.3f}
Tweet text: {media['text']}
Image description: {media['image_desc']}
Media URL: {media['media_url']}
Created at: {media['created_at']}
"""

def main():
    parser = argparse.ArgumentParser(description='Search tweets and media using embeddings')
    parser.add_argument('query', type=str, help='Search query text')
    parser.add_argument('--type', choices=['all', 'tweets', 'media'], default='all',
                      help='Type of content to search (default: all)')
    parser.add_argument('--limit', type=int, default=5,
                      help='Number of results to return (default: 5)')
    parser.add_argument('--force-cpu', action='store_true',
                      help='Force using CPU even if GPU is available')
    
    args = parser.parse_args()
    
    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Forcing CPU usage")
    
    print(f"Using device: {DEVICE}")
    print(f"Searching for: {args.query}")
    
    try:
        if args.type == 'all':
            tweet_results, media_results = search_all(args.query, args.limit)
            
            if tweet_results:
                print("\nTop tweets:")
                for tweet in tweet_results:
                    print(format_tweet_result(tweet))
            else:
                print("\nNo tweets found")
                
            if media_results:
                print("\nTop media:")
                for media in media_results:
                    print(format_media_result(media))
            else:
                print("\nNo media found")
                
        elif args.type == 'tweets':
            results = search_tweets(args.query, args.limit)
            if results:
                print("\nTop tweets:")
                for tweet in results:
                    print(format_tweet_result(tweet))
            else:
                print("\nNo tweets found")
                
        else:  # media
            results = search_media(args.query, args.limit)
            if results:
                print("\nTop media:")
                for media in results:
                    print(format_media_result(media))
            else:
                print("\nNo media found")
                
    except Exception as e:
        print(f"Error processing query: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 