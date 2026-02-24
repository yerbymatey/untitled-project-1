from flask import Flask, render_template, request, jsonify
from scripts.search_embeddings import (
    search_all,
    search_traditional_combined,
    search_tweets,
    search_media,
    search_media_image_only,
    rerank_text_results,
)
import os
import logging
import numpy as np

# Configure logger if needed in app context
# logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('search.html')

@app.route('/search')
def search():
    query = request.args.get('q', '')
    expanded = request.args.get('expanded', 'false') == 'true'
    mode = request.args.get('mode', 'weighted')

    if not query:
        return jsonify({
            'tweets': [],
            'media_combined': [],
            'media_image_only': [],
            'combined_results': [] 
        })

    limit = 10 if expanded else 5

    if mode == 'traditional':
        combined_results = search_traditional_combined(query, limit=limit)

        formatted_combined = []
        for item in combined_results:
            item_type = item.get('type', 'unknown')
            status_id = item.get('id') if item_type == 'tweet' else item.get('tweet_id')
            embed_html = ""
            if status_id:
                 embed_html = f'<blockquote class="twitter-tweet" data-conversation="none" data-dnt="true"><p lang="en" dir="ltr">Loading tweet...</p>&mdash; User <a href="https://twitter.com/x/status/{status_id}">Date</a></blockquote>'

            formatted_combined.append({
                **item,
                'tweet_url': f"https://twitter.com/x/status/{status_id}" if status_id else "#",
                'embed_html': embed_html
            })

        return jsonify({
            'combined_results': formatted_combined,
            'has_more': len(combined_results) == limit 
        })

    else:
        tweet_results, media_combined_results, media_image_only_results = search_all(query, limit=limit)

        formatted_tweets = [{
            **tweet,
            'tweet_url': f"https://twitter.com/x/status/{tweet['id']}",
            'embed_html': f'<blockquote class="twitter-tweet" data-conversation="none" data-dnt="true"><p lang="en" dir="ltr">Loading tweet...</p>&mdash; User <a href="https://twitter.com/x/status/{tweet["id"]}">Date</a></blockquote>'
        } for tweet in tweet_results]

        formatted_media_combined = [{
            **media,
            'tweet_url': f"https://twitter.com/x/status/{media['tweet_id']}",
            'embed_html': f'<blockquote class="twitter-tweet" data-conversation="none" data-dnt="true"><p lang="en" dir="ltr">Loading tweet...</p>&mdash; User <a href="https://twitter.com/x/status/{media["tweet_id"]}">Date</a></blockquote>'
        } for media in media_combined_results]

        formatted_media_image_only = [{
            **media,
            'tweet_url': f"https://twitter.com/x/status/{media['tweet_id']}",
            'embed_html': f'<blockquote class="twitter-tweet" data-conversation="none" data-dnt="true"><p lang="en" dir="ltr">Loading tweet...</p>&mdash; User <a href="https://twitter.com/x/status/{media["tweet_id"]}">Date</a></blockquote>'
        } for media in media_image_only_results]

        has_more_tweets = len(tweet_results) == limit
        has_more_media_combined = len(media_combined_results) == limit
        has_more_media_image_only = len(media_image_only_results) == limit

        return jsonify({
            'tweets': formatted_tweets,
            'media_combined': formatted_media_combined,
            'media_image_only': formatted_media_image_only,
            'has_more': has_more_tweets or has_more_media_combined or has_more_media_image_only
        })

def _sanitize_for_json(obj):
    """Recursively convert numpy/torch scalars to Python-native types."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _tweet_to_api_item(tweet):
    return _sanitize_for_json({
        'id': tweet.get('id'),
        'text': tweet.get('text'),
        'url': tweet.get('tweet_url'),
        'score': tweet.get('similarity'),
        'created_at': tweet.get('created_at'),
        'quoted_tweet': tweet.get('quoted_tweet'),
        'media_url': tweet.get('media_url'),
        'image_desc': tweet.get('image_desc'),
    })


def _media_to_api_item(media):
    return _sanitize_for_json({
        'id': media.get('tweet_id'),
        'text': media.get('text'),
        'url': media.get('tweet_url'),
        'score': media.get('similarity'),
        'created_at': media.get('created_at'),
        'quoted_tweet': media.get('quoted_tweet'),
        'media_url': media.get('media_url'),
        'image_desc': media.get('image_desc'),
    })


def _rerank_media(query, media_list):
    """Rerank a media result list using Voyage rerank-2.5, following search_cli.py pattern."""
    if not media_list:
        return media_list
    try:
        from utils.voyage import voyage_rerank
        docs = [f"{m.get('image_desc', '')}\n\n{m.get('text', '')}" for m in media_list]
        rr = voyage_rerank(query, docs, model='rerank-2.5', top_k=len(docs))
        order = [r['index'] for r in rr]
        return [media_list[i] for i in order]
    except Exception as e:
        logging.getLogger(__name__).warning(f"Media rerank failed: {e}")
        return media_list


@app.route('/api/search')
def api_search():
    query = request.args.get('q', '').strip()
    search_type = request.args.get('type', 'all')
    mode = request.args.get('mode', 'weighted')
    do_rerank = request.args.get('rerank', 'false').lower() == 'true'

    if not query:
        return jsonify({'error': 'q parameter is required'}), 400
    if search_type not in ('all', 'tweets', 'media', 'media_image'):
        return jsonify({'error': 'type must be one of: all, tweets, media, media_image'}), 400
    if mode not in ('weighted', 'traditional'):
        return jsonify({'error': 'mode must be one of: weighted, traditional'}), 400

    try:
        limit = int(request.args.get('limit', 5))
    except (ValueError, TypeError):
        return jsonify({'error': 'limit must be an integer'}), 400
    limit = max(1, min(limit, 50))

    try:
        if mode == 'traditional':
            results = search_traditional_combined(query, limit=limit)
            if do_rerank and results:
                results = rerank_text_results(query, results, text_key='text')
            formatted = []
            for r in results:
                item_type = r.get('type', 'unknown')
                base = _tweet_to_api_item(r) if item_type == 'tweet' else _media_to_api_item(r)
                base['type'] = item_type
                formatted.append(base)
            return jsonify({'query': query, 'mode': 'traditional', 'results': formatted})

        # weighted mode
        if search_type == 'tweets':
            tweets = search_tweets(query, limit=limit)
            if do_rerank and tweets:
                tweets = rerank_text_results(query, tweets, text_key='text')
            return jsonify({
                'query': query, 'mode': 'weighted', 'type': 'tweets',
                'results': {'tweets': [_tweet_to_api_item(t) for t in tweets]},
            })

        if search_type == 'media':
            media = search_media(query, limit=limit)
            if do_rerank:
                media = _rerank_media(query, media)
            return jsonify({
                'query': query, 'mode': 'weighted', 'type': 'media',
                'results': {'media_combined': [_media_to_api_item(m) for m in media]},
            })

        if search_type == 'media_image':
            media = search_media_image_only(query, limit=limit)
            if do_rerank:
                media = _rerank_media(query, media)
            return jsonify({
                'query': query, 'mode': 'weighted', 'type': 'media_image',
                'results': {'media_image_only': [_media_to_api_item(m) for m in media]},
            })

        # type == 'all'
        tweets, media_combined, media_image_only = search_all(query, limit=limit)
        if do_rerank:
            if tweets:
                tweets = rerank_text_results(query, tweets, text_key='text')
            media_combined = _rerank_media(query, media_combined)
            media_image_only = _rerank_media(query, media_image_only)

        return jsonify({
            'query': query, 'mode': 'weighted', 'type': 'all',
            'results': {
                'tweets': [_tweet_to_api_item(t) for t in tweets],
                'media_combined': [_media_to_api_item(m) for m in media_combined],
                'media_image_only': [_media_to_api_item(m) for m in media_image_only],
            },
        })

    except Exception as e:
        logging.getLogger(__name__).error(f"API search error: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True)