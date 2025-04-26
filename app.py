from flask import Flask, render_template, request, jsonify
from scripts.search_embeddings import (
    search_all,
    search_traditional_combined,
    search_tweets,
    search_media,
    search_media_image_only,
)
import os
import logging

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

if __name__ == '__main__':
    app.run(debug=True) 