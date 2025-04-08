from flask import Flask, render_template, request, jsonify
from scripts.search_embeddings import search_all
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('search.html')

@app.route('/search')
def search():
    query = request.args.get('q', '')
    expanded = request.args.get('expanded', 'false') == 'true'
    if not query:
        return jsonify({'tweets': [], 'media': []})
    
    # Get search results
    limit = 10 if expanded else 3
    tweet_results, media_results = search_all(query, limit=limit)
    
    # Format results for display
    formatted_tweets = [{
        **tweet,
        'tweet_url': f"https://twitter.com/x/status/{tweet['id']}",
        'embed_html': f'<a href="https://twitter.com/x/status/{tweet["id"]}"></a>'
    } for tweet in tweet_results]
    
    formatted_media = [{
        **media,
        'tweet_url': f"https://twitter.com/x/status/{media['tweet_id']}",
        'embed_html': f'<a href="https://twitter.com/x/status/{media["tweet_id"]}"></a>'
    } for media in media_results]
    
    return jsonify({
        'tweets': formatted_tweets,
        'media': formatted_media,
        'has_more': len(tweet_results) == limit or len(media_results) == limit
    })

if __name__ == '__main__':
    app.run(debug=True) 