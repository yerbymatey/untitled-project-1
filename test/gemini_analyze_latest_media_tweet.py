import os
import tempfile
import requests
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv
from db.session import get_db_session

load_dotenv()

def get_latest_media_tweet():
    with get_db_session() as session:
        query = """
        SELECT t.text, m.media_url, m.type as media_type
        FROM tweets t
        JOIN media m ON t.id = m.tweet_id
        ORDER BY t.created_at DESC
        LIMIT 1
        """
        session.execute(query)
        return session.fetchone()

def analyze_with_gemini(tweet_data):
    # Download the image to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    response = requests.get(tweet_data['media_url'])
    temp_file.write(response.content)
    temp_file.close()

    # Configure API key
    genai.configure(api_key=os.environ.get("google_api_key"))
    
    # Load the image file
    image_data = open(temp_file.name, "rb").read()
    
    # Create the model
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Prepare the system prompt
    system_prompt = """You are a high-context digital media interpreter trained to extract latent semantic structures and thematic payloads from complex multimodal objects. Given a digital object (e.g., tweet, image, meme, or quote tweet), return a conceptually precise and structurally coherent distillation in 1–2 dense sentences.

Do not interpret from a viewer's perspective or describe superficial visual elements unless they directly reinforce the object's rhetorical function. If the object is a quote tweet, treat the quoted tweet as the source artifact and the main tweet as framing commentary—analyze their interplay as a single compositional unit.

Focus on the function of each component within the object's internal logic. Disregard purely decorative or non-semantic features. Your goal is to encode the object's essential mechanics, generative intention, and structural affordances with high compression and minimal redundancy."""
    
    # Create the contents with image and text
    contents = [
        {"role": "user", 
         "parts": [
             {"text": system_prompt},
             {"inline_data": {"mime_type": "image/jpeg", "data": image_data}},
             {"text": f"Tweet text: {tweet_data['text']}"}
         ]
        }
    ]
    
    # Generate response
    response = model.generate_content(contents)
    interpretation = response.text
    print(interpretation)
    
    # Store interpretation in database
    with get_db_session() as session:
        query = """
        INSERT INTO tweet_interpretations (tweet_id, interpretation)
        SELECT t.id, %s
        FROM tweets t
        JOIN media m ON t.id = m.tweet_id
        ORDER BY t.created_at DESC
        LIMIT 1
        ON CONFLICT (tweet_id) 
        DO UPDATE SET interpretation = %s, created_at = CURRENT_TIMESTAMP
        """
        session.execute(query, (interpretation, interpretation))
        session.commit()
    
    # Clean up temporary file
    os.unlink(temp_file.name)

def view_tweet_with_interpretation():
    with get_db_session() as session:
        # Query for tweets with interpretations
        query = """
        SELECT t.id, t.text, m.media_url, ti.interpretation 
        FROM tweets t 
        JOIN tweet_interpretations ti ON t.id = ti.tweet_id 
        JOIN media m ON t.id = m.tweet_id 
        ORDER BY t.created_at DESC
        LIMIT 1
        """
        session.execute(query)
        result = session.fetchone()
        
        if result:
            print("\n=== TWEET WITH INTERPRETATION ===")
            print(f"Tweet ID: {result['id']}")
            print(f"Tweet Text: {result['text']}")
            print(f"Media URL: {result['media_url']}")
            print(f"Interpretation: {result['interpretation']}")
            print("==================================\n")
        else:
            print("No tweets with interpretations found")

def main():
    tweet_data = get_latest_media_tweet()
    if not tweet_data:
        print("No media tweets found")
        return
    
    print(f"Found tweet: {tweet_data['text']}")
    print(f"Media URL: {tweet_data['media_url']}")
    print("\nAnalyzing with Gemini...\n")
    
    analyze_with_gemini(tweet_data)
    view_tweet_with_interpretation

if __name__ == "__main__":
    main() 