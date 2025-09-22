import os
from typing import Dict, Optional
import re

from playwright.sync_api import sync_playwright

def format_env_value(value: str) -> str:
    """Format a value for .env file, handling special characters"""
    # Replace any newlines with spaces
    value = value.replace('\n', ' ')
    # Escape any quotes
    value = value.replace('"', '\\"')
    return value

def generate_env_updates(data: Dict[str, Optional[str]]) -> Dict[str, str]:
    """Build the values we want to persist to the .env file"""
    updates: Dict[str, str] = {}

    if data['auth']:
        auth_token = data['auth'].replace("Bearer ", "")
        updates['X_AUTH_TOKEN'] = format_env_value(auth_token)

    if data['cookie']:
        updates['X_COOKIE_STRING'] = format_env_value(data['cookie'])

    if data['csrf']:
        updates['X_CSRF_TOKEN'] = format_env_value(data['csrf'])

    if data['bookmarks_api_id']:
        updates['BOOKMARKS_API_ID'] = format_env_value(data['bookmarks_api_id'])

    return updates


def merge_env_file(updates: Dict[str, str], filepath: str = ".env") -> None:
    """Merge captured headers into the existing .env without dropping other variables"""
    preserved_lines = []

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.rstrip("\n")
                stripped = line.strip()

                if not stripped or stripped.startswith("#"):
                    preserved_lines.append(line)
                    continue

                if "=" not in stripped:
                    preserved_lines.append(line)
                    continue

                key = stripped.split("=", 1)[0].strip()
                if key in updates:
                    # Skip old value; we'll append the refreshed one later
                    continue

                preserved_lines.append(line)

    for key, value in updates.items():
        preserved_lines.append(f'{key}="{value}"')

    with open(filepath, "w", encoding="utf-8") as env_file:
        env_file.write("\n".join(preserved_lines))
        env_file.write("\n")

def save_twitter_headers():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        captured_data = {
            'bookmarks_api_id': None,
            'auth': None,
            'cookie': None,
            'csrf': None
        }

        def handle_request(request):
            url = request.url
            headers = request.headers
            
            # Check for bookmarks API ID
            bookmarks_pattern = re.compile(r'https://(?:x|twitter)\.com/i/api/graphql/([^/]+)/Bookmarks\?')
            match = bookmarks_pattern.search(url)
            if match:
                captured_data['bookmarks_api_id'] = match.group(1)
                print(f"✓ Found bookmarks API ID: {captured_data['bookmarks_api_id']}")

            # Capture auth headers
            if 'authorization' in headers:
                captured_data['auth'] = headers['authorization']
                print(f"✓ Found auth token")
            if 'cookie' in headers:
                captured_data['cookie'] = headers['cookie']
                print(f"✓ Found cookie")
            if 'x-csrf-token' in headers:
                captured_data['csrf'] = headers['x-csrf-token']
                print(f"✓ Found CSRF token")

        page.on("request", handle_request)

        # Navigate and wait for interaction
        print("\n=== Twitter Headers Capture ===")
        print("1. Opening Twitter bookmarks page...")
        page.goto("https://x.com/i/bookmarks")
        
        print("\n2. Please log in if needed and press Enter once you're on the bookmarks page...")
        input()
        
        # Try to get cookies directly from context
        cookies = context.cookies()
        if cookies:
            cookie_string = '; '.join([f"{c['name']}={c['value']}" for c in cookies])
            captured_data['cookie'] = cookie_string
            print("✓ Found cookies from context")

        # Trigger some scrolling for req capture
        print("\n3. Scrolling page to capture all headers...")
        for _ in range(3):
            page.mouse.wheel(0, 1000)
            page.wait_for_timeout(1000)

        # Generate and display .env content
        print("\n=== Generated .env Content ===")
        env_updates = generate_env_updates(captured_data)
        env_content = '\n'.join(f'{key}="{value}"' for key, value in env_updates.items())
        print(env_content)
        
        # Save to .env file
        if env_updates:
            print("\n4. Saving to .env file...")
            merge_env_file(env_updates)
        else:
            print("\n4. No new headers captured; existing .env left untouched.")
        
        print("\n=== Capture Summary ===")
        for key, value in captured_data.items():
            status = '✓' if value else '✗'
            if value and key != 'cookie':
                print(f"- {key}: {status} ({value[:50]}...)" if len(str(value)) > 50 else f"- {key}: {status} ({value})")
            else:
                print(f"- {key}: {status}")

        if env_updates:
            print("\n✓ Headers captured and saved to .env file")
        else:
            print("\n✓ Capture complete (no changes written)")
        browser.close()

if __name__ == "__main__":
    save_twitter_headers()
