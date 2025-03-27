from playwright.sync_api import sync_playwright
import re
from typing import Dict, Optional

def format_env_value(value: str) -> str:
    """Format a value for .env file, handling special characters"""
    # Replace any newlines with spaces
    value = value.replace('\n', ' ')
    # Escape any quotes
    value = value.replace('"', '\\"')
    return value

def generate_env_content(data: Dict[str, Optional[str]]) -> str:
    """Generate .env file content from captured data"""
    env_lines = []
    
    # Add auth token
    if data['auth']:
        auth_token = data['auth'].replace("Bearer ", "")
        env_lines.append(f'X_AUTH_TOKEN="{format_env_value(auth_token)}"')
    
    # Add cookie string
    if data['cookie']:
        env_lines.append(f'X_COOKIE_STRING="{format_env_value(data["cookie"])}"')
    
    # Add CSRF token
    if data['csrf']:
        env_lines.append(f'X_CSRF_TOKEN="{format_env_value(data["csrf"])}"')
    
    # Add API ID
    if data['bookmarks_api_id']:
        env_lines.append(f'BOOKMARKS_API_ID="{format_env_value(data["bookmarks_api_id"])}"')
    
    return '\n'.join(env_lines)

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
        env_content = generate_env_content(captured_data)
        print(env_content)
        
        # Save to .env file
        print("\n4. Saving to .env file...")
        with open(".env", "w") as f:
            f.write(env_content)
        
        print("\n=== Capture Summary ===")
        for key, value in captured_data.items():
            status = '✓' if value else '✗'
            if value and key != 'cookie':
                print(f"- {key}: {status} ({value[:50]}...)" if len(str(value)) > 50 else f"- {key}: {status} ({value})")
            else:
                print(f"- {key}: {status}")

        print("\n✓ Headers captured and saved to .env file")
        browser.close()

if __name__ == "__main__":
    save_twitter_headers()