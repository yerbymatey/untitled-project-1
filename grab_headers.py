from playwright.sync_api import sync_playwright
import json
import re

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
                print(f"Found bookmarks API ID: {captured_data['bookmarks_api_id']}")

            # Capture auth headers
            if 'authorization' in headers:
                captured_data['auth'] = headers['authorization']
                print(f"Found auth token")
            if 'cookie' in headers:
                captured_data['cookie'] = headers['cookie']
                print(f"Found cookie")
            if 'x-csrf-token' in headers:
                captured_data['csrf'] = headers['x-csrf-token']
                print(f"Found CSRF token")

        page.on("request", handle_request)

        # Navigate and wait for interaction
        print("Navigating to Twitter...")
        page.goto("https://x.com/i/bookmarks")
        
        # Wait for user manual log in
        input("Please log in if needed and press Enter once you're on the bookmarks page...")
        
        # Try to get cookies directly from context
        cookies = context.cookies()
        if cookies:
            cookie_string = '; '.join([f"{c['name']}={c['value']}" for c in cookies])
            captured_data['cookie'] = cookie_string
            print("Found cookies from context")

        # Trigger some scrolling for req capture
        print("Scrolling page to trigger requests...")
        for _ in range(3):
            page.mouse.wheel(0, 1000)
            page.wait_for_timeout(1000)

        # Write to .env
        print("\nWriting available data to .env file...")
        with open(".env", "w") as f:
            if captured_data['auth']:
                auth_token = captured_data['auth'].replace("Bearer ", "")
                f.write(f'X_AUTH_TOKEN={auth_token}\n')
            if captured_data['cookie']:
                f.write(f'X_COOKIE_STRING={captured_data["cookie"]}\n')
            if captured_data['csrf']:
                f.write(f'X_CSRF_TOKEN={captured_data["csrf"]}\n')
            if captured_data['bookmarks_api_id']:
                f.write(f'BOOKMARKS_API_ID={captured_data["bookmarks_api_id"]}\n')

        print("\nCaptured data status:")
        for key, value in captured_data.items():
            status = '✓' if value else '✗'
            if value and key != 'cookie':
                print(f"- {key}: {status} ({value[:50]}...)" if len(str(value)) > 50 else f"- {key}: {status} ({value})")
            else:
                print(f"- {key}: {status}")

        browser.close()

if __name__ == "__main__":
    save_twitter_headers()