from setuptools import setup, find_packages

setup(
    version="0.1.0",
    description="A tool to fetch and store Twitter/X bookmarks with vector search capabilities",
    packages=find_packages(),
    install_requires=[
        "requests>=2.31.0",
        "psycopg2-binary>=2.9.9",
        "python-dotenv>=1.0.0",
        "playwright>=1.42.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "twitter-bookmarks=scripts.run_scraper:main",
            "grab-headers=scrapers.grab_headers:save_twitter_headers",
        ],
    },
) 