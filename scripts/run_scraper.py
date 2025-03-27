from bookmarks.scraper import TwitterBookmarkScraper
from bookmarks.parser import parse_tweet_data
from pipelines.ingest import BookmarkIngester
from utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    """Main function to run the bookmark scraper and database ingestion"""
    try:
        # Initialize scraper
        scraper = TwitterBookmarkScraper()
        
        # Fetch bookmarks
        logger.info("Starting bookmark fetch...")
        bookmarks = scraper.scrape_all_bookmarks()
        
        if not bookmarks:
            logger.warning("No bookmarks were fetched!")
            return
        
        # Parse and clean the data
        logger.info("Parsing and cleaning bookmark data...")
        parsed_bookmarks = [parse_tweet_data(bookmark) for bookmark in bookmarks]
        
        # Ingest into database and save to file
        logger.info("Starting bookmark ingestion...")
        ingester = BookmarkIngester()
        file_path = ingester.ingest_bookmarks(parsed_bookmarks)
        
        if file_path:
            logger.info(f"Successfully processed {len(parsed_bookmarks)} bookmarks")
            logger.info(f"Saved to: {file_path}")
        
    except Exception as e:
        logger.error(f"Error running scraper: {e}")
        raise

if __name__ == "__main__":
    main() 