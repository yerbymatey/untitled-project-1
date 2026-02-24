import subprocess
import argparse
import sys
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def stream_output(pipe, log_level):
    """Reads lines from a pipe and logs them."""
    try:
        # Use iter to read lines without blocking indefinitely
        for line in iter(pipe.readline, b''):
            log_level(line.decode('utf-8', errors='replace').rstrip())
    except Exception as e:
        # Catch potential errors during stream reading
        logger.error(f"Error reading stream: {e}")
    finally:
        pipe.close()

def run_command(command):
    """Runs a command using subprocess, logs output in real-time."""
    logger.info(f"Running command: {' '.join(command)}")
    try:
        # Use Popen for real-time output
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=False # Read as bytes to handle potential encoding issues
        )

        # Use threads to read stdout and stderr concurrently without blocking
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, logger.info))
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, logger.error))

        stdout_thread.start()
        stderr_thread.start()

        # Wait for threads to finish (which means pipes are closed)
        stdout_thread.join()
        stderr_thread.join()

        # Wait for the process to terminate and get the exit code
        process.wait()
        return_code = process.returncode

        if return_code == 0:
            logger.info(f"Command finished successfully with exit code {return_code}.")
            return True
        else:
            logger.error(f"Command failed with exit code {return_code}: {' '.join(command)}")
            return False

    except FileNotFoundError:
        logger.error(f"Error: The command '{command[0]}' was not found. Make sure Python and the required modules are in your PATH.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while trying to run the command: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the full data processing pipeline.")
    parser.add_argument(
        "--skip-db-setup",
        action="store_true",
        help="Skip the database schema setup step (`python -m db.schema`).",
    )
    parser.add_argument(
        "--skip-scraping",
        action="store_true",
        help="Skip the data scraping step (`python -m scripts.run_scraper`).",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip the embedding generation step (`python -m scripts.encode_embeddings`).",
    )
    parser.add_argument(
        "--skip-image-descriptions",
        action="store_true",
        help="Skip the image description step (Gemini).",
    )
    # Removed image text extraction step (SmolDocling/MLX)

    args = parser.parse_args()

    python_executable = sys.executable # Use the same python that runs this script

    logger.info("Starting data pipeline execution...")

    # Step 1: Database Schema Setup (Optional)
    if not args.skip_db_setup:
        logger.info("--- Running Database Schema Setup ---")
        if not run_command([python_executable, "-m", "db.schema"]):
            logger.error("Database schema setup failed. Aborting pipeline.")
            sys.exit(1)
        logger.info("--- Database Schema Setup Finished ---")
    else:
        logger.info("--- Skipping Database Schema Setup ---")

    # Step 2: Scrape Data (Optional)
    if not args.skip_scraping:
        logger.info("--- Running Data Scraper ---")
        # Assuming run_scraper handles ingestion via BookmarkIngester internally
        if not run_command([python_executable, "-m", "scripts.run_scraper"]):
            logger.error("Data scraping failed. Aborting pipeline.")
            sys.exit(1)
        logger.info("--- Data Scraper Finished ---")
    else:
        logger.info("--- Skipping Data Scraper ---")

    # Removed: Image text extraction step (SmolDocling/MLX)

    # Step 4: Generate Image Descriptions (Default ON)
    if not args.skip_image_descriptions:
        logger.info("--- Running Gemini Image Description ---")
        if not run_command([python_executable, "-m", "scripts.image_descriptions_gemini"]):
            # Non-fatal: continue to embeddings even if descriptions fail
            logger.warning("Gemini image description step failed. Continuing pipeline.")
        logger.info("--- Gemini Image Description Finished ---")
    else:
        logger.info("--- Skipping Gemini Image Description ---")

    # Step 5: Generate Embeddings (Optional)
    if not args.skip_embeddings:
        logger.info("--- Running Embedding Generation ---")
        if not run_command([python_executable, "-m", "scripts.encode_embeddings"]):
            # Make this non-fatal? Depends on workflow. For now, let's make it fatal.
            logger.error("Embedding generation failed. Aborting pipeline.")
            sys.exit(1)
        logger.info("--- Embedding Generation Finished ---")
    else:
        logger.info("--- Skipping Embedding Generation ---")

    logger.info("Data pipeline execution completed.")

if __name__ == "__main__":
    main() 
