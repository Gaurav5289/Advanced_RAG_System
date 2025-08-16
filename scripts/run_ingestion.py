import sys
import os
from dotenv import load_dotenv

# This will find and load the .env file from your root directory
load_dotenv()

# This block adds the main project directory to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.pipeline import run_ingestion_pipeline

if __name__ == "__main__":
    run_ingestion_pipeline()