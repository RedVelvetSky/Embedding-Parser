import os
import time
import json
import logging
from openai import OpenAI
import schedule
import clickhouse_connect
from typing import List

if os.name == "nt":  # Windows
    from dotenv import load_dotenv
    load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler("embedding_processor.log")
    ]
)

OPENAI_EMBEDDING_MODEL = 'text-embedding-3-small'

client = clickhouse_connect.get_client(
    host=os.getenv('SLURP_HOST'),
    port=int(os.getenv('SLURP_PORT')),
    database=os.getenv('SLURP_DATABASE'),
    username=os.getenv('SLURP_USERNAME'),
    password=os.getenv('SLURP_PASSWORD'),
    connect_timeout=30,
    secure=False,
    server_host_name=os.getenv('SLURP_HOST'),
    settings={'session_timeout': 300}
)

clientai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def fetch_new_rows() -> List[dict]:
    query = f"""
    SELECT *
    FROM articles
    WHERE (embedding IS NULL OR empty(embedding))
    AND created_at >= now() - INTERVAL 3 DAY
    """
    try:
        result = client.query(query)
        rows = result.result_rows
        columns = result.column_names

        logging.info(f"Fetched {len(rows)} new row(s) from the database.")
        return [dict(zip(columns, row)) for row in rows]

    except Exception as e:
        logging.error(f"Error fetching new rows: {e}")
        return []

def generate_embedding(text: str) -> List[float]:
    try:
        response = clientai.embeddings.create(
            input=text,
            model=OPENAI_EMBEDDING_MODEL
        )
        embedding = response.data[0].embedding
        logging.info("Generated embedding successfully.")
        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return []

def update_embedding(row_id: int, embedding: List[float]):
    try:
        embedding_json = json.dumps(embedding)
        query = f"""
        ALTER TABLE articles
        UPDATE embedding = '{embedding_json}'
        WHERE id = '{row_id}'
        """
        client.command(query)
        logging.info(f"Updated embedding for row ID {row_id}")
    except Exception as e:
        logging.error(f"Error updating embedding for row ID {row_id}: {e}")

def process_new_rows():
    logging.info("Checking for new rows...")
    new_rows = fetch_new_rows()
    if not new_rows:
        logging.info("No new rows found.")
        return

    logging.info(f"Found {len(new_rows)} new row(s). Processing...")

    for row in new_rows:
        row_id = row.get('id')
        title = row.get('title', '')
        content = row.get('content', '')
        combined_text = f"{title} {content}".strip()

        if not combined_text:
            logging.warning(f"Row ID {row_id} has empty 'title' and 'content'. Skipping.")
            continue

        embedding = generate_embedding(combined_text)
        if embedding:
            update_embedding(row_id, embedding)
        else:
            logging.warning(f"Failed to generate embedding for row ID {row_id}.")

def main():
    schedule.every(1).minutes.do(process_new_rows)

    logging.info("Embedding processor started. Press Ctrl+C to exit.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Embedding processor stopped.")

if __name__ == "__main__":
    main()