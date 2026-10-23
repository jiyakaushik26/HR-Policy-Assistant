from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.rag import DB_PATH, PDF_PATH, create_vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the HR manual PDF into a ChromaDB vector store.")
    parser.add_argument("--pdf", type=str, default=str(PDF_PATH), help="Path to the HR policy PDF")
    parser.add_argument("--persist-dir", type=str, default=str(DB_PATH), help="Directory for the persistent ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Delete the existing persistent store before ingestion")
    args = parser.parse_args()

    persist_dir = Path(args.persist_dir)
    if args.reset and persist_dir.exists():
        shutil.rmtree(persist_dir)

    vector_store = create_vector_store(pdf_path=Path(args.pdf), persist_dir=persist_dir)
    print(f"Ingested {len(vector_store.get()['documents'])} chunks into {persist_dir}")


if __name__ == "__main__":
    main()
