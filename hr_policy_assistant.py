"""Compatibility wrapper for the HR Policy Assistant package."""

from src.rag import answer_question, create_vector_store, load_vector_store

__all__ = ["answer_question", "create_vector_store", "load_vector_store"]
