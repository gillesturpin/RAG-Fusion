"""
Agentic RAG - Optimized RAG Agent

Single agent implementation based on official LangChain tutorials:
- RAG Agent: Stateless agent with RAG Fusion (k=8 final docs) and improved prompt for completeness
"""

from .rag_agent import RAGAgent
from .evaluator import EvaluationEvaluator

__all__ = ["RAGAgent", "EvaluationEvaluator"]
__version__ = "2.0.0"
