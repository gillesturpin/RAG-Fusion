"""
Évaluateur RAGAS pour Agentic RAG
Évalue le système avec 2 métriques essentielles :
- context_precision (retrieval quality)
- answer_similarity (generation semantic similarity - more tolerant)
"""

from typing import Dict, Optional
from ragas import evaluate
from ragas.metrics import (
    context_precision,      # Retrieval - Quality of retrieved docs
    answer_similarity       # Generation - Semantic similarity (more tolerant than answer_correctness)
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)


class CertificationEvaluator:
    """
    Évaluateur pour certification avec 2 métriques RAGAS essentielles

    Métriques:
        - context_precision: Qualité du retrieval (docs pertinents récupérés)
        - answer_similarity: Similarité sémantique de la réponse (vs ground truth)
                            Plus tolérant qu'answer_correctness (embeddings-based)

    Pondération overall_score:
        - 30% context_precision (retrieval)
        - 70% answer_similarity (generation - plus important car résultat final)
    """

    def __init__(self, rag_agent):
        """
        Args:
            rag_agent: Instance du RAG Agent à évaluer
        """
        self.rag_agent = rag_agent

        # LLM pour RAGAS (utilise Claude au lieu d'OpenAI)
        langchain_llm = ChatAnthropic(
            model="claude-3-5-haiku-20241022",  # Haiku pour économie
            temperature=0
        )

        # Embeddings pour RAGAS (mêmes que le RAG)
        langchain_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Wrap pour RAGAS 0.3+
        self.llm = LangchainLLMWrapper(langchain_llm)
        self.embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

        # Métriques RAGAS (2 métriques essentielles)
        self.metrics = [
            context_precision,     # Retrieval quality
            answer_similarity      # Generation semantic similarity (requires ground_truth)
        ]

    def evaluate_single(
        self,
        question: str,
        ground_truth: str,
        thread_id: Optional[str] = None
    ) -> Dict:
        """
        Évalue une seule question avec toutes les métriques

        Args:
            question: La question à évaluer
            ground_truth: La réponse attendue (requis pour answer_correctness)
            thread_id: ID du thread pour mémoire conversationnelle

        Returns:
            Dict avec scores détaillés
        """

        logger.info(f"Evaluating question: {question[:60]}...")

        # 1. Invoke RAG Agent
        result = self.rag_agent.invoke(question, thread_id=thread_id)

        # 2. Extract contexts from messages
        contexts = self._extract_contexts(result["messages"])

        if not contexts or contexts == [""]:
            logger.warning("No contexts retrieved!")
            contexts = ["No context retrieved"]

        # 3. Prepare data for RAGAS
        data = {
            "question": [question],
            "answer": [result["answer"]],
            "contexts": [contexts],
            "ground_truth": [ground_truth]
        }

        # 4. Create Dataset
        dataset = Dataset.from_dict(data)

        # 5. Evaluate with RAGAS (using Claude)
        logger.info("Running RAGAS evaluation...")

        # RAGAS 0.3+ requires llm AND embeddings parameters
        scores = evaluate(
            dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )

        # 6. Convert RAGAS EvaluationResult to dict
        # RAGAS 0.3+ returns EvaluationResult object with to_pandas() method
        if hasattr(scores, 'to_pandas'):
            scores_df = scores.to_pandas()
            scores_dict = scores_df.iloc[0].to_dict()
        else:
            scores_dict = dict(scores)

        # Structure results (2 metrics only)
        evaluation_result = {
            "question": question,
            "answer": result["answer"],
            "ground_truth": ground_truth,
            "contexts": contexts,

            # Scores (simplified to 2 metrics)
            "scores": {
                "context_precision": float(scores_dict.get("context_precision", 0.0)),
                "answer_similarity": float(scores_dict.get("answer_similarity", 0.0)),
            },

            # Score global pondéré (30% retrieval, 70% generation)
            "overall_score": self._compute_overall_score(scores_dict),

            # Metadata
            "metadata": {
                "used_retrieval": result.get("used_retrieval", True),
                "num_contexts": len(contexts),
                "thread_id": thread_id
            }
        }

        logger.info(f"Evaluation complete. Overall score: {evaluation_result['overall_score']:.3f}")

        return evaluation_result

    def _extract_contexts(self, messages):
        """
        Extrait les contextes des tool messages

        Args:
            messages: Liste des messages du graph

        Returns:
            Liste des contextes (strings)
        """
        contexts = []

        for msg in messages:
            # Tool messages contiennent les documents récupérés
            if hasattr(msg, 'type') and msg.type == "tool":
                if hasattr(msg, 'content') and msg.content:
                    contexts.append(msg.content)

        return contexts if contexts else [""]

    def _compute_overall_score(self, scores: dict) -> float:
        """
        Calcule le score global pondéré (2 métriques)

        Pondération:
            - context_precision: 30%  (retrieval quality)
            - answer_similarity: 70% (generation semantic similarity - plus critique)

        Args:
            scores: Dict des scores RAGAS

        Returns:
            Score global entre 0 et 1
        """
        return (
            0.3 * scores.get("context_precision", 0.0) +
            0.7 * scores.get("answer_similarity", 0.0)
        )

    def get_certification_verdict(self, overall_score: float, scores: dict) -> Dict:
        """
        Détermine si le système passe la certification (2 métriques)

        Seuils de certification:
            - context_precision >= 0.70 (retrieval quality)
            - answer_similarity >= 0.75 (generation semantic similarity)

        Args:
            overall_score: Score global pondéré
            scores: Scores individuels

        Returns:
            Dict avec verdict de certification
        """

        thresholds = {
            "context_precision": 0.70,
            "answer_similarity": 0.75
        }

        # Vérifier chaque seuil
        passed = all(
            scores.get(metric, 0) >= threshold
            for metric, threshold in thresholds.items()
        )

        # Grade
        if overall_score >= 0.9:
            grade = "A+ (Excellent)"
        elif overall_score >= 0.8:
            grade = "A (Very Good)"
        elif overall_score >= 0.7:
            grade = "B (Good)"
        elif overall_score >= 0.6:
            grade = "C (Acceptable)"
        else:
            grade = "F (Failed)"

        return {
            "passed": passed,
            "overall_score": overall_score,
            "grade": grade,
            "requirements": {
                metric: {
                    "score": scores[metric],
                    "threshold": threshold,
                    "passed": scores[metric] >= threshold
                }
                for metric, threshold in thresholds.items()
            }
        }
