"""
RAGAS Evaluator for Agentic RAG
Evaluates the system with 2 essential metrics:
- context_precision (retrieval quality)
- answer_similarity (generation semantic similarity - more tolerant)
"""

from typing import Dict, Optional
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    answer_similarity
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)


class EvaluationEvaluator:
    """
    Evaluation evaluator with 2 essential RAGAS metrics

    Metrics:
        - context_precision: Retrieval quality (relevant docs retrieved)
        - answer_similarity: Semantic similarity of answer (vs ground truth)
                            More tolerant than answer_correctness (embeddings-based)

    Overall score weighting:
        - 30% context_precision (retrieval)
        - 70% answer_similarity (generation - more important as final result)
    """

    def __init__(self, rag_agent):
        """
        Args:
            rag_agent: RAG Agent instance to evaluate
        """
        self.rag_agent = rag_agent

        langchain_llm = ChatAnthropic(
            model="claude-3-5-haiku-20241022",
            temperature=0
        )

        langchain_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.llm = LangchainLLMWrapper(langchain_llm)
        self.embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

        self.metrics = [
            context_precision,
            answer_similarity
        ]

    def evaluate_single(
        self,
        question: str,
        ground_truth: str,
        thread_id: Optional[str] = None
    ) -> Dict:
        """
        Evaluate a single question with all metrics

        Args:
            question: Question to evaluate
            ground_truth: Expected answer (required for answer_correctness)
            thread_id: Thread ID for conversational memory

        Returns:
            Dict with detailed scores
        """

        logger.info(f"Evaluating question: {question[:60]}...")

        result = self.rag_agent.invoke(question)

        # Extract contexts from result (RAGFusion returns contexts directly)
        contexts = result.get("contexts", [])

        if not contexts or contexts == [""]:
            logger.warning("No contexts retrieved!")
            contexts = ["No context retrieved"]

        data = {
            "question": [question],
            "answer": [result["answer"]],
            "contexts": [contexts],
            "ground_truth": [ground_truth]
        }

        dataset = Dataset.from_dict(data)

        logger.info("Running RAGAS evaluation...")

        scores = evaluate(
            dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )

        if hasattr(scores, 'to_pandas'):
            scores_df = scores.to_pandas()
            scores_dict = scores_df.iloc[0].to_dict()
        else:
            scores_dict = dict(scores)

        evaluation_result = {
            "question": question,
            "answer": result["answer"],
            "ground_truth": ground_truth,
            "contexts": contexts,

            "scores": {
                "context_precision": float(scores_dict.get("context_precision", 0.0)),
                "answer_similarity": float(scores_dict.get("answer_similarity", 0.0)),
            },

            "overall_score": self._compute_overall_score(scores_dict),

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
        Extract contexts from tool messages

        Args:
            messages: List of messages from the graph

        Returns:
            List of contexts (strings)
        """
        contexts = []

        for msg in messages:
            if hasattr(msg, 'type') and msg.type == "tool":
                if hasattr(msg, 'content') and msg.content:
                    contexts.append(msg.content)

        return contexts if contexts else [""]

    def _compute_overall_score(self, scores: dict) -> float:
        """
        Compute weighted overall score (2 metrics)

        Weighting:
            - context_precision: 30%  (retrieval quality)
            - answer_similarity: 70% (generation semantic similarity - more critical)

        Args:
            scores: Dict of RAGAS scores

        Returns:
            Overall score between 0 and 1
        """
        return (
            0.3 * scores.get("context_precision", 0.0) +
            0.7 * scores.get("answer_similarity", 0.0)
        )

    def get_evaluation_verdict(self, overall_score: float, scores: dict) -> Dict:
        """
        Determine if the system passes evaluation (2 metrics)

        Evaluation thresholds:
            - context_precision >= 0.70 (retrieval quality)
            - answer_similarity >= 0.75 (generation semantic similarity)

        Args:
            overall_score: Weighted overall score
            scores: Individual scores

        Returns:
            Dict with evaluation verdict
        """

        thresholds = {
            "context_precision": 0.70,
            "answer_similarity": 0.75
        }

        passed = all(
            scores.get(metric, 0) >= threshold
            for metric, threshold in thresholds.items()
        )

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
