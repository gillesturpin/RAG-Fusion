"""
Évaluateur RAGAS pour Agentic RAG
Évalue le système avec 4 métriques : context_precision, faithfulness, answer_relevancy, answer_correctness
"""

from typing import Dict, Optional
from ragas import evaluate
from ragas.metrics import (
    context_precision,      # Retrieval
    faithfulness,           # Generation
    answer_relevancy,       # Generation
    answer_correctness      # Generation (nécessite ground_truth)
)
from datasets import Dataset
from langchain_anthropic import ChatAnthropic
import logging

logger = logging.getLogger(__name__)


class CertificationEvaluator:
    """
    Évaluateur pour certification avec 4 métriques RAGAS

    Métriques:
        - context_precision: Qualité du retrieval (docs pertinents)
        - faithfulness: Pas d'hallucinations (fidélité au contexte)
        - answer_relevancy: Pertinence de la réponse à la question
        - answer_correctness: Exactitude factuelle (vs ground truth)
    """

    def __init__(self, rag_agent):
        """
        Args:
            rag_agent: Instance du RAG Agent à évaluer
        """
        self.rag_agent = rag_agent

        # LLM pour RAGAS (utilise Claude au lieu d'OpenAI)
        self.llm = ChatAnthropic(
            model="claude-3-5-haiku-20241022",  # Haiku pour économie
            temperature=0
        )

        # Métriques RAGAS
        self.metrics = [
            context_precision,     # Sans GT
            faithfulness,          # Sans GT
            answer_relevancy,      # Sans GT
            answer_correctness     # AVEC GT (critique pour certification)
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

        # Configure metrics to use Claude
        from ragas.llms import LangchainLLMWrapper
        critic_llm = LangchainLLMWrapper(self.llm)

        # Update metrics with Claude LLM
        for metric in self.metrics:
            metric.llm = critic_llm

        scores = evaluate(dataset, metrics=self.metrics)

        # 6. Structure results
        evaluation_result = {
            "question": question,
            "answer": result["answer"],
            "ground_truth": ground_truth,
            "contexts": contexts,

            # Scores détaillés
            "scores": {
                "retrieval": {
                    "context_precision": float(scores["context_precision"]),
                },
                "generation": {
                    "faithfulness": float(scores["faithfulness"]),
                    "answer_relevancy": float(scores["answer_relevancy"]),
                    "answer_correctness": float(scores["answer_correctness"]),
                },
            },

            # Score global pondéré
            "overall_score": self._compute_overall_score(scores),

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
        Calcule le score global pondéré

        Pondération:
            - context_precision: 20%  (retrieval)
            - faithfulness: 30%       (pas d'hallucinations - critique)
            - answer_relevancy: 20%   (pertinence)
            - answer_correctness: 30% (exactitude - critique pour certification)

        Args:
            scores: Dict des scores RAGAS

        Returns:
            Score global entre 0 et 1
        """
        return (
            0.2 * scores["context_precision"] +
            0.3 * scores["faithfulness"] +
            0.2 * scores["answer_relevancy"] +
            0.3 * scores["answer_correctness"]
        )

    def get_certification_verdict(self, overall_score: float, scores: dict) -> Dict:
        """
        Détermine si le système passe la certification

        Seuils:
            - context_precision >= 0.70
            - faithfulness >= 0.80 (strict)
            - answer_relevancy >= 0.70
            - answer_correctness >= 0.75 (strict)

        Args:
            overall_score: Score global
            scores: Scores individuels

        Returns:
            Dict avec verdict de certification
        """

        thresholds = {
            "context_precision": 0.70,
            "faithfulness": 0.80,
            "answer_relevancy": 0.70,
            "answer_correctness": 0.75
        }

        # Vérifier chaque seuil
        passed = all(
            scores[metric] >= threshold
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
