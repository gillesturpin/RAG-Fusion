#!/usr/bin/env python3
"""
Script d'évaluation de certification - Agentic RAG
Évalue le système RAG avec RAGAS sur le dataset de certification
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import logging

# Setup path
sys.path.append(str(Path(__file__).parent.parent))

from rags.rag_agent import RAGAgent
from rags.evaluator import CertificationEvaluator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


def run_certification(
    dataset_file="certification_dataset.json",
    output_file=None,
    limit=None
):
    """
    Lance l'évaluation de certification complète

    Args:
        dataset_file: Fichier JSON du dataset
        output_file: Fichier de sortie (auto-généré si None)
        limit: Limiter à N questions (pour tests rapides)
    """

    print("=" * 80)
    print("🎓 CERTIFICATION EVALUATION - Agentic RAG")
    print("=" * 80)
    print()

    # 1. Load dataset
    dataset_path = Path(__file__).parent / dataset_file

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print(f"   Run: python generate_certification_dataset.py first")
        sys.exit(1)

    with open(dataset_path) as f:
        data = json.load(f)

    questions = data.get("questions", [])
    metadata = data.get("metadata", {})

    if limit:
        questions = questions[:limit]
        print(f"⚠️  Limiting to {limit} questions for testing")

    print(f"📊 Loaded {len(questions)} certification questions")
    print(f"📚 From {metadata.get('num_documents', '?')} documents")
    print(f"📦 Using {metadata.get('total_chunks', '?')} chunks")
    print()

    # 2. Initialize RAG Agent
    print("🚀 Initializing RAG Agent...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    persist_dir = str(Path(__file__).parent.parent.parent / "data" / "chroma_db")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    checkpointer = InMemorySaver()
    agent = RAGAgent(vectorstore, checkpointer=checkpointer)

    print("✅ RAG Agent initialized")
    print()

    # 3. Initialize Evaluator
    print("🎯 Initializing RAGAS Evaluator...")
    evaluator = CertificationEvaluator(agent)
    print("✅ Evaluator ready with 4 metrics:")
    print("   - context_precision (retrieval)")
    print("   - faithfulness (no hallucinations)")
    print("   - answer_relevancy (relevance)")
    print("   - answer_correctness (accuracy)")
    print()

    # 4. Run evaluation
    print("=" * 80)
    print("🧪 RUNNING EVALUATION")
    print("=" * 80)
    print()

    results = []
    passed_count = 0

    for i, item in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Evaluating...")
        print(f"   Q: {item['question'][:70]}...")
        print(f"   Category: {item.get('category', 'N/A')}, Difficulty: {item.get('difficulty', 'N/A')}")

        try:
            # Evaluate
            result = evaluator.evaluate_single(
                question=item["question"],
                ground_truth=item["ground_truth"],
                thread_id=f"cert-eval-{i}"
            )

            # Get verdict
            verdict = evaluator.get_certification_verdict(
                result["overall_score"],
                {
                    "context_precision": result["scores"]["retrieval"]["context_precision"],
                    "faithfulness": result["scores"]["generation"]["faithfulness"],
                    "answer_relevancy": result["scores"]["generation"]["answer_relevancy"],
                    "answer_correctness": result["scores"]["generation"]["answer_correctness"]
                }
            )

            # Add metadata
            result["verdict"] = verdict
            result["metadata"].update({
                "id": item["id"],
                "category": item.get("category"),
                "difficulty": item.get("difficulty"),
                "source_file": item.get("source_file")
            })

            results.append(result)

            # Show quick summary
            status = "✅ PASS" if verdict["passed"] else "❌ FAIL"
            print(f"   {status} - Overall: {verdict['overall_score']:.3f} ({verdict['grade']})")
            print(f"   Scores: CP={result['scores']['retrieval']['context_precision']:.2f} | "
                  f"F={result['scores']['generation']['faithfulness']:.2f} | "
                  f"AR={result['scores']['generation']['answer_relevancy']:.2f} | "
                  f"AC={result['scores']['generation']['answer_correctness']:.2f}")

            if verdict["passed"]:
                passed_count += 1

        except Exception as e:
            logger.error(f"Error evaluating question {i}: {e}")
            print(f"   ❌ ERROR: {e}")
            continue

    # 5. Compute aggregate statistics
    print()
    print("=" * 80)
    print("📈 CERTIFICATION RESULTS")
    print("=" * 80)
    print()

    if not results:
        print("❌ No results to report")
        sys.exit(1)

    # Average scores
    avg_scores = {
        "context_precision": sum(r["scores"]["retrieval"]["context_precision"] for r in results) / len(results),
        "faithfulness": sum(r["scores"]["generation"]["faithfulness"] for r in results) / len(results),
        "answer_relevancy": sum(r["scores"]["generation"]["answer_relevancy"] for r in results) / len(results),
        "answer_correctness": sum(r["scores"]["generation"]["answer_correctness"] for r in results) / len(results),
    }

    overall_avg = sum(r["overall_score"] for r in results) / len(results)

    print(f"📊 Average Scores (n={len(results)}):")
    print(f"   Context Precision:  {avg_scores['context_precision']:.3f}")
    print(f"   Faithfulness:       {avg_scores['faithfulness']:.3f}")
    print(f"   Answer Relevancy:   {avg_scores['answer_relevancy']:.3f}")
    print(f"   Answer Correctness: {avg_scores['answer_correctness']:.3f} ⭐")
    print()
    print(f"   Overall Score:      {overall_avg:.3f}")
    print()
    print(f"🎯 Pass Rate: {passed_count}/{len(results)} ({100*passed_count/len(results):.1f}%)")
    print()

    # 6. Determine final certification
    certification_passed = passed_count >= len(results) * 0.8  # 80% pass rate

    # Grade
    if overall_avg >= 0.9:
        final_grade = "A+ (Excellent)"
    elif overall_avg >= 0.8:
        final_grade = "A (Very Good)"
    elif overall_avg >= 0.7:
        final_grade = "B (Good)"
    elif overall_avg >= 0.6:
        final_grade = "C (Acceptable)"
    else:
        final_grade = "F (Failed)"

    print("=" * 80)
    if certification_passed:
        print("✅ CERTIFICATION: PASSED")
        print(f"   Grade: {final_grade}")
        print(f"   Overall Score: {overall_avg:.3f}")
    else:
        print("❌ CERTIFICATION: FAILED")
        print(f"   Pass rate too low: {100*passed_count/len(results):.1f}% < 80%")
        print(f"   Grade: {final_grade}")
    print("=" * 80)
    print()

    # 7. Save detailed report
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"certification_report_{timestamp}.json"

    output_path = Path(__file__).parent / output_file

    report = {
        "timestamp": datetime.now().isoformat(),
        "certification_passed": certification_passed,
        "summary": {
            "total_questions": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
            "pass_rate": passed_count / len(results),
            "average_scores": avg_scores,
            "overall_score": overall_avg,
            "grade": final_grade
        },
        "detailed_results": results,
        "dataset_metadata": metadata
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"💾 Detailed report saved: {output_path}")
    print()

    # 8. Show top/bottom performers
    if len(results) >= 3:
        print("=" * 80)
        print("🏆 TOP 3 QUESTIONS (Best Performance)")
        print("=" * 80)
        sorted_results = sorted(results, key=lambda x: x["overall_score"], reverse=True)
        for i, r in enumerate(sorted_results[:3], 1):
            print(f"\n{i}. Score: {r['overall_score']:.3f} - {r['metadata'].get('category', 'N/A')}")
            print(f"   Q: {r['question'][:70]}...")

        print()
        print("=" * 80)
        print("⚠️  BOTTOM 3 QUESTIONS (Needs Improvement)")
        print("=" * 80)
        for i, r in enumerate(sorted_results[-3:], 1):
            print(f"\n{i}. Score: {r['overall_score']:.3f} - {r['metadata'].get('category', 'N/A')}")
            print(f"   Q: {r['question'][:70]}...")
            # Show which metrics failed
            failed_metrics = [k for k, v in r["verdict"]["requirements"].items() if not v["passed"]]
            if failed_metrics:
                print(f"   Failed: {', '.join(failed_metrics)}")

    print()
    print("=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)
    print()

    # Return exit code for CI/CD
    return 0 if certification_passed else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run certification evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="certification_dataset.json",
        help="Dataset file (default: certification_dataset.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (default: auto-generated)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to N questions (for testing)"
    )

    args = parser.parse_args()

    try:
        exit_code = run_certification(
            dataset_file=args.dataset,
            output_file=args.output,
            limit=args.limit
        )
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
