#!/usr/bin/env python3
"""
Script de génération automatique du dataset de certification
Utilise Claude pour créer des questions/réponses basées sur les documents uploadés
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Setup path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

def generate_certification_dataset(num_questions=10, output_file="certification_dataset.json"):
    """
    Génère automatiquement un dataset de certification

    Args:
        num_questions: Nombre de questions à générer
        output_file: Nom du fichier de sortie
    """

    print("=" * 70)
    print("🎓 GÉNÉRATION DU DATASET DE CERTIFICATION")
    print("=" * 70)
    print()

    # 1. Initialize embeddings and vectorstore
    print("📚 Chargement du vectorstore...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    persist_dir = str(Path(__file__).parent.parent.parent / "data" / "chroma_db")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # 2. Get all documents
    all_docs = vectorstore.get()
    total_chunks = len(all_docs['documents'])

    if total_chunks == 0:
        print("❌ Aucun document dans le vectorstore!")
        print("   Uploadez d'abord des documents via l'interface.")
        sys.exit(1)

    print(f"✅ {total_chunks} chunks trouvés")

    # 3. Get unique sources
    sources = list(set(all_docs['metadatas'][i].get('source', 'unknown')
                      for i in range(len(all_docs['metadatas']))))
    print(f"✅ {len(sources)} documents sources: {', '.join(sources[:3])}...")
    print()

    # 4. Initialize LLM for generation
    print("🤖 Initialisation de Claude pour génération...")
    llm = ChatAnthropic(
        model="claude-3-5-haiku-20241022",  # Haiku pour rapidité
        temperature=0.3  # Un peu de créativité mais pas trop
    )
    print("✅ Claude prêt")
    print()

    # 5. Select diverse chunks
    print(f"📊 Sélection de {num_questions} chunks diversifiés...")
    step = max(1, total_chunks // num_questions)
    selected_indices = [i * step for i in range(num_questions)]
    selected_chunks = [all_docs['documents'][i] for i in selected_indices]
    selected_metadata = [all_docs['metadatas'][i] for i in selected_indices]

    print(f"✅ {len(selected_chunks)} chunks sélectionnés")
    print()

    # 6. Generate Q&A pairs
    print("🎯 Génération des questions/réponses...")
    print("-" * 70)

    dataset = []

    for i, (chunk, metadata) in enumerate(zip(selected_chunks, selected_metadata), 1):
        print(f"\n[{i}/{num_questions}] Source: {metadata.get('source', 'unknown')[:40]}...")

        # Truncate chunk if too long
        chunk_preview = chunk[:1500] if len(chunk) > 1500 else chunk

        prompt = f"""Based on this educational content, generate a certification-quality question and answer.

CONTENT:
{chunk_preview}

Generate a JSON object with these fields:
- "question": A clear, specific question that tests understanding of the content
- "ground_truth": A complete, accurate answer based ONLY on the content (2-4 sentences)
- "difficulty": One of "easy", "medium", or "hard"
- "category": The main topic (e.g., "Python Classes", "Git Basics", "Agile", etc.)

Requirements:
- Question must be answerable from the content
- Ground truth must be factually accurate and complete
- Use technical terms when appropriate
- Avoid yes/no questions

Output ONLY valid JSON, no markdown formatting."""

        try:
            response = llm.invoke(prompt)

            # Parse JSON from response
            content = response.content.strip()

            # Remove markdown code blocks if present
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            qa_data = json.loads(content)

            # Add metadata
            qa_data["id"] = i
            qa_data["source_file"] = metadata.get('source', 'unknown')
            qa_data["generated_at"] = datetime.now().isoformat()

            dataset.append(qa_data)

            # Preview
            print(f"   ✅ Q: {qa_data['question'][:60]}...")
            print(f"      Category: {qa_data.get('category', 'N/A')}, Difficulty: {qa_data.get('difficulty', 'N/A')}")

        except json.JSONDecodeError as e:
            print(f"   ⚠️  JSON parse error, retrying...")
            # Retry once
            try:
                response = llm.invoke(prompt + "\n\nIMPORTANT: Output ONLY the JSON object, no other text!")
                content = response.content.strip()
                if content.startswith('```'):
                    content = '\n'.join(content.split('\n')[1:-1])
                qa_data = json.loads(content)
                qa_data["id"] = i
                qa_data["source_file"] = metadata.get('source', 'unknown')
                qa_data["generated_at"] = datetime.now().isoformat()
                dataset.append(qa_data)
                print(f"   ✅ Q: {qa_data['question'][:60]}...")
            except Exception as e2:
                print(f"   ❌ Failed: {e2}")
                continue

        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

    print()
    print("-" * 70)
    print(f"✅ Généré {len(dataset)} questions/réponses")
    print()

    # 7. Add summary statistics
    categories = {}
    difficulties = {}
    for item in dataset:
        cat = item.get('category', 'Unknown')
        diff = item.get('difficulty', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1

    print("📊 Statistiques du dataset:")
    print(f"   Total questions: {len(dataset)}")
    print(f"   Catégories: {dict(categories)}")
    print(f"   Difficultés: {dict(difficulties)}")
    print()

    # 8. Save to file
    output_path = Path(__file__).parent / output_file

    # Add metadata to dataset
    final_output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "num_questions": len(dataset),
            "num_documents": len(sources),
            "total_chunks": total_chunks,
            "categories": categories,
            "difficulties": difficulties
        },
        "questions": dataset
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"💾 Dataset sauvegardé: {output_path}")
    print()

    # 9. Preview
    print("=" * 70)
    print("🔍 PREVIEW DES PREMIÈRES QUESTIONS")
    print("=" * 70)
    print()

    for i, item in enumerate(dataset[:3], 1):
        print(f"{i}. QUESTION:")
        print(f"   {item['question']}")
        print(f"\n   GROUND TRUTH:")
        print(f"   {item['ground_truth']}")
        print(f"\n   Metadata: {item.get('category')} | {item.get('difficulty')} | {item.get('source_file')}")
        print()

    if len(dataset) > 3:
        print(f"   ... et {len(dataset) - 3} autres questions")

    print()
    print("=" * 70)
    print("✅ GÉNÉRATION TERMINÉE")
    print("=" * 70)
    print()
    print(f"📄 Fichier: {output_path}")
    print(f"📊 Questions: {len(dataset)}")
    print()
    print("🎯 Next steps:")
    print("   1. Vérifiez le fichier JSON généré")
    print("   2. Ajustez manuellement si nécessaire")
    print("   3. Lancez l'évaluation avec: python run_certification.py")
    print()

    return dataset, output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Générer dataset de certification")
    parser.add_argument(
        "--num-questions",
        type=int,
        default=10,
        help="Nombre de questions à générer (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="certification_dataset.json",
        help="Nom du fichier de sortie (default: certification_dataset.json)"
    )

    args = parser.parse_args()

    try:
        generate_certification_dataset(
            num_questions=args.num_questions,
            output_file=args.output
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Génération interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
