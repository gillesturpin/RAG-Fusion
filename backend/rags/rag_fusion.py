#!/usr/bin/env python3
"""
RAG Fusion - Simple Chain Implementation
Based on Learning LangChain Ch3: d-rag-fusion.py

Simplified version without LangGraph/tools for better performance:
- 2 LLM calls instead of 3 (-33% API calls)
- Direct chain flow (no tool calling overhead)
- Same RAG Fusion quality (multi-query + RRF reranking)
"""

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from typing import List


class RAGFusion:
    """
    RAG Fusion with simple chain architecture (no tools, no LangGraph)

    Flow:
    1. User question → Generate query variations (LLM call 1)
    2. Multi-query retrieval + RRF reranking
    3. Generate answer with context (LLM call 2)

    Total: 2 LLM calls (vs 3 with tool-based approach)
    """

    def __init__(
        self,
        vectorstore,
        use_rag_fusion=True,
        temperature=1.0,
        k_documents=8
    ):
        """
        Initialize RAG Fusion chain

        Args:
            vectorstore: ChromaDB vectorstore
            use_rag_fusion: Enable RAG Fusion (multi-query + RRF) [default: True]
            temperature: Model temperature for generation [default: 1.0]
            k_documents: Number of documents to retrieve [default: 8]
        """
        self.vectorstore = vectorstore
        self.use_rag_fusion = use_rag_fusion
        self.temperature = temperature
        self.k_documents = k_documents

        # Initialize model
        self.model = init_chat_model(
            "claude-sonnet-4-5-20250929",
            model_provider="anthropic",
            temperature=temperature
        )

    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate 4 variations of the query for RAG Fusion"""
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant that generates multiple search queries based on a single input query.
            Generate multiple search queries related to: {question}
            Output (4 queries):"""
        )

        llm = init_chat_model(
            "claude-sonnet-4-5-20250929",
            model_provider="anthropic"
        )

        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"question": query})

        # Parse queries (split by newlines)
        variations = [line.strip() for line in result.split('\n') if line.strip()]

        # Keep original query + top 3 variations (total 4)
        return [query] + variations[:3]

    def _reciprocal_rank_fusion(self, results: List[List], k=60) -> List:
        """Reciprocal rank fusion on multiple lists of ranked documents"""
        fused_scores = {}
        documents = {}

        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = doc.page_content
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                    documents[doc_str] = doc
                # RRF formula
                fused_scores[doc_str] += 1 / (rank + k)

        # Sort by fused scores (descending)
        reranked_doc_strs = sorted(
            fused_scores, key=lambda d: fused_scores[d], reverse=True
        )

        return [documents[doc_str] for doc_str in reranked_doc_strs]

    def _retrieve_documents(self, query: str) -> List:
        """
        Retrieve documents using RAG Fusion or simple retrieval

        Returns list of documents
        """
        if self.use_rag_fusion:
            # RAG Fusion: Multi-query + RRF
            # Generates 4 total queries (1 original + 3 rewrites)
            query_variations = self._generate_query_variations(query)
            all_results = []
            for variation in query_variations:
                # Retrieve k=4 docs per query variation (4 queries × 4 docs = 16 total docs)
                docs = self.vectorstore.similarity_search(variation, k=4)
                all_results.append(docs)
            # Apply RRF reranking to all 16 documents
            fused_docs = self._reciprocal_rank_fusion(all_results, k=60)
            # Take top k_documents (default=8) from reranked results
            retrieved_docs = fused_docs[:self.k_documents]
        else:
            # Simple retrieval
            retrieved_docs = self.vectorstore.similarity_search(query, k=self.k_documents)

        return retrieved_docs

    def invoke(self, question: str) -> dict:
        """
        Invoke RAG Fusion chain

        Args:
            question: User question

        Returns:
            dict with answer, metadata, and contexts (for RAGAS evaluation)
        """
        # 1. Retrieve documents (includes LLM call for query generation if RAG Fusion)
        documents = self._retrieve_documents(question)

        # 2. Build context from documents
        context = "\n\n".join([
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in documents
        ])

        # Extract contexts for RAGAS evaluation (list of document contents)
        contexts = [doc.page_content for doc in documents]

        # 3. Generate answer with context (LLM call 2)
        system_msg = SystemMessage(
            "You are a teaching assistant for a SPECIFIC course. Your knowledge base contains the course materials.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Base your answer EXCLUSIVELY on the provided context - DO NOT use external knowledge\n\n"
            "QUESTION ANALYSIS (MANDATORY STEPS):\n"
            "Step 1: Break down the question into ALL its components\n"
            "  - If the question asks for 'three things', your answer MUST list exactly three things\n"
            "  - If it asks 'describe X and explain Y', you MUST address both X AND Y\n"
            "Step 2: For EACH component, verify the context provides the answer\n"
            "Step 3: Only proceed if ALL components are answerable from context\n\n"
            "COMPLETENESS CHECK:\n"
            "BEFORE answering, ask yourself: 'Does the context contain ALL information for EVERY part of the question?'\n"
            "- If YES (context covers ALL parts): Provide a complete answer addressing EVERY component\n"
            "- If NO (ANY part is missing): Respond ONLY with 'The course materials don't cover this topic'\n"
            "- NEVER give partial answers - it's ALL or NOTHING\n\n"
            "ANSWER REQUIREMENTS:\n"
            "1. Use EXACT quotes and terminology from the context (prefer direct citations over paraphrasing)\n"
            "2. Include ALL specific examples/details mentioned in the context\n"
            "3. Match the structure requested in the question (e.g., if it asks for a list, provide a list)\n"
            "4. Ensure your answer directly addresses EVERY part of the question\n"
            "5. Be comprehensive but concise (3-6 sentences depending on question complexity)"
        )

        # Build prompt with context
        messages = [
            system_msg,
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]

        # Call LLM
        response = self.model.invoke(messages)

        return {
            "answer": response.content,
            "used_retrieval": True,  # Always true in this simple implementation
            "num_documents": len(documents),
            "contexts": contexts  # For RAGAS evaluation
        }


# Test if running standalone
if __name__ == "__main__":
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from dotenv import load_dotenv

    load_dotenv()

    # Setup embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="../../data/chroma_db",
        embedding_function=embeddings
    )

    # Create RAG Fusion chain
    rag = RAGFusion(vectorstore)

    # Test questions
    print("=" * 60)
    print("RAG Fusion - Simple Chain")
    print("=" * 60)

    print("\n--- Test Question 1 ---")
    q1 = "What is a class in Python?"
    r1 = rag.invoke(q1)
    print(f"Q: {q1}")
    print(f"A: {r1['answer'][:200]}...")
    print(f"Used retrieval: {r1['used_retrieval']}")
    print(f"Num documents: {r1['num_documents']}")

    print("=" * 60)
