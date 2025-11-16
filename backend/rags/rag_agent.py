#!/usr/bin/env python3
"""
RAG Agent with RAG Fusion
Based on official LangChain docs: https://python.langchain.com/docs/tutorials/rag_agent/

ENHANCEMENTS:
- RAG Fusion: Multi-query retrieval + RRF reranking for improved retrieval
- Stateless mode: No InMemorySaver (each question independent for evaluation)
- Optimized configuration: k=8 documents, temperature=1.0

Inspired by:
- learning-langchain-master/ch3/py/d-rag-fusion.py (RAG Fusion)
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from typing import Literal, List


# Simple function to trim messages for memory management
def trim_messages(messages):
    """
    Keep only the last 10 messages to fit context window.
    IMPORTANT: Preserves tool_use/tool_result pairs to avoid API errors.
    """
    if len(messages) <= 10:
        return messages

    # Separate system message from the rest
    has_system = messages and messages[0].type == "system"
    system_msg = [messages[0]] if has_system else []
    non_system = messages[1:] if has_system else messages

    # How many non-system messages can we keep? (10 total - system message)
    max_keep = 10 - len(system_msg)  # Usually 9 if system exists, else 10

    # Get last max_keep non-system messages
    kept = non_system[-max_keep:]

    # Check if first kept message is an orphaned tool_result
    if kept and hasattr(kept[0], 'type') and kept[0].type == "tool":
        # Find its index in non_system messages
        tool_result_idx = non_system.index(kept[0])
        if tool_result_idx > 0:
            prev_msg = non_system[tool_result_idx - 1]
            # If previous message has tool_calls, we need to include it
            if hasattr(prev_msg, 'tool_calls') and prev_msg.tool_calls:
                # Keep from prev_msg onwards, but still only max_keep messages total
                # This sacrifices the oldest message to preserve the tool pair
                kept = non_system[tool_result_idx - 1:tool_result_idx - 1 + max_keep]

    # Check if last kept message has orphaned tool_use
    if kept and hasattr(kept[-1], 'tool_calls') and kept[-1].tool_calls:
        # Last message has tool_calls but its tool_result would be excluded
        # We must exclude this message to avoid orphaned tool_use
        kept = kept[:-1]

    return system_msg + kept


class RAGAgent:
    """RAG Agent - Pure implementation from official docs"""

    def __init__(self, vectorstore, checkpointer=None, use_rag_fusion=True, temperature=1.0, k_documents=8):
        """Initialize with existing vectorstore and optional checkpointer

        Args:
            vectorstore: ChromaDB vectorstore
            checkpointer: Optional checkpointer for state (None = stateless)
            use_rag_fusion: Enable RAG Fusion (multi-query + RRF) [default: True]
            temperature: Model temperature for generation [default: 1.0]
            k_documents: Number of documents to retrieve [default: 8]
        """
        self.vectorstore = vectorstore
        self.checkpointer = checkpointer

        self.use_rag_fusion = use_rag_fusion
        self.temperature = temperature
        self.k_documents = k_documents

        # Initialize model
        self.model = init_chat_model(
            "claude-sonnet-4-5-20250929",
            model_provider="anthropic",
            temperature=temperature
        )

        # Create retrieve tool with RAG Fusion
        @tool
        def retrieve(query: str):
            """Retrieve information related to a query using RAG Fusion."""

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

            # Return documents
            serialized = "\n\n".join(
                f"Source: {doc.metadata}\nContent: {doc.page_content}"
                for doc in retrieved_docs
            )
            return serialized

        # Create tools list
        self.tools = [retrieve]

        # Bind tools to model
        self.model_with_tools = self.model.bind_tools(self.tools)

        # Build the graph
        self.graph = self._build_graph()

    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate 4 variations of the query for RAG Fusion"""
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant that generates multiple search queries based on a single input query.
            Generate multiple search queries related to: {question}
            Output (4 queries):"""
        )

        # Use a simple model for query generation
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

    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(MessagesState)

        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self.tools))

        # Define the entry point
        workflow.add_edge(START, "agent")

        # Define conditional edges from agent
        def should_continue(state: MessagesState) -> Literal["tools", END]:
            messages = state["messages"]
            last_message = messages[-1]
            # If the last message has tool calls, we should execute tools
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            # Otherwise we're done
            return END

        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                END: END,
            }
        )

        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")

        # Compile with checkpointer if provided, otherwise stateless
        if self.checkpointer:
            return workflow.compile(checkpointer=self.checkpointer)
        else:
            return workflow.compile()

    def _call_model(self, state: MessagesState):
        """Call the model with tools and message trimming"""
        messages = state["messages"]

        # Trim messages to fit context window
        messages = trim_messages(messages)

        # Add system message if not present
        if not messages or messages[0].type != "system":
            system_msg = SystemMessage(
                "You are a teaching assistant for a SPECIFIC course. Your knowledge base contains the course materials.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. ALWAYS use the retrieval tool FIRST to check the course materials\n"
                "2. Base your answer EXCLUSIVELY on the retrieved context - DO NOT use external knowledge\n\n"
                "QUESTION ANALYSIS (MANDATORY STEPS):\n"
                "Step 1: Break down the question into ALL its components\n"
                "  - If the question asks for 'three things', your answer MUST list exactly three things\n"
                "  - If it asks 'describe X and explain Y', you MUST address both X AND Y\n"
                "Step 2: For EACH component, verify the context provides the answer\n"
                "Step 3: Only proceed if ALL components are answerable from context\n\n"
                "COMPLETENESS CHECK:\n"
                "BEFORE answering, ask yourself: 'Does the retrieved context contain ALL information for EVERY part of the question?'\n"
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
            messages = [system_msg] + messages

        # Call model with tools
        response = self.model_with_tools.invoke(messages)
        return {"messages": [response]}

    def invoke(self, question: str, thread_id: str = None) -> dict:
        """
        Invoke the agent and return the result
        Returns ONLY what's documented - no business metrics

        Args:
            question: The user question
            thread_id: Optional thread ID for conversation memory (only used if checkpointer is set)
        """
        # Build config with thread_id if provided (requires checkpointer)
        config = {}
        if thread_id and self.checkpointer:
            config = {"configurable": {"thread_id": thread_id}}

        # Call graph (stateless if no checkpointer)
        from langchain_core.messages import HumanMessage
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config
        )

        # Return only the essential information
        last_message = result["messages"][-1]

        # Check if retrieval was used
        used_retrieval = any(
            msg.type == "tool" for msg in result["messages"]
        )

        return {
            "answer": last_message.content,
            "messages": result["messages"],
            "used_retrieval": used_retrieval,
            "thread_id": thread_id  # Return thread_id for frontend tracking
        }


# Test if running standalone
if __name__ == "__main__":
    from langchain_huggingface import HuggingFaceEmbeddings

    # Setup embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="../../data/chroma_db",
        embedding_function=embeddings
    )

    # Create agent
    agent = RAGAgent(vectorstore)

    # Test questions
    print("=" * 60)
    print("RAG Agent - Stateless Mode")
    print("=" * 60)

    print("\n--- Test Question 1 ---")
    q1 = "What is Task Decomposition?"
    r1 = agent.invoke(q1)
    print(f"Q: {q1}")
    print(f"A: {r1['answer'][:200]}...")
    print(f"Used retrieval: {r1['used_retrieval']}")

    print("\n--- Test Question 2 ---")
    q2 = "What are the key learning objectives of this course?"
    r2 = agent.invoke(q2)
    print(f"Q: {q2}")
    print(f"A: {r2['answer'][:200]}...")
    print(f"Used retrieval: {r2['used_retrieval']}")

    print("=" * 60)