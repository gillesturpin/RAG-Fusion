#!/usr/bin/env python3
"""
RAG Agent - EXACTLY as documented in official LangChain docs
https://python.langchain.com/docs/tutorials/rag_agent/

WITH MEMORY SUPPORT - Using InMemorySaver for thread persistence
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from typing import Any, Literal


# Simple function to trim messages for memory management
def trim_messages(messages):
    """Keep only the last 10 messages to fit context window."""
    if len(messages) <= 10:
        return messages
    # Keep first (system) and last 9 messages
    return [messages[0]] + messages[-9:] if messages else []


class RAGAgent:
    """RAG Agent - Pure implementation from official docs"""

    def __init__(self, vectorstore, checkpointer=None):
        """Initialize with existing vectorstore and optional checkpointer"""
        self.vectorstore = vectorstore

        # Use provided checkpointer or create InMemorySaver
        self.checkpointer = checkpointer or InMemorySaver()

        # Initialize model
        self.model = init_chat_model(
            "claude-sonnet-4-5-20250929",
            model_provider="anthropic"
        )

        # Create retrieve tool EXACTLY as in docs
        @tool
        def retrieve(query: str):
            """Retrieve information related to a query."""
            retrieved_docs = self.vectorstore.similarity_search(query, k=4)
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

        # Compile with checkpointer for memory support
        return workflow.compile(checkpointer=self.checkpointer)

    def _call_model(self, state: MessagesState):
        """Call the model with tools and message trimming"""
        messages = state["messages"]

        # Trim messages to fit context window
        messages = trim_messages(messages)

        # Add system message if not present
        if not messages or messages[0].type != "system":
            system_msg = SystemMessage(
                "You have access to a tool that retrieves context from documents. "
                "Use the tool to help answer user queries. "
                "IMPORTANT: Provide COMPLETE and COMPREHENSIVE answers with ALL details from the retrieved context. "
                "If the context mentions multiple items (e.g., phases, steps, stages), include ALL of them with their full descriptions. "
                "Do not omit any information. Use proper Markdown formatting for readability."
            )
            messages = [system_msg] + messages

        # Call the model with tools
        response = self.model_with_tools.invoke(messages)
        return {"messages": [response]}

    def invoke(self, question: str, thread_id: str = None) -> dict:
        """
        Invoke the agent and return the result
        Returns ONLY what's documented - no business metrics

        Args:
            question: The user question
            thread_id: Optional thread ID for conversation memory
        """
        # Build config with thread_id if provided
        config = {}
        if thread_id:
            config = {"configurable": {"thread_id": thread_id}}

        # Call graph with memory support
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
    print("RAG Agent with Memory Support (InMemorySaver)")
    print("=" * 60)

    # Test WITHOUT memory (different thread_ids)
    print("\n--- Test WITHOUT memory (different threads) ---")
    q1 = "My name is Alice. What is Task Decomposition?"
    r1 = agent.invoke(q1, thread_id="thread-1")
    print(f"Q: {q1}")
    print(f"A: {r1['answer'][:100]}...")

    q2 = "What is my name?"
    r2 = agent.invoke(q2, thread_id="thread-2")  # Different thread
    print(f"\nQ: {q2}")
    print(f"A: {r2['answer']}")
    print("(Different thread - no memory of previous conversation)")

    # Test WITH memory (same thread_id)
    print("\n--- Test WITH memory (same thread) ---")
    q3 = "My name is Bob. What is Task Decomposition?"
    r3 = agent.invoke(q3, thread_id="thread-demo")
    print(f"Q: {q3}")
    print(f"A: {r3['answer'][:100]}...")

    q4 = "What is my name?"
    r4 = agent.invoke(q4, thread_id="thread-demo")  # Same thread
    print(f"\nQ: {q4}")
    print(f"A: {r4['answer']}")
    print("(Same thread - remembers the name Bob)")

    print("=" * 60)