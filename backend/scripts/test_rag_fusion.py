#!/usr/bin/env python3
"""
Test script for RAG Fusion simple chain
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from rags.rag_fusion import RAGFusion
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
persist_dir = str(Path(__file__).parent.parent.parent / "data" / "chroma_db")
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

rag = RAGFusion(vectorstore)

print('=' * 80)
print('TEST - RAG Fusion Simple Chain')
print('=' * 80)
print()

# Test 1: Git question
q1 = 'What are three key learning objectives for this Git and Github collaboration course?'
print(f'Q1: {q1}')
print()

result1 = rag.invoke(q1)
print(f'Answer: {result1["answer"][:300]}...')
print()
print(f'Documents retrieved: {result1["num_documents"]}')
print('=' * 80)
print()

# Test 2: Python function question
q2 = 'Describe the key components of defining a function in Python'
print(f'Q2: {q2}')
print()

result2 = rag.invoke(q2)
print(f'Answer: {result2["answer"][:300]}...')
print()
print(f'Documents retrieved: {result2["num_documents"]}')
print('=' * 80)
