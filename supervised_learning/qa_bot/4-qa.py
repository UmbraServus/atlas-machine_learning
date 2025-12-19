#!/usr/bin/env python3
"""
Question answering from corpus using semantic search + QA extraction
Integrates with existing semantic_search and question_answer functions
"""

# Import your existing functions
semantic_search = __import__('3-semantic_search').semantic_search
qa = __import__('0-qa').question_answer


def question_answer(corpus_path):
    """
    Answers questions from multiple reference texts.
    
    This function creates an interactive loop where users can ask questions.
    It uses semantic search to find the most relevant document from the corpus,
    then extracts the answer from that document using BERT QA.
    
    Args:
        corpus_path: Path to the corpus of reference documents
    
    The function runs in an interactive loop until the user types exit/quit.
    """
    print("Welcome! Ask me questions about the corpus.")
    print("Type 'exit', 'quit', 'goodbye', or 'bye' to stop.\n")
    
    while True:
        # Get user question
        question = input("Q: ").strip()
        
        # Check for exit commands
        if question.lower() in {"exit", "quit", "goodbye", "bye"}:
            print("A: Goodbye")
            break
        
        # Skip empty questions
        if not question:
            continue
        
        # Step 1: Use semantic search to find most relevant document
        reference = semantic_search(corpus_path, question)
        
        if reference is None:
            print("A: Sorry, I couldn't find any relevant documents.\n")
            continue
        
        # Step 2: Extract answer from the found document using QA
        answer = qa(question, reference)
        
        # Display answer
        if answer:
            print(f"A: {answer}\n")
        else:
            print("A: Sorry, I couldn't find an answer to that question.\n")
