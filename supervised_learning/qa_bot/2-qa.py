#!/usr/bin/env python3
""" simple loop to query qa method """
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    while True:
        question = input("Q: ").strip()

        if question.lower() in {"exit", "quit", "goodbye", "bye"}:
            print("A: Goodbye")
            break

        answer = question_answer(question, reference)
        print(f"A: {answer}" if answer else "Sorry, I do not know the answer.")
