#!/usr/bin/env python3
""" qa method that finds snippets in text that answer questions """
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a ref document to answer a question.

    Args:
        question: string containing the question to answer
        reference: string containing the ref document from which to find answer

    Returns:
        A string containing the answer, or None if no answer is found
    """
    # Load the model from TensorFlow Hub
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )

    # Tokenize the input
    tokens = tokenizer.encode_plus(
        question,
        reference,
        add_special_tokens=True,
        return_tensors='tf',
        max_length=512,
        truncation=True
    )

    input_ids = tokens['input_ids']
    token_type_ids = tokens['token_type_ids']
    attention_mask = tokens['attention_mask']

    # Get model predictions
    outputs = model([input_ids, attention_mask, token_type_ids])

    # The model returns a list: [start_logits, end_logits]
    start_logits = outputs[0][0]
    end_logits = outputs[1][0]

    # Get the most likely start and end positions
    start_idx = tf.argmax(start_logits).numpy()
    end_idx = tf.argmax(end_logits).numpy()

    # Find where the reference text starts (after the [SEP] token)
    # token_type_ids: 0 for question tokens, 1 for reference tokens
    sep_index = None
    for i, token_type in enumerate(token_type_ids[0]):
        if token_type == 1:  # Found first reference token
            sep_index = i
            break
    
    # Make sure answer is from reference, not question
    if sep_index is not None and start_idx < sep_index:
        # Answer is in the question part, which is wrong
        # Try to find answer in reference part
        ref_start_logits = start_logits[sep_index:]
        ref_end_logits = end_logits[sep_index:]
        
        start_idx = tf.argmax(ref_start_logits).numpy() + sep_index
        end_idx = tf.argmax(ref_end_logits).numpy() + sep_index

    # Basic validation - only check if start comes before end
    if start_idx > end_idx:
        return None
    
    # If start and end are the same, extend end by 1
    if start_idx == end_idx:
        end_idx = start_idx + 1

    # Extract the answer tokens
    answer_tokens = input_ids[0][start_idx:end_idx + 1]

    # Convert tokens back to string
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    # Clean up the answer
    answer = answer.strip()
    
    # Fix spacing around punctuation (BERT tokenization artifact)
    import re
    answer = re.sub(r'\s+([.,!?;:])', r'\1', answer)  # Remove space before punctuation
    answer = re.sub(r':\s+(\d)', r':\1', answer)  # Fix time format (9: 00 -> 9:00)
    answer = re.sub(r'\s+-\s+', r'-', answer)  # Fix hyphens (on - site -> on-site)
    answer = re.sub(r'\s+', ' ', answer)  # Multiple spaces to single space
    
    # Only return None if the answer is completely empty
    if not answer:
        return None
    
    # If answer is just special tokens or single character, return None
    if len(answer) <= 1:
        return None

    return answer
