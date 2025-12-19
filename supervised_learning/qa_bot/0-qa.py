#!/usr/bin/env python3
""" qa method that finds snippets in text that answer questions """
import tensorflow as tf
import transformers
import tensorflow_hub as hub


def question_answer(question, reference): 
    """ finds a snippet of text within a ref document to answer a question

    question: string containing the question to answer
    reference: string containing the ref document from which to find the answr

    Returns: a string containing the answer
        If no answer is found, return None
    """
    # load model from tensorflow hub
    model_url = "https://tfhub.dev/see--/bert-uncased-tf2-qa/1"
    model = hub.load(model_url)

    # load tokenizer from transformers
    tokenizer = transformers.BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )

    # encode question and reference
    inputs = tokenizer.encode_plus(
        question,
        reference,
        add_special_tokens=True,
        return_tensors="tf"
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    # get start and end scores from model
    start_scores, end_scores, _ = model(
        [input_ids, attention_mask, token_type_ids]
    )

    start_index = tf.argmax(start_scores, axis=1).numpy()[0]
    end_index = tf.argmax(end_scores, axis=1).numpy()[0] + 1

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer_tokens = all_tokens[start_index:end_index]
    answer = tokenizer.convert_tokens_to_string(answer_tokens).strip()

    return answer if answer else None
