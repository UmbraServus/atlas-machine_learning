#!/usr/bin/env python3
"""module for updating topics attribute in a collection of documents"""


def update_topics(mongo_collection, name, topics):
    """ update topics attribute in a collection of documents
    args:
    mongo_collection: pymongo collection object
    name (string): school name to update
    topics (list of strings): list of topics approached in the school"""

    mongo_collection.update_many(
        {'name': name},
        {'$set': {'topics': topics}}
    )
