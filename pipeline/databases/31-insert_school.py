#!/usr/bin/env python3
"""module for inserting a new document in a collection based on kwargs"""


def insert_school(mongo_collection, **kwargs):
    """ insert a new document in a collection based on kwargs
    args:
        mongo_collection: mongo collection
        kwargs: kwargs for inserting a new document

    Returns:
        mongo_collection_id"""

    results = mongo_collection.insert_one(kwargs)
    return results.inserted_id
