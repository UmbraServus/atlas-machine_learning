#!/usr/bin/env python3
"""method for listing all documents in a collection in mongodb"""


def list_all(mongo_collection):
    """ method for listing all documents in a collection in mongodb

args:
    mongo_collection: mongodb collection
returns:
        empty list if none or the document is in the collection"""
    if mongo_collection is None:
        return []
    else:
        return list(mongo_collection.find())
