#!/usr/bin/env python3
"""method for listing all documents in a collection in mongodb"""


def schools_by_topic(mongo_collection, topic):
    """ method for listing all documents in a collection in mongodb
args:
    mongo_collection: the pymongo collection object
    topic (string): topic searched
returns:
    list: documents in a collection in mongodb where topic is specified"""
    if mongo_collection is None or topic is None:
        return []
    else:
        return list(mongo_collection.find({"topics": topic}))
