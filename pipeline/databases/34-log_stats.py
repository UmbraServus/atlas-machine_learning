#!/usr/bin/env python3
"""
Script to provide some stats about Nginx logs stored in MongoDB.
"""
from pymongo import MongoClient


def main():
    """ main function"""
    # connect to mongoDB
    mongo_client = MongoClient("mongodb://localhost:27017/")
    mongo_db = mongo_client.logs
    collection = mongo_db.nginx

    # Total n of logs
    total = collection.count_documents({})
    print(f"{total} logs")

    # method stats
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print(f"Methods:")
    for method in methods:
        count = collection.count_documents({"method": method})
        print(f"\t{method}: {count}")

    # n of GET /status requests
    satus_count = collection.count_documents({"method": "GET", "path": "/status"})
    print(f"{status_count} status check")

    if __name__ == "__main__":
        main()
