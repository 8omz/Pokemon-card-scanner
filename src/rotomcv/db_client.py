import os
import pymongo
from pymongo import MongoClient
from datetime import datetime

class RotomDB:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RotomDB, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Use environment variable for MongoDB URI
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        self.client = MongoClient(mongo_uri)
        self.db = self.client["rotomcv"]
        self.cards = self.db["cards"]
        self.scans = self.db["scans"]
        
        # Ensure indexes
        # prevent duplicates
        self.cards.create_index([("set_id", 1), ("number", 1)], unique=True)
        # fast lookup by clean name
        self.cards.create_index("name_clean")
        
        print("Connected to MongoDB: rotomcv")

    def get_card(self, set_id, number):
        """Retrieve a specific card by set and number."""
        return self.cards.find_one({"set_id": set_id, "number": number})

    def search_cards_by_name(self, name_clean):
        """
        Find cards by clean name.
        """
        return list(self.cards.find({"name_clean": name_clean}))

    def log_scan(self, result):
        """
        Log a scan result to the 'scans' collection.
        """
        # Add timestamp if not present
        if "timestamp" not in result:
            result["timestamp"] = datetime.now()
            
        return self.scans.insert_one(result)
    
    def count_cards(self):
        return self.cards.count_documents({})
