import pymongo

client = pymongo.MongoClient("mongodb+srv://review_radar:abcd01@cluster0.yhvnzmm.mongodb.net/")

db = client["fake_review_detector"]
c1 = db["reviews"]

