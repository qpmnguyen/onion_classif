# This script contains the top 1000 posts for a subreddit of all time. 

import praw
import os 
import pandas as pd  
import numpy as np

reddit = praw.Reddit(client_id = os.environ['C_ID'], client_secret = os.environ['C_SECRET'], 
                        user_agent = "onion_scrape")

onion = reddit.subreddit("TheOnion")
nottheonion = reddit.subreddit("nottheonion")

with open("data/onion.txt", "w") as f:
    for submission in onion.top(limit = 1000):
        f.write(submission.title + "\t" + str(1) + "\n")

with open("data/nottheonion.txt", "w") as f:
    for submission in nottheonion.top(limit = 1000):
        f.write(submission.title + "\t" + str(0) + "\n")
        
# combine into one large csv file 
with open("data/combined.tsv", "w") as f:
    f.write("String \t Label \n")
    o_file = open("data/onion.txt")
    f.write(o_file.read())
    no_file = open("data/nottheonion.txt")
    f.write(no_file.read())

