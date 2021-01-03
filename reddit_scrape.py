import praw
import os 
import argparse

# arguments 
parser = argparse.ArgumentParser(description='Scraping reddit data')
parser.add_argument('nentries', metavar='N', type=int,
                    help='Number of entries')

args = parser.parse_args()
n_limit = args.nentries

instance = praw.Reddit(client_id = os.environ['C_ID'], client_secret = os.environ['C_SECRET'], 
                        user_agent = "onion_scrape")

onion = instance.subreddit("TheOnion")
nottheonion = instance.subreddit("nottheonion")

with open("data/onion.txt", "w") as f:
    for submission in onion.top(limit = n_limit):
        f.write(submission.title + "\t" + str(1) + "\n")

with open("data/nottheonion.txt", "w") as f:
    for submission in nottheonion.top(limit = n_limit):
        f.write(submission.title + "\t" + str(0) + "\n")
        
# combine into one large csv file 
with open("data/combined.tsv", "w") as f:
    f.write("String \t Label \n")
    o_file = open("data/onion.txt")
    f.write(o_file.read())
    no_file = open("data/nottheonion.txt")
    f.write(no_file.read())