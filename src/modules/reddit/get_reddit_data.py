import pandas as pd
import praw
import json
import io
import time
import pickle
from datetime import datetime

def get_post_data(subreddit_name, subreddit_type = 'new', post_limit = 100, comment_limmit = 100, reddit = None):
    print(f'Getting Reddit Data: Subreddit: {subreddit_name} --- Number of Posts: {post_limit} --- Comment Limit : {comment_limmit}')
    
    ## post id로 가져올 수도 있음 (id argument로)
    subreddit = reddit.subreddit(subreddit_name)
    
    if subreddit_type =='top':
        print('Getting top posts')
        posts = subreddit.top(limit=post_limit)  
    elif subreddit_type=='new':
        print('Getting new posts')
        posts = subreddit.new(limit=post_limit)  
    elif subreddit_type=='hot':
        print('Getting hot posts')
        posts = subreddit.hot(limit=post_limit)  
    posts_with_comments = []
    for post in posts:
        post.comments.replace_more(limit=comment_limmit)
        comments = []
        for comment in post.comments.list():
            comment_data = {
                'body': comment.body,
                'author': str(comment.author),
                'score': comment.score,
                'created_utc': comment.created_utc,
                'is_top_level': comment.is_root,
                'parent_id': comment.parent_id,
                'depth': comment.depth,
                'gilded': comment.gilded
            }
            comments.append(comment_data)

        post_data = {
            'title': post.title,
            'selftext': post.selftext,
            'score': post.score,
            'url': post.url,
            'author': str(post.author),
            'created_utc': post.created_utc,
            'num_comments': post.num_comments,
            'upvote_ratio': post.upvote_ratio,
            'subreddit': str(post.subreddit),
            'comments': comments
        }
        posts_with_comments.append(post_data)
        #stream_to_s3('reddit-project-data', subreddit_name, post_data)
    print('Got Reddit Data')
    return posts_with_comments


# if __name__ == '__main__':
#     # https://lovit.github.io/dataset/2019/01/16/get_reddit/
#     # https://praw.readthedocs.io/en/latest/index.html
#     print('Getting Reddit Credentials')
#     reddit_cred_file = 'utils/reddit_cred.json'
#     with open(reddit_cred_file, 'r') as file:
#         reddit_cred = json.load(file)

#     # Reddit app
#     client_id = reddit_cred['client_id']
#     client_secret = reddit_cred['client_secret']
#     user_agent = reddit_cred['user_agent']
#     reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
#     print('Got Reddit Credentials')

#     ## GET DATA
#     posts_with_comments = get_post_data(
#         subreddit_name = 'stocks',
#         subreddit_type = 'hot',
#         post_limit = 5,
#         comment_limmit = 10,
#         reddit = reddit
#         )
    
#     file_path = './reddit_data/reddit_data_stoks_hot_10.pkl'
#     with open(file_path, 'wb') as f:
#         pickle.dump(posts_with_comments, f)
#     print(f'Data saved to {file_path}')