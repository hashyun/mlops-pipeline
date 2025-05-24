import requests
from requests.auth import HTTPBasicAuth
import json

reddit_cred_file = '../../configreddit_cred.json'
with open(reddit_cred_file, 'r') as file:
    reddit_cred = json.load(file)

# Reddit app
client_id = reddit_cred['client_id']
client_secret = reddit_cred['client_secret']

# Reddit user in
username = reddit_cred['username']
password = reddit_cred['password']

# POST (password grant_type 사용)
data = {
    'grant_type': 'password',
    'username': username,
    'password': password
}

# POST
headers = {
    'User-Agent': username
}
auth = HTTPBasicAuth(client_id, client_secret)
response_auth = requests.post(
    'https://www.reddit.com/api/v1/access_token',
    headers=headers,
    data=data,
    auth=auth
)

if response_auth.status_code == 200:
    print("Access token:", response_auth.json())
else:
    print(f"Error: {response_auth.status_code}")
    print(response_auth.text)