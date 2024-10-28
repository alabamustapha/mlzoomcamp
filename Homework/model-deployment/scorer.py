import requests


url = "http://0.0.0.0:9696/score"
# client = {"job": "student", "duration": 280, "poutcome": "failure"}
client = {"job": "management", "duration": 400, "poutcome": "success"}
print(requests.post(url, json=client).json())