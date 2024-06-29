import requests
import json

url = 'http://localhost:5000/predict'
data = {'data': ["This is cool!"]}  # Replace with your actual data
headers = {'Content-Type': 'application/json'}

response = requests.post(url, data=json.dumps(data), headers=headers)
print(response.json())
