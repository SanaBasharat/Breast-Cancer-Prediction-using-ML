import requests
url = 'http://localhost:5000/api'
r = requests.post(url,json={'exp':[11.32, 27.08, 71.76, 395.7, 0.06883]})
print(r.json())