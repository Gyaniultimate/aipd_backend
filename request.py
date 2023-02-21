import requests
import csv


url = 'http://127.0.0.1:5000/predict_api'
csv="./GaCo01_0.csv"

with open(csv, 'r') as f:
    r = requests.post(url, files={csv: f})
    print(r)



