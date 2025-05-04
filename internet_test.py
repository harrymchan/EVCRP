import requests

try:
    r = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params={
        "address": "40.468254,-86.980963",
        "key": "AIzaSyBxcVtnDTNCFN4LVKWX5UXRnt8spMdnLVA"
    }, timeout=10)
    print(r.status_code)
    print(r.text)
except Exception as e:
    print("连接失败:", e)