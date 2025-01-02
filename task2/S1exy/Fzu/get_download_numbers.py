import requests

def get_download_numbers(url):
    headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0"
        }
    response = requests.get(url, headers=headers)
    a = response.text.split(",")[0].split(":")[1]
    return int(a)








