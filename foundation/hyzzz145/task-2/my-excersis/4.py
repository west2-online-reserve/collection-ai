import requests
url = "https://movie.douban.com/j/chart/top_list"
head = {
    "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36 Edg/154.0.0.0"
}
param = {

    
    "type":"11",
"interval_id":"100%3A90",
"action":"",
"start":"0",
"limit":"20",
}
resp = requests.get(url,params=param,headers=head)
print(resp.text)
resp.close()