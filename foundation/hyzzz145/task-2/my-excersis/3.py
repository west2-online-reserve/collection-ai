from urllib.request import urlopen

url = "http://www.baidu.com" 
resp = urlopen(url)
# print(resp.read())
with open("mybaidu.html",mode='wb') as f:
    f.write(resp.read())
print('over')