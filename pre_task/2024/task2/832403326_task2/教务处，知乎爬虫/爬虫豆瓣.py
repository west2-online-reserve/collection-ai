import requests  # 导入requests
from bs4 import BeautifulSoup  # 导入BeautifulSoup

url = "https://movie.douban.com/top250?"

my_headers = {
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}
with open("douban_top250.text", "w", encoding="utf-8") as f:

    for i in range(1, 11):
        new_url = url + f"start={(i-1)*25}"
        # print(new_url)
        resp = requests.get(new_url, headers=my_headers)

        page = BeautifulSoup(resp.text, "html.parser")

        li_list = page.find("ol").find_all("li")
        for li in li_list:
            # 在li的基础上, 进一步查找
            titles = li.find("span", attrs={"class": "title"}).text  # 获取文本.text
            scores = (
                li.find("div", attrs={"class": "star"})
                .find("span", attrs={"class": "rating_num"})
                .text
            )
            comments = li.find("div", attrs={"class": "star"}).find_all("span")[-1].text

            f.write(f"{titles}\t{scores}\t{comments}\n")

            print(titles, scores, comments)
