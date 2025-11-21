import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Optional, Any

import httpx
import pandas
from bs4 import BeautifulSoup, Tag, ResultSet

from attach import Attach
from constants import BASE_URL, HEADERS
from notice import Notice


def get_download_times(client: httpx.Client, wbnewsid: str, owner: str) -> int:
    parms: dict[str, Any] = {
        "wbnewsid": wbnewsid,
        "owner": owner,
        "type": "wbnewsfile",
        "randomid": "nattach"
    }
    response_json: httpx.Response = client.get("system/resource/code/news/click/clicktimes.jsp", params=parms,
                                               timeout=10).json()
    return response_json["wbshowtimes"]


def get_notice(client: httpx.Client, notice: Notice) -> None:
    print(f"Get: {notice.path}")
    response: httpx.Response = client.get(notice.path, timeout=10)
    notice_soup: BeautifulSoup = BeautifulSoup(response.text, 'html.parser')
    notice.body = notice_soup.find("div", class_="v_news_content").get_text(strip=True)
    attach_ul_tag: Optional[Tag] = notice_soup.find("ul", style="list-style-type:none;")
    if attach_ul_tag is not None:
        for attach_tag in attach_ul_tag:
            link_tag: Tag = attach_tag.find_next("a")

            name: str = link_tag.get_text()
            if name == "加入收藏" or not name:
                continue

            path: str = link_tag.get("href")
            download_time_tag: Tag = attach_tag.find_next("script")
            download_time_parms: list[str] = download_time_tag.get_text().replace("getClickTimes(", "").replace(
                ',"wbnewsfile","attach")', "").split(",")
            download_times: int = get_download_times(client, *download_time_parms)
            notice.attaches.append(Attach(name=name, path=path, download_times=download_times))


def get_notice_list(client: httpx.Client, notices: list[Notice], path: str) -> None:
    response: httpx.Response = client.get(path)
    jxtz_soup: BeautifulSoup = BeautifulSoup(response.text, 'html.parser')
    jxtz_tags: ResultSet[Tag] = jxtz_soup.find("ul", class_="list-gl").find_all("li")
    for tag in jxtz_tags:
        notices.append(Notice(author=re.search(r"【(.*)】", tag.get_text()).group(1),
                              title=tag.find("a").get("title"),
                              date=date.fromisoformat(tag.find("span").text.strip()),
                              path=tag.find("a").get("href").replace("../", ""),
                              body="",
                              attaches=[]))


def export_notices_csv(notices: list[Notice], filename: str):
    detailed_data = []
    for notice in notices:
        row = {
            'title': notice.title,
            'author': notice.author,
            'date': notice.date.isoformat(),
            'url': notice.url,
            'body': notice.body,
            'attaches': json.dumps([attach.to_dict() for attach in notice.attaches], ensure_ascii=False)
        }
        detailed_data.append(row)

    df = pandas.DataFrame(detailed_data)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"成功导出 {len(detailed_data)} 条记录到 {filename}")


def main() -> None:
    client: httpx.Client = httpx.Client(base_url=BASE_URL, headers=HEADERS)
    response: httpx.Response = client.get("jxtz.htm")
    jxtz_soup: BeautifulSoup = BeautifulSoup(response.text, 'html.parser')
    max_page: int = int(re.search(r"jxtz/(\d*).htm", response.text).group(1))
    notice_list_paths: list[str] = [f"jxtz/{page}.htm" for page in range(2, max_page)]
    notice_list_paths.append("jxtz.htm")
    notices: list[Notice] = []
    with ThreadPoolExecutor(max_workers=30) as pool:
        pool.map(lambda path: get_notice_list(client, notices, path), notice_list_paths)

    jxtz_tags: ResultSet[Tag] = jxtz_soup.find("ul", class_="list-gl").find_all("li")

    for tag in jxtz_tags:
        notices.append(Notice(author=re.search(r"【(.*)】", tag.get_text()).group(1),
                              title=tag.find("a").get("title"),
                              date=date.fromisoformat(tag.find("span").text.strip()),
                              path=tag.find("a").get("href"),
                              body="",
                              attaches=[]))

    with ThreadPoolExecutor(max_workers=50) as pool:
        pool.map(lambda notice: get_notice(client, notice), notices)

    export_notices_csv(notices, "notices.csv")


if __name__ == '__main__':
    main()
