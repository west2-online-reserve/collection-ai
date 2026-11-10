from pathlib import Path
from urllib.parse import urljoin
from lxml import etree
import csv
import requests


def parse_list(html_bytes, base_url):
    tree = etree.HTML(html_bytes)
    result = tree.xpath("//ul[@class='list-gl']/li")

    dict_list = []
    for li in result:
        publisher = "".join(li.xpath("./text()")).strip().strip("【】")
        title = li.xpath("./a/@title")[0] if li.xpath("./a/@title") else ""
        date = li.xpath("string(./span[@class='doclist_time'])").strip()
        href = li.xpath("./a/@href")[0] if li.xpath("./a/@href") else ""
        notice_url = urljoin(base_url, href) if href else ""
        dict_list.append(
            {
                "publisher": publisher,
                "title": title,
                "published_date": date,
                "notice_url": notice_url,
            }
        )

    next_hrefs = tree.xpath("//span[@class='p_next p_fun']/a/@href")
    if next_hrefs:
        next_url = urljoin(base_url, next_hrefs[0])
    else:
        next_url = None

    return dict_list, next_url


def main():
    directory = Path(__file__).resolve().parent
    html_path = directory / "samples" / "list.html"
    html_bytes = html_path.read_bytes()

    base_url = "https://jwch.fzu.edu.cn/jxtz.htm"
    records, next_url = parse_list(html_bytes, base_url)

    print("通知数：", len(records))
    print("第一条：", records[0])
    print("下一页：", next_url)


if __name__ == "__main__":
    main()
