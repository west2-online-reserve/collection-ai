import csv
from dataclasses import dataclass
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://jwch.fzu.edu.cn/jxtz.htm"
DOWNLOADS_API = "https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp"


@dataclass
class AttachmentInfo:
    file_name: str
    url: str
    downloads: int


@dataclass
class WebInfo:
    announcer: str
    title: str
    time: str
    url: str
    content: str
    attachments: list[AttachmentInfo]


def get_max_page() -> int:
    response = requests.get(BASE_URL)
    response.encoding = "utf-8"

    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    href = soup.find("a", {"href": "jxtz/1.htm"})
    return int(href.text) if href else 0


def get_urls() -> list[str]:  # type: ignore
    max_page = get_max_page()

    urls = []

    # 只爬30页，太多等死人
    for i in range(max_page, 0, -1):
        if i < max_page - 30:
            break

        current_page = (
            BASE_URL if i == max_page else f"https://jwch.fzu.edu.cn/jxtz/{i}.htm"
        )

        response = requests.get(current_page)
        response.encoding = "utf-8"

        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        all_notice = soup.find("ul", class_="list-gl")

        if all_notice:
            for a in all_notice.find_all("a"):
                href = a.get("href", "")
                if href:
                    urls.append(urljoin(BASE_URL, href))  # type: ignore

    return urls  # type: ignore


def get_content(soup: BeautifulSoup) -> str:
    content: list[str] = []

    all_news_text = soup.find("div", {"class": "v_news_content"}).find_all("p")  # type: ignore

    for news_text in all_news_text:
        content.append(news_text.get_text(strip=True))

    return "".join(content)


def get_downloads(fileid: str, owner: str):
    params = {
        "wbnewsid": fileid,
        "owner": owner,
        "type": "wbnewsfile",
        "randomid": "nattach",
    }
    try:
        response = requests.get(
            DOWNLOADS_API,
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        return int(response.json().get("wbshowtimes", 0))
    except Exception as e:  # noqa: BLE001
        print(f"获得下载次数失败 {fileid}: {e}")
        return 0


def get_attachments(url: str):
    attachments: list[AttachmentInfo] = []

    response = requests.get(url)
    response.encoding = "utf-8"

    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    ul = soup.find("ul", {"style": "list-style-type:none;"})

    if not ul:
        return attachments

    for li in ul.find_all("li"):
        a = li.find("a", href=True)

        if not a:
            continue

        href = a.get("href")

        if not isinstance(href, str):
            continue

        qs = parse_qs(urlparse(href).query)
        fileid = qs.get("wbfileid", [""])[0]
        owner = qs.get("owner", [""])[0]

        attachments.append(
            AttachmentInfo(
                file_name=a.get_text(strip=True),
                url=urljoin(url, href),
                downloads=get_downloads(fileid, owner),
            )
        )

    return attachments


def get_web_info(url: str):
    response = requests.get(url)
    response.encoding = "utf-8"

    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    announcer = soup.find("p", {"class": "w-main-dh-text"}).find_all("a")[-1].text  # type: ignore
    title = soup.find("h4", {"align": "center"}).text  # type: ignore
    time = soup.find("span", {"class": "xl_sj_icon"}).text.replace("发布时间：", "")  # type: ignore
    content = get_content(soup)
    attachments = get_attachments(url)

    return WebInfo(announcer, title, time, url, content, attachments)


def save_to_csv(infos: list[WebInfo], path: str = "notices.csv") -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        # 表头
        writer.writerow(
            [
                "发布单位",
                "标题",
                "发布时间",
                "链接",
                "正文",
                "附件",
            ]
        )

        for info in infos:
            attachment_info = "; ".join(
                f"{attachment.file_name}|{attachment.url}|{attachment.downloads}"
                for attachment in info.attachments
            )
            writer.writerow(
                [
                    info.announcer,
                    info.title,
                    info.time,
                    info.url,
                    info.content,
                    attachment_info,
                ]
            )
            f.flush()


if __name__ == "__main__":
    urls = get_urls()

    def crawl_all():
        for url in urls:
            print(f"正在爬取: {url}")
            yield get_web_info(url)

    save_to_csv(crawl_all())  # type: ignore
