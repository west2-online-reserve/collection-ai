from pathlib import Path
from urllib.parse import urljoin, parse_qs, urlsplit
from lxml import etree
import csv
import requests
import json


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


def clean_body(text):
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def parse_detail(html_bytes):
    tree = etree.HTML(html_bytes)
    main_content = tree.xpath("//div[@id='vsb_content']")
    if len(main_content) == 1:
        content_text = main_content[0].xpath("string()")
        cleaned_text = clean_body(content_text)
        return cleaned_text
    raise ValueError(f"正文容器应为 1 个，实际匹配 {len(main_content)} 个")


def parse_attachments(html_bytes, notice_url):
    tree = etree.HTML(html_bytes)
    links = tree.xpath("//a[contains(@href, 'download.jsp')]")
    attachments = []
    for a in links:
        name = a.xpath("string(.)").strip()
        attachment_url = urljoin(notice_url, a.get("href", ""))
        params = parse_qs(urlsplit(attachment_url).query)
        attachment_code = params.get("wbfileid", [""])[0]
        attachments.append(
            {
                "notice_url": notice_url,
                "attachment_name": name,
                "attachment_code": attachment_code,
                "attachment_url": attachment_url,
                "download_count": "",  # 暂时为空，后续获取
            }
        )
    return attachments


def parse_download_count(data, attachment_code):
    if not isinstance(data, dict):
        raise ValueError("下载次数响应应为 JSON 对象")

    if "wbnewsid" not in data or str(data["wbnewsid"]) != str(attachment_code):
        raise ValueError(f"响应编号与附件 {attachment_code} 不一致")

    count = data.get("wbshowtimes")
    # bool 也是 int 的子类，使用精确类型检查以排除 True 和 False。
    if type(count) is not int or count < 0:
        raise ValueError(f"附件 {attachment_code} 的下载次数必须是非负整数")

    return str(count)


def main():
    directory = Path(__file__).resolve().parent
    html_path = directory / "samples" / "list.html"
    html_bytes = html_path.read_bytes()

    base_url = "https://jwch.fzu.edu.cn/jxtz.htm"
    records, next_url = parse_list(html_bytes, base_url)

    print("通知数：", len(records))
    print("第一条：", records[0])
    print("下一页：", next_url)

    detail_path = directory / "samples" / "detail.html"
    detail_bytes = detail_path.read_bytes()
    body = parse_detail(detail_bytes)

    print("正文字符数：", len(body))
    print("正文开头：", body[:100])
    print("正文结尾：", body[-100:])

    attachments_bytes = (
        directory / "samples" / "detail_with_attachments.html"
    ).read_bytes()
    attachments = parse_attachments(
        attachments_bytes, "https://jwch.fzu.edu.cn/info/1036/14874.htm"
    )

    print("附件数：", len(attachments))
    for attachment in attachments:
        print(attachment)

    for attachment in attachments:
        code = attachment["attachment_code"]
        count_path = directory / "samples" / f"count_{code}.json"
        data = json.loads(count_path.read_bytes())
        attachment["download_count"] = parse_download_count(data, code)
        print(code, attachment["download_count"])


if __name__ == "__main__":
    main()
