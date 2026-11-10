from pathlib import Path
from urllib.parse import urljoin, parse_qs, urlsplit
from lxml import etree
import csv
import requests
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo


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
        params = parse_qs(urlsplit(attachment_url).query)  # 返回字典
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


def fetch_with_retry(url, error_records, params=None):
    for attempt in range(1, 4):
        try:
            time.sleep(1)  # 避免请求过快
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response

        except requests.HTTPError as error:
            error_records.append(
                {
                    "url": url,
                    "attempts": attempt,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            return None

        except (requests.Timeout, requests.ConnectionError) as error:
            if attempt == 3:
                error_records.append(
                    {
                        "url": url,
                        "attempts": attempt,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
    return None


def fetch_attachment_count(attachment, directory, error_records):
    code = attachment["attachment_code"]
    query = parse_qs(urlsplit(attachment["attachment_url"]).query)

    url = "https://jwch.fzu.edu.cn/system/resource/code/news/click/clicktimes.jsp"
    params = {
        "wbnewsid": code,
        "owner": query["owner"][0] if "owner" in query else "",
        "type": "wbnewsfile",
        "randomid": "nattach",
    }

    response = fetch_with_retry(url, error_records, params=params)
    if response is None:
        return None

    collected_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")

    raw_dir = directory / "raw"
    raw_dir.mkdir(exist_ok=True)
    raw_path = raw_dir / f"count_{code}_{time.time_ns()}.json"
    raw_path.write_bytes(response.content)

    download_count = parse_download_count(response.json(), code)

    # 整理本次采集记录
    metadata = {
        "attachment_code": code,
        "collected_at": collected_at,
        "download_count": download_count,
        "response_file": raw_path.name,
        "response_url": response.url,
        "status_code": response.status_code,
    }

    metadata_path = raw_path.with_suffix(".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return metadata


def parse_notice(html_bytes, metadata):
    # parse_detail 内部已经调用 clean_body，无需重复清理
    body = parse_detail(html_bytes)

    notice = {
        "publisher": metadata["publisher"],
        "title": metadata["title"],
        "published_date": metadata["published_date"],
        "notice_url": metadata["notice_url"],
        "body": body,
    }

    attachments = parse_attachments(
        html_bytes,
        metadata["notice_url"],
    )

    return notice, attachments


def save_csv(records, csv_path, temp_path, fieldnames):
    with open(temp_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    temp_path.replace(csv_path)


def save_checkpoint(directory, items):
    data_dir = directory / "data"
    data_dir.mkdir(exist_ok=True)

    state_path = data_dir / "state.json"
    temp_state_path = state_path.with_suffix(".tmp")
    with open(temp_state_path, "w", encoding="utf-8") as file:
        json.dump({"items": items}, file, ensure_ascii=False, indent=2)
    temp_state_path.replace(state_path)

    notice_rows = []
    attachment_rows = []

    for item in items.values():
        notice_rows.append(item["notice"])
        attachment_rows.extend(item["attachments"])

    # 保存通知数据到 CSV
    save_csv(
        notice_rows,
        data_dir / "notices.csv",
        data_dir / "notices.csv.tmp",
        ["publisher", "title", "published_date", "notice_url", "body"],
    )

    # 保存附件数据到 CSV
    save_csv(
        attachment_rows,
        data_dir / "attachments.csv",
        data_dir / "attachments.csv.tmp",
        [
            "notice_url",
            "attachment_name",
            "download_count",
            "attachment_code",
            "attachment_url",
        ],
    )


def load_checkpoint(directory):
    state_path = directory / "data" / "state.json"
    if not state_path.exists():
        return {}

    with open(state_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data["items"]


def main():
    directory = Path(__file__).resolve().parent

    items = load_checkpoint(directory)
    save_checkpoint(directory, items)

    target_url = "https://jwch.fzu.edu.cn/info/1036/14874.htm"
    notice_metadata = None

    if items.get(target_url):
        notice = items[target_url]["notice"]
        attachments = items[target_url]["attachments"]
    else:
        html_path = directory / "samples" / "list.html"
        html_bytes = html_path.read_bytes()

        base_url = "https://jwch.fzu.edu.cn/jxtz.htm"
        records, next_url = parse_list(html_bytes, base_url)

        for record in records:
            if record["notice_url"] == target_url:
                notice_metadata = record
                break

        if notice_metadata is None:
            print("未找到目标通知的元数据")
            return

        detail_path = directory / "samples" / "detail_with_attachments.html"
        detail_bytes = detail_path.read_bytes()

        notice, attachments = parse_notice(detail_bytes, notice_metadata)

        items[notice["notice_url"]] = {
            "notice": notice,
            "attachments": attachments,
        }

        save_checkpoint(directory, items)

    # 先解析通知和附件，再用已有本地 JSON 回填次数。
    for attachment in attachments:
        if attachment["download_count"] != "":
            print("跳过已知次数：", attachment["attachment_code"])
            continue
        code = attachment["attachment_code"]
        count_path = directory / "samples" / f"count_{code}.json"
        data = json.loads(count_path.read_bytes())
        attachment["download_count"] = parse_download_count(data, code)
        save_checkpoint(directory, items)  # 保存最终结果


if __name__ == "__main__":
    main()
