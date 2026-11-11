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
            print(f"  [请求失败] {url}：{error}", flush=True)
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
            print(
                f"  [请求异常 {attempt}/3] {url}：{error}"
                + ("；即将重试" if attempt < 3 else "；已达到尝试上限"),
                flush=True,
            )
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


def collect_list_pages(start_url, directory, error_records, max_pages=2):
    current_url = start_url
    all_records = []
    seen_urls = set()  # 记录已访问的 URL，避免重复抓取
    page_count = 0

    while current_url and page_count < max_pages:
        print(
            f"[列表 {page_count + 1}/{max_pages}] 正在获取：{current_url}",
            flush=True,
        )
        response = fetch_with_retry(current_url, error_records)
        if response is None:
            print("[列表失败] 停止翻页，保留当前地址供后续处理。", flush=True)
            break
        collected_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
            timespec="seconds"
        )

        raw_dir = directory / "raw"
        raw_dir.mkdir(exist_ok=True)

        raw_path = raw_dir / f"list_{time.time_ns()}.html"
        raw_path.write_bytes(response.content)

        response_metadata = {
            "requested_url": current_url,
            "response_url": response.url,
            "status_code": response.status_code,
            "collected_at": collected_at,
            "response_file": raw_path.name,
        }

        meta_path = raw_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(response_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        page_records, next_url = parse_list(response.content, response.url)

        if not page_records:
            print("[列表失败] 未解析到通知，原文已保存。", flush=True)
            raise ValueError(
                f"第 {page_count + 1} 页未解析到任何记录，URL: {current_url}"
            )

        for record in page_records:
            notice_url = record["notice_url"]
            if notice_url in seen_urls:
                continue  # 跳过已访问的通知 URL
            seen_urls.add(notice_url)

            all_records.append(record)

        current_url = next_url
        page_count += 1
        print(
            f"[列表 {page_count}/{max_pages}] 成功：本页 {len(page_records)} 条，"
            f"累计 {len(all_records)} 条唯一通知。",
            flush=True,
        )

    return all_records, current_url  # 返回所有记录和下一页的 URL


def collect_notice(notice_metadata, directory, items, error_records):
    notice_url = notice_metadata["notice_url"]

    if notice_url not in items:
        print("  正在获取详情正文……", flush=True)
        response = fetch_with_retry(notice_url, error_records)
        if response is None:
            return False  # 请求失败，无法继续处理

        collected_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
            timespec="seconds"
        )

        raw_dir = directory / "raw"
        raw_dir.mkdir(exist_ok=True)

        raw_path = raw_dir / f"detail_{time.time_ns()}.html"
        raw_path.write_bytes(response.content)

        response_metadata = {
            "requested_url": notice_url,
            "response_url": response.url,
            "status_code": response.status_code,
            "collected_at": collected_at,
            "response_file": raw_path.name,
        }

        meta_path = raw_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(response_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        try:
            notice, attachments = parse_notice(response.content, notice_metadata)
        except ValueError as error:
            error_records.append(
                {
                    "url": notice_url,
                    "attempts": "",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            return False  # 解析失败，无法继续处理

        items[notice_url] = {
            "notice": notice,
            "attachments": attachments,
            "detail_response": response_metadata,  # 保存详情页响应元数据
            "count_evidence": {},  # 初始化下载次数证据字典
        }

        save_checkpoint(directory, items)  # 保存中间结果
        print(f"  正文已保存，发现 {len(attachments)} 个附件。", flush=True)
    else:
        print("  复用已保存正文，检查附件次数。", flush=True)

    item = items[notice_url]
    attachment_total = len(item["attachments"])
    for index, attachment in enumerate(item["attachments"], start=1):
        if attachment["download_count"] != "":
            print(
                f"  [附件 {index}/{attachment_total}] 跳过已知次数："
                f"{attachment['attachment_name']}（{attachment['download_count']} 次）",
                flush=True,
            )
            continue  # 跳过已知次数

        print(
            f"  [附件 {index}/{attachment_total}] 正在获取次数："
            f"{attachment['attachment_name']}",
            flush=True,
        )
        try:
            count_record = fetch_attachment_count(attachment, directory, error_records)
        except ValueError as error:
            error_records.append(
                {
                    "url": notice_url,
                    "attempts": "",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            return False  # 下载次数解析失败，无法继续处理
        if count_record is None:
            return False  # 下载次数获取失败，无法继续处理

        attachment["download_count"] = count_record["download_count"]
        item.setdefault("count_evidence", {})[
            attachment["attachment_code"]
        ] = count_record
        save_checkpoint(directory, items)  # 保存最终结果
        print(
            f"  [附件 {index}/{attachment_total}] 成功："
            f"{attachment['download_count']} 次，已保存。",
            flush=True,
        )

    return True


def collect_notices(records, directory, items, error_records):
    success_count = 0
    failure_count = 0

    total = len(records)
    for index, record in enumerate(records, start=1):
        print(
            f"[通知 {index}/{total}] 开始：{record['title']}\n"
            f"  {record['notice_url']}",
            flush=True,
        )
        previous_errors = len(error_records)
        try:
            result = collect_notice(record, directory, items, error_records)
        except OSError as error:
            print(f"[通知 {index}/{total}] 文件操作失败，采集停止：{error}", flush=True)
            raise
        if not result:
            failure_count += 1
            reason = (
                error_records[-1]["error_message"]
                if len(error_records) > previous_errors
                else "未能完成采集"
            )
            print(f"[通知 {index}/{total}] 失败：{reason}", flush=True)
        else:
            success_count += 1
            print(f"[通知 {index}/{total}] 成功，数据已完整保存。", flush=True)
        print(f"  本轮累计：成功 {success_count}，失败 {failure_count}。", flush=True)

    return success_count, failure_count


def main():
    directory = Path(__file__).resolve().parent

    items = load_checkpoint(directory)
    save_checkpoint(directory, items)
    print(f"[启动] 已加载 {len(items)} 条通知状态。", flush=True)

    error_records = []

    start_url = "https://jwch.fzu.edu.cn/jxtz.htm"
    records, next_url = collect_list_pages(
        start_url, directory, error_records, max_pages=50
    )

    success_count, failure_count = collect_notices(
        records, directory, items, error_records
    )

    data_dir = directory / "data"
    save_csv(
        error_records,
        data_dir / "errors.csv",
        data_dir / "errors.csv.tmp",
        ["url", "attempts", "error_type", "error_message"],
    )

    print(f"[结束] 本轮处理完毕，数据目录：{data_dir}", flush=True)
    print("列表条数：", len(records))
    print("成功数：", success_count)
    print("失败数：", failure_count)
    print("错误数：", len(error_records))
    print("下一待处理列表页：", next_url)


if __name__ == "__main__":
    main()
