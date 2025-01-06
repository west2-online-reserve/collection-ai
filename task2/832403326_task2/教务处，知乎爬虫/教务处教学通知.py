import csv
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 写表头
with open("notifications.csv", mode="w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(
        [
            "发布日期",
            "发布部门",
            "通知标题",
            "通知链接",
            "附件名",
            "附件链接",
            "下载次数",
        ]
    )

web = Chrome()

web.get("https://jwch.fzu.edu.cn/jxtz.htm")  ##爬取教学通知所有内容

notices = web.find_elements(
    By.XPATH, "/html/body/div[1]/div[2]/div[2]/div/div/div[3]/div[1]/ul/li"
)
# /html/body/div[1]/div[2]/div[2]/div/div/div[3]/div[1]/ul/li[1]/a
for notice in notices:
    department = web.execute_script(
        "return arguments[0].nextSibling.textContent.trim();",
        notice.find_element(By.XPATH, './span[@class="doclist_time"]'),
    )

    title = notice.find_element(
        By.XPATH, "./a"
    ).text  # 使用相对 XPath 获取 <a> 标签的文本
    date = notice.find_element(By.XPATH, './span[@class="doclist_time"]').text

    # 获取通知链接
    link = notice.find_element(By.XPATH, "./a").get_attribute(
        "href"
    )  # 获取 <a> 标签的 href 属性
    web.get(link)
    # 等待详情页加载完成
    WebDriverWait(web, 30).until(
        EC.presence_of_element_located(
            (By.XPATH, '//div[@class="xl_news"]')
        )  # 确保详情页加载
    )
    # 检查附件是否存在
    try:
        # 获取所有附件元素
        attachments = web.find_elements(
            By.XPATH, "/html/body/div/div[2]/div[2]/form/div/div[1]/div/ul/li"
        )
        if attachments:  # 如果存在附件
            for attachment in attachments:
                # 提取附件名称
                file_name = attachment.find_element(By.XPATH, "./a").text

                # 提取附件链接
                file_link = attachment.find_element(By.XPATH, "./a").get_attribute(
                    "href"
                )

                # 提取附件下载次数
                download_count = attachment.find_element(By.XPATH, "./span").text
                # 将数据写入 CSV 文件
                with open(
                    "notifications.csv", mode="a", encoding="utf-8", newline=""
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [
                            date,
                            department,
                            title,
                            link,
                            file_name,
                            file_link,
                            download_count,
                        ]
                    )

                # 打印附件信息
                # print(f"通知标题：{title}")
                # print(f"附件名：{file_name}")
                # print(f"附件链接：{file_link}")
                # print(f"下载次数：{download_count}")
                # print("-" * 40)
        else:
            with open(
                "notifications.csv", mode="a", encoding="utf-8", newline=""
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        date,
                        department,
                        title,
                        link,
                        "无附件",
                        "无附件链接",
                        "无下载次数",
                    ]
                )

        # print(f"通知标题：{title}")
        # print("无附件")
        # print("-" * 40)
    except Exception as e:
        with open("notifications.csv", mode="a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    date,
                    department,
                    title,
                    link,
                    "附件提取失败",
                    "附件提取失败",
                    "附件提取失败",
                ]
            )

        # print(f"通知标题：{title}")
        # print("附件提取失败，可能不存在附件")
        # print(f"错误信息：{e}")
        # print("-" * 40)

    # 返回上一页
    web.back()

    # 等待通知列表页加载完成
    WebDriverWait(web, 30).until(
        EC.presence_of_all_elements_located((By.XPATH, '//ul[@class="list-gl"]/li'))
    )

    # print(f"发布日期：{date}")
    # print(f"发布部门：{department}")
    # print(f"通知标题：{title}")
    # print(f"通知链接：{link}")
    # print("-" * 40)


for g in range(1, 198):
    url = "https://jwch.fzu.edu.cn/jxtz/"
    new_url = url + f"{198-g}.htm"
    web.get(new_url)
    notices = web.find_elements(
        By.XPATH, "/html/body/div[1]/div[2]/div[2]/div/div/div[3]/div[1]/ul/li"
    )
    for notice in notices:
        department = web.execute_script(
            "return arguments[0].nextSibling.textContent.trim();",
            notice.find_element(By.XPATH, './span[@class="doclist_time"]'),
        )

        title = notice.find_element(
            By.XPATH, "./a"
        ).text  # 使用相对 XPath 获取 <a> 标签的文本
        date = notice.find_element(By.XPATH, './span[@class="doclist_time"]').text

        # 获取通知链接
        link = notice.find_element(By.XPATH, "./a").get_attribute(
            "href"
        )  # 获取 <a> 标签的 href 属性
        web.get(link)
        # 检查页面是否有效（避免 404 报错）
        if "404" in web.title or "Not Found" in web.page_source:
            print(f"通知标题：{title}")
            print(f"通知链接：{link}")
            print("页面无效（404 错误），跳过处理")
            print("-" * 40)
            # 返回上一页，继续处理其他通知
            web.back()
            WebDriverWait(web, 10).until(
                EC.presence_of_all_elements_located(
                    (By.XPATH, '//ul[@class="list-gl"]/li')
                )
            )
            continue  # 跳过当前通知，处理下一个

        # 等待详情页加载完成
        WebDriverWait(web, 30).until(
            EC.presence_of_element_located(
                (By.XPATH, '//div[@class="xl_news"]')
            )  # 确保详情页加载
        )
        # 检查附件是否存在
        try:
            # 获取所有附件元素
            attachments = web.find_elements(
                By.XPATH, "/html/body/div/div[2]/div[2]/form/div/div[1]/div/ul/li"
            )
            if attachments:  # 如果存在附件
                for attachment in attachments:
                    # 提取附件名称
                    file_name = attachment.find_element(By.XPATH, "./a").text

                    # 提取附件链接
                    file_link = attachment.find_element(By.XPATH, "./a").get_attribute(
                        "href"
                    )

                    # 提取附件下载次数
                    download_count = attachment.find_element(By.XPATH, "./span").text
                    with open(
                        "notifications.csv", mode="a", encoding="utf-8", newline=""
                    ) as file:
                        writer = csv.writer(file)
                        writer.writerow(
                            [
                                date,
                                department,
                                title,
                                link,
                                file_name,
                                file_link,
                                download_count,
                            ]
                        )

                    # 打印附件信息
                    # print(f"通知标题：{title}")
                    # print(f"附件名：{file_name}")
                    # print(f"附件链接：{file_link}")
                    # print(f"下载次数：{download_count}")
                    # print("-" * 40)
            else:
                with open(
                    "notifications.csv", mode="a", encoding="utf-8", newline=""
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [
                            date,
                            department,
                            title,
                            link,
                            "无附件",
                            "无附件链接",
                            "无下载次数",
                        ]
                    )
                # print(f"通知标题：{title}")
                # print("无附件")
                # print("-" * 40)
        except Exception as e:
            with open(
                "notifications.csv", mode="a", encoding="utf-8", newline=""
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        date,
                        department,
                        title,
                        link,
                        "附件提取失败",
                        "附件提取失败",
                        "附件提取失败",
                    ]
                )

            # print(f"通知标题：{title}")
            # print("附件提取失败，可能不存在附件")
            # print(f"错误信息：{e}")
            # print("-" * 40)

        # 返回上一页
        web.back()

        # 等待通知列表页加载完成
        WebDriverWait(web, 30).until(
            EC.presence_of_all_elements_located((By.XPATH, '//ul[@class="list-gl"]/li'))
        )

        # print(f"发布日期：{date}")
        # print(f"发布部门：{department}")
        # # print(f"通知标题：{title}")
        # print(f"通知链接：{link}")
        # print("-" * 40)
web.quit()


# /html/body/div/div[2]/div[2]/form/div/div[1]/div/ul/li/a
# /html/body/div/div[2]/div[2]/form/div/div[1]/div/ul/li[1]/a
# /html/body/div/div[2]/div[2]/form/div/div[1]/div/ul/li/a
# /html/body/div/div[2]/div[2]/form/div/div[1]/div/ul/li/a
# //*[@id="nattach15572980"]
