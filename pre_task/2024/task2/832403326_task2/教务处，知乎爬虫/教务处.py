from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
import time
from lxml import html

etree = html.etree

web = Chrome()
web.get("https://jwch.fzu.edu.cn/")


web.find_element(By.XPATH, "/html/body/div/div[1]/div/div[3]/ul/li[3]/span/a").click()

web.switch_to.window(web.window_handles[-1])
print("新窗口标题：", web.title)

web.find_element(
    By.XPATH, "/html/body/div[1]/div[2]/div[2]/div/div/div[2]/div/div/li/a[2]"
).click()


notices = web.find_elements(
    By.XPATH, "/html/body/div[1]/div[2]/div[2]/div/div/div[3]/div[1]/ul/li"
)

# 遍历所有通知
for notice in notices:
    # 获取通知标题
    # department = notice.find_element(By.XPATH,'./span[@class="doclist_time"]/following-sibling::text()').strip('[]')
    # 获取整个 <li> 的文本
    # li_text = notice.text

    # # 分割文本，提取日期和部门
    # # 假设日期在第一部分，发布部门紧跟日期
    # lines = li_text.split("\n")  # 按行分割文本
    # date = lines[0]  # 第 1 行是日期
    # department = lines[1].strip("[]")  # 第 2 行是发布部门，去掉方括号
    # 提取完整的 <li> 文本内容
    # li_text = notice.text

    # # 提取发布部门（利用 [ 和 ] 定位）
    # department_start = li_text.find("[")  # 找到 [ 的位置
    # department_end = li_text.find("]")  # 找到 ] 的位置
    # if department_start != -1 and department_end != -1:  # 确保找到 [ 和 ]
    #     department = li_text[department_start:department_end + 1]  # 包括 [ 和 ]
    # else:
    #     department = "未知部门"  # 如果没有找到部门信息
    # 使用 JavaScript 提取部门文本
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
    print(f"发布日期：{date}")
    print(f"发布部门：{department}")
    print(f"通知标题：{title}")
    print(f"通知链接：{link}")
    print("-" * 40)


# 关闭浏览器
web.quit()
