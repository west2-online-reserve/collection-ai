# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.edge.options import Options
import time

# 核心：配置 Edge 浏览器选项，规避知乎的反检测
edge_options = Options()

# 1. 禁用自动化提示条和扩展，隐藏 Selenium 特征
edge_options.add_experimental_option("excludeSwitches", ["enable-automation"])
edge_options.add_experimental_option("useAutomationExtension", False)

# 2. 设置浏览器窗口大小（模拟正常用户的窗口，避免被判定为异常）
edge_options.add_argument("--window-size=1920,1080")

# 3. 设置用户代理（模拟正常浏览器的 UA，可选但推荐）
edge_options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0"
)

# 4. 可选：禁用图片加载（提升速度，不影响核心功能）
# edge_options.add_argument("blink-settings=imagesEnabled=false")

# 创建 Edge 驱动对象（传入配置好的选项）
wd = webdriver.Edge(options=edge_options)

# 关键：执行 JS 脚本，重写 navigator.webdriver 属性，使其返回 undefined
wd.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
    "source": """
    Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined
    });
    """
})

try:
    # 访问知乎首页
    wd.get("https://www.zhihu.com/")
    print("正在加载知乎首页...")
    # 等待页面加载完成（给足够的时间让页面渲染）
    time.sleep(3)

    # 等待用户登录（扫码/手机号登录）
    input("登录后按回车继续...")

except Exception as e:
    print("执行出错：", e)

finally:
    # 关闭浏览器
    wd.quit()
    print("浏览器已关闭")
