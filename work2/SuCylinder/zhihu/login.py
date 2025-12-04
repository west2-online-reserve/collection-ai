from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait
from time import sleep
import pathlib
import json

edge_options = Options()
edge_options.add_experimental_option("excludeSwitches", ["enable-automation"])
edge_options.add_experimental_option("useAutomationExtension", False)
edge_options.add_argument("--disable-blink-features=AutomationControlled")
path = pathlib.Path(__file__).parent / "msedgedriver.exe"
service = EdgeService(executable_path=path)

edge = webdriver.Edge(service=service,options=edge_options)

edge.get("https://www.zhihu.com/")
print(edge.title)
print("扫码登录")

a = WebDriverWait(edge,1000).until(
    ec.presence_of_element_located((By.XPATH,'//div[contains(text(),"提问题")]'))
)

json_path = pathlib.Path(__file__).parent / "cookies.json"
cookies = edge.get_cookies()

with open(json_path,"w") as f:
    json.dump(cookies,f)

edge.quit()
