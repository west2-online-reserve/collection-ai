from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time

# 感觉这个挺关键，一添加上就跑通了
opt = Options()

opt.add_argument("--disable-blink-features=AutomationControlled")

web = Chrome(options=opt)

web.get("https://www.zhihu.com/")

web.implicitly_wait(100)


web.find_element(By.XPATH, '//*[@id="Popover1-toggle"]').send_keys("庄子", Keys.ENTER)

web.implicitly_wait(5)


web.find_element(
    By.XPATH, '//*[@id="root"]/div/main/div/div[1]/div/div/div'
).click()  # 筛选注意加括号，死几次了

web.implicitly_wait(5)

web.find_element(
    By.XPATH, '//*[@id="root"]/div/main/div/div[1]/div[2]/ul[1]/li[2]/div'
).click()  # 筛选，只看问答


web.find_element(
    By.XPATH,
    '//*[@id="SearchMain"]/div/div/div/div[2]/div/div/div/h2/span/div/div/div/a/span',
)


# //*[@id="SearchMain"]/div/div/div/div[3]/div/div/div/h2/span/div/div/div/a/span
# //*[@id="SearchMain"]/div/div/div/div[4]/div/div/div/div/div[2]/div[1]/a/span/em
# //*[@id="SearchMain"]/div/div/div/div[5]/div/div/div/h2/span/div/div/div/a/span


# //*[@id="SearchMain"]/div/div/div/div[2]/div/div/div/h2/span/div/div/div/a/span
# //*[@id="SearchMain"]/div/div/div/div[2]/div/div/div/h2/span/div/div/div/a/span
# //*[@id="SearchMain"]/div/div/div/div[3]/div/div/div/h2/span/div/div/div/a/span
# //*[@id="SearchMain"]/div/div/div/div[4]/div/div/div/h2/span/div/div/div/a/span
