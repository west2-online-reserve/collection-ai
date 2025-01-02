from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import base64
import json
import requests


def base64_api(img ,uname='zks', pwd='Zm001889@', typeid=20):
    # with open(img, 'rb') as f:
    #     base64_data = base64.b64encode(f.read())
    #     b64 = base64_data.decode()
    data = {"username": uname, "password": pwd, "typeid": typeid, "image": img}#修改
    result = json.loads(requests.post("http://api.ttshitu.com/predict", json=data).text)
    if result['success']:
        return result["data"]["result"]
    else:
        #！！！！！！！注意：返回 人工不足等 错误情况 请加逻辑处理防止脚本卡死 继续重新 识别
        return result["message"]
    return ""

opt = Options()

opt.add_argument("--disable-blink-features=AutomationControlled")

web = Chrome(options=opt)
web.get('https://www.zhihu.com/signin?next=%2F')

web.implicitly_wait(10)

#切换到账号登陆
web.find_element(By.XPATH,'//*[@id="root"]/div/main/div/div/div/div/div[2]/div/div[1]/div/div[1]/form/div[1]/div[2]').click()
#输入账号密码
web.find_element(By.XPATH,'//*[@id="root"]/div/main/div/div/div/div/div[2]/div/div[1]/div/div[1]/form/div[2]/div/label/input').send_keys('17380672607')
web.find_element(By.XPATH,'//*[@id="root"]/div/main/div/div/div/div/div[2]/div/div[1]/div/div[1]/form/div[3]/div/label/input').send_keys('Zm12345678')
web.find_element(By.XPATH,'//*[@id="root"]/div/main/div/div/div/div/div[2]/div/div[1]/div/div[1]/form/button').click()
#点击后会弹出验证码
img = web.find_element(By.XPATH,'/html/body/div[4]/div[2]/div/div/div[2]/div')
b64_verify_code = img.screenshot_as_base64

result = base64_api(b64_verify_code)

print(result)