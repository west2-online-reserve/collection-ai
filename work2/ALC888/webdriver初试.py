from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, json, os, csv

# 简化脚本：抓取 20 个问题，每题最多 10 个回答，保存为 CSV
COOKIES = 'cookies.json'
QUERY = '福州大学'
OUT = 'zhihu_20x10.csv'

def init_driver():
    opts = webdriver.EdgeOptions()
    opts.add_experimental_option('excludeSwitches', ['enable-automation'])
    opts.add_experimental_option('useAutomationExtension', False)
    opts.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
    driver = webdriver.Edge(options=opts)
    try:
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument',
            {'source': "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"})
    except Exception:
        pass
    return driver

def load_cookies(driver):
    if not os.path.exists(COOKIES):
        return False
    with open(COOKIES, 'r', encoding='utf-8') as f:
        cookies = json.load(f)
    driver.get('https://www.zhihu.com')
    for c in cookies:
        cookie = {'name': c['name'], 'value': c['value']}
        if c.get('path'): cookie['path'] = c['path']
        if c.get('domain'): cookie['domain'] = c['domain']
        exp = c.get('expiry') if c.get('expiry') is not None else c.get('expirationDate')
        if exp is not None:
            try: cookie['expiry'] = int(exp)
            except: pass
        try:
            driver.add_cookie(cookie)
        except Exception:
            cookie.pop('domain', None)
            try: driver.add_cookie(cookie)
            except: pass
    driver.refresh()
    time.sleep(1)
    return True

def get_question_urls(driver, query, n=20):
    q = query.replace(' ', '+')
    driver.get(f'https://www.zhihu.com/search?q={q}&type=content')
    urls = []
    for _ in range(12):
        elems = driver.find_elements(By.CSS_SELECTOR, "a[href*='/question/']")
        for e in elems:
            href = e.get_attribute('href')
            if not href:
                continue
            # 规范化：去掉 query/hash，并去掉尾部斜杠，减少重复
            href = href.split('?')[0].split('#')[0].rstrip('/')
            if href and href not in urls:
                urls.append(href)
                if len(urls) >= n: return urls[:n]
        # 有时页面未能找到 body 元素，使用 JS 滚动作为更稳健的替代
        try:
            driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        except Exception:
            try:
                driver.find_element(By.TAG_NAME, 'html').send_keys(Keys.END)
            except Exception:
                pass
        time.sleep(1)
    return urls[:n]

def expand_full_texts(driver):
    xps = ["//button[contains(text(),'展开') or contains(text(),'查看全部')]"]
    for xp in xps:
        for el in driver.find_elements(By.XPATH, xp):
            try:
                driver.execute_script('arguments[0].scrollIntoView(true);', el)
                time.sleep(0.2)
                el.click()
                time.sleep(0.3)
            except Exception:
                pass

def scrape_answers(driver, url, m=10):
    driver.get(url)
    try:
        WebDriverWait(driver, 6).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[class*='Answer']")))
    except Exception:
        time.sleep(1)
    expand_full_texts(driver)
    time.sleep(0.8)
    # 只在每个答案容器内提取一次富文本节点，避免抓到作者、meta 或页面碎片
    items = driver.find_elements(By.XPATH, "//div[contains(@class,'AnswerItem') or contains(@class,'Answer') or @data-testid='answer']")
    res = []
    seen = set()
    for it in items:
        t = ''
        try:
            # 精确选择答案正文的富文本节点
            try:
                sub = it.find_element(By.CSS_SELECTOR, ".RichText, .RichContent, .ztext, [data-testid='answer-content']")
                t = sub.text.strip()
            except Exception:
                # 回退到容器文本
                t = it.text.strip()
        except Exception:
            t = ''

        if not t:
            continue
        norm = ' '.join(t.split())
        # 仅保留去重逻辑，不过滤短文本或关键词
        if norm in seen:
            continue
        seen.add(norm)
        res.append(t)
        if len(res) >= m:
            break

    return res[:m]

def main():
    driver = init_driver()
    try:
        ok = load_cookies(driver)
        if not ok:
            input('请在打开的浏览器登录知乎并完成人机验证，完成后回车继续...')
        qurls = get_question_urls(driver, QUERY, n=20)
        with open(OUT, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            # 不输出链接：只保存问题标题、回答编号和回答文本
            writer.writerow(['question_title', 'answer_index', 'answer_text'])
            for idx, qu in enumerate(qurls, 1):
                ans = scrape_answers(driver, qu, m=10)
                title = driver.title if driver.title else ''
                for i, a in enumerate(ans, 1):
                    writer.writerow([title, i, a])
                print(f'已抓取 {idx}/{len(qurls)}: {len(ans)} 回答')
    finally:
        driver.quit()

if __name__ == '__main__':
    import csv
    main()

   