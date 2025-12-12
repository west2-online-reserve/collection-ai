import requests
import time
import csv
import os
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

API_LIST = 'https://summer-ospp.ac.cn/api/getProList'
API_DETAIL = 'https://summer-ospp.ac.cn/api/getProDetail'
COOKIES = {}
HEADERS = {
    'accept': '*/*',
    'content-type': 'application/json',
    'user-agent': 'Mozilla/5.0',
}

OUT_CSV = 'summer_projects_full.csv'
PDF_DIR = 'pdfs'

DEFAULT_PAYLOAD = {
    'supportLanguage': [],
    'techTag': [],
    'programmingLanguageTag': [],
    'programName': '',
    'difficulty': [],
    'pageNum': 1,
    'pageSize': 50,
    'lang': 'zh',
    'orgName': [],
}

def fetch_list(session: requests.Session, page: int = 1, page_size: int = 50) -> List[Dict[str,Any]]:
    payload = dict(DEFAULT_PAYLOAD)
    payload['pageNum'] = page
    payload['pageSize'] = page_size
    r = session.post(API_LIST, json=payload, timeout=15)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        d = data.get('data', data)
        if isinstance(d, dict) and isinstance(d.get('list'), list):
            return d['list']
        if isinstance(d, list):
            return d
        for key in ('list','records','rows','items'):
            if key in data and isinstance(data[key], list):
                return data[key]
    if isinstance(data, list):
        return data
    return []

def get_candidate_id(item: Dict[str,Any]) -> Optional[str]:
    for k in ('id','projectId','proId','pro_id'):
        if k in item and item[k]:
            return str(item[k])
    for k in ('url','detailUrl','projectUrl'):
        if k in item and item[k]:
            v = item[k]      
            import re
            m = re.search(r"(\d{5,})", v)
            if m: return m.group(1)
    return None

def extract_fields_from_item(item: Dict[str,Any]) -> Dict[str,str]:
    name = item.get('projectName') or item.get('proName') or item.get('name') or item.get('programName') or item.get('title') or ''
    difficulty = item.get('difficulty') or item.get('level') or ''
    tags = item.get('techTag') or item.get('tech_tags') or item.get('programmingLanguageTag') or item.get('supportLanguage') or ''
    if isinstance(tags, list):
        tags = ','.join([str(t) for t in tags])
    return {'name': str(name).strip(), 'difficulty': str(difficulty), 'tags': str(tags)}

def fetch_detail_via_api(session: requests.Session, pid: str) -> Optional[Dict[str,Any]]:
    for key in ('projectId','id','proId'):
        try:
            r = session.post(API_DETAIL, json={key: pid}, timeout=12)
            if r.status_code == 200:
                try:
                    data = r.json()

                    if isinstance(data, dict):
                        d = data.get('data', data)
                        if isinstance(d, dict):
                            return d
                        return data
                except Exception:
                    continue
        except Exception:
            continue
    return None

def fetch_detail_via_page(session: requests.Session, pid: str) -> Dict[str,str]:
    candidates = [f'https://summer-ospp.ac.cn/project/{pid}', f'https://summer-ospp.ac.cn/project/detail/{pid}']
    out = {'desc':'', 'requirements':'', 'pdf':''}
    for url in candidates:
        try:
            r = session.get(url, timeout=12)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, 'html.parser')
            sel_candidates = [".description", ".project-desc", ".proj-intro", "#description", "[data-testid=project-desc]"]
            texts = []
            for s in sel_candidates:
                el = soup.select_one(s)
                if el and el.get_text(strip=True):
                    texts.append(el.get_text('\n', strip=True))
            if not texts:
                ps = soup.find_all('p')
                if ps:
                    longest = max(ps, key=lambda p: len(p.get_text(strip=True)))
                    texts.append(longest.get_text('\n', strip=True))
            if texts:
                out['desc'] = '\n\n'.join(texts)

            req_texts = []
            for h in soup.find_all(['h2','h3','h4']):
                txt = h.get_text(strip=True)
                if any(k in txt for k in ('产出','要求','output','成果')):
        
                    sib = h.find_next_sibling()
                    if sib:
                        req_texts.append(sib.get_text('\n', strip=True))
            if req_texts:
                out['requirements'] = '\n\n'.join(req_texts)

            a_tags = soup.find_all('a')
            for a in a_tags:
                href = a.get('href') or ''
                if href.lower().endswith('.pdf') or '申请书' in (a.get_text() or ''):
                    out['pdf'] = requests.compat.urljoin(url, href)
                    break

            if any(out.values()):
                return out
        except Exception:
            continue
    return out

def main():
    s = requests.Session()
    s.headers.update(HEADERS)
    s.cookies.update(COOKIES)

    all_items = []
    for p in range(1, 6):
        print('fetch list page', p)
        rows = fetch_list(s, page=p, page_size=50)
        if not rows:
            print('list page empty, stop')
            break
        all_items.extend(rows)
        time.sleep(0.6)

    print('total items collected', len(all_items))

    out_rows = []
    for item in all_items:
        base = extract_fields_from_item(item)
        pid = get_candidate_id(item) or ''
        base['id'] = pid
        # try detail via api
        detail = None
        if pid:
            detail = fetch_detail_via_api(s, pid)
        desc = ''
        reqs = ''
        pdf_url = ''
        if isinstance(detail, dict):
            desc = detail.get('description') or detail.get('intro') or detail.get('summary') or detail.get('projectDesc') or ''
            reqs = detail.get('requirements') or detail.get('output') or detail.get('projectOutput') or ''
            for k in ('pdf','file','applyFile','applyUrl'):
                if k in detail and detail[k]:
                    pdf_url = detail[k]
                    break
        if not any((desc, reqs, pdf_url)) and pid:
            page_detail = fetch_detail_via_page(s, pid)
            desc = desc or page_detail.get('desc','')
            reqs = reqs or page_detail.get('requirements','')
            pdf_url = pdf_url or page_detail.get('pdf','')

        out_rows.append({
            'id': pid,
            'name': base.get('name',''),
            'difficulty': base.get('difficulty',''),
            'tags': base.get('tags',''),
            'description': desc,
            'requirements': reqs,
            'pdf_url': pdf_url,
        })
    if out_rows:
        keys = list(out_rows[0].keys())
        with open(OUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in out_rows:
                writer.writerow(r)
        print('saved', len(out_rows), 'to', OUT_CSV)

if __name__ == '__main__':
    main()
