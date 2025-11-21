from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any

import httpx
from pandas import DataFrame

from page import Page

BASEURL: str = "https://wiki.biligame.com/blhx/"

client = httpx.Client(base_url=BASEURL, event_hooks={
    "request": [lambda request: print(f"{request.method} {request.url}")],
    "response": [lambda response: None if response.is_success else print(
        f"[Error]{response.request.method} {response.url} ({response.status_code}): \n{response.text}")]
})


def get_page(page: Page) -> None:
    params = {
        "action": "parse",
        "pageid": page.page_id,
        "format": "json",
        "formatversion": "2",
        "maxlag": "5"
    }
    json_response: dict[str, Any] = client.get("/api.php?action=&page=Pet_door&format=json", params=params).json()
    page.source = json_response["parse"]["text"]


def main():
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmpageid": 16858,
        "cmlimit": "500",
        "format": "json",
        "formatversion": "2",
        "maxlag": "5"
    }

    json_response: dict[str, Any] = client.get("/api.php", params=params).json()

    pages: list[Page] = [Page(name=i["title"], page_id=i["pageid"], source="") for i in
                         json_response["query"]["categorymembers"]]

    with ThreadPoolExecutor(max_workers=5) as pool:
        pool.map(get_page, pages)

    df = DataFrame([i.model_dump() for i in pages])
    df.to_csv("pages.csv", index=False)
    print(f"Export {len(df)} records...")


if __name__ == "__main__":
    main()
