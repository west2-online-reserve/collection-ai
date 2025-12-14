"""
Pytest fixtures for FZU crawler tests.
"""
from typing import cast

import pytest
from lxml import html
from lxml.html import HtmlElement


@pytest.fixture
def sample_announcement_li() -> HtmlElement:
    """Sample li element for announcement entry testing."""
    html_content = """
    <li>
        <span class="doclist_time">2025-11-14</span>
        【教学运行】
        <a href="info/1036/14352.htm" target="_blank" title="关于公布2025-2026学年各学院转专业实施细则的通知">关于公布2025-2026学年各学院转专业实施细则的通知</a>
    </li>
    """
    tree = html.fromstring(html_content)
    li_elements = cast(list[HtmlElement], tree.xpath("//li"))
    return li_elements[0]


@pytest.fixture
def sample_attachment_li() -> HtmlElement:
    """Sample li element for attachment testing."""
    html_content = """
    <li>附件【<a href="/system/_content/download.jsp?urltype=news.DownloadAttachUrl&amp;owner=1744984858&amp;wbfileid=16738452" target="_blank">2501高等数学A（下）期末考-12.7上午.xlsx</a>】已下载<span id="nattach16738452">146</span>次</li>
    """
    tree = html.fromstring(html_content)
    li_elements = cast(list[HtmlElement], tree.xpath("//li"))
    return li_elements[0]


@pytest.fixture
def sample_announcement_list_html():
    """Sample full announcement list page HTML matching the real XPath structure."""
    # XPath: /html/body/div[1]/div[2]/div[2]/div/div/div[3]/div[1]/ul/li
    return """
    <html>
    <body>
        <div>
            <div></div>
            <div>
                <div></div>
                <div>
                    <div>
                        <div>
                            <div></div>
                            <div></div>
                            <div>
                                <div>
                                    <ul>
                                        <li>
                                            <span class="doclist_time">2025-11-14</span>
                                            【教学运行】
                                            <a href="info/1036/14352.htm" target="_blank" title="通知1">通知1</a>
                                        </li>
                                        <li>
                                            <span class="doclist_time">2025-11-13</span>
                                            【考试通知】
                                            <a href="info/1036/14351.htm" target="_blank" title="通知2">通知2</a>
                                        </li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def sample_announcement_detail_html():
    """Sample announcement detail page HTML matching real XPath structure."""
    # Body XPath: //*[@id="vsb_content"]/div
    # Attachments XPath: /html/body/div/div[2]/div[2]/form/div/div[1]/div/ul
    return """
    <html>
    <body>
        <div>
            <div></div>
            <div>
                <div></div>
                <div>
                    <form>
                        <div>
                            <div>
                                <div>
                                    <ul style="list-style-type:none;">
                                        <li>附件【<a href="/system/_content/download.jsp?urltype=news.DownloadAttachUrl&amp;owner=1744984858&amp;wbfileid=16738452" target="_blank">test.xlsx</a>】已下载<span id="nattach16738452">146</span>次</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>
        <div id="vsb_content">
            <div>这是公告正文内容。This is the announcement body content.</div>
        </div>
    </body>
    </html>
    """
