# Summer OSPP Crawler

## 快速开始

```bash
uv sync
uv run summer-ospp-crawler
```

该命令会：
1.  自动爬取所有开源之夏项目。
2.  **不**下载 PDF 附件（默认）。
3.  在当前目录下生成 `projects.json` 文件。

## 其他用法

```bash
usage: summer-ospp-crawler [-h] [-o OUTPUT] [--pdf-dir PDF_DIR] [--max-pages MAX_PAGES] [--page-size PAGE_SIZE]
                           [--cookies COOKIES] [--save-pdf] [-v] [-c COUNT] [-j CONCURRENCY]

Summer OSPP Crawler

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   Output JSON file
  --pdf-dir PDF_DIR     Directory to save PDFs
  --max-pages MAX_PAGES
                        Max pages to crawl (-1 for all)
  --page-size PAGE_SIZE
                        Number of items per page
  --cookies COOKIES     Cookies string for requests
  --save-pdf            Download PDFs
  -v, --verbose         Enable verbose output
  -c, --count COUNT     Number of items to crawl
  -j, --concurrency CONCURRENCY
                        Concurrency level
```