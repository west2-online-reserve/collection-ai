import argparse
import logging
import json
import os

from rich.logging import RichHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
from summer_ospp_crawler.crawler import OSPPCrawler
from summer_ospp_crawler.models import Project
from typing import cast, Protocol, Any

logging.basicConfig(
    level=logging.INFO, 
    format="%(message)s", 
    datefmt="[%X]", 
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("summer_ospp_crawler.main")

class Args(Protocol):
    output: str
    pdf_dir: str
    max_pages: int
    page_size: int
    cookies: str | None
    save_pdf: bool
    verbose: bool
    count: int
    concurrency: int

def process_project(crawler: OSPPCrawler, project: Project, save_pdf: bool, pdf_dir: str):
    try:
        _ = crawler.fetch_project_detail(project)
        if save_pdf:
            crawler.download_pdf(project, pdf_dir)
    except Exception as e:
        logger.error(f"Error processing project {project.program_id}: {e}")
    return project

def main():
    parser = argparse.ArgumentParser(description="Summer OSPP Crawler")
    _ = parser.add_argument("-o", "--output", default="projects.json", help="Output JSON file")
    _ = parser.add_argument("--pdf-dir", default="pdfs", help="Directory to save PDFs")
    _ = parser.add_argument("--max-pages", type=int, default=-1, help="Max pages to crawl (-1 for all)")
    _ = parser.add_argument("--page-size", type=int, default=50, help="Number of items per page")
    _ = parser.add_argument("--cookies", help="Cookies string for requests")
    _ = parser.add_argument("--save-pdf", action="store_true", help="Download PDFs")
    _ = parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    _ = parser.add_argument("-c", "--count", type=int, default=-1, help="Number of items to crawl")
    _ = parser.add_argument("-j", "--concurrency", type=int, default=4, help="Concurrency level")
    
    # Parse args and cast to typed Namespace
    args = cast(Args, cast(Any, parser.parse_args())) # pyright: ignore[reportExplicitAny]
    
    if args.verbose:
        logging.getLogger("summer_ospp_crawler").setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("httpcore").setLevel(logging.INFO)
    else:
        logging.getLogger("summer_ospp_crawler").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        
    if args.save_pdf and not os.path.exists(args.pdf_dir):
        os.makedirs(args.pdf_dir)
        
    crawler = OSPPCrawler(cookies=args.cookies)
    
    try:
        logger.info("Starting crawl...")
        projects = crawler.fetch_project_list(page_size=args.page_size, max_pages=args.max_pages, limit=args.count)
        logger.info(f"Fetched list of {len(projects)} projects.")
        
        if args.concurrency > 1:
            logger.info(f"Fetching details with concurrency {args.concurrency}...")
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {executor.submit(process_project, crawler, p, args.save_pdf, args.pdf_dir): p for p in projects}
                
                count = 0
                total = len(projects)
                for future in as_completed(futures):
                    count += 1
                    try:
                        p = future.result()
                        logger.debug(f"Processed {count}/{total}: {p.program_name}")
                    except Exception as e:
                        logger.error(f"Task failed: {e}")
        else:
            logger.info("Fetching details...")
            for i, project in enumerate(projects):
                logger.debug(f"Processing {i+1}/{len(projects)}: {project.program_name}")
                _ = crawler.fetch_project_detail(project)
                
                if args.save_pdf:
                    crawler.download_pdf(project, args.pdf_dir)
                
        logger.info(f"Saving results to {args.output}...")
        projects.sort(key=lambda p: p.org_program_id if p.org_program_id is not None else -1)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([p.to_dict() for p in projects], f, ensure_ascii=False, indent=2)
            
        logger.info("Done!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        crawler.close()

if __name__ == "__main__":
    main()