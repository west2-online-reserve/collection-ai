import httpx
import logging
import os
import time
from typing import ClassVar, Any, cast

from summer_ospp_crawler.models import Project, parse_tech_tags, ProjectOutputRequirement

logger = logging.getLogger(__name__)

class OSPPCrawler:
    BASE_URL: ClassVar[str] = "https://summer-ospp.ac.cn"
    API_LIST: ClassVar[str] = "/api/getProList"
    API_DETAIL: ClassVar[str] = "/api/getProDetail"
    PDF_URL_TEMPLATE: ClassVar[str] = "https://summer-ospp.ac.cn/previewPdf/{}"

    client: httpx.Client
    headers: dict[str, str]

    def __init__(self, cookies: str | None = None):
        self.headers = {
            "accept": "*/*",
            "content-type": "application/json",
            "origin": self.BASE_URL,
            "referer": f"{self.BASE_URL}/org/projectlist",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
        }
        if cookies:
            self.headers["cookie"] = cookies
            
        self.client = httpx.Client(headers=self.headers, timeout=10.0, follow_redirects=True)

    def fetch_project_list(self, page_size: int = 50, max_pages: int = -1, limit: int = -1) -> list[Project]:
        projects: list[Project] = []
        page_num = 1
        optimized = False
        
        while True:
            logger.debug(f"Fetching page {page_num}...")
            payload: dict[str, Any] = {  # pyright: ignore[reportExplicitAny]
                "supportLanguage": [],
                "techTag": [],
                "programmingLanguageTag": [],
                "programName": "",
                "difficulty": [],
                "pageNum": str(page_num),
                "pageSize": str(page_size),
                "lang": "zh",
                "orgName": []
            }
            
            try:
                resp = self.client.post(f"{self.BASE_URL}{self.API_LIST}", json=payload)
                _ = resp.raise_for_status()
                data = cast(dict[str, Any], resp.json())  # pyright: ignore[reportExplicitAny]
                
                rows = cast(list[dict[str, Any]], data.get("rows", []))  # pyright: ignore[reportExplicitAny]
                if not rows:
                    break
                
                total = int(cast(int | str, data.get("total", 0)))

                # Optimization: resizing page to fetch all in one go
                if page_num == 1 and not optimized and max_pages == -1:
                    target = limit if limit != -1 else total
                    if target > page_size:
                        logger.info(f"Optimizing: Adjusting page size from {page_size} to {target} to fetch items in one request.")
                        page_size = target
                        optimized = True
                        continue

                for row in rows:
                    project = Project(
                        program_id=str(row.get("programCode")),
                        program_name=str(row.get("programName")),
                        difficulty=str(row.get("difficulty")),
                        tech_tags=parse_tech_tags(str(row.get("techTag"))),
                        org_id=str(row.get("orgId"))
                    )
                    projects.append(project)

                    if limit != -1 and len(projects) >= limit:
                        break

                
                logger.info(f"Fetched {len(rows)} projects. Total: {total}")

                if max_pages != -1 and page_num >= max_pages:
                    break
                    
                if limit != -1 and len(projects) >= limit:
                    break

                if len(projects) >= total:
                    break
                
                page_num += 1
                time.sleep(1) # Be coding citizen

            except Exception as e:
                logger.error(f"Error fetching page {page_num}: {e}")
                break
                
        return projects

    def fetch_project_detail(self, project: Project) -> Project:
        logger.debug(f"Fetching details for project {project.program_id}...")
        payload = {
            "programId": project.program_id,
            "type": "org"
        }
        
        try:
            resp = self.client.post(f"{self.BASE_URL}{self.API_DETAIL}", json=payload)
            _ = resp.raise_for_status()
            data = cast(dict[str, Any], resp.json())  # pyright: ignore[reportExplicitAny]
            
            project.program_desc = str(cast(object, data.get("programDesc", "")))
            project.org_program_id = cast(int | None, data.get("orgProgramId"))
            
            reqs = cast(list[dict[str, Any] | None], data.get("outputRequirement", []))  # pyright: ignore[reportExplicitAny]
            parsed_reqs: list[ProjectOutputRequirement] = []
            if reqs:
                for r in reqs:
                    if r: # Could be null in the list based on curl output
                        parsed_reqs.append(ProjectOutputRequirement(
                            title=str(cast(object, r.get("title", ""))),
                            children=cast(list[Any], r.get("children", []))  # pyright: ignore[reportExplicitAny]
                        ))
            project.output_requirements = parsed_reqs
            
        except Exception as e:
            logger.error(f"Error fetching detail for {project.program_id}: {e}")
            
        return project

    def download_pdf(self, project: Project, save_dir: str):
        if not project.org_program_id:
            logger.warning(f"No orgProgramId for project {project.program_id}, cannot download PDF.")
            return

        pdf_url = self.PDF_URL_TEMPLATE.format(project.org_program_id)
        
        # Check if file exists mainly to avoid re-downloading if interrupted? 
        # But user requested "download", so we will maximize success.
        filename = f"{project.program_id}_{project.program_name.replace('/', '_')}.pdf"
        # Sanitize filename further if needed
        filename = "".join([c for c in filename if c.isalpha() or c.isdigit() or c in (' ', '.', '_', '-')]).rstrip()
        filepath = os.path.join(save_dir, filename)

        logger.debug(f"Downloading PDF from {pdf_url} to {filepath}...")
        try:
            resp = self.client.get(pdf_url)
            _ = resp.raise_for_status()
            
            # Check content type if possible, or just write
            with open(filepath, "wb") as f:
                _ = f.write(resp.content)
                
        except Exception as e:
            logger.error(f"Failed to download PDF for {project.program_id}: {e}")

    def close(self):
        self.client.close()
