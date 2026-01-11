import requests
from bs4 import BeautifulSoup
from collections import deque
import json
import time
import os
from urllib.parse import urljoin, urlparse
from agents.base import BaseAgent
import pandas as pd

class ScraperAgent(BaseAgent):
    def __init__(self, output_dir="data"):
        super().__init__("ScraperAgent")
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def is_valid_url(self, url, base_domain):
        parsed = urlparse(url)
        return bool(parsed.netloc) and parsed.netloc == base_domain

    def clean_text(self, text):
        return " ".join(text.split())

    def get_domain_name(self, url):
        return urlparse(url).netloc.replace("www.", "")

    def scrape(self, start_urls, max_pages=50, progress_callback=None):
        """
        Scrapes a list of websites.
        start_urls: List of URL strings.
        max_pages: Max pages per site.
        progress_callback: Optional function(status_msg) to update UI.
        """
        if isinstance(start_urls, str):
            start_urls = [start_urls]

        results = {}

        for start_url in start_urls:
            domain = self.get_domain_name(start_url)
            self.log(f"Starting scrape for {domain}")
            if progress_callback: progress_callback(f"Starting crawl of {domain}...")

            pages_data = self._crawl_domain(start_url, max_pages, progress_callback)
            
            # Save to disk
            domain_file = os.path.join(self.output_dir, f"{domain}.json")
            with open(domain_file, 'w', encoding='utf-8') as f:
                json.dump(pages_data, f, indent=2)
            
            results[domain] = domain_file
            if progress_callback: progress_callback(f"Completed {domain}. Saved {len(pages_data)} pages.")

        return results

    def _crawl_domain(self, start_url, max_pages, progress_callback):
        visited = set()
        queue = deque([start_url])
        pages_data = []
        base_domain = urlparse(start_url).netloc

        while queue and len(visited) < max_pages:
            url = queue.popleft()
            if url in visited: continue
            visited.add(url)

            try:
                response = requests.get(url, timeout=5)
                if response.status_code != 200: continue

                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Cleanup
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()

                text = self.clean_text(soup.get_text(separator=' '))
                title = soup.title.string if soup.title else url

                if len(text) > 100:
                    pages_data.append({
                        "url": url,
                        "title": self.clean_text(title),
                        "content": text
                    })
                    if progress_callback and len(pages_data) % 5 == 0:
                        progress_callback(f"Scraped {len(pages_data)} pages from {base_domain}...")

                # Find and store links
                internal_links = []
                for link in soup.find_all('a', href=True):
                    full_url = urljoin(url, link['href']).split('#')[0]
                    
                    if self.is_valid_url(full_url, base_domain):
                        internal_links.append(full_url)
                        if full_url not in visited:
                            queue.append(full_url)
                
                # Update page data with links
                pages_data[-1]["links"] = list(set(internal_links)) # unique links
                
                time.sleep(0.1) 

            except Exception as e:
                self.log(f"Error {url}: {e}")

        return pages_data
