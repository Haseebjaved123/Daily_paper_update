#!/usr/bin/env python3
"""
Daily Paper Update Fetcher
Automatically fetches and formats one paper abstract per day from arXiv
"""

import requests
import xml.etree.ElementTree as ET
import json
import os
from datetime import datetime, timedelta
import random
import re
from typing import Dict, List, Optional

class DailyPaperFetcher:
    def __init__(self):
        self.arxiv_base_url = "http://export.arxiv.org/api/query"
        self.domains = {
            'cs.AI': 'AI',
            'cs.LG': 'ML',
            'cs.CL': 'NLP', 
            'cs.CV': 'CV',
            'cs.RO': 'RL',
            'cs.HC': 'Healthcare AI',
            'cs.MM': 'Multimodal',
            'stat.ML': 'Theory',
            'cs.NE': 'Neural Networks',
            'cs.IR': 'Information Retrieval'
        }
        
    def get_today_papers(self, max_results: int = 50) -> List[Dict]:
        """Fetch today's papers from arXiv"""
        # Get papers from the last 2 days to ensure we have recent content
        yesterday = datetime.now() - timedelta(days=1)
        date_str = yesterday.strftime("%Y%m%d")
        
        # Search for recent papers in our target categories
        categories = list(self.domains.keys())
        category_query = " OR ".join([f"cat:{cat}" for cat in categories])
        
        query_params = {
            'search_query': f"({category_query}) AND submittedDate:[{date_str}0000 TO {date_str}2359]",
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.arxiv_base_url, params=query_params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                paper = self._parse_paper_entry(entry)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Error fetching papers: {e}")
            return []
    
    def _parse_paper_entry(self, entry) -> Optional[Dict]:
        """Parse individual paper entry from arXiv XML"""
        try:
            # Extract paper details
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
            published = entry.find('{http://www.w3.org/2005/Atom}published').text
            
            # Get authors
            authors = []
            for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                name = author.find('{http://www.w3.org/2005/Atom}name').text
                authors.append(name)
            
            # Get arXiv ID and link
            arxiv_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
            arxiv_link = f"https://arxiv.org/abs/{arxiv_id}"
            
            # Get primary category
            primary_category = entry.find('{http://arxiv.org/schemas/atom}primary_category')
            category = primary_category.get('term') if primary_category is not None else 'cs.AI'
            domain = self.domains.get(category, 'AI')
            
            # Clean up title and abstract
            title = re.sub(r'\s+', ' ', title)
            summary = re.sub(r'\s+', ' ', summary)
            
            return {
                'title': title,
                'authors': authors,
                'abstract': summary,
                'arxiv_id': arxiv_id,
                'arxiv_link': arxiv_link,
                'domain': domain,
                'published': published
            }
            
        except Exception as e:
            print(f"Error parsing paper entry: {e}")
            return None
    
    def select_paper(self, papers: List[Dict]) -> Optional[Dict]:
        """Select one paper from the list using intelligent criteria"""
        if not papers:
            return None
        
        # Filter papers by quality indicators
        quality_papers = []
        for paper in papers:
            # Skip papers that are too short or too long
            if len(paper['abstract']) < 200 or len(paper['abstract']) > 2000:
                continue
            
            # Skip papers with very short titles
            if len(paper['title']) < 20:
                continue
                
            # Prefer papers with reasonable number of authors (1-10)
            if 1 <= len(paper['authors']) <= 10:
                quality_papers.append(paper)
        
        if not quality_papers:
            quality_papers = papers
        
        # Select randomly from quality papers to ensure variety
        return random.choice(quality_papers)
    
    def format_paper_markdown(self, paper: Dict, date: str) -> str:
        """Format paper into markdown following the repository format"""
        authors_str = ", ".join(paper['authors'])
        
        markdown = f"""# Date: {date}

## {paper['title']}
**Authors**: {authors_str}  
**Link**: [arXiv]({paper['arxiv_link']})  
**Domain**: {paper['domain']}  

**Abstract**: 
{paper['abstract']}

---
"""
        return markdown
    
    def get_file_path(self, date: str) -> str:
        """Get the file path for the given date"""
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        year = date_obj.year
        month = date_obj.strftime("%B").lower()
        day = date_obj.strftime("%d-%m-%Y")
        
        return f"{year}/{month}/{day}.md"
    
    def update_daily_paper(self) -> bool:
        """Main function to update daily paper"""
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = self.get_file_path(today)
        
        # Check if file already exists
        if os.path.exists(file_path):
            print(f"Paper for {today} already exists at {file_path}")
            return True
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Fetch papers
        print("Fetching today's papers...")
        papers = self.get_today_papers()
        
        if not papers:
            print("No papers found for today")
            return False
        
        # Select one paper
        selected_paper = self.select_paper(papers)
        if not selected_paper:
            print("No suitable paper found")
            return False
        
        # Format and save
        markdown_content = self.format_paper_markdown(selected_paper, today)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"Successfully updated daily paper: {file_path}")
        print(f"Selected paper: {selected_paper['title']}")
        print(f"Domain: {selected_paper['domain']}")
        
        return True

def main():
    """Main execution function"""
    fetcher = DailyPaperFetcher()
    success = fetcher.update_daily_paper()
    
    if success:
        print("Daily paper update completed successfully!")
    else:
        print("Daily paper update failed!")
        exit(1)

if __name__ == "__main__":
    main()
