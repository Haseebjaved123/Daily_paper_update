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
import urllib.parse
from pathlib import Path

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
        """Fetch recent papers from arXiv (last 6 hours for 2-hourly updates)"""
        # Get papers from the last 6 hours to ensure we have fresh content
        now = datetime.now()
        six_hours_ago = now - timedelta(hours=6)
        
        # Format dates for arXiv API
        start_date = six_hours_ago.strftime("%Y%m%d%H%M")
        end_date = now.strftime("%Y%m%d%H%M")
        
        # Search for recent papers in our target categories
        categories = list(self.domains.keys())
        category_query = " OR ".join([f"cat:{cat}" for cat in categories])
        
        query_params = {
            'search_query': f"({category_query}) AND submittedDate:[{start_date} TO {end_date}]",
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
            
            # Get all categories
            categories = []
            for cat in entry.findall('{http://arxiv.org/schemas/atom}category'):
                categories.append(cat.get('term'))
            
            # Clean up title and abstract
            title = re.sub(r'\s+', ' ', title)
            summary = re.sub(r'\s+', ' ', summary)
            
            # Extract key information from abstract
            word_count = len(summary.split())
            sentence_count = len([s for s in summary.split('.') if s.strip()])
            
            return {
                'title': title,
                'authors': authors,
                'abstract': summary,
                'arxiv_id': arxiv_id,
                'arxiv_link': arxiv_link,
                'domain': domain,
                'published': published,
                'categories': categories,
                'word_count': word_count,
                'sentence_count': sentence_count
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
    
    def extract_first_figure(self, arxiv_id: str) -> Optional[str]:
        """Extract the first figure from arXiv paper"""
        try:
            # Try to get the PDF and extract first figure
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # For now, we'll use a placeholder approach
            # In a full implementation, you'd use PDF parsing libraries
            # This is a simplified version that creates a figure placeholder
            
            # Check if we can access the paper's source
            response = requests.head(pdf_url, timeout=10)
            if response.status_code == 200:
                # Create a figure placeholder with paper info
                figure_placeholder = f"![Paper Figure](https://img.shields.io/badge/Figure-{arxiv_id}-blue?style=for-the-badge&logo=arxiv)"
                return figure_placeholder
            else:
                return None
                
        except Exception as e:
            print(f"Error extracting figure for {arxiv_id}: {e}")
            return None
    
    def get_paper_insights(self, paper: Dict) -> Dict:
        """Extract insights and key information from the paper"""
        abstract = paper['abstract']
        
        # Extract key phrases (simplified approach)
        key_phrases = []
        if 'deep learning' in abstract.lower():
            key_phrases.append('Deep Learning')
        if 'neural network' in abstract.lower():
            key_phrases.append('Neural Networks')
        if 'transformer' in abstract.lower():
            key_phrases.append('Transformers')
        if 'attention' in abstract.lower():
            key_phrases.append('Attention Mechanism')
        if 'reinforcement learning' in abstract.lower():
            key_phrases.append('Reinforcement Learning')
        if 'computer vision' in abstract.lower():
            key_phrases.append('Computer Vision')
        if 'natural language' in abstract.lower():
            key_phrases.append('Natural Language Processing')
        
        # Extract methodology mentions
        methodologies = []
        if 'benchmark' in abstract.lower():
            methodologies.append('Benchmarking')
        if 'evaluation' in abstract.lower():
            methodologies.append('Evaluation')
        if 'experiment' in abstract.lower():
            methodologies.append('Experimentation')
        if 'dataset' in abstract.lower():
            methodologies.append('Dataset Creation')
        
        return {
            'key_phrases': key_phrases[:5],  # Top 5 key phrases
            'methodologies': methodologies[:3],  # Top 3 methodologies
            'complexity': 'High' if paper['word_count'] > 300 else 'Medium' if paper['word_count'] > 200 else 'Low'
        }
    
    def format_paper_markdown(self, paper: Dict, date: str) -> str:
        """Format paper into markdown following the repository format"""
        authors_str = ", ".join(paper['authors'])
        
        # Get paper insights
        insights = self.get_paper_insights(paper)
        
        # Try to extract first figure
        figure = self.extract_first_figure(paper['arxiv_id'])
        
        # Format categories
        categories_str = ", ".join(paper['categories'][:3]) if paper['categories'] else paper['domain']
        
        # Create enhanced markdown
        markdown = f"""# 📅 Date: {date}

## 📄 {paper['title']}

### 👥 Authors
{authors_str}

### 🔗 Links
- **arXiv**: [{paper['arxiv_id']}]({paper['arxiv_link']})
- **PDF**: [Download PDF](https://arxiv.org/pdf/{paper['arxiv_id']}.pdf)

### 🏷️ Classification
- **Primary Domain**: {paper['domain']}
- **Categories**: {categories_str}
- **Complexity**: {insights['complexity']}

### 📊 Paper Statistics
- **Word Count**: {paper['word_count']} words
- **Sentences**: {paper['sentence_count']} sentences
- **Authors**: {len(paper['authors'])} researchers

### 🔍 Key Topics
{', '.join(insights['key_phrases']) if insights['key_phrases'] else 'General AI/ML'}

### 🛠️ Methodologies
{', '.join(insights['methodologies']) if insights['methodologies'] else 'Research Paper'}

### 🖼️ Figure
{figure if figure else '![No Figure Available](https://img.shields.io/badge/Figure-Not_Available-lightgrey?style=for-the-badge)'}

### 📝 Abstract
{paper['abstract']}

---
*Generated by Daily Paper Update System*
"""
        return markdown
    
    def get_file_path(self, date: str) -> str:
        """Get the file path for the given date and time"""
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        year = date_obj.year
        month = date_obj.strftime("%B").lower()
        
        # Get current time for 2-hourly updates
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        
        # Create time-based filename
        time_str = f"{hour:02d}-{minute:02d}"
        day = date_obj.strftime("%d-%m-%Y")
        
        return f"{year}/{month}/{day}_{time_str}.md"
    
    def update_daily_paper(self) -> bool:
        """Main function to update paper (every 2 hours)"""
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = self.get_file_path(today)
        
        # Check if file already exists
        if os.path.exists(file_path):
            print(f"Paper for {today} at current time already exists at {file_path}")
            return True
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Fetch recent papers (last 6 hours)
        print("Fetching recent papers (last 6 hours)...")
        papers = self.get_today_papers()
        
        if not papers:
            print("No recent papers found in the last 6 hours")
            # Fallback: try last 24 hours
            print("Trying to fetch papers from last 24 hours...")
            papers = self.get_fallback_papers()
            
        if not papers:
            print("No papers found")
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
        
        print(f"Successfully updated paper: {file_path}")
        print(f"Selected paper: {selected_paper['title']}")
        print(f"Domain: {selected_paper['domain']}")
        print(f"Time: {datetime.now().strftime('%H:%M UTC')}")
        
        return True
    
    def get_fallback_papers(self, max_results: int = 50) -> List[Dict]:
        """Fallback method to get papers from last 24 hours"""
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        
        start_date = yesterday.strftime("%Y%m%d%H%M")
        end_date = now.strftime("%Y%m%d%H%M")
        
        categories = list(self.domains.keys())
        category_query = " OR ".join([f"cat:{cat}" for cat in categories])
        
        query_params = {
            'search_query': f"({category_query}) AND submittedDate:[{start_date} TO {end_date}]",
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.arxiv_base_url, params=query_params, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                paper = self._parse_paper_entry(entry)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Error fetching fallback papers: {e}")
            return []

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
