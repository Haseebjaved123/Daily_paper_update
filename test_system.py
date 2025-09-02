#!/usr/bin/env python3
"""
Test script for the Daily Paper Update system
"""

import os
import sys
from datetime import datetime
from daily_paper_fetcher import DailyPaperFetcher

def test_paper_fetcher():
    """Test the paper fetcher functionality"""
    print("Testing Daily Paper Fetcher...")
    
    fetcher = DailyPaperFetcher()
    
    # Test fetching papers
    print("1. Testing paper fetching...")
    papers = fetcher.get_today_papers(max_results=10)
    
    if papers:
        print(f"   ✓ Successfully fetched {len(papers)} papers")
        
        # Test paper selection
        print("2. Testing paper selection...")
        selected = fetcher.select_paper(papers)
        
        if selected:
            print(f"   ✓ Selected paper: {selected['title'][:50]}...")
            print(f"   ✓ Domain: {selected['domain']}")
            print(f"   ✓ Authors: {len(selected['authors'])}")
            
            # Test markdown formatting
            print("3. Testing markdown formatting...")
            today = datetime.now().strftime("%Y-%m-%d")
            markdown = fetcher.format_paper_markdown(selected, today)
            
            if markdown and len(markdown) > 100:
                print("   ✓ Markdown formatting successful")
                print(f"   ✓ Generated {len(markdown)} characters")
                
                # Test file path generation
                print("4. Testing file path generation...")
                file_path = fetcher.get_file_path(today)
                print(f"   ✓ File path: {file_path}")
                
                return True
            else:
                print("   ✗ Markdown formatting failed")
                return False
        else:
            print("   ✗ Paper selection failed")
            return False
    else:
        print("   ✗ No papers fetched")
        return False

def create_sample_paper():
    """Create a sample paper for testing"""
    print("\nCreating sample paper...")
    
    fetcher = DailyPaperFetcher()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Create a sample paper
    sample_paper = {
        'title': 'Sample Paper: Testing the Enhanced Daily Paper Update System',
        'authors': ['Test Author 1', 'Test Author 2'],
        'abstract': 'This is a sample abstract for testing the enhanced daily paper update system. It demonstrates how the system formats and stores paper information with figures, insights, and detailed statistics. The abstract contains enough text to meet the minimum length requirements and provides a realistic example of what the system will process. This paper focuses on deep learning and neural networks with attention mechanisms.',
        'arxiv_id': '2025.01001',
        'arxiv_link': 'https://arxiv.org/abs/2025.01001',
        'domain': 'AI',
        'published': '2025-01-01T00:00:00Z',
        'categories': ['cs.AI', 'cs.LG'],
        'word_count': 67,
        'sentence_count': 4
    }
    
    # Format and save
    markdown_content = fetcher.format_paper_markdown(sample_paper, today)
    file_path = fetcher.get_file_path(today)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"✓ Sample paper created at: {file_path}")
    return file_path

def main():
    """Main test function"""
    print("Daily Paper Update System Test")
    print("=" * 40)
    
    # Test the fetcher
    if test_paper_fetcher():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        return False
    
    # Create sample paper
    sample_path = create_sample_paper()
    
    print(f"\nTest completed successfully!")
    print(f"Sample paper created at: {sample_path}")
    print("\nTo test manually, run: python daily_paper_fetcher.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
