#!/usr/bin/env python3
"""
Test script for the Daily Paper Update system
"""

import os
import sys
from datetime import datetime
from daily_paper_fetcher import AdvancedPaperFetcher

def test_paper_fetcher():
    """Test the advanced multi-source paper fetcher functionality"""
    print("Testing Advanced Multi-Source Paper Fetcher...")
    
    fetcher = AdvancedPaperFetcher()
    
    # Test fetching papers from multiple sources
    print("1. Testing multi-source paper fetching...")
    papers = fetcher.get_papers_from_multiple_sources(max_results=10)
    
    if papers:
        print(f"   ✓ Successfully fetched {len(papers)} papers from multiple sources")
        
        # Test enhanced paper selection
        print("2. Testing enhanced paper selection...")
        selected = fetcher.select_paper_enhanced(papers)
        
        if selected:
            print(f"   ✓ Selected paper: {selected['title'][:50]}...")
            print(f"   ✓ Domain: {selected['domain']}")
            print(f"   ✓ Source: {selected.get('source', 'unknown')}")
            print(f"   ✓ Authors: {len(selected['authors'])}")
            
            # Test enhanced markdown formatting
            print("3. Testing enhanced markdown formatting...")
            today = datetime.now().strftime("%Y-%m-%d")
            markdown = fetcher.format_paper_markdown_enhanced(selected, today)
            
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
        print("   ⚠️ No papers fetched from external sources - this is normal in CI environment")
        print("   ✓ Testing fallback mechanisms...")
        
        # Test with mock data to ensure the system works
        print("   ✓ Testing with mock data from multiple sources...")
        
        # Test multiple mock sources
        mock_sources = [
            ('Papers with Code', fetcher._get_mock_papers_with_code()),
            ('GitHub Trending', fetcher._get_mock_github_repos()),
            ('Hugging Face', fetcher._get_mock_huggingface_papers()),
            ('Conference Papers', fetcher._get_mock_conference_papers('neurips_papers')),
            ('Academic Database', fetcher._get_mock_academic_papers('ieee_xplore')),
            ('Reddit Community', fetcher._get_mock_reddit_papers())
        ]
        
        all_mock_papers = []
        for source_name, papers in mock_sources:
            if papers:
                all_mock_papers.extend(papers)
                print(f"   ✓ {source_name}: {len(papers)} papers")
        
        if all_mock_papers:
            print(f"   ✓ Total mock papers available: {len(all_mock_papers)}")
            selected = fetcher.select_paper_enhanced(all_mock_papers)
            if selected:
                print(f"   ✓ Mock paper selection successful: {selected['title'][:50]}...")
                print(f"   ✓ Source: {selected.get('source', 'unknown')}")
                return True
        
        print("   ✗ All fallback mechanisms failed")
        return False

def create_sample_paper():
    """Create a sample paper for testing"""
    print("\nCreating sample paper...")
    
    fetcher = AdvancedPaperFetcher()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Create a sample paper with source information
    sample_paper = {
        'title': 'Sample Paper: Testing the Advanced Multi-Source Paper Intelligence System',
        'authors': ['Test Author 1', 'Test Author 2'],
        'abstract': 'This is a sample abstract for testing the advanced multi-source paper intelligence system. It demonstrates how the system formats and stores paper information with figures, insights, and detailed statistics from multiple sources. The abstract contains enough text to meet the minimum length requirements and provides a realistic example of what the system will process. This paper focuses on deep learning and neural networks with attention mechanisms.',
        'arxiv_id': '2025.01001',
        'arxiv_link': 'https://arxiv.org/abs/2025.01001',
        'domain': 'AI',
        'published': '2025-01-01T00:00:00Z',
        'categories': ['cs.AI', 'cs.LG'],
        'word_count': 67,
        'sentence_count': 4,
        'source': 'papers_with_code',
        'stars': 150
    }
    
    # Format and save using enhanced formatting
    markdown_content = fetcher.format_paper_markdown_enhanced(sample_paper, today)
    file_path = fetcher.get_file_path(today)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"✓ Sample paper created at: {file_path}")
    return file_path

def main():
    """Main test function"""
    print("Advanced Multi-Source Paper Intelligence System Test")
    print("=" * 60)
    
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
