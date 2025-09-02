# Daily Paper Update - Setup Guide

This repository automatically fetches and updates one paper abstract every day from arXiv, covering various AI/ML domains.

## 🚀 Quick Start

### 1. Repository Setup
The repository is already configured with:
- ✅ Directory structure for 2025 (all months)
- ✅ Python script for fetching papers
- ✅ GitHub Actions workflow for automation
- ✅ Configuration files

### 2. Manual Testing
To test the system locally:

```bash
# Install dependencies
py -m pip install -r requirements.txt

# Test the system
py test_system.py

# Run daily paper fetcher manually
py daily_paper_fetcher.py
```

### 3. GitHub Actions Setup
The automation is configured to run daily at 6:00 AM UTC. To enable:

1. **Push to GitHub**: Commit and push all files to your repository
2. **Enable Actions**: Go to your repository → Actions tab → Enable workflows
3. **Manual Trigger**: You can manually trigger the workflow from the Actions tab

## 📁 File Structure

```
Daily_paper_update/
├── 2025/                          # Year directory
│   ├── january/                   # Month directories
│   │   ├── 02-01-2025.md         # Daily paper files
│   │   └── ...
│   ├── february/
│   └── ...
├── .github/
│   └── workflows/
│       └── daily-paper-update.yml # GitHub Actions workflow
├── daily_paper_fetcher.py         # Main Python script
├── test_system.py                 # Test script
├── requirements.txt               # Python dependencies
├── config.json                    # Configuration
└── README.md                      # Repository documentation
```

## 🔧 Configuration

### Paper Selection Criteria (`config.json`)
- **Max results**: 50 papers fetched per day
- **Abstract length**: 200-2000 characters
- **Title length**: Minimum 20 characters
- **Authors**: 1-10 authors per paper

### Domains Covered
- **AI**: Artificial Intelligence
- **ML**: Machine Learning
- **NLP**: Natural Language Processing
- **CV**: Computer Vision
- **RL**: Reinforcement Learning
- **Healthcare AI**: Medical AI applications
- **Multimodal**: Cross-domain models
- **Theory**: ML theory and optimization

### Schedule
- **Frequency**: Daily
- **Time**: 6:00 AM UTC
- **Timezone**: UTC

## 📝 Paper Format

Each daily file follows this format:

```markdown
# Date: YYYY-MM-DD

## Paper Title
**Authors**: Author1, Author2, Author3  
**Link**: [arXiv](https://arxiv.org/abs/paper-id)  
**Domain**: NLP/CV/ML/etc  

**Abstract**: 
[Original abstract from the paper]

---
```

## 🛠️ Troubleshooting

### No Papers Found
If the system reports "No papers found":
1. Check arXiv API status
2. Verify date range (looks for papers from yesterday)
3. Check paper selection criteria in `config.json`

### GitHub Actions Not Running
1. Ensure Actions are enabled in repository settings
2. Check workflow file syntax
3. Verify repository permissions

### Local Testing Issues
1. Install Python 3.9+ and pip
2. Install dependencies: `py -m pip install -r requirements.txt`
3. Check internet connection for arXiv API access

## 🔄 Manual Updates

To manually add a paper:

1. Create a new file: `2025/[month]/[DD-MM-YYYY].md`
2. Follow the format above
3. Commit and push to repository

## 📊 Monitoring

The system will:
- ✅ Create daily paper files automatically
- ✅ Commit changes with descriptive messages
- ✅ Create GitHub issues if no papers are found
- ✅ Log all activities in GitHub Actions

## 🎯 Next Steps

1. **Push to GitHub**: Commit all files and push to your repository
2. **Enable Actions**: Go to Actions tab and enable the workflow
3. **Monitor**: Check the Actions tab for daily runs
4. **Customize**: Modify `config.json` for different selection criteria

## 📞 Support

If you encounter issues:
1. Check the GitHub Actions logs
2. Review the configuration files
3. Test locally with `py test_system.py`
4. Create an issue in the repository

---

**Note**: The system is designed to be robust and will handle edge cases like weekends, holidays, and API outages gracefully.
