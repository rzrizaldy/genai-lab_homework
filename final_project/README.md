# Final Project: Generating Product Images from Customer Reviews

CMU 94-844 Generative AI Lab (Fall 2025)

## Project Overview

This project explores the intersection of text and image generation by:
1. **Q1**: Collecting product reviews using web scraping (Selenium)
2. **Q2**: Analyzing reviews with LLMs to extract visual and sentiment information
3. **Q3**: Generating product images using diffusion models based on extracted information

## Files

- **final_project.ipynb** - Main notebook with all code (step-by-step)
- **data/** - Directory for collected product data (CSV files)
- **94844-Final Project-F25-v2.pdf** - Project requirements

## Selected Products

1. **Baggy Jeans** (Clothing) - Amazon
   - Rich visual descriptions in reviews
   - Link: https://a.co/d/1Rj5tqt

2. **Chicken Drumsticks** (Food) - Walmart
   - Quality/freshness focused reviews
   - Link: https://www.walmart.com/ip/158751412

3. **[To be added]** - Different category

## Setup

### Install Dependencies

```bash
pip install selenium webdriver-manager beautifulsoup4 pandas requests
pip install openai anthropic  # For Q2 LLM analysis
```

### Usage

1. Open `final_project.ipynb` in Jupyter
2. Run cells sequentially
3. For Q1 data collection:
   - Browser will open automatically
   - Log in to Amazon/Walmart when prompted
   - Press ENTER in notebook to continue
   - Script scrapes data automatically

## Data Collection Method

**Selenium WebDriver with Manual Login:**
- Opens real browser (not headless)
- Allows manual login to bypass anti-bot protection
- Scrapes multiple pages of reviews
- Saves to CSV format

## Project Structure

```
final_project/
├── final_project.ipynb       # Main notebook
├── README.md                  # This file
├── data/                      # Collected data
│   ├── baggy_jeans_info.csv
│   ├── baggy_jeans_reviews.csv
│   ├── chicken_drumsticks_info.csv
│   └── chicken_drumsticks_reviews.csv
└── 94844-Final Project-F25-v2.pdf
```

## Progress Checklist

- [x] Q1 Setup: Selenium scraping functions
- [ ] Q1 Data: Collect Product 1 (Baggy Jeans)
- [ ] Q1 Data: Collect Product 2 (Chicken Drumsticks)
- [ ] Q1 Data: Select and collect Product 3
- [ ] Q2 Setup: Configure LLM API
- [ ] Q2 Analysis: Visual information extraction
- [ ] Q2 Analysis: Sentiment analysis
- [ ] Q2 Analysis: Feature extraction
- [ ] Q3 Setup: Configure image generation API
- [ ] Q3 Generation: Generate images (Model 1)
- [ ] Q3 Generation: Generate images (Model 2)
- [ ] Q3 Analysis: Compare with actual products
- [ ] Final Report: Write methodology and findings
- [ ] Final Presentation: Create slides

## Notes

- **Anti-bot Protection**: Amazon and Walmart block basic scrapers. This notebook uses Selenium with manual login to bypass this.
- **API Keys Needed**:
  - LLM API (OpenAI or Anthropic) for Q2
  - Image generation API (DALL-E, Stable Diffusion) for Q3
- **Review Count**: Aim for 30-50 reviews per product for better LLM analysis

## Deadlines

- Final Presentation: 12/01 and 12/03
- Final Report: 12/05
