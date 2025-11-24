# How to Run Q2: LLM Analysis with RAG

## Prerequisites
1. `.env` file with `OPENAI_API_KEY=your-key-here`
2. Reviews collected in `parsed_data/` (from Q1)

## Step-by-Step Execution

### 1. Install Dependencies (First Time Only)
```bash
pip install sentence-transformers faiss-cpu openai python-dotenv nltk
```

### 2. Run Notebook Cells in Order
Open `final_project.ipynb` and run:

- **Q2 STEP 1**: Setup (load libraries, API keys)
- **Q2 STEP 2**: Define semantic chunking functions
- **Q2 STEP 3**: Define FAISS vector store functions  
- **Q2 STEP 4**: Define RAG query functions
- **Q2 STEP 5**: Process both products (creates vector stores)
  - This will take 2-3 minutes
  - Downloads embedding model on first run
- **Q2 STEP 6**: Extract visual information
  - Makes OpenAI API calls (~8 requests)
- **Q2 STEP 7**: Extract product features
  - Makes OpenAI API calls (~8 requests)
- **Q2 STEP 8**: Generate image prompts
  - Makes OpenAI API calls (~2 requests)
- **Q2 STEP 9**: Summary and verification

## Expected Outputs

### Directory: `rag_vector_store/`
- `baggy_jeans/faiss_index.bin` (~50KB)
- `baggy_jeans/embeddings.npy` (~100KB)
- `baggy_jeans/sentences.csv` (~30KB)
- `chicken_drumsticks/faiss_index.bin` (~150KB)
- `chicken_drumsticks/embeddings.npy` (~300KB)
- `chicken_drumsticks/sentences.csv` (~80KB)

### Directory: `review_analysis_output/`
Each product folder contains:
- `visual_information.json` - Colors, textures, visual details
- `product_features.json` - Features, pros, cons
- `image_generation_prompts.json` - 3 prompts for Q3
- `image_generation_prompts.txt` - Same prompts in readable format

## Estimated Costs
- Embedding model: Free (runs locally)
- OpenAI API: ~$0.10-0.20 total
  - Visual extraction: ~2,000 tokens per product
  - Features extraction: ~2,000 tokens per product
  - Prompt generation: ~1,500 tokens per product
  - Total: ~11,000 tokens @ gpt-4o-mini

## Estimated Time
- First run: ~5 minutes (downloads model)
- Subsequent runs: ~3 minutes

## Troubleshooting

### Error: "OpenAI API key not found"
- Check `.env` file exists in project root
- Verify `OPENAI_API_KEY` is set correctly

### Error: "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### Error: "No module named 'faiss'"
```bash
pip install faiss-cpu
```

### Slow embedding generation?
- Normal on first run (downloads ~90MB model)
- Subsequent runs use cached model
- M1/M2 Mac: ~30 seconds
- Other systems: ~60 seconds

## Outputs Ready for Q3
After Q2 completes, you'll have:
✅ 6 image generation prompts (3 per product)
✅ Detailed visual descriptions
✅ Product feature summaries
✅ Ready to use in DALL-E or Stable Diffusion
