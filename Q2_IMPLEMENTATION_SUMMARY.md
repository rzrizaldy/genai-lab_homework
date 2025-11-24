# Q2: LLM Analysis with RAG - Implementation Summary

## Overview
Implemented RAG (Retrieval-Augmented Generation) system inspired by HW2 to analyze customer reviews using:
- **Semantic Chunking**: Split reviews into sentences
- **Vector Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **FAISS Index**: Fast similarity search
- **OpenAI GPT-4**: Analysis and prompt generation

## Directory Structure

```
final_project/
├── rag_vector_store/              # Vector stores for semantic search
│   ├── baggy_jeans/
│   │   ├── faiss_index.bin        # FAISS index for jeans reviews
│   │   ├── embeddings.npy         # Sentence embeddings
│   │   └── sentences.csv          # Chunked sentences with metadata
│   └── chicken_drumsticks/
│       ├── faiss_index.bin
│       ├── embeddings.npy
│       └── sentences.csv
│
└── review_analysis_output/         # Analysis results
    ├── baggy_jeans/
    │   ├── visual_information.json      # Colors, textures, materials
    │   ├── product_features.json        # Features, pros, cons
    │   ├── image_generation_prompts.json
    │   └── image_generation_prompts.txt
    └── chicken_drumsticks/
        ├── visual_information.json
        ├── product_features.json
        ├── image_generation_prompts.json
        └── image_generation_prompts.txt
```

## Implementation Steps

### Step 1: Setup
- Load OpenAI API key from .env
- Install: sentence-transformers, faiss-cpu, openai, nltk
- Create output directories

### Step 2: Semantic Chunking
- Split reviews into sentences using NLTK
- Preserve metadata (rating, product_name, review_id)
- Filter out very short sentences (<10 chars)

### Step 3: Create Embeddings
- Use SentenceTransformer('all-MiniLM-L6-v2')
- Generate 384-dimensional embeddings
- Convert to numpy arrays

### Step 4: Build FAISS Index
- Create IndexFlatL2 for exact search
- Add embeddings to index
- Save index, embeddings, and sentences

### Step 5: RAG Query System
- `rag_retrieve()`: Find k most relevant sentences
- `rag_llm_analysis()`: Analyze with OpenAI GPT-4o-mini
- Context-aware prompting

### Step 6: Visual Information Module
Queries:
- What colors are mentioned?
- What textures and materials?
- Visual appearance details?

Output: visual_information.json

### Step 7: Product Features Module
Queries:
- Main features and characteristics?
- What customers like most?
- Complaints and issues?
- Unique selling points?

Output: product_features.json

### Step 8: Image Generation Prompts
- Combine visual + feature analysis
- Generate 3 detailed prompts per product
- Optimized for DALL-E/Stable Diffusion

Output: image_generation_prompts.json + .txt

## Key Features

✅ **Semantic Search**: FAISS enables fast similarity search
✅ **Context-Aware**: RAG retrieves relevant sentences for each query
✅ **Structured Output**: JSON files for easy parsing
✅ **Reusable**: Vector stores can be loaded for future queries
✅ **Comprehensive**: 3 analysis modules cover different aspects

## Technical Highlights

- **Chunking Strategy**: Sentence-level (vs character/token chunks)
- **Embedding Model**: all-MiniLM-L6-v2 (fast, good quality)
- **Search Algorithm**: L2 distance (exact nearest neighbor)
- **LLM**: GPT-4o-mini (cost-effective, high quality)
- **k-value**: 15 sentences retrieved per query

## Next: Q3
Ready to use generated prompts for image generation with diffusion models!
