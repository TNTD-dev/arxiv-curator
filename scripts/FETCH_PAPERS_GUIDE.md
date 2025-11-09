# 📥 Hướng Dẫn Fetch Papers từ arXiv

## 🎯 Tổng Quan

Có 2 scripts để fetch papers:
1. **`fetch_and_index_papers.py`** - Full pipeline (fetch + process + index)
2. **`fetch_papers_quick.py`** - Quick fetch (chỉ download)

---

## 🚀 Quick Start

### Option 1: Full Pipeline (Recommended)

**Fetch 15 papers mới nhất (AI/ML/NLP) và index luôn**:
```bash
python scripts/fetch_and_index_papers.py
```

**Kết quả**:
- ✅ Download 15 papers mới nhất
- ✅ Process với Docling
- ✅ Index vào vector store + BM25
- ✅ Ready to use!

---

### Option 2: Quick Fetch

**Chỉ download papers (không process)**:
```bash
python scripts/fetch_papers_quick.py
```

**Interactive menu**:
```
1. AI/ML/NLP papers (15 papers)
2. Transformer papers (10 papers)
3. LLM papers (10 papers)
0. Custom query
```

Sau đó index riêng:
```bash
python scripts/fetch_and_index_papers.py --index-only
```

---

## 📖 Chi Tiết Sử Dụng

### `fetch_and_index_papers.py`

#### 1. **Fetch mặc định (15 papers AI/ML/NLP mới nhất)**
```bash
python scripts/fetch_and_index_papers.py
```

#### 2. **Fetch nhiều hơn**
```bash
python scripts/fetch_and_index_papers.py --max-results 20
```

#### 3. **Custom query**
```bash
# Transformer papers
python scripts/fetch_and_index_papers.py --query "all:transformer" --max-results 10

# LLM papers
python scripts/fetch_and_index_papers.py --query "all:GPT OR all:LLM" --max-results 15

# Computer Vision papers
python scripts/fetch_and_index_papers.py --query "cat:cs.CV" --max-results 10

# Specific topic
python scripts/fetch_and_index_papers.py --query "all:BERT AND all:NLP" --max-results 5
```

#### 4. **Sort options**
```bash
# Newest papers (default)
python scripts/fetch_and_index_papers.py --sort-by submitted

# Most relevant
python scripts/fetch_and_index_papers.py --sort-by relevance

# Recently updated
python scripts/fetch_and_index_papers.py --sort-by updated
```

#### 5. **Only fetch (no indexing)**
```bash
python scripts/fetch_and_index_papers.py --fetch-only --max-results 20
```

#### 6. **Only index (already have PDFs)**
```bash
python scripts/fetch_and_index_papers.py --index-only
```

#### 7. **Force rebuild index**
```bash
python scripts/fetch_and_index_papers.py --force-rebuild
```

---

### `fetch_papers_quick.py`

**Interactive mode**:
```bash
python scripts/fetch_papers_quick.py
```

Chọn option:
- `1` - AI/ML/NLP (15 papers)
- `2` - Transformer (10 papers)
- `3` - LLM (10 papers)
- `0` - Custom query

---

## 🔍 arXiv Query Examples

### By Category
```bash
# AI
--query "cat:cs.AI"

# Machine Learning
--query "cat:cs.LG"

# Natural Language Processing
--query "cat:cs.CL"

# Computer Vision
--query "cat:cs.CV"

# Combined
--query "cat:cs.AI OR cat:cs.LG OR cat:cs.CL"
```

### By Keywords
```bash
# Transformers
--query "all:transformer"

# Attention mechanisms
--query "all:attention"

# BERT
--query "all:BERT"

# LLMs
--query "all:GPT OR all:LLM OR all:large language model"

# Combined
--query "all:transformer AND (all:NLP OR all:text)"
```

### By Author
```bash
--query "au:Vaswani"  # Attention is All You Need author
```

### By Date Range
```bash
# Papers from 2024
--query "cat:cs.AI AND submittedDate:[202401* TO 202412*]"
```

### Complex Queries
```bash
# Transformers in NLP, recent
--query "all:transformer AND cat:cs.CL"

# LLM papers, exclude specific
--query "all:LLM AND NOT all:survey"
```

---

## 📊 Recommended Configurations

### For General AI/ML Research (Current Default)
```bash
Query: cat:cs.AI OR cat:cs.LG OR cat:cs.CL
Max: 15 papers
Sort: submitted (newest)
```

### For Transformer Focus
```bash
python scripts/fetch_and_index_papers.py \
  --query "all:transformer OR all:attention OR all:BERT" \
  --max-results 20
```

### For LLM Focus
```bash
python scripts/fetch_and_index_papers.py \
  --query "all:GPT OR all:LLM OR all:language model" \
  --max-results 20
```

### For Diverse Coverage
```bash
# Get 10 from each category
python scripts/fetch_and_index_papers.py --query "cat:cs.AI" --max-results 10
python scripts/fetch_and_index_papers.py --query "cat:cs.LG" --max-results 10
python scripts/fetch_and_index_papers.py --query "cat:cs.CL" --max-results 10
```

---

## 💡 Tips

### 1. **Start Small**
```bash
# Test với 5 papers first
python scripts/fetch_and_index_papers.py --max-results 5
```

### 2. **Check Before Download**
```bash
# Quick fetch shows papers before download
python scripts/fetch_papers_quick.py
```

### 3. **Incremental Updates**
```bash
# Fetch new papers periodically
python scripts/fetch_and_index_papers.py --max-results 5 --sort-by submitted
```

### 4. **Rebuild When Needed**
```bash
# If index corrupted
python scripts/fetch_and_index_papers.py --index-only --force-rebuild
```

---

## 🔧 Troubleshooting

### Issue: SSL Error on Windows
**Solution**: Already handled in ArxivClient (SSL verification disabled for arXiv)

### Issue: Download Slow
**Solution**: 
- Reduce `--max-results`
- Check internet connection
- arXiv may rate limit (wait a bit)

### Issue: Indexing Fails
**Solution**:
```bash
# Clear and rebuild
rm -rf data/vector_db/*
python scripts/fetch_and_index_papers.py --index-only --force-rebuild
```

### Issue: Out of Disk Space
**Solution**: Papers are large! Each PDF ~2-5MB, processed ~500KB-1MB
- 15 papers ≈ 50-100MB
- Monitor disk space

---

## 📈 Current Status

After running default script:
```
✓ Papers: 15 (AI/ML/NLP, newest)
✓ Processed: texts, tables, figures extracted
✓ Indexed: Vector DB + BM25
✓ Ready: Can query immediately
```

Check:
```bash
# Count PDFs
ls data/raw/*.pdf | wc -l

# Count processed texts
ls data/processed/texts/*.md | wc -l

# Check vector DB
# (Vector store will show count when you query)
```

---

## 🔄 Workflow Examples

### Scenario 1: First Time Setup
```bash
# 1. Fetch and index
python scripts/fetch_and_index_papers.py --max-results 15

# 2. Test
python tests/test_phase3_retrieval.py

# 3. Launch UI
python demo_ui.py
```

### Scenario 2: Add More Papers
```bash
# Already have some papers, want more
python scripts/fetch_papers_quick.py  # Choose option
python scripts/fetch_and_index_papers.py --index-only
```

### Scenario 3: Different Topic
```bash
# Want computer vision papers instead
python scripts/fetch_and_index_papers.py \
  --query "cat:cs.CV" \
  --max-results 15 \
  --force-rebuild
```

### Scenario 4: Weekly Update
```bash
# Get latest 5 papers every week
python scripts/fetch_and_index_papers.py --max-results 5
```

---

## 📚 References

- [arXiv API Docs](https://info.arxiv.org/help/api/index.html)
- [arXiv Category Taxonomy](https://arxiv.org/category_taxonomy)
- [arXiv Python Package](https://pypi.org/project/arxiv/)

---

## ✅ Summary

**Quick Commands**:
```bash
# Default (best for most cases)
python scripts/fetch_and_index_papers.py

# Quick interactive
python scripts/fetch_papers_quick.py

# Custom query
python scripts/fetch_and_index_papers.py --query "your_query" --max-results 20

# Index existing
python scripts/fetch_and_index_papers.py --index-only
```

**Default Config**:
- Query: AI + ML + NLP
- Count: 15 papers
- Sort: Newest first
- Auto: Fetch + Process + Index

---

**Happy Paper Fetching! 📚✨**

