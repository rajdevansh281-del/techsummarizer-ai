---
title: TechSummarizer AI
emoji: 📝
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.31.0
python_version: 3.12
app_file: app.py
pinned: false
---

# TechSummarizer AI

Summarize pasted text, text-based PDF files, and public HTML article URLs using `facebook/bart-large-cnn`.

Features:
- Short, Medium, and Detailed summaries
- Token-based chunking with overlap and multi-pass reduction
- Basic frequency-based keyword extraction
- PDF text extraction (scanned PDFs/OCR are not supported)
- Public HTML article extraction with size, timeout, redirect, and local-network checks

The model is downloaded from Hugging Face when the Space starts. The first startup may take several minutes. Free hosting may sleep when inactive and may have limited GPU quota. AI summaries should be checked against the original source.
