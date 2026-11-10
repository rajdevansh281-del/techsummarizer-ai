import ipaddress
import os
import re
import socket
from urllib.parse import urljoin, urlparse

import gradio as gr
import requests
import spaces
import torch
from bs4 import BeautifulSoup
from pypdf import PdfReader
from transformers import AutoTokenizer, pipeline

MODEL_NAME = "facebook/bart-large-cnn"
MAX_INPUT_TOKENS = 900
CHUNK_OVERLAP_TOKENS = 40
MAX_REDUCTION_ROUNDS = 8
MAX_WEBSITE_BYTES = 2_000_000
MAX_WEBSITE_REDIRECTS = 3
SUMMARY_SETTINGS = {
    "Short": (35, 80),
    "Medium": (80, 150),
    "Detailed": (130, 220),
}

print("Loading TechSummarizer AI model...")
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# ZeroGPU supports CUDA emulation at startup; the actual GPU is allocated
# when a function decorated with @spaces.GPU is called.
summarizer = pipeline(
    "summarization",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)
print("TechSummarizer AI model loaded.")


def clean_text(text):
    if text is None:
        return ""
    text = str(text).replace("\x00", " ")
    return re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()


def _token_count(text, add_special_tokens=False):
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens, truncation=False))


def split_into_chunks(text, max_tokens=MAX_INPUT_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS):
    text = clean_text(text)
    if not text:
        return []
    max_tokens = int(max_tokens)
    overlap_tokens = int(overlap_tokens)
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1.")
    overlap_tokens = max(0, min(overlap_tokens, max_tokens - 1))
    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if not token_ids:
        return []
    step = max_tokens - overlap_tokens
    chunks = []
    for start in range(0, len(token_ids), step):
        piece_ids = token_ids[start:start + max_tokens]
        if not piece_ids:
            break
        piece = clean_text(tokenizer.decode(piece_ids, skip_special_tokens=True))
        if piece:
            chunks.append(piece)
        if start + max_tokens >= len(token_ids):
            break
    return chunks


def _summarize_chunk(chunk, min_len, max_len, compression=0.45):
    input_tokens = _token_count(chunk, add_special_tokens=True)
    if input_tokens < 35:
        return clean_text(chunk)
    output_max = min(max_len, max(16, int(input_tokens * compression)))
    output_min = min(min_len, max(5, output_max // 2))
    output_min = min(output_min, output_max - 1)
    result = summarizer(
        chunk,
        max_length=output_max,
        min_length=output_min,
        do_sample=False,
        truncation=True,
    )
    if not result or not isinstance(result, list) or not result[0].get("summary_text"):
        raise RuntimeError("The summarization model returned an empty result.")
    summary = clean_text(result[0]["summary_text"])
    if not summary:
        raise RuntimeError("The summarization model returned an empty summary.")
    return summary


@spaces.GPU(duration=120)
def summarize_text(text, length="Medium"):
    text = clean_text(text)
    if not text:
        return "Please provide some text to summarize."
    if length not in SUMMARY_SETTINGS:
        raise ValueError("Choose Short, Medium, or Detailed summary length.")
    min_len, max_len = SUMMARY_SETTINGS[length]
    source_chunks = split_into_chunks(text)
    if not source_chunks:
        return "Please provide some readable text to summarize."
    partial_summaries = [
        _summarize_chunk(chunk, min_len, max_len, compression=0.45)
        for chunk in source_chunks
    ]
    combined = clean_text(" ".join(partial_summaries))
    for _ in range(MAX_REDUCTION_ROUNDS):
        current_tokens = _token_count(combined, add_special_tokens=False)
        if current_tokens <= MAX_INPUT_TOKENS:
            return combined
        reducer_chunks = split_into_chunks(combined, MAX_INPUT_TOKENS, 0)
        reduced_parts = [
            _summarize_chunk(chunk, min_len, max_len, compression=0.35)
            for chunk in reducer_chunks
        ]
        reduced_text = clean_text(" ".join(reduced_parts))
        reduced_tokens = _token_count(reduced_text, add_special_tokens=False)
        if not reduced_text or reduced_tokens >= current_tokens:
            reduced_parts = [
                _summarize_chunk(chunk, min_len, max_len, compression=0.20)
                for chunk in reducer_chunks
            ]
            reduced_text = clean_text(" ".join(reduced_parts))
            reduced_tokens = _token_count(reduced_text, add_special_tokens=False)
        if not reduced_text or reduced_tokens >= current_tokens:
            raise RuntimeError("The document could not be reduced safely. Try a shorter document.")
        combined = reduced_text
    if _token_count(combined, add_special_tokens=False) > MAX_INPUT_TOKENS:
        raise RuntimeError("The document is too large to summarize in the allowed reduction passes.")
    return combined


STOPWORDS = set("""
a an the and or but if while is are was were be been being to of in on for with by
from as at this that these those it its into about than then we you they he she them
his her their our your can could should would will may might
""".split())


def extract_keywords(text, top_n=10):
    text = clean_text(text).lower()
    if not text:
        return []
    words = re.findall(r"\b[a-z][a-z'-]{2,}\b", text)
    counts = {}
    for word in words:
        if word not in STOPWORDS:
            counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ranked[:top_n]]


def extract_pdf_text(pdf_path):
    if not pdf_path:
        raise ValueError("Upload a PDF first.")
    if not os.path.isfile(os.fspath(pdf_path)):
        raise ValueError("The uploaded PDF file could not be found.")
    try:
        reader = PdfReader(os.fspath(pdf_path), strict=False)
        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")
            except Exception:
                decrypted = 0
            if not decrypted:
                raise ValueError("This PDF is password-protected and cannot be opened.")
        page_texts = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_texts.append(page.extract_text() or "")
            except Exception as exc:
                raise ValueError(f"Text could not be extracted from PDF page {page_number}.") from exc
        extracted = clean_text(" ".join(page_texts))
        if not extracted:
            raise ValueError("No selectable text was found. This may be a scanned PDF; OCR is not included.")
        return extracted
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"The PDF could not be read ({type(exc).__name__}).") from exc


def _validate_public_host(hostname):
    hostname = (hostname or "").rstrip(".")
    if not hostname:
        raise ValueError("The URL must include a hostname.")
    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        try:
            resolved = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
            addresses = [ipaddress.ip_address(item[4][0].split("%")[0]) for item in resolved]
        except (socket.gaierror, ValueError, OSError) as exc:
            raise ValueError("The website hostname could not be resolved.") from exc
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("For safety, local, private, and reserved network addresses are blocked.")


def extract_url_text(url):
    if not url or not str(url).strip():
        raise ValueError("Enter a website URL first.")
    current_url = str(url).strip()
    if len(current_url) > 2048:
        raise ValueError("The URL is too long.")
    for redirect_number in range(MAX_WEBSITE_REDIRECTS + 1):
        parsed = urlparse(current_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("Enter a valid URL beginning with http:// or https://")
        if parsed.username or parsed.password:
            raise ValueError("URLs containing a username or password are not supported.")
        _validate_public_host(parsed.hostname)
        try:
            response = requests.get(
                current_url,
                headers={"User-Agent": "TechSummarizerAI/1.0"},
                timeout=(5, 15),
                stream=True,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise ValueError(f"The website request failed ({type(exc).__name__}).") from exc
        try:
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("The website returned a redirect without a destination.")
                if redirect_number >= MAX_WEBSITE_REDIRECTS:
                    raise ValueError("The website redirected too many times.")
                current_url = urljoin(current_url, location)
                continue
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type not in ("text/html", "application/xhtml+xml"):
                raise ValueError("The URL did not return an HTML page.")
            content_length = response.headers.get("Content-Length")
            try:
                declared_bytes = int(content_length) if content_length else None
            except (TypeError, ValueError):
                declared_bytes = None
            if declared_bytes is not None and declared_bytes > MAX_WEBSITE_BYTES:
                raise ValueError("The webpage is larger than the 2 MB safety limit.")
            body_parts, total_bytes = [], 0
            for part in response.iter_content(chunk_size=65536):
                if part:
                    total_bytes += len(part)
                    if total_bytes > MAX_WEBSITE_BYTES:
                        raise ValueError("The webpage is larger than the 2 MB safety limit.")
                    body_parts.append(part)
            if not body_parts:
                raise ValueError("The website returned an empty page.")
            soup = BeautifulSoup(b"".join(body_parts), "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                tag.decompose()
            article = soup.find("article") or soup.find("main") or soup
            paragraphs = [p.get_text(" ", strip=True) for p in article.find_all("p")]
            extracted = clean_text(" ".join(p for p in paragraphs if p) or article.get_text(" ", strip=True))
            if not extracted:
                raise ValueError("No readable article text was found on this webpage.")
            return extracted
        finally:
            response.close()
    raise ValueError("The website redirected too many times.")


def text_tab(text, length):
    cleaned = clean_text(text)
    if not cleaned:
        return "Please paste some text to summarize.", ""
    try:
        return summarize_text(cleaned, length), ", ".join(extract_keywords(cleaned))
    except Exception as exc:
        return f"Text summarization failed: {type(exc).__name__}: {exc}", ""


def pdf_tab(pdf_path, length):
    if not pdf_path:
        return "Upload a PDF first.", ""
    try:
        extracted = extract_pdf_text(pdf_path)
        return summarize_text(extracted, length), ", ".join(extract_keywords(extracted))
    except Exception as exc:
        return f"PDF processing failed: {type(exc).__name__}: {exc}", ""


def url_tab(url, length):
    try:
        extracted = extract_url_text(url)
        return summarize_text(extracted, length), ", ".join(extract_keywords(extracted))
    except Exception as exc:
        return f"Website processing failed: {type(exc).__name__}: {exc}", ""


with gr.Blocks(title="TechSummarizer AI") as demo:
    gr.Markdown("# TechSummarizer AI")
    gr.Markdown("Summarize text, text-based PDFs, and public web articles.")
    with gr.Tab("Text"):
        text_input = gr.Textbox(label="Paste your text", lines=12)
        text_length = gr.Radio(["Short", "Medium", "Detailed"], value="Medium", label="Summary length")
        text_button = gr.Button("Generate Summary", variant="primary")
        text_output = gr.Textbox(label="Summary", lines=8)
        text_keywords = gr.Textbox(label="Important keywords")
        text_button.click(text_tab, [text_input, text_length], [text_output, text_keywords])
    with gr.Tab("PDF"):
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
        pdf_length = gr.Radio(["Short", "Medium", "Detailed"], value="Medium", label="Summary length")
        pdf_button = gr.Button("Summarize PDF", variant="primary")
        pdf_output = gr.Textbox(label="Summary", lines=8)
        pdf_keywords = gr.Textbox(label="Important keywords")
        pdf_button.click(pdf_tab, [pdf_input, pdf_length], [pdf_output, pdf_keywords])
    with gr.Tab("Website URL"):
        url_input = gr.Textbox(label="Article URL", placeholder="https://example.com/article")
        url_length = gr.Radio(["Short", "Medium", "Detailed"], value="Medium", label="Summary length")
        url_button = gr.Button("Summarize Article", variant="primary")
        url_output = gr.Textbox(label="Summary", lines=8)
        url_keywords = gr.Textbox(label="Important keywords")
        url_button.click(url_tab, [url_input, url_length], [url_output, url_keywords])

if __name__ == "__main__":
    demo.queue().launch()
