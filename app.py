#!/usr/bin/env python3
import os
import re
import json
import time
import logging
from datetime import datetime
from io import BytesIO

import streamlit as st
import google.generativeai as genai
import openai
import pandas as pd
from dotenv import load_dotenv
from transformers import pipeline
from PyPDF2 import PdfReader
import docx
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING

# =========================
# Config & LLMOps Logging
# =========================
LOG_FILE = "metrics_log.jsonl"
logging.basicConfig(
    filename="smartlegal_metrics.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def _ensure_metrics_state():
    if "metrics" not in st.session_state:
        st.session_state.metrics = {
            "total_requests": 0,
            "failed_requests": 0,
            "total_latency": 0.0,
            "avg_latency": 0.0,
        }

def _bump_success(latency_s: float):
    _ensure_metrics_state()
    st.session_state.metrics["total_requests"] += 1
    st.session_state.metrics["total_latency"] += latency_s
    tr = st.session_state.metrics["total_requests"]
    st.session_state.metrics["avg_latency"] = (
        st.session_state.metrics["total_latency"] / max(tr, 1)
    )

def _bump_failure():
    _ensure_metrics_state()
    st.session_state.metrics["failed_requests"] += 1
    st.session_state.metrics["total_requests"] += 1

def log_metric(event_type, latency, tokens, cost, status="SUCCESS", model="gemini-2.0-flash"):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event_type,
        "model": model,
        "latency_s": latency,
        "tokens_used": tokens,
        "cost_gbp": cost,
        "status": status,
    }
    # Append to JSONL
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    # Also to streamlit sidebar counters
    if status == "SUCCESS":
        _bump_success(latency)
    else:
        _bump_failure()
    # System log file
    logging.info(entry)

# =========================
# Document Formatting Helper
# =========================
def create_formatted_agreement(draft_text, tenant, landlord):
    doc = Document()

    # Title
    title = doc.add_paragraph("RENTAL AGREEMENT")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.runs[0]
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(16)

    # Subtitle
    subtitle = doc.add_paragraph("(Under the Model Tenancy Act, 2021)")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].font.size = Pt(12)
    doc.add_paragraph()  # spacing

    # Normal style
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Times New Roman"
    normal_style.font.size = Pt(12)
    normal_style.paragraph_format.line_spacing = 1.15

    # === Clean and process text ===
    clean_text = re.sub(r"#+", "", draft_text)           # remove markdown ## headings
    clean_text = re.sub(r"\*\*", "", clean_text)         # remove ** markers
    lines = [ln.strip() for ln in clean_text.split("\n") if ln.strip()]

    for line in lines:
        if re.match(r"^(WHEREAS|NOW THEREFORE|IN WITNESS|THIS RENTAL AGREEMENT|BETWEEN|AND|IMPORTANT NOTES|SIGNED)", line, re.IGNORECASE):
            p = doc.add_paragraph(line)
            r = p.runs[0]
            r.bold = True
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(4)
        elif re.match(r"^[0-9]+\.", line):  # numbered clauses
            p = doc.add_paragraph(line)
            p.paragraph_format.left_indent = Inches(0.3)
            r = p.runs[0]
            r.bold = True
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        elif line.startswith("*"):  # bullet lines
            clean = line.lstrip("* ").strip()
            p = doc.add_paragraph(clean, style="List Bullet")
            p.paragraph_format.left_indent = Inches(0.6)
        else:
            p = doc.add_paragraph(line)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.space_after = Pt(6)

    # Signature block
    doc.add_page_break()
    doc.add_paragraph(
        "IN WITNESS WHEREOF, the parties hereto have executed this Agreement.",
        style="Normal",
    )
    table = doc.add_table(rows=2, cols=2)
    table.autofit = True
    table.cell(0, 0).text = f"By the Landlord:\n\n(Signature)\n\n{landlord}"
    table.cell(0, 1).text = f"By the Tenant:\n\n(Signature)\n\n{tenant}"

    filename = f"Formatted_Rental_{tenant}.docx"
    doc.save(filename)
    return filename

# =========================
# App Init & Models
# =========================
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def load_risk_model():
    return pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512,
    )

risk_pipe = load_risk_model()

st.title("SmartLegal Rental Assistant")
st.markdown("**Draft • Review • Fix — Based on Model Tenancy Act 2021**")

tab1, tab2 = st.tabs(["Draft New Agreement", "Review & Fix Agreement"])

# =========================
# TAB 1: Draft New Agreement
# =========================
with tab1:
    st.header("Create New Rental Agreement")

    # Input form (prevents field refresh glitches)
    with st.form("rental_form", clear_on_submit=False):
        landlord = st.text_input("Landlord Name", key="landlord_name")
        tenant = st.text_input("Tenant Name", key="tenant_name")
        rent = st.number_input("Monthly Rent ₹", min_value=1000, key="rent_amount")
        deposit = st.number_input("Security Deposit ₹", min_value=0, key="deposit_amount")
        address = st.text_area("Property Address", key="property_address")
        start = st.date_input("Start Date", key="lease_start_date")
        months = st.selectbox("Duration", ["11 months", "2 years", "3 years"], key="lease_duration")
        amenities = st.text_area("Amenities (optional)", key="amenities_text")
        submitted = st.form_submit_button("Generate Agreement")

    if submitted:
        # Show loading spinner
        with st.spinner('Generating rental agreement... Please wait.'):
            prompt = f"""
            Draft a complete rental agreement under the Model Tenancy Act 2021 with:
            Landlord: {landlord}
            Tenant: {tenant}
            Property: {address}
            Rent: ₹{rent}/month
            Deposit: ₹{deposit}
            Duration: {months}, starting from {start}.
            Amenities: {amenities if amenities else 'None'}
            Include all mandatory clauses such as registration, police verification,
            maintenance, notice period, and eviction.
            
            IMPORTANT: Do not include any examples, placeholders, or "e.g." text. 
            Use specific values provided above. Do not use brackets with examples.
            Write in clear, formal legal language suitable for immediate use.
            """

            try:
                model = genai.GenerativeModel("models/gemini-2.0-flash")
                t0 = time.time()
                response = model.generate_content(prompt)
                t1 = time.time()

                draft = (response.text or "").strip()

                # Trim "Sure/Okay/Here is …" style prefixes if present
                for prefix in ("sure", "okay", "here", "below", "this is"):
                    if draft.lower().startswith(prefix):
                        parts = draft.split("\n", 1)
                        draft = parts[1].strip() if len(parts) > 1 else draft
                        break

                # Metrics
                latency = round(t1 - t0, 2)
                token_count = len(prompt.split()) + len(draft.split())
                cost = round(token_count * 0.0005 / 1000, 6)

                st.caption(f"Latency: {latency}s | Tokens: {token_count} | Cost: £{cost}")
                log_metric("DraftAgreement", latency, token_count, cost, status="SUCCESS")
            except Exception as e:
                st.error(f"Gemini API error: {e}")
                draft = "Unable to generate agreement text."
                log_metric("DraftAgreement", 0, 0, 0, status="FAILED")

        # Show results after loading completes
        st.success("Agreement Drafted Successfully!")
        st.text_area("Preview", draft, height=400, key="draft_preview")

        # Show loader for document formatting
        with st.spinner('Creating formatted document...'):
            file_name = create_formatted_agreement(draft, tenant, landlord)
            
        with open(file_name, "rb") as f:
            st.download_button(
                label="Download Formatted Word File",
                data=f,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

# =========================
# TAB 2: Review & Fix Agreement
# =========================
with tab2:
    st.header("Review & Suggest Amendments")
    uploaded = st.file_uploader(
        "Upload PDF / DOCX / TXT",
        type=["pdf", "docx", "txt"],
        key="review_uploader",
    )

    if uploaded:
        # Show loader for text extraction
        with st.spinner('Processing uploaded document...'):
            # Extract text
            if uploaded.type == "application/pdf":
                text = " ".join([(p.extract_text() or "") for p in PdfReader(uploaded).pages])
            elif uploaded.type in (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword",
            ):
                d = docx.Document(uploaded)
                text = "\n".join([p.text for p in d.paragraphs])
            else:
                text = uploaded.read().decode("utf-8", errors="ignore")

        model = genai.GenerativeModel("models/gemini-2.0-flash")

        # Summary + metrics with loader
        with st.spinner('Generating summary...'):
            try:
                t0 = time.time()
                summary = model.generate_content(
                    f"Summarize this rental agreement in 100 words:\n{text[:4000]}"
                ).text
                t1 = time.time()
                latency_summary = round(t1 - t0, 2)
                tokens_summary = len(summary.split()) + len(text.split()[:4000])
                cost_summary = round(tokens_summary * 0.0005 / 1000, 6)
                
                st.subheader("Summary")
                st.text_area("Summary Preview", summary, height=150, key="review_summary")
                st.caption(f"Latency: {latency_summary}s | Tokens: {tokens_summary} | Cost: £{cost_summary}")
                log_metric("Summarisation", latency_summary, tokens_summary, cost_summary, status="SUCCESS")
            except Exception as e:
                st.error(f"Gemini summary error: {e}")
                log_metric("Summarisation", 0, 0, 0, status="FAILED")

        # Risk classifier with loader
        with st.spinner('Analyzing risk level...'):
            try:
                risk = risk_pipe(text[:10000])[0]
                st.subheader("Risk Level")
                st.write(f"**{risk['label']}** (Confidence: {risk['score']:.1%})")
                log_metric("RiskClassification", 0, 0, 0, status="SUCCESS", model="distilbert-sst2")
            except Exception as e:
                st.error(f"Risk model error: {e}")
                log_metric("RiskClassification", 0, 0, 0, status="FAILED", model="distilbert-sst2")

        # Amendments + metrics with loader
        with st.spinner('Generating amendment suggestions...'):
            try:
                t0 = time.time()
                amendments = model.generate_content(
                    f"List missing or incorrect clauses according to Model Tenancy Act 2021:\n{text[:5000]}"
                ).text
                t1 = time.time()
                latency_amend = round(t1 - t0, 2)
                tokens_amend = len(amendments.split()) + len(text.split()[:5000])
                cost_amend = round(tokens_amend * 0.0005 / 1000, 6)
                
                st.subheader("Suggested Amendments")
                st.text_area("Amendments Preview", amendments, height=200, key="review_amendments")
                st.caption(f"Latency: {latency_amend}s | Tokens: {tokens_amend} | Cost: £{cost_amend}")
                log_metric("AmendmentReview", latency_amend, tokens_amend, cost_amend, status="SUCCESS")
            except Exception as e:
                st.error(f"Gemini amendment error: {e}")
                log_metric("AmendmentReview", 0, 0, 0, status="FAILED")

        # Successful queries counter
        st.session_state.success_count = st.session_state.get("success_count", 0) + 1
        st.sidebar.metric("Total Successful Queries", st.session_state.success_count)

# =========================
# Sidebar: Voice + Metrics
# =========================
with st.sidebar:
    st.header("Voice to Clause")
    audio = st.file_uploader("Upload voice note", type=["mp3", "wav"])
    if audio:
        try:
            transcript = openai.audio.transcriptions.create(model="whisper-1", file=audio).text
            st.code(transcript)
            clause = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": f"Convert to legal clause (Model Tenancy Act 2021): {transcript}"
                }],
            ).choices[0].message.content
            st.success(clause)
            # Approx metrics for voice step (no latency tracked here)
            log_metric("VoiceToClause", 0, len(transcript.split()), 0.0001, status="SUCCESS", model="whisper+gpt-4o-mini")
        except Exception as e:
            st.error(f"Voice-to-clause error: {e}")
            log_metric("VoiceToClause", 0, 0, 0, status="FAILED", model="whisper+gpt-4o-mini")

    st.subheader("📈 LLMOps Metrics Summary")
    _ensure_metrics_state()
    st.metric("Total Requests", st.session_state.metrics["total_requests"])
    st.metric("Avg Latency (s)", round(st.session_state.metrics["avg_latency"], 2))
    st.metric("Failed Requests", st.session_state.metrics["failed_requests"])

    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_json(LOG_FILE, lines=True)
            st.dataframe(df.tail(5), use_container_width=True)
            st.caption(f"Total Logs: {len(df)} | Last Updated: {df.iloc[-1]['timestamp']}")
        except Exception:
            st.info("Metrics log exists but could not be parsed yet.")
    else:
        st.info("No metrics logged yet. Generate an agreement to start.")

