#!/usr/bin/env python3
import streamlit as st
import google.generativeai as genai
import openai
import os
from dotenv import load_dotenv
from transformers import pipeline
from PyPDF2 import PdfReader
import docx
from docx import Document
from io import BytesIO

# For better formatting 
#from docx import Document
from docx.shared import Pt, Inches 
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING

#from docx import Document
#from docx.shared import Pt
#from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
#import re

#from docx import Document
#from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
import re
# For generating the metrics
import time

# Generating Formatted Document
def create_formatted_agreement(draft_text, tenant):
    doc = Document()

    # === Title ===
    title = doc.add_paragraph("RENTAL AGREEMENT")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.runs[0]
    run.bold = True
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16)

    subtitle = doc.add_paragraph("(Under the Model Tenancy Act, 2021)")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].font.size = Pt(12)
    doc.add_paragraph()  # spacing

    # === Body Style ===
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Times New Roman"
    normal_style.font.size = Pt(12)

    # === Clean and process text ===
    clean_text = re.sub(r"#+", "", draft_text)           # remove markdown ## headings
    clean_text = re.sub(r"\*\*", "", clean_text)         # remove ** markers
    lines = [ln.strip() for ln in clean_text.split("\n") if ln.strip()]

    for line in lines:
        # === Detect and style headings ===
        if re.match(r"^(WHEREAS|NOW THEREFORE|IN WITNESS|THIS RENTAL AGREEMENT|BETWEEN|AND|IMPORTANT NOTES|SIGNED)", line, re.IGNORECASE):
            p = doc.add_paragraph(line)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.runs[0]
            r.bold = True
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(4)
        elif re.match(r"^[0-9]+\.", line):  # numbered clauses (e.g., 1., 2.)
            p = doc.add_paragraph(line)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.left_indent = Inches(0.3)  # indent numbered clauses
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        elif line.startswith("*"):  # bulleted points
            clean = line.lstrip("* ").strip()
            p = doc.add_paragraph(clean, style="List Bullet")
            p.paragraph_format.left_indent = Inches(0.6)
            p.paragraph_format.space_after = Pt(2)
        else:
            p = doc.add_paragraph(line)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

    # === Signature Section ===
    doc.add_page_break()
    doc.add_paragraph("IN WITNESS WHEREOF, the parties hereto have executed this Agreement.", style="Normal")

    table = doc.add_table(rows=2, cols=2)
    table.autofit = True
    table.cell(0, 0).text = "By the Landlord:\n\n(Signature)\n\nVivek Bhadra"
    table.cell(0, 1).text = f"By the Tenant:\n\n(Signature)\n\n{tenant}"

    filename = f"Formatted_Rental_{tenant}.docx"
    doc.save(filename)
    return filename

# === SmartLegal Rental Assistant ===

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load fine-tuned model (Hugging Face)
@st.cache_resource
def load_risk_model():
    """
    Loads the Legal Risk Classification model.
    Uses a reliable public model for this assignment.
    """
    from transformers import pipeline

    pipe = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512,
    )
    return pipe

risk_pipe = load_risk_model()

st.title("SmartLegal Rental Assistant")
st.markdown("**Draft • Review • Fix — Based on Model Tenancy Act 2021**")

tab1, tab2 = st.tabs(["Draft New Agreement", "Review & Fix Agreement"])

# === TAB 1: Draft New Agreement ===
with tab1:
    st.header("Create New Rental Agreement")

    # === Input Form ===
    with st.form("rental_form", clear_on_submit=False):
        landlord = st.text_input("Landlord Name", key="landlord_name")
        tenant = st.text_input("Tenant Name", key="tenant_name")
        rent = st.number_input("Monthly Rent ₹", min_value=1000, key="rent_amount")
        deposit = st.number_input("Security Deposit ₹", min_value=0, key="deposit_amount")
        address = st.text_area("Property Address", key="property_address")
        start = st.date_input("Start Date", key="lease_start_date")
        months = st.selectbox("Duration", ["11 months", "2 years", "3 years"], key="lease_duration")
        amenities = st.text_area("Amenities (optional)", key="amenities_text")

        # === Submit Form ===
        submitted = st.form_submit_button("Generate Agreement")

    if submitted:
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
        """

        try:
            model = genai.GenerativeModel("models/gemini-2.0-flash")
            start_time = time.time()
            response = model.generate_content(prompt)
            end_time = time.time()

            # === Extract and clean the model output ===
            draft = response.text.strip()

            # Remove conversational prefixes like "Sure," or "Okay, here..."
            unwanted_prefixes = [
                "Sure, here",
                "Okay, here",
                "Here is",
                "Here’s",
                "Below is",
                "This is"
            ]
            for prefix in unwanted_prefixes:
                if draft.lower().startswith(prefix.lower()):
                    draft = draft.split("\n", 1)[-1].strip()
                    break

            # === LLMOps Metrics ===
            latency = round(end_time - start_time, 2)
            token_count = len(prompt.split()) + len(draft.split())   # Approx token count
            cost = round(token_count * 0.0005 / 1000, 6)             # Approx cost in £

            st.caption(f"Latency: {latency}s | Tokens: {token_count} | Cost: £{cost}")

        except Exception as e:
            st.error(f"Gemini API error: {e}")
            draft = "Unable to generate agreement text. Please try again later."

        st.success("Agreement Drafted Successfully!")
        st.text_area("Preview", draft, height=400)

        file_name = create_formatted_agreement(draft, tenant)

        st.success("Agreement Drafted Successfully!")
        with open(file_name, "rb") as f:
            st.download_button(
                label="Download Formatted Word File",
                data=f,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        #st.text_area("Preview", draft, height=400)
        st.text_area("Preview", draft, height=400, key="draft_preview")


# === TAB 2: Review & Fix Agreement ===
with tab2:
    st.header("Review & Suggest Amendments")

    uploaded = st.file_uploader(
        "Upload PDF / DOCX / TXT",
        type=["pdf", "docx", "txt"],
        key="review_uploader"
    )

    if uploaded:
        # === Extract text from uploaded document ===
        if uploaded.type == "application/pdf":
            text = " ".join([p.extract_text() or "" for p in PdfReader(uploaded).pages])
        elif uploaded.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded)
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            text = uploaded.read().decode("utf-8")

        # === Load verified Gemini model ===
        model = genai.GenerativeModel("models/gemini-2.0-flash")

        # ===  Summary Generation with LLMOps Metrics ===
        import time
        start = time.time()
        summary = model.generate_content(
            f"Summarize this rental agreement in 100 words:\n{text[:4000]}"
        ).text
        end = time.time()
        latency_summary = round(end - start, 2)
        tokens_summary = len(summary.split()) + len(text.split()[:4000])
        cost_summary = round(tokens_summary * 0.0005 / 1000, 6)

        st.subheader("Summary")
        st.text_area("Summary Preview", summary, height=150, key="review_summary")
        st.caption(f" Latency: {latency_summary}s | Tokens: {tokens_summary} | Cost: £{cost_summary}")

        # === Risk Level (BERT classification) ===
        risk = risk_pipe(text[:10000])[0]
        st.subheader("Risk Level")
        st.write(f"**{risk['label']}** (Confidence: {risk['score']:.1%})")
        if risk['label'] == "SAFE":
            st.caption("Classified as SAFE – no major missing clauses.")
        else:
            st.caption("Classified as RISKY – review required.")

        # === Amendment Suggestions with LLMOps Metrics ===
        start = time.time()
        amendments = model.generate_content(
            f"List missing or incorrect clauses according to Model Tenancy Act 2021:\n{text[:5000]}"
        ).text
        end = time.time()
        latency_amend = round(end - start, 2)
        tokens_amend = len(amendments.split()) + len(text.split()[:5000])
        cost_amend = round(tokens_amend * 0.0005 / 1000, 6)

        st.subheader("Suggested Amendments")
        st.text_area("Amendments Preview", amendments, height=200, key="review_amendments")
        st.caption(f"Latency: {latency_amend}s | Tokens: {tokens_amend} | Cost: £{cost_amend}")

        # === Track query success ===
        if "success_count" not in st.session_state:
            st.session_state.success_count = 0
        st.session_state.success_count += 1
        st.sidebar.metric("Total Successful Queries", st.session_state.success_count)


# === SIDEBAR: Voice to Clause ===
with st.sidebar:
    st.header("Voice to Clause")
    audio = st.file_uploader("Upload voice note", type=["mp3","wav"])
    if audio:
        transcript = openai.audio.transcriptions.create(model="whisper-1", file=audio).text
        st.code(transcript)
        clause = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":f"Convert to legal clause (Model Tenancy Act 2021): {transcript}"}]
        ).choices[0].message.content
        st.success(clause)
