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

# === SmartLegal Rental Assistant ===

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load fine-tuned model (Hugging Face)
@st.cache_resource
def load_risk_model():
    return pipeline("text-classification",
                    model="yourusername/india-rental-simple-bert",
                    tokenizer="yourusername/india-rental-simple-bert",
                    truncation=True, max_length=512)

risk_pipe = load_risk_model()

st.title("SmartLegal Rental Assistant")
st.markdown("**Draft • Review • Fix — Based on Model Tenancy Act 2021**")

tab1, tab2 = st.tabs(["Draft New Agreement", "Review & Fix Agreement"])

# === TAB 1: Draft New Agreement ===
with tab1:
    st.header("Create New Rental Agreement")
    with st.form("draft_form"):
        c1, c2 = st.columns(2)
        with c1:
            landlord = st.text_input("Landlord Name")
            tenant = st.text_input("Tenant Name")
            rent = st.number_input("Monthly Rent ₹", min_value=1000)
            deposit = st.number_input("Security Deposit ₹", min_value=0)
        with c2:
            address = st.text_area("Property Address")
            start = st.date_input("Start Date")
            months = st.selectbox("Duration", ["11 months", "2 years", "3 years"])
        
        amenities = st.text_area("Amenities (optional)")
        submitted = st.form_submit_button("Generate Agreement")
        
        if submitted:
            prompt = f"""
            Draft a complete rental agreement under Model Tenancy Act 2021 for:
            Landlord: {landlord}
            Tenant: {tenant}
            Property: {address}
            Rent: ₹{rent}/month
            Deposit: ₹{deposit}
            Duration: {months} from {start}
            Amenities: {amenities}
            Include all mandatory clauses: registration, police verification, maintenance, eviction.
            """
            resp = openai.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.3
            )
            draft = resp.choices[0].message.content
            
            doc = Document()
            for line in draft.split("\n"):
                doc.add_paragraph(line)
            bio = BytesIO()
            doc.save(bio)
            bio.seek(0)
            
            st.success("Agreement Ready!")
            st.download_button("Download Word File", bio, f"Rental_{tenant}.docx", 
                             "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            st.text_area("Preview", draft, height=400)

# === TAB 2: Review & Fix Agreement ===
with tab2:
    st.header("Review & Suggest Amendments")
    uploaded = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf","docx","txt"])
    if uploaded:
        if uploaded.type == "application/pdf":
            text = " ".join([p.extract_text() or "" for p in PdfReader(uploaded).pages])
        elif uploaded.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded)
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            text = uploaded.read().decode("utf-8")
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        summary = model.generate_content(f"Summarize this rental agreement in 100 words: {text[:4000]}").text
        st.write("### Summary")
        st.write(summary)
        
        risk = risk_pipe(text[:10000])[0]
        st.write("### Risk Level")
        st.write(f"**{risk['label']}** (Confidence: {risk['score']:.1%})")
        
        amendments = model.generate_content(
            f"List missing or incorrect clauses according to Model Tenancy Act 2021:\n{text[:5000]}"
        ).text
        st.write("### Suggested Amendments")
        st.warning(amendments)

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

