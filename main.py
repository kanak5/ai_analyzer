import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio
import nest_asyncio
import re

# ----------------- Fix Streamlit Async Conflict -----------------
nest_asyncio.apply()

# ----------------- Load Environment Variables -----------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("⚠ GEMINI_API_KEY not found in .env file. Please add it before running.")
    st.stop()

# Configure Gemini client
genai.configure(api_key=api_key)

# ----------------- Streamlit Session State -----------------
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if "resume" not in st.session_state:
    st.session_state.resume = ""
if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""

# ----------------- Page Title -----------------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="🧠", layout="wide")
st.title("🧠 AI Resume Analyzer")
st.caption("Compare your resume with a job description using AI similarity and Gemini feedback.")

# ----------------- Helper Functions -----------------
def extract_pdf_text(uploaded_file):
    try:
        return extract_text(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error extracting text from PDF: {str(e)}")
        return ""

@st.cache_resource
def get_bert_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def calculate_similarity_bert(text1, text2):
    model = get_bert_model()
    embeddings1 = model.encode([text1])
    embeddings2 = model.encode([text2])
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return similarity

# Generate Gemini Report
def get_report(resume, job_desc):
    try:
        prompt = f"""
        You are an AI Resume Evaluator.

        Compare the resume and job description.
        Return ONLY very short bullet points (max 3–5 words each).

        STRICT FORMAT:

        - Skills Matched:
        • <3–5 short bullets>
        - Missing Skills:
        • <3–5 short bullets>
        - Partial Matches:
        • <3–5 short bullets>
        - Suggestions:
        • <3–5 short bullets>

        Rules:
        • No sentences
        • No explanations
        • No paragraphs
        • ONLY minimal bullets
        • MAX 5 words per bullet

        Resume:
        {resume}

        Job Description:
        {job_desc}
        """
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"❌ Error generating AI report: {str(e)}")
        return ""

# ----------------- Streamlit Workflow -----------------
if not st.session_state.form_submitted:
    with st.form("resume_form"):
        resume_file = st.file_uploader("📄 Upload Resume (PDF)", type="pdf")
        st.session_state.job_desc = st.text_area("💼 Enter Job Description", placeholder="Paste job description here...")
        submitted = st.form_submit_button("🔍 Analyze")

        if submitted:
            if resume_file and st.session_state.job_desc.strip():
                st.session_state.resume = extract_pdf_text(resume_file)
                st.session_state.form_submitted = True
                st.rerun()
            else:
                st.warning("⚠ Please upload a resume and enter the job description.")

# ----------------- After Submission -----------------
if st.session_state.form_submitted:
    with st.spinner("⏳ Analyzing your resume using AI..."):
        ats_score = calculate_similarity_bert(st.session_state.resume, st.session_state.job_desc)
        report = get_report(st.session_state.resume, st.session_state.job_desc)

    st.success("✅ Analysis Complete!")

    # Display Key Scores
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📈 ATS Similarity Score", f"{round(ats_score * 100, 2)}%")
    with col2:
        st.metric("🤖 AI Evaluation", "See feedback below")

    # Display Clean Gemini Report
    st.subheader("🧾 Gemini AI Feedback")
    st.markdown(report)

    # Reset Button
    st.markdown("---")
    if st.button("🔄 Run Another Analysis"):
        st.session_state.form_submitted = False
        st.session_state.resume = ""
        st.session_state.job_desc = ""
        st.rerun()