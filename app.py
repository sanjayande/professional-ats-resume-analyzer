import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="Professional ATS Analyzer", layout="wide")

st.title("📊 Professional ATS Resume Analyzer")

st.sidebar.header("Navigation")
st.sidebar.write("Upload Resume and Paste Job Description")

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type="pdf")
job_description = st.text_area("📝 Paste Job Description")

if uploaded_file and job_description:

    reader = PyPDF2.PdfReader(uploaded_file)
    resume_text = ""
    for page in reader.pages:
        resume_text += page.extract_text()

    documents = [resume_text, job_description]

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(documents)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("🎯 Overall ATS Score", f"{round(score*100,2)} %")

    resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
    jd_words = set(re.findall(r'\b\w+\b', job_description.lower()))

    matched = resume_words.intersection(jd_words)
    missing = jd_words.difference(resume_words)

    with col2:
        st.metric("🔑 Matched Keywords", len(matched))

    st.subheader("✅ Matched Keywords")
    st.write(list(matched)[:20])

    st.subheader("❌ Missing Keywords")
    st.write(list(missing)[:20])