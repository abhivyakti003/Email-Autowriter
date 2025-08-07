import os
import re
from io import BytesIO
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import docx2txt
import fitz  # PyMuPDF
import spacy
from app.chains import generate_prompt, fetch_job_description_from_url, extract_text_from_links

# Load environment variables
load_dotenv()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize HF Client
client = InferenceClient(
    provider="nebius",
    api_key=os.getenv("HF_TOKEN")
)

# Streamlit UI
st.set_page_config(page_title="Email AutoWriter", page_icon="📧")
st.title("📧 Email AutoWriter (Hugging Face LLaMA)")

uploaded_file = st.file_uploader("📂 Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

# Resume text extraction helpers
def extract_text_from_resume(file):
    if file.name.endswith(".pdf"):
        pdf_doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in pdf_doc:
            text += page.get_text()
        return text
    elif file.name.endswith(".docx"):
        return docx2txt.process(BytesIO(file.read()))
    return ""

def extract_details(text):
    # 1. Extract name using NER with fallback
    name = ""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break
    if not name:
        for ent in doc.ents:
            if ent.label_ in ["GPE", "ORG"] and len(ent.text.split()) == 1:
                name = ent.text
                break
    if not name:
        # Fallback: use the first capitalized word from resume
        capital_words = re.findall(r"\b[A-Z][a-z]+\b", text)
        if capital_words:
            name = capital_words[0]

    # 2. Skills
    skills_match = re.findall(r"(skills|technologies|technical skills)[\s:]*([\s\S]{0,400})", text, re.IGNORECASE)
    skills = skills_match[0][1].strip() if skills_match else ""

    # 3. Experience
    exp_match = re.findall(r"(experience|worked at|internship|employment)[\s:]*([\s\S]{0,500})", text, re.IGNORECASE)
    experience = exp_match[0][1].strip() if exp_match else ""

    # 4. Languages
    lang_match = re.findall(r"(languages|spoken languages)[\s:]*([\s\S]{0,300})", text, re.IGNORECASE)
    languages = lang_match[0][1].strip() if lang_match else ""

    # 5. University
    university = ""
    for uni in ["Galgotias", "Banasthali", "IIT", "NIT", "University", "Institute", "College"]:
        if uni.lower() in text.lower():
            university = uni
            break

    return {
        "name": name,
        "skills": skills,
        "experience": experience,
        "languages": languages,
        "university": university
    }

# Manual fallback inputs
name = ""
university = ""
skills = ""
experience = ""
languages = ""

# Extract and fill if resume is uploaded
if uploaded_file:
    resume_text = extract_text_from_resume(uploaded_file)
    extracted = extract_details(resume_text)

    name = st.text_input("👤 Your Name", value=extracted["name"])
    university = st.text_input("🏫 Your University", value=extracted["university"])
    skills = st.text_area("🧠 Your Skills", value=extracted["skills"])
    experience = st.text_area("💼 Your Experience", value=extracted["experience"])
    languages = st.text_input("🗣️ Languages You Speak", value=extracted["languages"])
    st.success("📄 Resume uploaded and fields auto-filled.")
else:
    name = st.text_input("👤 Your Name", placeholder="e.g. Abhivyakti")
    university = st.text_input("🏫 Your University", placeholder="e.g. Banasthali University")
    languages = st.text_input("🗣️ Languages You Speak")
    skills = st.text_area("🧠 Your Skills", placeholder="e.g. Python, Machine Learning, Deep Learning")
    experience = st.text_area("💼 Your Experience", placeholder="e.g. 2 internships, 3 ML projects")

job_link = st.text_input("🔗 Job Description URL")
portfolio_links = st.text_area("🌐 Portfolio Links", placeholder="One per line")

if st.button("✉️ Generate Email"):
    if not all([name, university, skills, experience, job_link, portfolio_links]):
        st.warning("⚠️ Please fill in all fields.")
    else:
        job_description = fetch_job_description_from_url(job_link)
        portfolio_urls = [link.strip() for link in portfolio_links.strip().split("\n") if link.strip()]
        portfolio_text = extract_text_from_links(portfolio_urls)

        prompt = generate_prompt(
            job_description=job_description,
            portfolio_text=portfolio_text,
            name=name,
            university=university,
            languages=languages
        )

        try:
            with st.spinner("Generating email..."):
                response = client.chat.completions.create(
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    messages=[{"role": "user", "content": prompt}]
                )
                email = response.choices[0].message.content.strip()
                st.success("✅ Email Generated!")
                st.subheader("📩 Output:")
                st.text_area("Generated Email", email, height=300)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")