import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load .env variables
load_dotenv()

# Streamlit UI
st.title("📧 Email AutoWriter with LLaMA 3.1")

name = st.text_input("Your Name")
skills = st.text_area("Your Skills")
experience = st.text_area("Your Experience")
job_link = st.text_input("Job Description URL")

if st.button("Generate Email"):
    if not all([name, skills, experience, job_link]):
        st.error("Please fill all fields.")
    else:
        # Construct the prompt
        prompt = f"""
Write a professional job application email using the following details:

Name: {name}
Skills: {skills}
Experience: {experience}
Job Description URL: {job_link}

Make the email concise, polite, and persuasive.
"""

        try:
            # Initialize client
            client = InferenceClient(
                provider="nebius",
                api_key=os.environ["HF_TOKEN"],
            )

            # Send prompt
            completion = client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=[{"role": "user", "content": prompt}],
            )

            # Show result
            generated_email = completion.choices[0].message.content
            st.subheader("📩 Generated Email")
            st.text_area("Output", generated_email, height=300)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
