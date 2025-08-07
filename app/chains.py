import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import requests
from bs4 import BeautifulSoup

# Load environment variables
env_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path=env_path)

# 🔧 Prompt generator function
def generate_prompt(job_description, portfolio_text, name, university, languages):
    return f"""
### JOB DESCRIPTION:
{job_description}

### CANDIDATE:
You are {name}, an undergraduate student at {university} with a strong background in machine learning and multiple projects under your belt.

You can speak the following languages: {languages}

### PORTFOLIO CONTENT (Extracted from GitHub, LinkedIn, etc.):
{portfolio_text}

### INSTRUCTION:
Write a cold email to a recruiter based on the job description above and the candidate's profile.
✅ The email must include:
- A clear and relevant **Subject line**
- A professional **greeting**
- A concise but strong **introduction** of the candidate
- Incorporate relevant **skills**, **languages**, and **portfolio achievements** that align with the job description
- Be **concise**, **professional**, and **persuasive**
- Include a **call-to-action** (e.g., requesting a call or interview)
- A **polite closing** and proper **signature**

Do not include markdown or HTML. Return plain email text.

### FINAL EMAIL:
"""
def extract_text_from_links(link_list):
    

    extracted = []
    for url in link_list:
        try:
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40)
            extracted.append(f"\n--- Content from {url} ---\n{text[:1000]}")
        except Exception as e:
            extracted.append(f"\n--- Failed to fetch {url} ---\nReason: {e}")

    return "\n".join(extracted)
    
# ✅ Extract job description from a given job link
def fetch_job_description_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return f"❌ Failed to fetch content. Status code: {response.status_code}"
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40])
        return text[:2000]
    except Exception as e:
        return f"❌ Error fetching job description: {e}"
# 🔗 Main class
class Chain:
    def __init__(self):
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise EnvironmentError("HF_TOKEN not found in .env")
        self.client = InferenceClient(
            provider="nebius",
            api_key=hf_token
        )
        self.model = "meta-llama/Llama-3.1-8B-Instruct"

    

    def extract_jobs(self, cleaned_text: str):
        prompt = f"""
You are an expert data extractor.

### SCRAPED TEXT FROM WEBSITE:
{cleaned_text}

### INSTRUCTION:
The scraped text is from the careers page of a company website.
Extract the job postings and return them as a JSON array.
Each job must have:
- role
- experience
- skills
- description

Return only valid JSON array. Do NOT include explanations or preamble.
        """

        print("🔍 Invoking model for job extraction...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        output = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(output)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            print("❌ Failed to parse response as JSON.")
            print("🔎 Raw output:\n", output)
            raise ValueError("Model output is not valid JSON.")

    def write_mail(self, job: dict, links: str, name="Abhivyakti", university="Banasthali University", languages="English"):
        prompt = generate_prompt(
            job_description=json.dumps(job, indent=2),
            link_list=links,
            name=name,
            university=university,
            languages=languages
        )

        print("✉️ Invoking model to generate email...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content.strip()


# 🧪 Test
if __name__ == "__main__":
    chain = Chain()

    sample_text = """
We are hiring!

Position: Machine Learning Engineer
Experience: 2+ years
Skills: Python, TensorFlow, Scikit-learn
Description: Build and optimize ML pipelines in production.

Position: AI Research Intern
Experience: 0-1 years
Skills: PyTorch, NLP
Description: Assist the research team in training deep learning models.
    """

    print("\n--- Extracted Jobs ---")
    jobs = chain.extract_jobs(sample_text)
    print(json.dumps(jobs, indent=2))

    print("\n--- Generated Email ---")
    email = chain.write_mail(
        job=jobs[0],
        links="https://github.com/abhivyakti003/ml-projects\nhttps://abhivyakti-portfolio.streamlit.app"
    )
    print(email)
