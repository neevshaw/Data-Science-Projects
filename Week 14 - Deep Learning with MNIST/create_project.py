"""
Student-Project Matching System
--------------------------------
This script matches students to projects based on:
1. Student interests (non-technical)
2. Student technical skills/courses
3. Past project experience

Pipeline:
- A small open-source LLM (FLAN-T5-small) extracts relevant interests + technical skills
  from business-style project descriptions.
- A sentence-transformer computes similarity between student interests and project interests.
- Scores are combined into a weighted ranking.

Dependencies (install once):
    pip install transformers sentence-transformers torch
"""

import json
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util


# ------------------ Model Loading ------------------

print("Loading models... this may take a minute on first run.")
extractor = pipeline("text2text-generation", model="google/flan-t5-small")
embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")


# ------------------ Skill Extraction ------------------

# Controlled vocabularies for consistency
ALLOWED_INTERESTS = [
    "building websites", "building apps", "online ordering systems",
    "data dashboards", "image classification", "machine learning",
    "automation workflows", "customer analytics", "recommendation systems"
]

ALLOWED_SKILLS = [
    "HTML", "CSS", "JavaScript", "React", "APIs", "SQL", "Flask",
    "Django", "Python", "Git", "GitHub", "UI/UX", "Data Visualization",
    "TensorFlow", "PyTorch"
]


def extract_skills_and_interests(project_text):
    """
    Use FLAN-T5 to convert a plain project description into structured data:
    - relevant nontechnical interests
    - required technical skills
    """
    prompt = f"""
    You are a skill extraction engine.
    Given a business project description, return two JSON lists:
    1. "interests": choose from {ALLOWED_INTERESTS}
    2. "skills": choose from {ALLOWED_SKILLS}
    Only include items that are relevant.
    Project: {project_text}
    Return JSON:
    """

    raw = extractor(prompt, do_sample=False)[0]["generated_text"]

    try:
        data = json.loads(raw)
    except:
        # fallback if model output is messy
        data = {"interests": [], "skills": []}

    return data


# ------------------ Matching Engine ------------------

def match_students_to_projects(students, projects):
    """
    Rank students for each project by:
    - Interest similarity (cosine embedding)
    - Course overlap (skills match)
    - Experience (log-scaled project count)
    """
    results = {}
    for proj in projects:
        proj_text = f"{proj['title']} {proj['short_desc']} {proj['long_desc']}"
        proj_info = extract_skills_and_interests(proj_text)

        proj_interests = " ".join(proj_info["interests"])
        proj_skills = set(proj_info["skills"])

        proj_emb = embedder.encode(proj_interests, convert_to_tensor=True)

        scores = []
        for student in students:
            # Interest similarity
            student_interests = " ".join(student["interests"])
            student_emb = embedder.encode(student_interests, convert_to_tensor=True)
            interest_score = float(util.cos_sim(proj_emb, student_emb))

            # Skill/course overlap
            overlap = proj_skills.intersection(set(student["courses"]))
            course_score = len(overlap) / len(proj_skills) if proj_skills else 0.5

            # Experience (scaled to avoid bias)
            exp_score = np.log1p(student["projects_completed"]) / np.log1p(10)

            # Weighted score
            final_score = 0.4 * interest_score + 0.4 * course_score + 0.2 * exp_score
            scores.append((student["name"], final_score))

        results[proj["title"]] = sorted(scores, key=lambda x: x[1], reverse=True)

    return results


# ------------------ Example Run ------------------

if __name__ == "__main__":
    students = [
        {
            "name": "Alice",
            "interests": ["building websites", "data dashboards"],
            "courses": ["HTML", "CSS", "JavaScript", "React", "SQL"],
            "projects_completed": 2
        },
        {
            "name": "Bob",
            "interests": ["machine learning", "automation workflows"],
            "courses": ["Python", "TensorFlow"],
            "projects_completed": 0
        },
        {
            "name": "Charlie",
            "interests": ["building apps", "UI/UX"],
            "courses": ["JavaScript", "React", "Flask"],
            "projects_completed": 1
        },
    ]

    projects = [
        {
            "title": "Restaurant Online Ordering",
            "short_desc": "Set up an online ordering system",
            "long_desc": "We need a website where customers can order food online and track their order status."
        },
        {
            "title": "Customer Data Dashboard",
            "short_desc": "Dashboard for customer data",
            "long_desc": "Analyze customer ordering behavior and present insights with simple visualizations."
        }
    ]

    matches = match_students_to_projects(students, projects)

    for proj, ranked_students in matches.items():
        print(f"\nProject: {proj}")
        for student, score in ranked_students:
            print(f"  {student}: {score:.2f}")
