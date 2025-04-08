from gen_model import run
import json


def safe_json_parse(response_text):
    try:
        # If extra text is added accidentally, extract JSON array only
        json_start = response_text.find('[')
        json_end = response_text.rfind(']')
        json_str = response_text[json_start:json_end+1]
        return json.loads(json_str)
    except Exception as e:
        print("⚠️ JSON parsing failed:", e)
        return []

def find_most_accurate(matches, job_query):
    # Build a structured prompt with job query and matched assessments
    prompt = f"""
You are an expert recruiter AI. A user has provided a job query, and we’ve already shortlisted some relevant assessments based on embedding similarity.

Your task: From the provided list, **rank and return the top 10 most relevant assessments** for this job query based on the following parameters:
- Match to job description or skill requirements
- Suitability of duration
- Match to job level
- Fit for remote/adaptive testing needs (if relevant)

### Job Query:
{job_query}

### Matched Assessments:
"""

    for i, match in enumerate(matches):
        prompt += f"""
Assessment {i+1}:
- Assessment Name: {match['Assessment Name']}
- Description: {match['Description']}
- Duration: {match['Duration']}
- Job Level: {match['Job Level']}
- Remote Testing: {match['Remote Testing']}
- Adaptive Testing: {match['Adaptive Testing']}
- Language: {match['Language']}
"""

    prompt += """
Now return the **top 1 to 10 assessments** in JSON list format like this:

[
  {{
    "Assessment_Name": "...",
    "Link": "...",
    "Duration": "...",
    "Remote_Testing": "...",
    "Adaptive_Testing": "...",
    "Language": "..."
  }},
  ...
]
Only return the JSON list. Do not include explanation.
"""

    # Send the prompt to the LLM
    response = run(prompt)
    top_matches = safe_json_parse(response)
    return top_matches
