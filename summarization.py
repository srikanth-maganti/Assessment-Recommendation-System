from gen_model import run


def summarizer(job_query):
    prompt= f"""
    You are an AI that summarizes assessment details in the style of natural job requirement queries.

    You are given the following assessment information query:

    {job_query}

    Write a **one-paragraph recruiter-style summary** that sounds like a job request.

    Use these examples as your tone and style reference:

    1. "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."

    2. "Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript. Need an assessment package that can test all skills with max duration of 60 minutes."

    3. "Here is a JD text, can you recommend some assessment that can help me screen applications. Time limit is less than 30 minutes."

    4. "I am hiring for an analyst and want applications to be screened using Cognitive and personality tests, what options are available within 45 mins."

    Your output must:
    - Summarize the kind of hiring situation this assessment fits
    - Include job level, relevant skills or focus areas, test type and total duration
    - Sound conversational and natural
    - Be written in one concise paragraph
    - Match the **style** of the examples above so it's useful for embedding-based similarity

    Only output the summary text.
    """

    return run(prompt)

