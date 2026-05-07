import os
import requests
import chromadb
import google.generativeai as genai

from sentence_transformers import SentenceTransformer

# =========================================================
# CONFIGURATION
# =========================================================

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}"
}

# =========================================================
# GEMINI CONFIGURATION
# =========================================================

genai.configure(api_key=GOOGLE_API_KEY)

llm = genai.GenerativeModel("gemini-2.5-flash")

# =========================================================
# TROUBLESHOOTING DOCUMENTS
# =========================================================

documents = [
    "ExecutorLostFailure often indicates executor crashes due to memory pressure or skew.",
    "OutOfMemoryError typically occurs during shuffle-heavy joins or aggregations.",
    "BroadcastTimeout happens when broadcast joins exceed timeout thresholds.",
    "Data skew can create uneven partition distribution causing slow stages.",
    "AnalysisException usually indicates SQL or schema mismatch problems.",
    "DeltaTableNotFoundException means the Delta table path is invalid or missing.",
    "Shuffle spill issues can often be solved using repartitioning.",
    "Py4JJavaError usually wraps underlying Spark exceptions."
]

# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =========================================================
# CHROMADB SETUP
# =========================================================

client = chromadb.Client()

collection = client.create_collection(
    name="spark_troubleshooting"
)

# =========================================================
# INSERT DOCUMENTS
# =========================================================

for i, doc in enumerate(documents):

    embedding = embedding_model.encode(doc).tolist()

    collection.add(
        ids=[str(i)],
        documents=[doc],
        embeddings=[embedding]
    )

# =========================================================
# FETCH JOBS
# =========================================================

def fetch_all_jobs():

    url = f"{DATABRICKS_HOST}/api/2.1/jobs/list"

    response = requests.get(
        url,
        headers=HEADERS
    )

    if response.status_code != 200:
        raise Exception(response.text)

    return response.json()

# =========================================================
# FETCH FAILED RUNS
# =========================================================

def fetch_failed_runs(job_id):

    url = f"{DATABRICKS_HOST}/api/2.1/jobs/runs/list"

    params = {
        "job_id": job_id,
        "limit": 5
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params
    )

    if response.status_code != 200:
        raise Exception(response.text)

    runs_json = response.json()

    failed_runs = []

    for run in runs_json.get("runs", []):

        state = run.get("state", {})

        if state.get("result_state") == "FAILED":

            failed_runs.append(
                run["run_id"]
            )

    return failed_runs

# =========================================================
# FETCH RUN OUTPUT
# =========================================================

def fetch_run_output(run_id):

    url = f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get-output"

    params = {
        "run_id": run_id
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params
    )

    if response.status_code != 200:
        raise Exception(response.text)

    return response.json()

# =========================================================
# EXTRACT LOGS
# =========================================================

def extract_log_text(output_json):

    logs = ""

    metadata = output_json.get("metadata", {})

    state = metadata.get("state", {})

    logs += f"""
STATE MESSAGE:
{state.get("state_message", "")}
"""

    notebook_output = output_json.get(
        "notebook_output",
        {}
    )

    logs += f"""

NOTEBOOK OUTPUT:
{notebook_output.get("result", "")}
"""

    return logs

# =========================================================
# RULE ENGINE
# =========================================================

def detect_issue(log_text):

    if "OutOfMemoryError" in log_text:
        return "Memory Failure"

    if "ExecutorLostFailure" in log_text:
        return "Executor Failure"

    if "BroadcastTimeout" in log_text:
        return "Broadcast Failure"

    if "AnalysisException" in log_text:
        return "SQL/Schema Error"

    if "DeltaTableNotFoundException" in log_text:
        return "Delta Table Missing"

    if "skew" in log_text.lower():
        return "Data Skew"

    return "Unknown Failure"

# =========================================================
# CHROMADB RETRIEVAL
# =========================================================

def retrieve_docs(query, top_k=3):

    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0]

# =========================================================
# LLM REASONING
# =========================================================

def llm_reasoning(issue_type, log_text, retrieved_docs):

    docs_text = "\n".join(retrieved_docs)

    prompt = f"""
You are an expert Databricks and Spark ETL engineer.

Analyze the following ETL failure.

ISSUE TYPE:
{issue_type}

LOGS:
{log_text}

RETRIEVED DOCS:
{docs_text}

Provide:
1. Root cause
2. Technical explanation
3. Severity
4. Recommended fixes
5. Prevention best practices
"""

    response = llm.generate_content(prompt)

    return response.text

# =========================================================
# MAIN AGENT
# =========================================================

def run_agent():

    print("Fetching Databricks Jobs...")

    jobs_json = fetch_all_jobs()

    jobs = jobs_json.get("jobs", [])

    for job in jobs:

        job_id = job["job_id"]

        job_name = job["settings"]["name"]

        print(f"\nANALYZING JOB: {job_name}")

        failed_runs = fetch_failed_runs(job_id)

        if not failed_runs:
            continue

        for run_id in failed_runs:

            output_json = fetch_run_output(run_id)

            log_text = extract_log_text(output_json)

            issue_type = detect_issue(log_text)

            retrieved_docs = retrieve_docs(issue_type)

            analysis = llm_reasoning(
                issue_type,
                log_text,
                retrieved_docs
            )

            print(f"""
==================================================
AI INCIDENT REPORT
==================================================

JOB:
{job_name}

RUN ID:
{run_id}

ISSUE:
{issue_type}

ANALYSIS:
{analysis}
""")

# =========================================================
# EXECUTION
# =========================================================

if __name__ == "__main__":

    run_agent()