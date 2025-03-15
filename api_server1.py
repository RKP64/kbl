import os
import io
import re
import uvicorn
import tempfile
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine
from dotenv import load_dotenv
import boto3
import sqlite3

# LangChain imports
from langchain_experimental.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

# ---------------------------------------------------------------------------
# 1. Load Environment Variables
# ---------------------------------------------------------------------------
load_dotenv()
# Ensure your .env contains credentials and settings for:
# PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DB_NAME,
# DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME,
# S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION,
# SQLITE_DB_PATH,
# OPENAI_API_KEY

# ---------------------------------------------------------------------------
# 2. Initialize FastAPI
# ---------------------------------------------------------------------------
app = FastAPI()

# Optionally, add CORS middleware if your client (e.g., a React app) is hosted on a different origin:
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# ---------------------------------------------------------------------------
# 3. Define the Request Model
# ---------------------------------------------------------------------------
class NaturalPromptRequest(BaseModel):
    prompt: str  # e.g., "Load from MySQL and compute the average sales"

# ---------------------------------------------------------------------------
# 4. Naive Natural Language Parser for Data Source
# ---------------------------------------------------------------------------
def parse_data_source_from_prompt(prompt: str) -> (str, str):
    """
    Examines the prompt text to detect keywords that indicate the data source.
    Returns a tuple: (data_source, question).
    Defaults to "PostgreSQL Database" if no known keyword is found.
    """
    prompt_lower = prompt.lower()
    if "mysql" in prompt_lower or "mariadb" in prompt_lower:
        data_source = "MySQL/MariaDB Database"
        question = re.sub(r"(mysql|mariadb)", "", prompt, flags=re.IGNORECASE).strip()
    elif "postgres" in prompt_lower:
        data_source = "PostgreSQL Database"
        question = re.sub(r"(postgresql|postgres)", "", prompt, flags=re.IGNORECASE).strip()
    elif "s3" in prompt_lower:
        data_source = "AWS S3 (Excel)"
        question = re.sub(r"s3", "", prompt, flags=re.IGNORECASE).strip()
    elif "sqlite" in prompt_lower:
        data_source = "SQLite Database"
        question = re.sub(r"sqlite", "", prompt, flags=re.IGNORECASE).strip()
    elif "local file" in prompt_lower:
        data_source = "Local File"
        question = re.sub(r"local file", "", prompt, flags=re.IGNORECASE).strip()
    else:
        # Default to PostgreSQL if no keyword is detected
        data_source = "PostgreSQL Database"
        question = prompt.strip()
    return data_source, question

# ---------------------------------------------------------------------------
# 5. Data Loading Functions for Each Data Source
# ---------------------------------------------------------------------------
def load_postgresql_data() -> pd.DataFrame:
    db_host = os.getenv("PG_HOST", "your_postgres_host")
    db_port = os.getenv("PG_PORT", "5432")
    db_user = os.getenv("PG_USER", "your_postgres_user")
    db_password = os.getenv("PG_PASSWORD", "your_postgres_password")
    db_name = os.getenv("PG_DB_NAME", "your_postgres_db")
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(connection_string)
    query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    tables_df = pd.read_sql(query, engine)
    if tables_df.empty:
        raise Exception("No tables found in PostgreSQL.")
    loaded_dfs = []
    for table_name in tables_df["table_name"]:
        temp_df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        loaded_dfs.append(temp_df)
    merged_df = pd.concat(loaded_dfs, ignore_index=True)
    return merged_df

def load_mysql_data() -> pd.DataFrame:
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_user = os.getenv("DB_USER", "your_username")
    db_password = os.getenv("DB_PASSWORD", "your_password")
    db_name = os.getenv("DB_NAME", "your_database")
    connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(connection_string)
    tables = pd.read_sql("SHOW TABLES", engine)
    loaded_dfs = []
    for table in tables.iloc[:, 0]:
        temp_df = pd.read_sql(f"SELECT * FROM {table}", engine)
        loaded_dfs.append(temp_df)
    merged_df = pd.concat(loaded_dfs, ignore_index=True)
    return merged_df

def load_s3_data() -> pd.DataFrame:
    s3_bucket = os.getenv("S3_BUCKET")
    if not s3_bucket:
        raise Exception("S3_BUCKET environment variable is not set.")
    s3 = boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    response = s3.list_objects_v2(Bucket=s3_bucket)
    loaded_dfs = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if key.lower().endswith(".csv") or key.lower().endswith(".xlsx"):
                s3_response = s3.get_object(Bucket=s3_bucket, Key=key)
                file_data = s3_response["Body"].read()
                file_like = io.BytesIO(file_data)
                if key.lower().endswith(".csv"):
                    temp_df = pd.read_csv(file_like)
                else:
                    temp_df = pd.read_excel(file_like)
                loaded_dfs.append(temp_df)
    if loaded_dfs:
        return pd.concat(loaded_dfs, ignore_index=True)
    else:
        raise Exception("No CSV/Excel files found in S3.")

def load_sqlite_data() -> pd.DataFrame:
    db_path = os.getenv("SQLITE_DB_PATH", "data.db")
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    loaded_dfs = []
    for table_name in tables["name"]:
        temp_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        loaded_dfs.append(temp_df)
    conn.close()
    if loaded_dfs:
        return pd.concat(loaded_dfs, ignore_index=True)
    else:
        raise Exception("No tables found in SQLite.")

def load_local_file_data() -> pd.DataFrame:
    # Placeholder: implement based on how you want to handle local file input.
    raise Exception("Local file data source not implemented yet.")

# Master loader that dispatches based on the data source string.
def load_data(data_source: str) -> pd.DataFrame:
    if data_source == "PostgreSQL Database":
        return load_postgresql_data()
    elif data_source == "MySQL/MariaDB Database":
        return load_mysql_data()
    elif data_source == "AWS S3 (Excel)":
        return load_s3_data()
    elif data_source == "SQLite Database":
        return load_sqlite_data()
    elif data_source == "Local File":
        return load_local_file_data()
    else:
        raise Exception(f"Unknown data source '{data_source}'")

# ---------------------------------------------------------------------------
# 6. FastAPI Endpoint: Natural Language Prompt
# ---------------------------------------------------------------------------
@app.post("/process-natural-prompt")
async def process_natural_prompt(request: NaturalPromptRequest):
    try:
        # 1. Parse the prompt to determine the data source and extract the actual question.
        data_source, question = parse_data_source_from_prompt(request.prompt)

        # 2. Load the data using the determined data source.
        df = load_data(data_source)

        # 3. Write the DataFrame to a temporary CSV file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name

        # 4. Get the OpenAI API key.
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise Exception("OPENAI_API_KEY is not set.")

        # 5. Create the LangChain CSV Agent.
        agent = create_csv_agent(
            ChatOpenAI(
                temperature=0,
                model="gpt-4-turbo",  # or "gpt-3.5-turbo"
                openai_api_key=openai_api_key
            ),
            tmp_file_path,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            allow_dangerous_code=True
        )

        # 6. Run the agent with the parsed question.
        answer = agent.run(question)
        return {"data_source": data_source, "parsed_question": question, "answer": answer}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        # Clean up the temporary CSV file.
        if "tmp_file_path" in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

# ---------------------------------------------------------------------------
# 7. Run the FastAPI Server (Configured for Render)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Render sets the port in the PORT environment variable.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


