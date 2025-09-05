import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import os
from dotenv import load_dotenv
import json

# Load API key from .env file
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize Groq/OpenAI client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

DB_PATH = "fx_trades.db"

SCHEMA_CONTEXT = """
You are working with the following FX trading database:

Table: trades
 - trade_id (INTEGER)
 - cp_id (INTEGER): links to counterparties.cp_id
 - px_type (TEXT): FX product type - can be 'spot', 'fwd', 'swap', 'ndf'
 - notl (REAL): Notional value
 - ccy_pair (TEXT): Currency pair
 - near_dt (TEXT): Near leg date (used in all products)
 - far_dt (TEXT): Far leg date (used only in 'swap' trades)
 - rate (REAL): Executed FX rate

Table: counterparties
 - cp_id (INTEGER)
 - cp_name (TEXT): Name of the counterparty
 - region (TEXT): Region of the counterparty
"""

def create_prompt(user_question: str) -> str:
    return f'''
You are an expert SQL assistant that translates natural language questions into precise SQL queries based on this database schema:

{SCHEMA_CONTEXT}

Follow these rules:

- If the question is clear, generate only the SQL query without explanation.
- If the question is ambiguous or incomplete, respond with a question asking the user for clarification.
- Your output must be a JSON object with keys:
  - "sql": the SQL query (or empty string if clarifying)
  - "clarification": clarification question (or empty string if none)
  - "explanation": short explanation of the query or what you want to clarify

Examples:

Q: "Show total notional by product type."
A:
{{
  "sql": "SELECT px_type, SUM(notl) AS total_notional FROM trades GROUP BY px_type;",
  "clarification": "",
  "explanation": "Aggregates total notional amount grouped by FX product type."
}}

Q: "What are the trades?"
A:
{{
  "sql": "",
  "clarification": "Which trade attributes are you interested in? For example, product type, dates, or counterparties?",
  "explanation": "The question is too broad, asking for clarification."
}}

Now answer the following question:

Q: "{user_question}"
A:
'''

# Call Groq AI to get SQL or ask for clarification
def generate_sql(user_question: str) -> dict:
    prompt = create_prompt(user_question)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=350,
    )
    content = response.choices[0].message.content.strip()
    try:
        result = json.loads(content)
    except Exception:
        result = {
            "sql": content,
            "clarification": "",
            "explanation": ""
        }
    return result

# Execute SQL query
def execute_sql(query: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)

# Plotting logic
def plot_dataframe(df: pd.DataFrame):
    if df.empty:
        st.warning("No data returned to plot.")
        return

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.dataframe(df)
        st.info("No numeric columns to plot.")
        return

    st.subheader("📊 Data Visualization")
    chart_type = st.selectbox("Choose chart type", ["Bar", "Line", "Area"], key="chart_type")

    if chart_type == "Bar":
        st.bar_chart(df)
    elif chart_type == "Line":
        st.line_chart(df)
    elif chart_type == "Area":
        st.area_chart(df)

# Main Streamlit app logic
def main():
    st.set_page_config(page_title="FX Text-to-SQL Assistant", layout="wide")
    st.title("💱 FX Text-to-SQL Assistant")

    st.markdown("Ask questions about FX trades in plain English. The AI will convert them into SQL and visualize the results.")

    # Initialize state
    if "awaiting_clarification" not in st.session_state:
        st.session_state.awaiting_clarification = False
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""
    if "clarification_prompt" not in st.session_state:
        st.session_state.clarification_prompt = ""

    if not st.session_state.awaiting_clarification:
        user_input = st.text_input("🔍 Ask your question:")
        if st.button("Submit") and user_input.strip():
            with st.spinner("Thinking..."):
                result = generate_sql(user_input)

            if result.get("clarification"):
                st.session_state.awaiting_clarification = True
                st.session_state.pending_question = user_input
                st.session_state.clarification_prompt = result["clarification"]
                st.info(f"🤖 AI needs clarification: {result['clarification']}")
                return

            # Proceed with valid SQL
            show_results(result)
    else:
        st.info(f"❓ AI asked: {st.session_state.clarification_prompt}")
        clarification = st.text_input("Your clarification:")

        if st.button("Submit Clarification"):
            full_question = f"{st.session_state.pending_question}. Clarification: {clarification}"
            with st.spinner("Processing clarified question..."):
                result = generate_sql(full_question)

            st.session_state.awaiting_clarification = False
            st.session_state.pending_question = ""
            st.session_state.clarification_prompt = ""

            # Show results
            show_results(result)

# Display results (data + chart + SQL)
def show_results(result):
    if result.get("sql") == "":
        st.warning("AI couldn't generate a query.")
        return

    st.success("✅ SQL query generated and executed.")
    st.caption("💬 " + result.get("explanation", ""))

    df, error = execute_sql(result["sql"])
    if error:
        st.error(f"SQL execution failed: {error}")
        return

    st.dataframe(df)
    plot_dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download as CSV", data=csv, file_name="query_results.csv", mime="text/csv")

    with st.expander("🧾 View / Edit SQL Query"):
        sql_editor = st.text_area("Edit the generated SQL:", value=result["sql"])
        if st.button("Run Edited SQL"):
            new_df, new_error = execute_sql(sql_editor)
            if new_error:
                st.error(new_error)
            else:
                st.dataframe(new_df)
                plot_dataframe(new_df)

                csv_edit = new_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Edited Results", data=csv_edit, file_name="edited_query_results.csv", mime="text/csv")

if __name__ == "__main__":
    main()
