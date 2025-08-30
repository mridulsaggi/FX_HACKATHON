import streamlit as st
import sqlite3
import pandas as pd
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

# === Load API Key ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

DB_PATH = "fx_trades.db"

# === Schema Description for Prompt ===
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

# === Prompt Builder ===
def create_prompt(user_question: str) -> str:
    return f'''
You are an expert SQL assistant that translates natural language questions into SQL queries for the FX trading database.

{SCHEMA_CONTEXT}

Respond ONLY in JSON with the following keys:
- "sql": SQL query string (or empty string if clarification is needed)
- "clarification": follow-up question if clarification is needed
- "explanation": short explanation of the query (or the ambiguity)

Now answer the following question:

Q: "{user_question}"
A:
'''

# === Generate SQL or Clarification ===
def generate_sql(user_question: str) -> dict:
    prompt = create_prompt(user_question)
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=400
    )

    content = response.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except:
        return {
            "sql": "",
            "clarification": "Sorry, I couldn't understand your question. Could you rephrase it?",
            "explanation": ""
        }

# === Execute SQL ===
def execute_sql(query: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)

# === Plot Graph ===
def plot_dataframe(df: pd.DataFrame):
    if df.empty:
        st.warning("No data to display.")
        return

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns found for plotting. Displaying table.")
        st.dataframe(df)
        return

    chart_type = st.selectbox("Chart type", ["Bar", "Line", "Area"], key="chart_type")

    st.subheader("📊 Data Visualization")
    if chart_type == "Bar":
        st.bar_chart(df)
    elif chart_type == "Line":
        st.line_chart(df)
    elif chart_type == "Area":
        st.area_chart(df)

# === Show Result ===
def show_results(sql_result: dict):
    st.success("✅ Query executed successfully!")
    st.caption("🧠 " + sql_result.get("explanation", ""))

    df, error = execute_sql(sql_result["sql"])
    if error:
        st.error(f"SQL Error: {error}")
        return

    plot_dataframe(df)
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", data=csv, file_name="results.csv", mime="text/csv")

    with st.expander("🧾 View / Edit SQL Query"):
        edited_sql = st.text_area("SQL Query:", value=sql_result["sql"], height=150)
        if st.button("Run Edited SQL"):
            df_edit, err_edit = execute_sql(edited_sql)
            if err_edit:
                st.error(f"SQL Error: {err_edit}")
            else:
                st.success("Edited query ran successfully!")
                plot_dataframe(df_edit)
                st.dataframe(df_edit)

# === Streamlit App ===
def main():
    st.set_page_config(page_title="FX Text-to-SQL", layout="wide")
    st.title("💱 FX Text-to-SQL Assistant")
    st.markdown("Ask any FX trade-related question in plain English and see the results in charts and tables.")

    if "clarification_mode" not in st.session_state:
        st.session_state.clarification_mode = False
    if "original_question" not in st.session_state:
        st.session_state.original_question = ""
    if "clarification_prompt" not in st.session_state:
        st.session_state.clarification_prompt = ""

    if not st.session_state.clarification_mode:
        user_question = st.text_input("🔍 Ask your question:")
        if st.button("Submit"):
            if not user_question.strip():
                st.warning("Please enter a question.")
                return

            with st.spinner("Thinking..."):
                result = generate_sql(user_question)

            if result["clarification"]:
                st.session_state.clarification_mode = True
                st.session_state.original_question = user_question
                st.session_state.clarification_prompt = result["clarification"]
                st.info(f"🤖 Clarification needed: {result['clarification']}")
            else:
                show_results(result)

    else:
        st.info(f"🤖 Clarification: {st.session_state.clarification_prompt}")
        clarification_input = st.text_input("✏️ Your clarification:")
        if st.button("Submit Clarification"):
            full_question = f"{st.session_state.original_question}. Clarification: {clarification_input}"
            with st.spinner("Thinking..."):
                result = generate_sql(full_question)

            if result["clarification"]:
                st.session_state.clarification_prompt = result["clarification"]
                st.info(f"🤖 More clarification needed: {result['clarification']}")
            else:
                st.session_state.clarification_mode = False
                st.session_state.original_question = ""
                st.session_state.clarification_prompt = ""
                show_results(result)

if __name__ == "__main__":
    main()
