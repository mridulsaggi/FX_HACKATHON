import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import numpy as np

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

# Custom CSS for enhanced UI
def load_custom_css():
    st.markdown("""
    <style>
    /* Global Dark Theme */
    .stApp {
        background: linear-gradient(to right, #1e272e, #2f3640);
        color: #f1f2f6;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Main Container */
    .main-container {
        background: #2f3542;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
        margin-bottom: 2rem;
        border: 1px solid #57606f;
    }

    /* Section Containers */
    .query-section, .clarification-section, .results-section {
        background: #3d3d3d;
        color: #f1f2f6;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #57606f;
    }

    .query-section {
        background: #1e90ff;
        color: white;
    }

    .clarification-section {
        background: #ff6b6b;
        color: white;
    }

    /* SQL Query Display */
    .sql-display {
        background: #1e272e;
        color: #dfe6e9;
        padding: 1rem;
        border-radius: 10px;
        font-family: monospace;
        margin: 1rem 0;
        border-left: 4px solid #70a1ff;
    }

    /* Metric Cards */
    .metric-card {
        background: #57606f;
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.4);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(to right, #70a1ff, #3742fa);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 30px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(to right, #5352ed, #70a1ff);
        box-shadow: 0 0 10px rgba(112, 161, 255, 0.5);
        transform: translateY(-1px);
    }

    /* Visualization Container */
    .visualization-container {
        background: #2f3640;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        margin: 1rem 0;
        border: 1px solid #57606f;
    }

    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.25rem;
    }

    .status-success {
        background: #2ed573;
        color: #1e272e;
    }

    .status-warning {
        background: #ffa502;
        color: #1e272e;
    }

    .status-error {
        background: #ff4757;
        color: white;
    }

    /* Inputs */
    input, textarea {
        background-color: #2f3640 !important;
        color: white !important;
        border: 1px solid #57606f !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """, unsafe_allow_html=True)



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

def generate_sql(user_question: str) -> dict:
    """Generate SQL from natural language question"""
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

def execute_sql(query: str):
    """Execute SQL query and return results"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)

def create_interactive_visualization(df: pd.DataFrame, chart_key: str = "main"):
    """Create interactive visualization with real-time controls"""
    if df.empty:
        st.warning("📭 No data to visualize")
        return
    
    # Get column information
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    all_cols = df.columns.tolist()
    
    if not numeric_cols and not categorical_cols:
        st.info("📊 No visualizable columns found")
        return
    
    st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
    st.subheader("📊 Interactive Data Visualization")
    
    # Create columns for controls
    control_col1, control_col2, control_col3 = st.columns(3)
    
    with control_col1:
        chart_type = st.selectbox(
            "📈 Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Box Plot", "Heatmap", "Histogram"],
            key=f"chart_type_{chart_key}",
            help="Choose the type of visualization"
        )
    
    with control_col2:
        if numeric_cols:
            y_column = st.selectbox(
                "📊 Y-Axis (Values)",
                numeric_cols,
                key=f"y_col_{chart_key}",
                help="Select the numeric column for Y-axis"
            )
        else:
            y_column = None
    
    with control_col3:
        if categorical_cols and chart_type in ["Bar Chart", "Line Chart", "Box Plot", "Pie Chart"]:
            x_column = st.selectbox(
                "📋 X-Axis (Categories)",
                categorical_cols + [None],
                key=f"x_col_{chart_key}",
                help="Select the categorical column for X-axis"
            )
        elif chart_type == "Scatter Plot" and len(numeric_cols) >= 2:
            x_column = st.selectbox(
                "📍 X-Axis (Numeric)",
                [col for col in numeric_cols if col != y_column],
                key=f"x_col_{chart_key}",
                help="Select the numeric column for X-axis"
            )
        else:
            x_column = None
    
    # Additional controls row
    if chart_type in ["Scatter Plot", "Bar Chart", "Line Chart"] and categorical_cols:
        color_col1, size_col1 = st.columns(2)
        
        with color_col1:
            color_column = st.selectbox(
                "🎨 Color By",
                [None] + categorical_cols,
                key=f"color_col_{chart_key}",
                help="Group data by color"
            )
        
        with size_col1:
            if chart_type == "Scatter Plot" and numeric_cols:
                size_column = st.selectbox(
                    "📏 Size By",
                    [None] + [col for col in numeric_cols if col not in [x_column, y_column]],
                    key=f"size_col_{chart_key}",
                    help="Size points by this column"
                )
            else:
                size_column = None
    else:
        color_column = None
        size_column = None
    
    # Generate the chart based on selections
    try:
        fig = None
        
        if chart_type == "Bar Chart" and x_column and y_column:
            # Aggregate data if needed
            if df[x_column].dtype == 'object':
                agg_df = df.groupby(x_column)[y_column].sum().reset_index()
                fig = px.bar(
                    agg_df, 
                    x=x_column, 
                    y=y_column,
                    title=f"{y_column} by {x_column}",
                    color=x_column if not color_column else color_column,
                    template="plotly_dark"
                )
            else:
                fig = px.bar(
                    df, 
                    x=x_column, 
                    y=y_column,
                    color=color_column,
                    title=f"{y_column} by {x_column}",
                    template="plotly_dark"
                )
        
        elif chart_type == "Line Chart" and y_column:
            if x_column:
                fig = px.line(
                    df, 
                    x=x_column, 
                    y=y_column,
                    color=color_column,
                    title=f"{y_column} over {x_column}",
                    markers=True,
                    template="plotly_dark"
                )
            else:
                fig = px.line(
                    df.reset_index(), 
                    x='index', 
                    y=y_column,
                    title=f"{y_column} Trend",
                    markers=True,
                    template="plotly_dark"
                )
        
        elif chart_type == "Scatter Plot" and x_column and y_column:
            fig = px.scatter(
                df, 
                x=x_column, 
                y=y_column,
                color=color_column,
                size=size_column,
                title=f"{y_column} vs {x_column}",
                template="plotly_dark"
            )
        
        elif chart_type == "Pie Chart" and x_column:
            # Create pie chart from value counts
            value_counts = df[x_column].value_counts()
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {x_column}",
                template="plotly_dark"
            )
        
        elif chart_type == "Box Plot" and y_column:
            if x_column:
                fig = px.box(
                    df, 
                    x=x_column, 
                    y=y_column,
                    color=color_column,
                    title=f"{y_column} Distribution by {x_column}",
                    template="plotly_dark"
                )
            else:
                fig = px.box(
                    df, 
                    y=y_column,
                    title=f"{y_column} Distribution",
                    template="plotly_dark"
                )
        
        elif chart_type == "Histogram" and y_column:
            fig = px.histogram(
                df, 
                x=y_column,
                color=color_column,
                title=f"Distribution of {y_column}",
                nbins=30,
                template="plotly_dark"
            )
        
        elif chart_type == "Heatmap" and len(numeric_cols) >= 2:
            # Create correlation heatmap
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap",
                template="plotly_dark",
                color_continuous_scale="RdBu_r"
            )
        
        if fig:
            # Customize the figure
            fig.update_layout(
                height=500,
                showlegend=True,
                font=dict(size=12),
                title_font_size=16,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{chart_key}")
        else:
            st.warning("⚠️ Cannot create chart with selected parameters. Please try different columns.")
    
    except Exception as e:
        st.error(f"❌ Error creating visualization: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_data_summary(df: pd.DataFrame):
    """Display data summary and statistics"""
    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    st.subheader("📋 Data Summary")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(df)}</h3>
            <p>Total Rows</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(df.columns)}</h3>
            <p>Columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{numeric_cols}</h3>
            <p>Numeric Columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        categorical_cols = len(df.select_dtypes(include=['object', 'string']).columns)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{categorical_cols}</h3>
            <p>Text Columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Data preview
    st.subheader("📊 Data Preview")
    st.dataframe(df, use_container_width=True, height=300)
    
    # Download options
    st.subheader("💾 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download CSV",
            data=csv,
            file_name=f"fx_query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # JSON export
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            "📄 Download JSON",
            data=json_data,
            file_name=f"fx_query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="FX Text-to-SQL Assistant",
        layout="wide",
        page_icon="💱",
        initial_sidebar_state="expanded"
    )
    
    load_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-container">
        <h1 style="text-align: center; color: #2d3436; margin-bottom: 0.5rem;">💱 FX Text-to-SQL Assistant</h1>
        <p style="text-align: center; color: #636e72; font-size: 1.1rem;">
            Ask questions in natural language and get instant SQL insights with interactive visualizations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with schema info
    with st.sidebar:
        st.header("📊 Database Schema")
        
        with st.expander("🏦 Trades Table", expanded=True):
            st.markdown("""
            - `trade_id`: Unique identifier
            - `cp_id`: Counterparty ID
            - `px_type`: Product type (spot, fwd, swap, ndf)
            - `notl`: Notional value
            - `ccy_pair`: Currency pair
            - `near_dt`: Near leg date
            - `far_dt`: Far leg date (swaps)
            - `rate`: FX rate
            """)
        
        with st.expander("🏢 Counterparties Table"):
            st.markdown("""
            - `cp_id`: Counterparty ID
            - `cp_name`: Name
            - `region`: Geographic region
            """)
        
        # Example queries
        st.header("💡 Example Queries")
        examples = [
            "Show total notional by product type",
            "Top 5 currency pairs by volume",
            "Trading activity by region",
            "Average rates by currency pair",
            "Monthly trading volumes"
        ]
        
        for example in examples:
            if st.button(example, key=f"ex_{hash(example)}", use_container_width=True):
                st.session_state.user_question = example
                st.rerun()
    
    # Initialize session state
    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = "asking"  # asking, clarifying, results
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""
    if "clarification_question" not in st.session_state:
        st.session_state.clarification_question = ""
    if "sql_result" not in st.session_state:
        st.session_state.sql_result = {}
    if "query_data" not in st.session_state:
        st.session_state.query_data = None
    if "final_question" not in st.session_state:
        st.session_state.final_question = ""
    
    # Main conversation flow
    if st.session_state.conversation_state == "asking":
        # Initial question input
        st.markdown('<div class="query-section">', unsafe_allow_html=True)
        st.subheader("🔍 Ask Your Question")
        
        user_input = st.text_input(
            "💬 What would you like to know about your FX trades?",
            value=st.session_state.user_question,
            placeholder="e.g., Show me the total notional by product type",
            key="user_input"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🚀 Submit", type="primary"):
                if user_input.strip():
                    st.session_state.user_question = user_input
                    
                    with st.spinner("🤖 Analyzing your question..."):
                        result = generate_sql(user_input)
                    
                    if result.get("clarification"):
                        st.session_state.clarification_question = result["clarification"]
                        st.session_state.conversation_state = "clarifying"
                    else:
                        st.session_state.sql_result = result
                        st.session_state.final_question = user_input
                        st.session_state.conversation_state = "results"
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.conversation_state == "clarifying":
        # Clarification needed
        st.markdown('<div class="clarification-section">', unsafe_allow_html=True)
        
        st.subheader("❓ Need Clarification")
        st.info(f"**Your question:** {st.session_state.user_question}")
        st.warning(f"**AI asks:** {st.session_state.clarification_question}")
        
        clarification = st.text_input(
            "💬 Please provide more details:",
            placeholder="Add specific information to clarify your question"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("✅ Submit Clarification", type="primary"):
                if clarification.strip():
                    full_question = f"{st.session_state.user_question}. Clarification: {clarification}"
                    
                    with st.spinner("🔄 Processing clarified question..."):
                        result = generate_sql(full_question)
                    
                    st.session_state.sql_result = result
                    st.session_state.final_question = full_question
                    st.session_state.conversation_state = "results"
                    st.rerun()
        
        with col2:
            if st.button("🔄 Start Over"):
                st.session_state.conversation_state = "asking"
                st.session_state.user_question = ""
                st.session_state.clarification_question = ""
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.conversation_state == "results":
        # Display results
        result = st.session_state.sql_result
        
        if result.get("sql"):
            # Show status and question
            st.markdown(f"""
            <div class="results-section">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div>
                        <span class="status-badge status-success">✅ Query Generated</span>
                        <span class="status-badge status-success">📊 Data Retrieved</span>
                    </div>
                    <button onclick="window.location.reload()" style="background: #ddd; border: none; padding: 0.5rem; border-radius: 5px; cursor: pointer;">
                        🔄 New Query
                    </button>
                </div>
                <p><strong>Question:</strong> {st.session_state.final_question}</p>
                <p><strong>Explanation:</strong> {result.get('explanation', 'Query executed successfully')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Execute SQL if not already done
            if st.session_state.query_data is None:
                with st.spinner("⚡ Executing SQL query..."):
                    df, error = execute_sql(result["sql"])
                
                if error:
                    st.error(f"❌ SQL Error: {error}")
                    
                    # Show SQL for debugging
                    st.markdown('<div class="sql-display">', unsafe_allow_html=True)
                    st.code(result["sql"], language="sql")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if st.button("🔄 Try Again"):
                        st.session_state.conversation_state = "asking"
                        st.rerun()
                    return
                else:
                    st.session_state.query_data = df
            
            df = st.session_state.query_data
            
            # Show SQL query
            with st.expander("🔍 View SQL Query", expanded=False):
                st.markdown('<div class="sql-display">', unsafe_allow_html=True)
                st.code(result["sql"], language="sql")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Interactive visualization (this updates without page reload)
            create_interactive_visualization(df, "main")
            
            # Data summary
            display_data_summary(df)
            
            # New query button
            if st.button("🔄 Ask New Question", type="primary"):
                st.session_state.conversation_state = "asking"
                st.session_state.user_question = ""
                st.session_state.query_data = None
                st.session_state.sql_result = {}
                st.rerun()

if __name__ == "__main__":
    main()