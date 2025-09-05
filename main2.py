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
import time

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

# Dark Theme CSS
def load_dark_theme_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
        color: #e0e6ed;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Landing Page Styles */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 15, 35, 0.95) 0%, rgba(26, 26, 46, 0.95) 100%);
        padding: 4rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #64ffda 0%, #1de9b6 50%, #00bcd4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    .hero-subtitle {
        font-size: 1.4rem;
        color: #b0bec5;
        margin-bottom: 2rem;
        font-weight: 400;
        line-height: 1.6;
    }
    
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(100, 255, 218, 0.1) 0%, rgba(29, 233, 182, 0.05) 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(100, 255, 218, 0.2);
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 16px 40px rgba(100, 255, 218, 0.2);
        border-color: rgba(100, 255, 218, 0.4);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #64ffda;
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        color: #90a4ae;
        line-height: 1.5;
    }
    
    /* Main Container */
    .main-container {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.9) 0%, rgba(22, 33, 62, 0.9) 100%);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.4);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
    }
    
    /* Navigation */
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background: rgba(15, 15, 35, 0.95);
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(100, 255, 218, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .nav-logo {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #64ffda 0%, #1de9b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .nav-links {
        display: flex;
        gap: 2rem;
    }
    
    .nav-link {
        color: #b0bec5;
        text-decoration: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .nav-link:hover {
        background: rgba(100, 255, 218, 0.1);
        color: #64ffda;
    }
    
    /* Query Section */
    .query-section {
        background: linear-gradient(135deg, rgba(29, 233, 182, 0.15) 0%, rgba(0, 188, 212, 0.15) 100%);
        color: #e0e6ed;
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(29, 233, 182, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Clarification Section */
    .clarification-section {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, rgba(255, 152, 0, 0.15) 100%);
        color: #e0e6ed;
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 193, 7, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Results Section */
    .results-section {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.8) 0%, rgba(22, 33, 62, 0.8) 100%);
        padding: 2.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
        backdrop-filter: blur(15px);
    }
    
    /* SQL Display */
    .sql-display {
        background: linear-gradient(135deg, rgba(15, 15, 35, 0.95) 0%, rgba(26, 26, 46, 0.95) 100%);
        color: #64ffda;
        padding: 1.5rem;
        border-radius: 12px;
        font-family: 'JetBrains Mono', monospace;
        margin: 1rem 0;
        border-left: 4px solid #1de9b6;
        border: 1px solid rgba(100, 255, 218, 0.2);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(100, 255, 218, 0.1) 0%, rgba(29, 233, 182, 0.1) 100%);
        color: #e0e6ed;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(100, 255, 218, 0.2);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(100, 255, 218, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #64ffda;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: #90a4ae;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #64ffda 0%, #1de9b6 50%, #00bcd4 100%);
        color: #0f0f23;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(100, 255, 218, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(100, 255, 218, 0.4);
        background: linear-gradient(135deg, #1de9b6 0%, #00bcd4 50%, #64ffda 100%);
    }
    
    /* Visualization Container */
    .visualization-container {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.8) 0%, rgba(22, 33, 62, 0.8) 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4);
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-success {
        background: linear-gradient(135deg, #1de9b6 0%, #64ffda 100%);
        color: #0f0f23;
        box-shadow: 0 4px 12px rgba(29, 233, 182, 0.3);
    }
    
    .status-warning {
        background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%);
        color: #0f0f23;
        box-shadow: 0 4px 12px rgba(255, 193, 7, 0.3);
    }
    
    .status-error {
        background: linear-gradient(135deg, #f44336 0%, #e91e63 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(244, 67, 54, 0.3);
    }
    
    /* Input Styles */
    .stTextInput > div > div > input {
        background: rgba(26, 26, 46, 0.8);
        color: #e0e6ed;
        border: 2px solid rgba(100, 255, 218, 0.2);
        border-radius: 12px;
        padding: 0.8rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #64ffda;
        box-shadow: 0 0 0 3px rgba(100, 255, 218, 0.1);
    }
    
    /* Selectbox Styles */
    .stSelectbox > div > div > select {
        background: rgba(26, 26, 46, 0.8);
        color: #e0e6ed;
        border: 2px solid rgba(100, 255, 218, 0.2);
        border-radius: 12px;
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(15, 15, 35, 0.95) 0%, rgba(26, 26, 46, 0.95) 100%);
        border-right: 1px solid rgba(100, 255, 218, 0.2);
    }
    
    /* Animation Classes */
    .fade-in {
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-up {
        animation: slideUp 0.6s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(100, 255, 218, 0.3);
        border-radius: 50%;
        border-top-color: #64ffda;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Quick Start Cards */
    .quick-start-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .quick-start-card {
        background: linear-gradient(135deg, rgba(100, 255, 218, 0.05) 0%, rgba(29, 233, 182, 0.05) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(100, 255, 218, 0.2);
        cursor: pointer;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .quick-start-card:hover {
        background: linear-gradient(135deg, rgba(100, 255, 218, 0.1) 0%, rgba(29, 233, 182, 0.1) 100%);
        border-color: rgba(100, 255, 218, 0.4);
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(100, 255, 218, 0.2);
    }
    
    .quick-start-title {
        color: #64ffda;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .quick-start-description {
        color: #90a4ae;
        font-size: 0.9rem;
        line-height: 1.4;
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
    
    st.markdown('<div class="visualization-container slide-up">', unsafe_allow_html=True)
    st.markdown("### 📊 Interactive Data Visualization")
    
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
        
        # Dark theme template for Plotly
        dark_template = {
            "layout": {
                "paper_bgcolor": "rgba(26, 26, 46, 0.8)",
                "plot_bgcolor": "rgba(15, 15, 35, 0.8)",
                "font": {"color": "#e0e6ed"},
                "colorway": ["#64ffda", "#1de9b6", "#00bcd4", "#26c6da", "#4dd0e1", "#80deea", "#b2ebf2", "#e0f7fa"]
            }
        }
        
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
            # Customize the figure with dark theme
            fig.update_layout(
                height=500,
                showlegend=True,
                font=dict(size=12, color="#e0e6ed"),
                title_font_size=16,
                margin=dict(l=40, r=40, t=60, b=40),
                paper_bgcolor="rgba(26, 26, 46, 0.8)",
                plot_bgcolor="rgba(15, 15, 35, 0.8)",
                colorway=["#64ffda", "#1de9b6", "#00bcd4", "#26c6da", "#4dd0e1", "#80deea", "#b2ebf2", "#e0f7fa"]
            )
            
            # Update axes colors
            fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.1)", color="#e0e6ed")
            fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.1)", color="#e0e6ed")
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{chart_key}")
        else:
            st.warning("⚠️ Cannot create chart with selected parameters. Please try different columns.")
    
    except Exception as e:
        st.error(f"❌ Error creating visualization: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_data_summary(df: pd.DataFrame):
    """Display data summary and statistics"""
    st.markdown('<div class="results-section slide-up">', unsafe_allow_html=True)
    st.markdown("### 📋 Data Summary")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Total Rows</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df.columns)}</div>
            <div class="metric-label">Columns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{numeric_cols}</div>
            <div class="metric-label">Numeric Columns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        categorical_cols = len(df.select_dtypes(include=['object', 'string']).columns)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{categorical_cols}</div>
            <div class="metric-label">Text Columns</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data preview
    st.markdown("### 📊 Data Preview")
    
    # Style the dataframe for dark theme
    styled_df = df.style.set_properties(**{
        'background-color': 'rgba(26, 26, 46, 0.8)',
        'color': '#e0e6ed',
        'border-color': 'rgba(100, 255, 218, 0.2)'
    })
    
    st.dataframe(df, use_container_width=True, height=300)
    
    # Download options
    st.markdown("### 💾 Export Data")
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

def create_navigation():
    """Create top navigation bar"""
    st.markdown("""
    <div class="nav-container fade-in">
        <div class="nav-logo">💱 FX Analytics Hub</div>
        <div class="nav-links">
            <a href="#" class="nav-link" onclick="window.location.reload()">🏠 Home</a>
            <a href="#features" class="nav-link">✨ Features</a>
            <a href="#database" class="nav-link">📊 Database</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create enhanced sidebar"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #64ffda; margin-bottom: 0.5rem;">📊 Database Schema</h2>
            <p style="color: #90a4ae; font-size: 0.9rem;">Explore your FX trading data</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🏦 Trades Table", expanded=True):
            st.markdown("""
            **Core Fields:**
            - `trade_id`: Unique identifier
            - `cp_id`: Counterparty link
            - `px_type`: Product type
                - Spot, Forward, Swap, NDF
            - `notl`: Notional value
            - `ccy_pair`: Currency pair
            - `near_dt`: Near leg date
            - `far_dt`: Far leg date (swaps)
            - `rate`: Executed FX rate
            """)
        
        with st.expander("🏢 Counterparties Table"):
            st.markdown("""
            **Organization Data:**
            - `cp_id`: Counterparty ID
            - `cp_name`: Company name
            - `region`: Geographic region
            """)
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Reset Query", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith(('conversation_state', 'user_question', 'query_data')):
                    del st.session_state[key]
            st.session_state.show_landing = True
            st.rerun()
        
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.show_landing = True
            st.session_state.conversation_state = "asking"
            st.rerun()
        
        st.markdown("---")
        
        # Example queries
        st.markdown("### 💡 Example Queries")
        examples = [
            "Show total notional by product type",
            "Top 5 currency pairs by volume", 
            "Trading activity by region",
            "Average rates by currency pair",
            "Monthly trading volumes",
            "Largest trades this month"
        ]
        
        for example in examples:
            if st.button(example, key=f"sidebar_ex_{hash(example)}", use_container_width=True):
                st.session_state.user_question = example
                st.session_state.show_landing = False
                st.rerun()
        
        st.markdown("---")
        
        # Stats
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: rgba(100, 255, 218, 0.1); border-radius: 12px; border: 1px solid rgba(100, 255, 218, 0.2);">
            <h4 style="color: #64ffda; margin-bottom: 0.5rem;">🎯 AI Powered</h4>
            <p style="color: #90a4ae; font-size: 0.9rem; margin: 0;">
                Natural language to SQL<br>
                Lightning fast results
            </p>
        </div>
        """, unsafe_allow_html=True)

def create_typing_animation(text: str, key: str):
    """Create typing animation effect"""
    placeholder = st.empty()
    displayed_text = ""
    
    for char in text:
        displayed_text += char
        placeholder.markdown(f"**{displayed_text}**<span class='loading-spinner'></span>", unsafe_allow_html=True)
        time.sleep(0.02)
    
    placeholder.markdown(f"**{text}**")

def main():
    st.set_page_config(
        page_title="FX Analytics Hub - AI-Powered SQL Assistant",
        layout="wide",
        page_icon="💱",
        initial_sidebar_state="expanded"
    )
    
    load_dark_theme_css()
    
    # Initialize session state
    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = "asking"
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
    
    # Main application
    create_navigation()
    create_sidebar()
    
    # Main conversation flow
    if st.session_state.conversation_state == "asking":
        # Initial question input
        st.markdown('<div class="query-section fade-in">', unsafe_allow_html=True)
        st.markdown("### 🔍 Ask Your Question")
        st.markdown("*Transform your thoughts into powerful SQL insights*")
        
        user_input = st.text_input(
            "",
            value=st.session_state.user_question,
            placeholder="e.g., Show me the total notional by product type, or find the top performing currency pairs...",
            key="user_input",
            help="Ask any question about your FX trading data in natural language"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 Generate SQL", type="primary"):
                if user_input.strip():
                    st.session_state.user_question = user_input
                    
                    # Show loading animation
                    with st.spinner("🤖 AI is analyzing your question..."):
                        result = generate_sql(user_input)
                    
                    if result.get("clarification"):
                        st.session_state.clarification_question = result["clarification"]
                        st.session_state.conversation_state = "clarifying"
                    else:
                        st.session_state.sql_result = result
                        st.session_state.final_question = user_input
                        st.session_state.conversation_state = "results"
                    st.rerun()
                else:
                    st.warning("⚠️ Please enter a question")
        
        with col2:
            if st.button("🎲 Random Example"):
                examples = [
                    "Show total notional by product type",
                    "Top 5 currency pairs by volume",
                    "Trading activity by region",
                    "Average rates by currency pair",
                    "Largest trades last month"
                ]
                st.session_state.user_question = np.random.choice(examples)
                st.rerun()
        
        with col3:
            if st.button("🔄 Reset"):
                st.session_state.conversation_state = "asking"
                st.session_state.user_question = ""
                st.session_state.query_data = None
                st.session_state.sql_result = {}
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tips section
        st.markdown("""
        <div class="main-container fade-in">
            <h4 style="color: #64ffda;">💡 Pro Tips</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div style="padding: 1rem; background: rgba(100, 255, 218, 0.05); border-radius: 8px; border-left: 4px solid #64ffda;">
                    <strong>🎯 Be Specific:</strong> "Top 5 EUR/USD trades by notional" works better than "show trades"
                </div>
                <div style="padding: 1rem; background: rgba(100, 255, 218, 0.05); border-radius: 8px; border-left: 4px solid #1de9b6;">
                    <strong>📊 Ask for Analysis:</strong> "Average rates by region" or "Monthly trading patterns"
                </div>
                <div style="padding: 1rem; background: rgba(100, 255, 218, 0.05); border-radius: 8px; border-left: 4px solid #00bcd4;">
                    <strong>🔍 Filter Data:</strong> "Swap trades over $1M" or "Trades from last quarter"
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    elif st.session_state.conversation_state == "clarifying":
        # Clarification needed
        st.markdown('<div class="clarification-section slide-up">', unsafe_allow_html=True)
        
        st.markdown("### ❓ Need More Details")
        st.info(f"**Your question:** {st.session_state.user_question}")
        st.warning(f"**AI clarification:** {st.session_state.clarification_question}")
        
        clarification = st.text_input(
            "💬 Please provide more details:",
            placeholder="Add specific information to clarify your question",
            help="The AI needs more context to generate the right SQL query"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("✅ Submit Clarification", type="primary"):
                if clarification.strip():
                    full_question = f"{st.session_state.user_question}. Clarification: {clarification}"
                    
                    with st.spinner("🔄 Processing your clarified question..."):
                        result = generate_sql(full_question)
                    
                    st.session_state.sql_result = result
                    st.session_state.final_question = full_question
                    st.session_state.conversation_state = "results"
                    st.rerun()
                else:
                    st.warning("⚠️ Please provide clarification")
        
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
            <div class="results-section fade-in">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; flex-wrap: wrap;">
                    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                        <span class="status-badge status-success">✅ SQL Generated</span>
                        <span class="status-badge status-success">📊 Data Retrieved</span>
                        <span class="status-badge status-success">🎨 Visualization Ready</span>
                    </div>
                </div>
                <div style="background: rgba(100, 255, 218, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(100, 255, 218, 0.2); margin-bottom: 1rem;">
                    <p><strong style="color: #64ffda;">Question:</strong> {st.session_state.final_question}</p>
                    <p><strong style="color: #1de9b6;">Analysis:</strong> {result.get('explanation', 'Query executed successfully')}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Execute SQL if not already done
            if st.session_state.query_data is None:
                with st.spinner("⚡ Executing SQL query and preparing visualizations..."):
                    progress_bar = st.progress(0)
                    progress_bar.progress(25)
                    
                    df, error = execute_sql(result["sql"])
                    progress_bar.progress(75)
                    
                    if error:
                        st.error(f"❌ SQL Error: {error}")
                        
                        # Show SQL for debugging
                        st.markdown('<div class="sql-display">', unsafe_allow_html=True)
                        st.code(result["sql"], language="sql")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🔄 Try Again"):
                                st.session_state.conversation_state = "asking"
                                st.rerun()
                        with col2:
                            if st.button("🔄 Reset"):
                                st.session_state.conversation_state = "asking"
                                st.session_state.user_question = ""
                                st.session_state.query_data = None
                                st.session_state.sql_result = {}
                                st.rerun()
                        return
                    else:
                        st.session_state.query_data = df
                    
                    progress_bar.progress(100)
                    progress_bar.empty()
            
            df = st.session_state.query_data
            
            # Show SQL query
            with st.expander("🔍 View Generated SQL Query", expanded=False):
                st.markdown('<div class="sql-display">', unsafe_allow_html=True)
                st.code(result["sql"], language="sql")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Success message
            st.success(f"🎉 Successfully retrieved {len(df)} rows with {len(df.columns)} columns!")
            
            # Interactive visualization
            create_interactive_visualization(df, "main")
            
            # Data summary
            display_data_summary(df)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 New Query", type="primary"):
                    st.session_state.conversation_state = "asking"
                    st.session_state.user_question = ""
                    st.session_state.query_data = None
                    st.session_state.sql_result = {}
                    st.rerun()
            
            with col2:
                if st.button("📋 Copy SQL"):
                    st.code(result["sql"], language="sql")
                    st.success("✅ SQL query displayed above for copying!")
            
            with col3:
                if st.button("🔄 Reset All"):
                    st.session_state.conversation_state = "asking"
                    st.session_state.user_question = ""
                    st.session_state.query_data = None
                    st.session_state.sql_result = {}
                    st.rerun()

if __name__ == "__main__":
    main()