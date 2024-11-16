import streamlit as st
import os
from huggingface_hub import InferenceClient
from duckduckgo_search import DDGS
import yfinance as yf

# Initialize the client using environment variable
client = InferenceClient(api_key=st.secrets["HUGGINGFACE_API_KEY"])

# Define custom tool classes
class DuckDuckGo:
    def __init__(self):
        self.ddgs = DDGS()
    
    def search(self, query, max_results=5):
        return list(self.ddgs.text(query, max_results=max_results))

class YFinanceTools:
    def __init__(self, stock_price=True, analyst_recommendations=True, company_info=True):
        self.stock_price = stock_price
        self.analyst_recommendations = analyst_recommendations
        self.company_info = company_info
    
    def get_stock_data(self, symbol):
        stock = yf.Ticker(symbol)
        data = {}
        if self.stock_price:
            data['price'] = stock.info.get('regularMarketPrice')
        if self.analyst_recommendations:
            data['recommendations'] = stock.recommendations
        if self.company_info:
            data['info'] = {k: v for k, v in stock.info.items() if k in ['sector', 'industry', 'longBusinessSummary']}
        return data

# Define agents
web_agent = {
    "name": "Web Agent",
    "role": "Search the web for information",
    "instructions": ["Always include sources"],
    "tools": [DuckDuckGo()]
}

finance_agent = {
    "name": "Finance Agent",
    "role": "Get financial data",
    "instructions": ["Use tables to display data"],
    "tools": [YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)]
}

# Create Streamlit interface
st.title("AI Chat Assistant")

# Add agent selector
agent_type = st.sidebar.selectbox(
    "Select Agent Type",
    ["Web Agent", "Finance Agent"]
)

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display current agent type
st.write(f"Currently using: {agent_type}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Prepare system message based on selected agent
        selected_agent = web_agent if agent_type == "Web Agent" else finance_agent
        system_message = f"You are a {selected_agent['role']}. {' '.join(selected_agent['instructions'])}"
        
        # Create the stream with system message
        stream = client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            messages=[
                {"role": "system", "content": system_message},
                *st.session_state.messages
            ],
            max_tokens=3000,
            stream=True
        )
        
        # Process the stream
        for chunk in stream:
            full_response += chunk.choices[0].delta.content or ""
            message_placeholder.write(full_response + "▌")
        
        message_placeholder.write(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})