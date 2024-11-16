import streamlit as st
from huggingface_hub import InferenceClient

# Initialize the client using secrets
client = InferenceClient(api_key=st.secrets["huggingface"]["api_key"])

# Create Streamlit interface
st.title("AI Chat Assistant")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

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
        
        # Create the stream
        stream = client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            messages=st.session_state.messages,
            max_tokens=500,
            stream=True
        )
        
        # Process the stream
        for chunk in stream:
            full_response += chunk.choices[0].delta.content or ""
            message_placeholder.write(full_response + "▌")
        
        message_placeholder.write(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})