# Adapted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
import os
import json

import gc
import re
import uuid
import subprocess
import nest_asyncio
from dotenv import load_dotenv

import streamlit as st

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

import ast
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

# Initialize environment and event loop
load_dotenv()
nest_asyncio.apply()

def load_embedding_model(
    model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cpu"
) -> HuggingFaceBgeEmbeddings:
    """Loads and returns a HuggingFaceBgeEmbeddings model for embedding text."""
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

def generate_repo_ast(repo_path: str):
    """Generates an AST summary for all Python files in the given repository path."""
    repo_summary = {}
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        tree = ast.parse(f.read())
                    node_counts = {}
                    for node in ast.walk(tree):
                        node_type = type(node).__name__
                        node_counts[node_type] = node_counts.get(node_type, 0) + 1
                    repo_summary[file_path] = node_counts
                except Exception:
                    repo_summary[file_path] = "Unable to parse file"
    return repo_summary

# Initialize the 'id' attribute if it doesn't exist
if 'id' not in st.session_state:
    st.session_state.id = uuid.uuid4()  # or set to some default value

# Initialize chat history and query engine if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "repo_ast" not in st.session_state:
    st.session_state.repo_ast = {}
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

# Setting up the llm
llm = Ollama(model="llama3", request_timeout=60.0)

# Utility functions
def parse_github_url(url):
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    return match.groups() if match else (None, None)

def clone_repo(repo_url, target_path):
    return subprocess.run(["git", "clone", repo_url, target_path], check=True, text=True, capture_output=True)

def validate_owner_repo(owner, repo):
    return bool(owner) and bool(repo)

def reset_chat():
    st.session_state.messages = []
    st.session_state.repo_ast = {}
    st.session_state.query_engine = None
    gc.collect()

with st.sidebar:
    # Input for GitHub URL
    github_url = st.text_input("GitHub Repository URL")

    # Button to load and process the GitHub repository
    process_button = st.button("Load")

    message_container = st.empty()  # Placeholder for dynamic messages

    if process_button and github_url:
        owner, repo = parse_github_url(github_url)
        if validate_owner_repo(owner, repo):
            with st.spinner(f"Loading {repo} repository by {owner}..."):
                try:
                    cloned_repos_dir = "cloned_repos"
                    os.makedirs(cloned_repos_dir, exist_ok=True)
                    input_dir_path = os.path.join(cloned_repos_dir, repo)
                    
                    if not os.path.exists(input_dir_path):
                        clone_repo(github_url, input_dir_path)

                    if os.path.exists(input_dir_path):
                        loader = SimpleDirectoryReader(
                            input_dir=input_dir_path,
                            required_exts=[".py", ".ipynb", ".js", ".ts", ".md"],
                            recursive=True
                        )
                    else:    
                        st.error('Error occurred while cloning the repository, carefully check the url')
                        st.stop()

                    docs = loader.load_data()

                    # ====== Create vector store and upload data ======
                    Settings.embed_model = LangchainEmbedding(load_embedding_model())
                    index = VectorStoreIndex.from_documents(docs)

                    # ====== Setup a query engine ======
                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)
                    
                    # ====== Customize prompt template ======
                    qa_prompt_tmpl_str = (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "You are llama3, a large language model developed by Meta AI. Surya has integrated you into this environment to answer the user's coding questions. Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )

                    if docs:
                        message_container.success("Data loaded successfully!!")
                    else:
                        message_container.write(
                            "No data found, check if the repository is not empty!"
                        )
                    st.session_state.query_engine = query_engine
                    st.session_state.repo_ast = generate_repo_ast(input_dir_path)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.stop()

                st.success("Ready to Chat!")
        else:
            st.error('Invalid owner or repository')
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.header("Chat With Your Code! Powered by LLama3 🦙🚀")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    if st.session_state.query_engine is None:
        st.info("Load a GitHub repository before asking a question.")
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            query_engine = st.session_state.query_engine
            streaming_response = query_engine.query(
                f'Given the repository AST:\n{json.dumps(st.session_state.repo_ast, indent=2)}\n\nAnd the following question: {prompt}'
            )

            for chunk in streaming_response.response_gen:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
