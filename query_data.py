import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv() 

CHROMA_PATH = "chroma"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}

While answering the question, prioritize the connected documents I have added!
"""

@st.cache_resource

def load_chroma():
    embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    return db

def search_and_respond(query_text, db):
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.3:
        return "Unable to find matching results.", None

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    # formatted_response = f"{response_text}\nSources: {sources}"
    formatted_response = f"{response_text}"
    return formatted_response, sources

def main():
    db = load_chroma()
    # Sidebar design
    st.sidebar.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2E8B57, #3CB371);
            color: white;
        }
        .sidebar .sidebar-content .block-container {
            padding-top: 20px;
        }
        .sidebar .sidebar-content .block-container h2 {
            color: white;
        }
        .sidebar .sidebar-content .block-container img {
            margin: 10px 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Sidebar content
    st.sidebar.image("data/2a74b9a7608ac2b58673c4e98ce3ff6e.jpg", use_column_width=True)
    st.sidebar.header("Bangladesh")
    st.sidebar.write("Bangladesh, to the east of India on the Bay of Bengal, is a South \
                      Asian country marked by lush greenery and many waterways. Its Padma \
                     (Ganges), Meghna and Jamuna rivers create fertile plains, and travel by\
                      boat is common. On the southern coast, the Sundarbans, an enormous mangrove\
                      forest shared with Eastern India, is home to the royal Bengal tiger.")

    # Main content
    st.title("Welcome to the Bangladesh Info App")
    st.write("This application provides information about Bangladesh.")
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    query_text = st.text_input("Enter your query here:")
    if st.button("Search"):
        response, sources = search_and_respond(query_text, db)
        st.session_state['history'].append((query_text, response))
        if sources:
            st.write("Response:", response)
        else:
            st.warning(response)

    if st.session_state['history']:
        st.subheader("Conversation History")
        for i, (query, resp) in enumerate(st.session_state['history']):
            with st.expander(f"Query {i+1}"):
                st.text_area(f"Query {i+1}", value=query, height=75, disabled=True)
                st.text_area(f"Response {i+1}", value=resp, height=150, disabled=True)

if __name__ == "__main__":
    main()
