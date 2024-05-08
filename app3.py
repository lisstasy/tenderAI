import streamlit as st
import tempfile
from operator import itemgetter
from langchain.document_loaders import JSONLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader, UnstructuredExcelLoader, PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

openai_api_key = "###"


def load_xlsx(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name
    loader = UnstructuredExcelLoader(temp_file_path)
    doc = loader.load()
    return doc  

#doc=load_xlsx()

def process_data(doc, openai_api_key): 

    # Use the OpenAIEmbeddings with the provided API key
    embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    
    # Create a database for temporal data from the documents
    db_for_temporal_data = Chroma.from_documents(doc, embedding_function)
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    
    # Retrieve a retriever instance from the database
    retriever_temp = db_for_temporal_data.as_retriever()    
    retriever = db.as_retriever()
    
    # Return the retriever and any other objects you might need outside this function
    return retriever_temp, retriever


def generate(retriever, retriever_temp, openai_api_key):
    print("Entering generate function")
    
    # Assuming extraction_template and generation_template are correctly defined as before
    extraction_template = ChatPromptTemplate.from_template("""Extract from the {context} key and values. Possible keys: value pair are capacity_kg, service_life_years, lifting_speed_mm_s, engine_type, lifting_height_mm, dimensions_mm.
        instructions: {instructions}""")
    print(f"Extraction template defined: {extraction_template}")

    # Similarly for generation_template
    generation_template = ChatPromptTemplate.from_template("""Given the specifications {base_response}, answer the question using the following context to help: {context}.
        question: {question}""")
    print(f"Generation template defined: {generation_template}")
    
    # Correctly constructing the extraction chain
    extraction_chain = (
        extraction_template
        | ChatOpenAI(model="gpt-4-0125-preview", openai_api_key=openai_api_key)
        | StrOutputParser()
    )
    print("Extraction chain constructed successfully")
    
    # Correctly constructing the generation chain
    generation_chain = (
        generation_template
        | ChatOpenAI(model="gpt-4-0125-preview", openai_api_key=openai_api_key)
        | StrOutputParser()
    )
    print("Generation chain constructed successfully")
    
    try:
        # Invocation might need to ensure that data flows correctly between chains
        # Start by invoking the extraction chain with the appropriate context
        extraction_response = extraction_chain.invoke({"context": retriever_temp, "instructions": "extract in format of the table in English"})
        
        # Then, invoke the generation chain with the output from the extraction step
        # Ensure that 'base_response' or equivalent is used correctly
        generation_response = generation_chain.invoke({"context": retriever, "question": "You are an experienced salesperson. Write a commercial offer in Russian that proposes the machinery matching the specifications from base_response the most. Choose the particular model that matches specification the best. Include in your offer key data such as: capacity_kg,service_life_years,lifting_speed_mm_s,engine_type,lifting_height_mm,dimensions_mm.", "base_response": extraction_response})
        
        print("Chain invoked successfully")
        return generation_response
    except Exception as e:
        print(f"Error invoking chain: {e}")
        return None




import streamlit as st
# Ensure all necessary imports are included at the beginning of your script

# Define or import your functions here (load_xlsx, process_data, generate)

def main():
    st.title("Big Tender")
    uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'xls', 'pdf'])

    if uploaded_file is not None:
        doc = load_xlsx(uploaded_file)
        retriever_temp, retriever = process_data(doc, openai_api_key)  # Make sure openai_api_key is defined or accessible
        answer = generate(retriever, retriever_temp, openai_api_key)  # Adjusted to use the combined generate function
        
        st.write("Generated Answer:")
        st.write(answer)

if __name__ == "__main__":
    main()
