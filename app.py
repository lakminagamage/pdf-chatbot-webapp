import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import pickle
import os
from langchain.chat_models import ChatOpenAI


def main():
    # this app uses streamlit config(.streamlit/config.toml) to control maximum upload size for PDF.

    # environment variables for app [openai API KEY]
    load_dotenv()

    #this codechunk is used to add a sidebar of information to the user.
    st.header('SnapRead v0.1')
    st.sidebar.header(":blue[Welcome to SnapRead!]")
    
    st.sidebar.write('SnapRead is a web app that allows you to upload a PDF file and ask questions about it. You can use this service to shorten lengthy PDF documents into quickly readble texts. You can also ask questions about the PDF you uploaded and get answers from the app.')
    st.sidebar.write("")
    st.sidebar.write("&#9210; :red[ Please only upload PDFs with text. App will not be able to recognize text from images.]")
    st.sidebar.write("&#9210; :red[ Please only upload PDFs with English text. No other language will be understood by the model.")
    st.sidebar.write("&#9210; :red[ Please do not upload PDFs with more than 50 pages. App will not be able to process it.]")
    st.sidebar.write("&#9210; :green[ Please note that app is running on OpenAI API's free tier. Maximum of 3 requests per minute is allowed. If you get an error, please wait a minute and try again.]")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write(" :blue[Crafted with :heart: by Luke @Hypercube] | 2023")

    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("Technologies used: [Streamlit](https://streamlit.io), PyPDF2, [OpenAI API](https://platform.openai.com), FAISS, [Langchain](https://langchain.com), Python")
    
    

    ## this codechunk is used to hide streamlit logo as we using free tier of streamlit.
    hide_streamlit = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit, unsafe_allow_html=True)


    txt = ""
    pdf = st.file_uploader('Upload a PDF file with text in English. PDFs that only contain images will not be recognized.', type=['pdf'])
    
    
    #geting content the pdf file from user

    try:
        pdf_doc = PdfReader(pdf)
        for page in pdf_doc.pages:
            txt += page.extract_text()
  
        # creating chunks of text from the pdf file
        #here chunk_overlap is used to avoid missing of context while in the middle of a paragraph.
        text_split = RecursiveCharacterTextSplitter(
            chunk_size=1000, # number of characters per chunk
            chunk_overlap=200, # used to keep the context of a chunk intact with previous and next chunks
            length_function=len
        )
        chunks = text_split.split_text(text=txt)
        

        # getting file name for the vector store file be stored on the disk as a pickle.
        store_name = pdf.name[:-4]
        st.write("You entered file : " + store_name)
        
        
        

        # checking whether the vector store file is already present on the disk or not.
        if os.path.exists(f"{store_name}.pkl"):
            st.write("Loading from disk.")
            with open(f"{store_name}.pkl", "rb") as f:
                vs = pickle.load(f)
        
        else:
            # if no pickle file found, a new vector store is created and stored on the disk.
            embeddings = OpenAIEmbeddings()
            st.write("Creating vector store using OpenAI api.")
            vs = FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vs, f)
        
        # asking user to enter a query about the pdf file.
        query = st.text_input('Ask question about the PDF you entered!', max_chars=300)

        if query != "":
            #getting vector store checked for the query and getting the response within pickle file.
            docs = vs.similarity_search(query=query)
            # use Openai model to create a semantic response to the query
            llm = ChatOpenAI(model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            # using openai callbacks to get the cost per query.
            with get_openai_callback() as callback:
                response = chain.run(input_documents=docs, question=query)
            
            st.write(response)
            # displaying the cost per query.
            st.write(f"This query costs - {callback.total_cost} USD for OpenAI API.")

            

            

    except Exception as e:
        if pdf is None:
            st.write("Please upload a PDF file and ask me about it.")
            
        else:
            st.error(str(e))
            st.write(f"You have encountered an error. Please give {str(e)} as feedback to the developer.")

        

if __name__ == '__main__':
    main()