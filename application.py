import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever, MultiQueryRetriever, ContextualCompressionRetriever, EnsembleRetriever, SelfQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter, DocumentCompressorPipeline
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from typing import List
import faiss
from datetime import datetime, timedelta
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
import time
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from operator import itemgetter

#-------------------------------COMMON-ADDS--------------------------------
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4-turbo-preview",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens = 800,
    model_kwargs = {
        "top_p":0,
        "frequency_penalty": 0,
        "presence_penalty": 0
    },
)

#------------------------------MULTI_QUERY_RETRIEVER-ADDS------------------------------
multi_query_retriever_template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

final_rag_multi_query_template = """Answer the following question based on this context:

{context}

Question: {question}
"""

# Streamlit app
def main():

    # Load data
    with st.spinner('Loading documents...'):
        loader = TextLoader("./dataset-creation-docs/outputs/databricks-dolly-15K.txt")
        data3 = loader.load()
        docs =  data3
        embedding = OpenAIEmbeddings()
        time.sleep(2)

    st.title("Document Retrieval App")
    retriever_options = ["Parent Document Retriever", 
                         "Multi-Query Retriever", 
                         "Contextual Compression Retriever", 
                         "Ensemble Retriever",]
    selected_retriever = st.sidebar.selectbox("Select a retriever", retriever_options)
    query = st.text_input("Enter your query")
    submit_button = st.button("Submit")
    embedding = OpenAIEmbeddings()

    if submit_button:
        if selected_retriever == "Parent Document Retriever":
            # Code for Parent Document Retriever
            childSplitter = RecursiveCharacterTextSplitter(chunk_size=200, 
                                                           chunk_overlap=30)
            ParentSplitter = RecursiveCharacterTextSplitter(chunk_size=500, 
                                                            chunk_overlap=60)
            vectorstore = Chroma(collection_name="Project_documents", 
                                 embedding_function=embedding)
            store = InMemoryStore()
            retriever = ParentDocumentRetriever(vectorstore=vectorstore, 
                                                splitter=ParentSplitter, 
                                                docstore=store, 
                                                child_splitter=childSplitter, 
                                                parent_splitter=ParentSplitter)
            retriever.add_documents(docs, ids=None)
            llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-4-turbo-preview",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens = 800,
                model_kwargs = {
                    "top_p":0,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                },
            )
            qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=retriever)
            #Query which is used by user
            result_from_retriever = qa.invoke(query)
            #retrieved_docs = retriever.get_relevant_documents(query)
            prompt_template = PromptTemplate(input_variables=["query"], template="{query}")
            chain = LLMChain(llm=llm, prompt=prompt_template)
            result_llm = chain.invoke(query)
            cols = st.columns(2)
            cols[0].write(f"Answer from Retriever: '{result_from_retriever['result']}'")
            cols[1].write(f"Answer from LLM: '{result_llm['text']}'")
            #st.write(f"Answer for the Query: '{result['result']}'")

        elif selected_retriever == "Multi-Query Retriever":
            # Code for Multi-Query Retriever
            #Using the RecursiceCharaterTextSplitter for splitting docsuments in chunks for the better search 
            splitters = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 30)
            #SPlitting the documents
            splits = splitters.split_documents(docs)
            #Creating the Vector Store for the chunks we created
            vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
            #Using that has the base_retriever
            retriever = vectorstore.as_retriever()
            #Want to find which documents were the most relevant ones
            unique_docs = retriever.get_relevant_documents("Differentiator between Mountain Bike and Road Bike?")
            #print(len(unique_docs))
            llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-4-turbo-preview",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens = 800,
                model_kwargs = {
                    "top_p":0,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                },
            )
            #Use the multi_query_prompt_template for using the template to write prompt for Multi_query 
            prompt_perspectives = ChatPromptTemplate.from_template(multi_query_retriever_template)
            generate_queries = (
                prompt_perspectives 
                | llm
                | StrOutputParser() 
                | (lambda x: x.split("\n"))
            )
            #Create the Retriever Chain for using the Multi-Query Template
            retrieval_chain = generate_queries | retriever.map() | get_unique_union
            # Tesing a single retriever
            #docs = retrieval_chain.invoke({"question":question})
            #print(docs)
            prompt = ChatPromptTemplate.from_template(final_rag_multi_query_template)
            prompt_template = PromptTemplate(input_variables=["query"], template="{query}")
            chain = LLMChain(llm=llm, prompt=prompt_template)
            final_rag_chain = (
                {"context": retrieval_chain, 
                "question": itemgetter("question")} 
                | prompt
                | llm
                | StrOutputParser()
            )

            result = final_rag_chain.invoke({"question":query})
            result_llm = chain.invoke(query)
            cols = st.columns(2)
            cols[0].write(f"Answer for the Query: '{result}'")
            cols[1].write(f"Answer from LLM: '{result['text']}'")

        elif selected_retriever == "Contextual Compression Retriever":
            # Code for Contextual Compression Retriever
            #Create the Split of the documents
            splitters = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 30)
            #Doing the Split of the documents
            splits = splitters.split_documents(docs)
            #Create the ChromaDB vector Store
            vectorstore = Chroma(collection_name="Project_Docs_Compression", embedding_function=embedding)
            #Adding the vector store documents
            vectorstore.add_documents(splits)
            #Using it as the base retriever
            retriever = vectorstore.as_retriever()
            # Set up document compression pipeline
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding)
            relevant_filter = EmbeddingsFilter(embeddings=embedding, similarity_threshold=0.76)
            pipeline_compressor = DocumentCompressorPipeline(transformers=[splitter, redundant_filter, relevant_filter])
            # Create ContextualCompressionRetriever
            compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)
            # Create the RetrievalQA chain
            llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-4-turbo-preview",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens = 800,
                model_kwargs = {
                    "top_p":0,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                },
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=compression_retriever,
                return_source_documents=True
            )
            result = qa_chain({"query": query})
            prompt_template = PromptTemplate(input_variables=["query"], template="{query}")
            chain = LLMChain(llm=llm, prompt=prompt_template)
            #st.write(f"Answer for the Query: '{result['result']}'")
            cols = st.columns(2)
            cols[0].write(f"Answer for the Query: '{result['result']}'")
            cols[1].write(f"Answer from LLM: '{chain.invoke(query['result'])}'")

        elif selected_retriever == "Ensemble Retriever":
                       #Create the Split of the documents
            splitters = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 30)
            #Doing the Split of the documents
            splits = splitters.split_documents(docs)
            #Create the ChromaDB vector Store
            vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = 5

            chrome_vectorstore = Chroma.from_documents(docs, embedding)
            chrome_retriever = chrome_vectorstore.as_retriever()

            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, chrome_retriever], weights=[0.5,0.5]
            )
            llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-4-turbo-preview",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens = 800,
                model_kwargs = {
                    "top_p":0,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                },
            )
            # Initialize the RetrievalQA chain
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=ensemble_retriever)
            response = qa.invoke(query)
            prompt_template = PromptTemplate(input_variables=["query"], template="{query}")
            chain = LLMChain(llm=llm, prompt=prompt_template)
            #st.write(f"Answer for the Query: '{response['result']}'")
            cols = st.columns(2)
            cols[0].write(f"Answer for the Query: '{response['result']}'")
            cols[1].write(f"Answer from LLM: '{chain.invoke(query['result'])}'")

if __name__ == "__main__":
    main()