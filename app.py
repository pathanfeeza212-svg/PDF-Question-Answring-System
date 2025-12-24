import streamlit as st  #web app UI banane ke liye
import os   #folder & file handle karne ke liye

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader  #PDF se text extract karta hai
from langchain.text_splitter import RecursiveCharacterTextSplitter #Long text ko small chunks me todta hai
from langchain.embeddings import HuggingFaceEmbeddings #text ko vector me convert karta hai
from langchain.vectorstores import FAISS #Embeddings ko vector database me store karta hai
from langchain.chains import RetrievalQA #RAG QA chain banata hai
from langchain.llms import HuggingFacePipeline #HuggingFace models ko LLM ke roop me use karta hai
from langchain.prompts import PromptTemplate #Custom prompt template banane ke liye

# HuggingFace
from transformers import pipeline #HuggingFace models ke saath kaam karne ke liye


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="PDF Q&A System") #Web app ka title set karta hai
st.title("PDF Question Answering System") #Web app ka heading
st.write(" AI sirf uploaded PDF ke content se hi answer dega")#Instructions for users

pdf_file = st.file_uploader("Upload PDF", type=["pdf"]) #PDF upload karne ke liye file uploader
question = st.text_input("Ask a question from the PDF")#User se question input lene ke liye text input field


# ---------------- MAIN LOGIC ----------------
if pdf_file and question:        ##Check if PDF is uploaded and question is asked
    # Save uploaded PDF to a temporary location

    os.makedirs("data", exist_ok=True)
    pdf_path = "data/temp.pdf"

    # Save PDF
    with open(pdf_path, "wb") as f: #Write binary mode me file open karta hai
        f.write(pdf_file.read()) #PDF content ko temporary file me likhta hai

    # Load PDF
    loader = PyPDFLoader(pdf_path) #PDF loader se PDF load karta hai
    documents = loader.load() #PDF se text extract karta hai

    # Split text
    splitter = RecursiveCharacterTextSplitter( #Text splitter ko initialize karta hai
        chunk_size=500, #Har chunk ka size 500 characters
        chunk_overlap=100 #Chunks ke beech 100 characters ka overlap
    )
    chunks = splitter.split_documents(documents) #Long text ko small chunks me todta hai

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )  #Text ko vector me convert karta hai

    # Vector DB
    vector_db = FAISS.from_documents(chunks, embeddings) #Embeddings ko vector database me store karta hai

    # Load LLM
    llm_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    # Custom Prompt Template - Creative PDF responses
    prompt_template = """
You are a creative assistant who explains PDF content in an engaging, human-like way.

RULES:
1. If answer exists in PDF context: Explain creatively like a friendly teacher
2. Use simple, conversational language with examples from the PDF
3. Make the answer interesting and easy to understand
4. Add relevant details from PDF to make answer complete
5. If answer NOT in context: Say EXACTLY "The answer of this question is not present in your PDF"
6. ONLY use information from the provided PDF context
7. Be enthusiastic and engaging in your explanations

PDF Context: {context}
Question: {question}

Creative Answer (PDF only):"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # RAG QA Chain with custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5, "score_threshold": 0.5}
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    # Get Answer with strict validation
    with st.spinner("Searching answer from pdf..."):
        # Check similarity score threshold
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(question)
        
        # Calculate similarity scores
        if relevant_docs:
            # Get similarity scores for validation
            query_embedding = embeddings.embed_query(question)
            doc_embeddings = [embeddings.embed_query(doc.page_content) for doc in relevant_docs]
            
            # Simple similarity check (basic implementation)
            max_similarity = 0
            for doc_emb in doc_embeddings:
                # Basic dot product similarity
                similarity = sum(a * b for a, b in zip(query_embedding, doc_emb))
                max_similarity = max(max_similarity, similarity)
        
        # Balanced validation for creative answers
        if not relevant_docs or max_similarity < 0.3:
            answer = "The answer of this question is not present in your PDF"
        else:
            response = qa_chain({"query": question})
            answer = response["result"]
            
            # Check for external knowledge indicators
            external_indicators = [
                "generally", "typically", "usually", "in general", "commonly", "often", 
                "not found", "no information", "cannot find", "not mentioned", 
                "i don't", "i cannot", "unable to", "sorry"
            ]
            
            if any(phrase in answer.lower() for phrase in external_indicators):
                answer = "The answer of this question is not present in your PDF"
            
            # If answer is too short, try to enhance it with PDF context
            elif len(answer.strip()) < 20 and relevant_docs:
                # Create enhanced context for better creative answer
                enhanced_context = " ".join([doc.page_content for doc in relevant_docs[:3]])
                creative_prompt = f"""Based on this PDF content, explain creatively: {enhanced_context[:800]}
                
Question: {question}
                
Give a detailed, engaging explanation using only the PDF information above:"""
                
                enhanced_response = qa_chain({"query": creative_prompt})
                enhanced_answer = enhanced_response["result"]
                
                if len(enhanced_answer) > len(answer) and not any(phrase in enhanced_answer.lower() for phrase in external_indicators):
                    answer = enhanced_answer

    # Show Creative Answer
    st.subheader("AI Assistant Creative reply")
    if "The answer of this question is not present in your PDF" in answer:
        st.error(" " + answer)
        st.info(" Ask question related to pdf content")
    else:
        # Display creative answer with formatting
        st.success("✨ " + answer)
        
        # Show source confidence
        if len(answer) > 50:
            st.info("This creative content is created by your pdf")
        
        # Show related PDF content for reference
        if relevant_docs:
            with st.expander("Show related PDF content for reference"):
                for i, doc in enumerate(relevant_docs[:2]):
                    st.write(f"**Reference {i+1}:**")
                    st.write(doc.page_content[:200] + "...")
                    st.write("---")


# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Creative answers strictly based on PDF content – engaging and informative!")
