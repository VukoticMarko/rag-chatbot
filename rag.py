from langchain.chains import RetrievalQA
from vector_db import get_vector_db
from llm import load_llm
from langchain_core.prompts import PromptTemplate 

def build_rag_pipeline():
    llm = load_llm()
    vector_db = get_vector_db()


    template = """
    
    **Role**: You are an expert researcher. Answer using ONLY the provided context. If context is irrelevant, say so explicitly.
    Use ONLY the context below to answer the question. If the answer is not in the context, say "I don't know."

    **Context**:  
    {context}

    **Question**:  
    {question}

    **Response**:

    1. **Summary**  
    Concise 1-2 sentence overview of key answer.

    2. **Core Explanation**  
    - Detailed technical breakdown  
    - Key mechanisms or principles  
    - Mathematical notation: $E=mc^2$  

    3. **Key Concepts**  
    - Concept 1: Brief definition  
    - Concept 2: Brief definition  

    4. **Examples/Applications**  
    ```python
    # Practical code example if relevant
    def example():
        return "Hello World"

    - Real-world use case 1
    - Real-world use case 2

    5. **Limitations**  
    - When this doesn't apply
    - Edge cases to consider

    6. **Sources**  
    - Page X: [Relevant section name]
    - Page Y: [Additional reference]

    7. **Related Topics (Optional)**  
    - Topic A: Short description  
    - Topic B: Short description

    Rules:
    Never invent facts
    Cite exact page numbers
    If context doesn't answer: "I cannot answer from provided materials"
    """

    QA_PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever = vector_db.as_retriever(
            search_type="mmr", search_kwargs={"k": 12}
        ),
        return_source_documents=True,

        chain_type_kwargs={
        "prompt": QA_PROMPT
        }
    )