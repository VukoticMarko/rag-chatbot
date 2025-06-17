from rag import build_rag_pipeline

def main():
    qa_pipeline = build_rag_pipeline()
    print("Hey how are you? Have any interesting questions to ask me? (type 'quit' to exit)")
    
    while True:
        query = input("\nQuestion: ")
        if query.lower() == "quit":
            break
            
        response = qa_pipeline.invoke({"query": query})
        answer = response['result'].replace("Answer:", "").strip()
        print(f"\n{answer}")
        
        # print("\nSources:")
        # for doc in response['source_documents']:
        #     print(f"- Page {doc.metadata['page']}: {doc.page_content[:100]}...")

if __name__ == "__main__":
    main()