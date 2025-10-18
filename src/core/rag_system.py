class RagSystem:
    def __init__(self,embedding_engine,vector_database):
        self.embedding_engine = embedding_engine
        self.vector_database = vector_database
        
        
    def retriever(self,query):
        query_embedding = self.embedding_engine.generate_embeddings([query])[0]
        retrieved_chunks = self.vector_database.search(query_embedding, k=5)
        return retrieved_chunks
        
        
        