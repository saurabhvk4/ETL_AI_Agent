import chromadb

from sentence_transformers import SentenceTransformer

# =====================================================
# STEP 1 — DOCUMENTS
# =====================================================

documents = [

    "OutOfMemoryError happens when Spark executors run out of memory.",

    "Data skew occurs when one partition has much more data than others.",

    "BroadcastTimeout happens when broadcast joins become too large.",

    "ExecutorLostFailure usually indicates executor crashes."

]

# =====================================================
# STEP 2 — LOAD EMBEDDING MODEL
# =====================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =====================================================
# STEP 3 — CREATE CHROMADB CLIENT
# =====================================================

client = chromadb.PersistentClient(
    path="./ETL_db"
)

# =====================================================
# STEP 4 — CREATE COLLECTION
# =====================================================

collection = client.create_collection(
    name="spark_docs"
)

# =====================================================
# STEP 5 — CONVERT DOCUMENTS TO VECTORS
# =====================================================

for i, doc in enumerate(documents):

    # Convert text into embedding vector
    embedding = embedding_model.encode(doc).tolist()

    print("\nDOCUMENT:")
    print(doc)

    print("\nVECTOR:")
    print(embedding[:10])   # show first 10 numbers only

    # Store inside ChromaDB
    collection.add(
        ids=[str(i)],
        documents=[doc],
        embeddings=[embedding]
    )

# =====================================================
# STEP 6 — SEARCH QUERY
# =====================================================

query = "Spark memory issue"

# Convert query into vector
query_embedding = embedding_model.encode(
    query
).tolist()

# Search similar documents
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

# =====================================================
# STEP 7 — SHOW RESULTS
# =====================================================

print("\n" + "="*50)
print("SEARCH QUERY")
print("="*50)

print(query)

print("\n" + "="*50)
print("MOST SIMILAR DOCUMENTS")
print("="*50)

for doc in results["documents"][0]:

    print(doc)