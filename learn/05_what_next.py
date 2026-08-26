"""
LESSON 5: WHAT YOU NOW KNOW + WHAT'S NEXT
Run: python learn/05_what_next.py
"""
print("""
╔══════════════════════════════════════════════════════════════╗
║          YOU ARE NOW AN AI ENGINEER                          ║
╚══════════════════════════════════════════════════════════════╝

Here's what you built and the concepts behind each piece:

  PIECE                FILE                  CONCEPT
  ─────────────────────────────────────────────────────────────
  Embedding text    →  src/embedder.py     → Dense vector representations
  Storing vectors   →  src/vector_store.py → HNSW approximate nearest neighbour
  Finding similar   →  src/predictor.py   → Cosine similarity search
  Predicting ratings→  src/predictor.py   → k-NN weighted regression
  Visualizing taste →  src/taste_profile  → PCA/UMAP dimensionality reduction
  Grouping films    →  src/taste_profile  → k-Means unsupervised clustering
  Caching API calls →  src/tmdb_client.py → Cache-aside pattern (SQLite)
  3-space design    →  config.py          → Ensemble / multi-view retrieval

══════════════════════════════════════════════════════════════

STACK COMPARISON — Where you are vs the industry:

  YOURS                         INDUSTRY EQUIVALENT
  ─────────────────────────────────────────────────────────────
  nomic-embed-text              OpenAI ada-002, Cohere embed
  ChromaDB                      Pinecone, Weaviate, Qdrant
  k-NN weighted regression      Collaborative filtering, RecSys
  PCA/UMAP scatter              t-SNE, UMAP in production pipelines
  K-Means                       HDBSCAN, GMM, spectral clustering
  SQLite cache                  Redis, Memcached, DynamoDB

  You're using the open-source local equivalent of every paid tool.
  The concepts are IDENTICAL. Switch ChromaDB → Pinecone?
  Same API, same concepts, just a different host.

══════════════════════════════════════════════════════════════

YOUR 30-DAY AI ENGINEERING ROADMAP:

  Week 1 — Master what you have
  ─────────────────────────────
  ✅ Run pipeline.py --limit 50 (done)
  ✅ Run all 5 lessons (done)
  □  Run the full pipeline (all 3507 movies)
  □  Tune the weights in config.py — see how predictions change
  □  Read sentence-transformers docs — understand the model architecture
  □  Add backtesting: hold out 200 movies, measure prediction error

  Week 2 — Go deeper on retrieval
  ─────────────────────────────────
  □  Learn about bi-encoders vs cross-encoders
     → Your system uses bi-encoder (embed both sides separately)
     → Cross-encoders are slower but more accurate (used for reranking)
  □  Build a reranker: first retrieve 50, then rerank with cross-encoder
  □  Add Faiss (Facebook's vector library) as alternative to ChromaDB
  □  Experiment with mxbai-embed-large (1024-dim) — does it help?

  Week 3 — Add an LLM (RAG)
  ─────────────────────────
  □  Install Ollama model: ollama pull llama3
  □  Build RAG on top: "Why would I like Parasite?" →
     retrieve similar movies → feed to LLM → get explanation
  □  This is literally how ChatGPT plugins work
  □  Learn LangChain or LlamaIndex (they automate this pipeline)

  Week 4 — Fine-tuning
  ─────────────────────
  □  Collect positive/negative pairs from your ratings:
     (Inception review, Interstellar review) → similar (both 5★)
     (Inception review, Dil Chahta Hai review) → dissimilar
  □  Fine-tune nomic-embed-text on these pairs
  □  Your rating-predicting embedding, trained on YOUR taste
  □  Use sentence-transformers for this — it's 20 lines of code

══════════════════════════════════════════════════════════════

THE MATH YOU SHOULD LEARN (in order):

  1. Linear Algebra (3B1B on YouTube — best visual explanation)
     - Vectors, dot products, matrix multiplication
     - Eigenvectors (used in PCA)
     - This is the foundation of everything

  2. Probability & Statistics
     - Distributions, expectation, variance
     - Bayes theorem (foundation of generative models)

  3. Calculus (specifically derivatives + chain rule)
     - How neural networks learn (backpropagation)
     - You don't need to derive it, just understand it

  4. Information Theory
     - Entropy, cross-entropy loss
     - How language models measure prediction quality

  Resources:
  ───────────────────────────────────────────
  fast.ai         → Best practical ML course (free)
  3Blue1Brown     → Best visual math explanations (YouTube)
  Andrej Karpathy → Best deep learning from scratch (YouTube)
  Hugging Face    → Best resource for transformers/embeddings
  Papers with Code→ State of the art research + code

══════════════════════════════════════════════════════════════

RIGHT NOW — most impactful next step:

  python pipeline.py --limit 200
  streamlit run app.py
  # Type a movie into Predict Rating
  # See which similar films influenced the prediction
  # That IS AI engineering — you built it.

══════════════════════════════════════════════════════════════
""")
