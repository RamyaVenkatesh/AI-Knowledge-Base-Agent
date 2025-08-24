import json
import sqlite3
import requests
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class KnowledgeBaseAgent:
    def __init__(self, knowledge_db_path="knowledge.db"):
        # Ollama setup
        self.base_url = "http://localhost:11434"
        self.model = "llama3.2"
        
        # Knowledge base setup
        self.db_path = knowledge_db_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, lightweight
        self.init_database()
        
        # Vector search setup
        self.vector_index = None
        self.document_chunks = []
        self.load_or_create_vector_index()
        
    def init_database(self):
        """Initialize SQLite database for document storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                content TEXT,
                source TEXT,
                chunk_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_document(self, title: str, content: str, source: str = "manual"):
        """Add a document to the knowledge base"""
        # Split content into chunks (simple approach)
        chunks = self.chunk_text(content, chunk_size=500, overlap=50)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for i, chunk in enumerate(chunks):
            cursor.execute('''
                INSERT INTO documents (title, content, source, chunk_id)
                VALUES (?, ?, ?, ?)
            ''', (title, chunk, source, i))
        
        conn.commit()
        conn.close()
        
        # Rebuild vector index after adding documents
        self.build_vector_index()
        print(f"✅ Added document: {title} ({len(chunks)} chunks)")
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
                
        return chunks
    
    def build_vector_index(self):
        """Build FAISS vector index from all documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, content FROM documents')
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            print("No documents found. Add some documents first!")
            return
        
        # Store document chunks and their IDs
        self.document_chunks = [(row[0], row[1]) for row in rows]
        
        # Generate embeddings
        texts = [row[1] for row in rows]
        embeddings = self.embedding_model.encode(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings.astype('float32'))
        self.vector_index.add(embeddings.astype('float32'))
        
        print(f"🔍 Built vector index with {len(texts)} document chunks")
    
    def load_or_create_vector_index(self):
        """Load existing vector index or create new one"""
        try:
            self.build_vector_index()
        except Exception as e:
            print(f"Creating new vector index: {e}")
    
    def search_knowledge_base(self, query: str, top_k: int = 3) -> str:
        """Search knowledge base using vector similarity"""
        if self.vector_index is None or len(self.document_chunks) == 0:
            return "Knowledge base is empty. Please add some documents first."
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding.astype('float32'))
        
        # Search for similar documents
        scores, indices = self.vector_index.search(query_embedding.astype('float32'), top_k)
        
        # Retrieve matching document chunks
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.document_chunks):
                doc_id, content = self.document_chunks[idx]
                
                # Get document metadata
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT title, source FROM documents WHERE id = ?', (doc_id,))
                metadata = cursor.fetchone()
                conn.close()
                
                if metadata:
                    title, source = metadata
                    results.append({
                        'content': content,
                        'title': title,
                        'source': source,
                        'relevance_score': float(score)
                    })
        
        # Format results for the LLM
        if results:
            formatted_results = "Found relevant information:\n\n"
            for i, result in enumerate(results, 1):
                formatted_results += f"Document {i}: {result['title']}\n"
                formatted_results += f"Content: {result['content']}\n"
                formatted_results += f"Source: {result['source']}\n"
                formatted_results += f"Relevance: {result['relevance_score']:.3f}\n\n"
            return formatted_results
        else:
            return "No relevant information found in the knowledge base."
    
    def call_ollama(self, prompt: str) -> str:
        """Make a request to local Ollama instance"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            return response.json().get("response", "")
        except Exception as e:
            return f"Error calling Ollama: {e}"
    
    def chat(self, user_message: str) -> str:
        """Main chat function with knowledge base integration"""
        
        # First, search the knowledge base
        search_results = self.search_knowledge_base(user_message)
        
        # Create enhanced prompt with search results
        prompt = f"""
You are a helpful company assistant with access to the company knowledge base.

User Question: {user_message}

Knowledge Base Search Results:
{search_results}

Instructions:
1. Use the search results to answer the user's question accurately
2. If the search results contain relevant information, base your answer on them
3. If no relevant information is found, say so politely
4. Always be helpful and conversational
5. Cite which documents you're referencing when possible

Answer:"""

        # Get response from Ollama
        response = self.call_ollama(prompt)
        return response

# Helper functions for adding different types of documents
def add_sample_company_docs(agent: KnowledgeBaseAgent):
    """Add sample company documents for testing"""
    
    # HR Policy Document
    hr_policy = """
    Employee Handbook - Remote Work Policy
    
    Our company supports flexible work arrangements to promote work-life balance.
    
    Remote Work Guidelines:
    - Employees may work remotely up to 3 days per week
    - Core collaboration hours are 10 AM - 3 PM in company timezone
    - All remote workers must have reliable internet connection
    - Home office setup stipend of $500 available annually
    
    Vacation Policy:
    - All full-time employees receive 25 vacation days per year
    - Vacation days accrue monthly at 2.08 days per month
    - Maximum carryover is 5 days into the following year
    - Vacation requests require 2 weeks advance notice for trips over 5 days
    
    Benefits Package:
    - Health insurance with 90% company coverage
    - Dental and vision insurance included
    - 401(k) with 4% company matching
    - Professional development budget of $2,000 per year
    - Flexible spending account (FSA) available
    """
    
    # Tech Stack Document
    tech_stack = """
    Engineering Guidelines - Technology Stack
    
    Backend Technologies:
    - Primary language: Python 3.9+
    - Web framework: FastAPI for APIs, Django for web applications
    - Database: PostgreSQL for production, SQLite for development
    - Caching: Redis for session storage and caching
    - Message queue: RabbitMQ for async processing
    
    Frontend Technologies:
    - Framework: React 18+ with TypeScript
    - State management: Redux Toolkit
    - Styling: Tailwind CSS
    - Build tool: Vite
    
    DevOps and Infrastructure:
    - Cloud platform: AWS (EC2, S3, RDS)
    - Containerization: Docker with Docker Compose
    - CI/CD: GitHub Actions
    - Monitoring: DataDog for application monitoring
    - Version control: Git with GitHub
    
    Security Requirements:
    - All APIs must use JWT authentication
    - Database connections must use SSL
    - Environment variables for all secrets
    - Regular security audits quarterly
    """
    
    # Sales Process Document
    sales_process = """
    Sales Team Playbook - Lead Management
    
    Lead Qualification Process:
    1. Initial contact within 24 hours of lead submission
    2. BANT qualification (Budget, Authority, Need, Timeline)
    3. Discovery call to understand use case and requirements
    4. Technical demo tailored to prospect's needs
    5. Proposal and pricing discussion
    6. Contract negotiation and closing
    
    Pricing Structure:
    - Starter Plan: $99/month for up to 10 users
    - Professional Plan: $299/month for up to 50 users
    - Enterprise Plan: Custom pricing for 50+ users
    - Annual subscriptions receive 20% discount
    
    Key Performance Metrics:
    - Lead response time target: Under 4 hours
    - Demo-to-close rate target: 25%
    - Average sales cycle: 45 days
    - Customer acquisition cost (CAC): $1,200
    
    CRM Usage:
    - All prospect interactions must be logged in Salesforce
    - Lead scoring system: Hot (>80), Warm (50-80), Cold (<50)
    - Weekly pipeline reviews every Tuesday at 2 PM
    """
    
    # Add documents to the knowledge base
    agent.add_document("Employee Handbook - HR Policies", hr_policy, "HR Department")
    agent.add_document("Engineering Guidelines", tech_stack, "Engineering Team")
    agent.add_document("Sales Playbook", sales_process, "Sales Team")
    
    print("✅ Added sample company documents!")

# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = KnowledgeBaseAgent()
    
    # Add sample documents (you can skip this if you have real documents)
    add_sample_company_docs(agent)
    
    print("🤖 Company Knowledge Base Agent started!")
    print("Try asking:")
    print("- 'What's our remote work policy?'")
    print("- 'What technologies do we use for backend?'")
    print("- 'How long is our average sales cycle?'")
    print("- 'What are our vacation benefits?'")
    print("\nType 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        response = agent.chat(user_input)
        print(f"Agent: {response}\n")
