import json
import sqlite3
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime, timedelta
import os
import pickle
import re

# Google API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class EnhancedKnowledgeAgent:
    def __init__(self, knowledge_db_path="knowledge.db"):
        # Ollama setup
        self.base_url = "http://localhost:11434"
        self.model = "llama3.2"
        
        # Knowledge base setup
        self.db_path = knowledge_db_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.init_database()
        
        # Vector search setup
        self.vector_index = None
        self.document_chunks = []
        self.load_or_create_vector_index()
        
        # Google Services setup
        self.SCOPES = [
            'https://www.googleapis.com/auth/calendar.readonly',
            'https://www.googleapis.com/auth/gmail.compose'
        ]
        self.calendar_service = None
        self.gmail_service = None
        self.setup_google_services()
        
        # Conversation context tracking
        self.recent_knowledge_searches = []
        
        # Keep Ollama warm
        self.keep_model_warm()

    def keep_model_warm(self):
        """Keep Ollama model loaded in memory"""
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "Hi",
                    "stream": False,
                    "keep_alive": "10m"
                },
                timeout=5
            )
            print("🔥 Ollama model warmed up")
        except Exception as e:
            print(f"⚠️ Could not warm up Ollama model: {e}")

    def setup_google_services(self):
        """Setup Google Calendar and Gmail API authentication"""
        try:
            creds = None
            if os.path.exists('token.pickle'):
                with open('token.pickle', 'rb') as token:
                    creds = pickle.load(token)
            
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if os.path.exists('credentials.json'):
                        flow = InstalledAppFlow.from_client_secrets_file(
                            'credentials.json', self.SCOPES)
                        creds = flow.run_local_server(port=0)
                    else:
                        print("⚠️ Google services setup incomplete. Please add credentials.json")
                        return
                
                with open('token.pickle', 'wb') as token:
                    pickle.dump(creds, token)
            
            # Build services
            self.calendar_service = build('calendar', 'v3', credentials=creds)
            self.gmail_service = build('gmail', 'v1', credentials=creds)
            print("✅ Google Calendar and Gmail connected successfully")
            
        except Exception as e:
            print(f"❌ Google services setup failed: {e}")

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
            return
        
        self.document_chunks = [(row[0], row[1]) for row in rows]
        
        texts = [row[1] for row in rows]
        embeddings = self.embedding_model.encode(texts)
        
        dimension = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatIP(dimension)
        
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
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding.astype('float32'))
        
        scores, indices = self.vector_index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.document_chunks):
                doc_id, content = self.document_chunks[idx]
                
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
        
        if results:
            # Store recent searches for email context
            search_info = {
                'query': query,
                'results': results,
                'timestamp': datetime.now()
            }
            self.recent_knowledge_searches.append(search_info)
            # Keep only last 5 searches
            self.recent_knowledge_searches = self.recent_knowledge_searches[-5:]
            
            formatted_results = "Found relevant information:\n\n"
            for i, result in enumerate(results, 1):
                formatted_results += f"Document {i}: {result['title']}\n"
                formatted_results += f"Content: {result['content']}\n"
                formatted_results += f"Relevance: {result['relevance_score']:.3f}\n\n"
            return formatted_results
        else:
            return "No relevant information found in the knowledge base."

    def build_conversation_context(self, chat_history: List[tuple], max_exchanges: int = 3) -> str:
        """Build conversation context from recent chat history"""
        if not chat_history:
            return ""
        
        recent_history = chat_history[-max_exchanges*2:] if len(chat_history) > max_exchanges*2 else chat_history
        
        context = "Recent conversation:\n"
        for role, message in recent_history:
            if role == "user":
                context += f"User: {message}\n"
            else:
                context += f"Agent: {message[:200]}...\n" if len(message) > 200 else f"Agent: {message}\n"
        
        return context + "\n"

    def detect_user_intent_with_context(self, message: str, chat_history: List[tuple] = None) -> str:
        """Use LLM to detect what the user wants to do with conversation context"""
        context = self.build_conversation_context(chat_history) if chat_history else ""
        
        prompt = f"""
        {context}Current user message: "{message}"

        Based on the conversation context and current message, determine the user's intent:

        Possible intents:
        - SHOW_CALENDAR: User explicitly asks about calendar, meetings, events, schedule, or wants to see what's on their calendar
        - CREATE_EMAIL: User wants to create, draft, compose, or send an email
        - KNOWLEDGE_SEARCH: User asks about company policies, procedures, information, or wants to learn about something

        Examples:
        - "What's my calendar tomorrow?" -> SHOW_CALENDAR
        - "Show my meetings" -> SHOW_CALENDAR  
        - "Draft an email" -> CREATE_EMAIL
        - "What's our WFH policy?" -> KNOWLEDGE_SEARCH
        - "How many vacation days?" -> KNOWLEDGE_SEARCH

        Return only one word: SHOW_CALENDAR, CREATE_EMAIL, or KNOWLEDGE_SEARCH
        """
        
        response = self.call_ollama(prompt).strip().upper()
        
        if "SHOW_CALENDAR" in response:
            return "SHOW_CALENDAR"
        elif "CREATE_EMAIL" in response:
            return "CREATE_EMAIL"
        else:
            return "KNOWLEDGE_SEARCH"

    def extract_time_context(self, message: str) -> int:
        """Use LLM to extract time context from user message"""
        prompt = f"""
        Extract the candidate name and role from this message: "{message}"

        Return in this format:
        {{
            "candidate_name": "extracted name or 'Unknown'",
            "role_position": "extracted role or 'Unknown'", 
            "event_title": "extracted event title or meeting reference",
            "source": "current_message" or "chat_history"
        }}

        Examples:
        - Recent context shows "John Smith - Quality Head Interview", user says "analyze that candidate" 
        → {{"candidate_name": "John Smith", "role_position": "Quality Head", "event_title": "Quality Head Interview", "source": "chat_history"}}
        """
        prompt = f"""
        What time period is mentioned in this message: "{message}"

        Reply with only one number:
        1 for today
        2 for tomorrow
        7 for this week
        14 for next week  
        30 for this month
        7 if no time mentioned

        Message: "{message}"
        Answer:
        """
        
        response = self.call_ollama(prompt).strip()
        
        numbers = re.findall(r'\d+', response)
        if numbers:
            return int(numbers[0])
        else:
            return 7

    def get_upcoming_calendar_events(self, days_ahead: int = 7) -> str:
        """Get upcoming calendar events with relevant knowledge base context"""
        if not self.calendar_service:
            return "Google Calendar not connected. Please set up authentication first."
        
        try:
            # Get calendar events (existing code)
            now = datetime.utcnow().isoformat() + 'Z'
            end_time = (datetime.utcnow() + timedelta(days=days_ahead)).isoformat() + 'Z'
            
            events_result = self.calendar_service.events().list(
                calendarId='primary',
                timeMin=now,
                timeMax=end_time,
                maxResults=20,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            if events:
                # Proactively search knowledge base for event-related information
                event_titles = [event.get('summary', '') for event in events]
                combined_titles = ' '.join(event_titles)
                relevant_context = self.search_knowledge_base(combined_titles, top_k=2)
                
                result = f"Found {len(events)} upcoming events:\n\n"
                for i, event in enumerate(events, 1):
                    summary = event.get('summary', 'No title')
                    start_time = event['start'].get('dateTime', event['start'].get('date'))
                    description = event.get('description', '')
                    html_link = event.get('htmlLink', '')
                    
                    if 'T' in start_time:
                        dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                        formatted_time = dt.strftime('%Y-%m-%d %I:%M %p')
                    else:
                        formatted_time = start_time
                    
                    result += f"{i}. {summary}\n"
                    result += f"   Time: {formatted_time}\n"
                    
                    if html_link:
                        result += f"   📅 [Open in Calendar]({html_link})\n"
                    
                    if description:
                        desc_preview = description[:100] + "..." if len(description) > 100 else description
                        result += f"   📝 Description: {desc_preview}\n"
                    
                    result += "\n"
                
                # Add relevant knowledge context if found
                if "Found relevant information" in relevant_context:
                    result += f"💡 Relevant company information:\n{relevant_context}\n"
                
                return result
            else:
                return "No upcoming calendar events found in the specified time period."
                
        except HttpError as error:
            return f"An error occurred while fetching calendar events: {error}"
        except Exception as e:
            return f"Error accessing Google Calendar: {str(e)}"
    
    def extract_email_context(self, message: str, chat_history: List[tuple] = None) -> Dict:
        """Extract email details from user message and conversation context"""
        context = self.build_conversation_context(chat_history) if chat_history else ""
        
        prompt = f"""
        {context}Current user message: "{message}"
        
        Extract email details from the user's request. Consider conversation context for any referenced information.
        
        What email does the user want to create? Extract:
        - Who should receive it (recipient)
        - What should be the subject
        - What should be included in the content
        - Any specific information from previous conversation to include
        
        Describe in natural language what email should be created.
        
        Example: "User wants to send an email to john@company.com about the WFH policy they just searched for, including the policy details in the email body"
        """
        
        response = self.call_ollama(prompt)
        return {"email_intent": response.strip()}

    def create_gmail_draft(self, email_details: Dict) -> str:
        """Create a Gmail draft with proactive knowledge base search"""
        if not self.gmail_service:
            return "Gmail not connected. Please set up authentication first."
        
        try:
            email_intent = email_details.get('email_intent', '')
            
            # Proactively search knowledge base for relevant information
            relevant_knowledge = self.search_knowledge_base(email_intent, top_k=2)
            
            # Create draft content using AI with proactive knowledge
            draft_prompt = f"""
            Create a professional email draft based on this request:
            
            Email Request: {email_intent}
            
            Relevant Company Information Found:
            {relevant_knowledge}
            
            Instructions:
            - Create a complete, professional email
            - Include relevant company information if it helps the email purpose
            - If company information is relevant, reference the source document
            - If no relevant company info, create email without it
            - Don't force irrelevant information into the email
            
            Format as:
            Subject: [subject line]
            
            [email body]
            """
            
            draft_content = self.call_ollama(draft_prompt)
            
            # Parse subject and body
            lines = draft_content.split('\n')
            subject = "Email Draft"
            body = draft_content
            
            for line in lines:
                if line.startswith('Subject:'):
                    subject = line.replace('Subject:', '').strip()
                    body = '\n'.join(lines[lines.index(line)+1:]).strip()
                    break
            
            # Create Gmail draft
            draft_message = {
                'message': {
                    'raw': self.create_email_raw(subject, body)
                }
            }
            
            draft = self.gmail_service.users().drafts().create(
                userId='me',
                body=draft_message
            ).execute()
            
            draft_id = draft['id']
            draft_url = f"https://mail.google.com/mail/u/0/#drafts/{draft_id}"
            
            return f"✅ Email draft created successfully!\n\n📧 Subject: {subject}\n\n📝 Preview:\n{body[:300]}{'...' if len(body) > 300 else ''}\n\n🔗 [Open in Gmail]({draft_url})"
            
        except Exception as e:
            return f"Error creating email draft: {str(e)}"
    def create_email_raw(self, subject: str, body: str, to_email: str = "") -> str:
        """Create raw email message for Gmail API"""
        import base64
        from email.mime.text import MIMEText
        
        message = MIMEText(body)
        message['subject'] = subject
        if to_email:
            message['to'] = to_email
        
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        return raw_message

    def call_ollama(self, prompt: str) -> str:
        """Make a request to local Ollama instance with keep-alive"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "10m"
                }
            )
            return response.json().get("response", "")
        except Exception as e:
            return f"Error calling Ollama: {e}"

    def chat(self, user_message: str, chat_history: List[tuple] = None) -> str:
        """Enhanced chat with flexible natural language processing"""
        try:
            if chat_history is None:
                chat_history = []
            
            # Use AI to detect user intent with conversation context
            intent = self.detect_user_intent_with_context(user_message, chat_history)
            
            if intent == "SHOW_CALENDAR":
                days = self.extract_time_context(user_message)
                events = self.get_upcoming_calendar_events(days)
                return events
            
            elif intent == "CREATE_EMAIL":
                # Extract email context from message and conversation
                email_details = self.extract_email_context(user_message, chat_history)
                
                # Create draft with knowledge base context
                draft_result = self.create_gmail_draft(
                    email_details
                )
                return draft_result
            
            else:  # KNOWLEDGE_SEARCH
                # Regular knowledge base search with conversation context
                search_results = self.search_knowledge_base(user_message)
                context = self.build_conversation_context(chat_history)
                
                prompt = f"""
                {context}
                You are a helpful company assistant with access to the company knowledge base and Google services.
                
                Current User Question: {user_message}
                
                Knowledge Base Results:
                {search_results}
                
                Provide a natural, helpful response based on the available information and conversation context.
                """
                
                response = self.call_ollama(prompt)
                return response
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

# Helper function to add sample job descriptions
def add_sample_job_descriptions(agent: EnhancedKnowledgeAgent):
    """Add sample job descriptions and policies for testing"""
    
    wfh_policy = """
    Work From Home Policy - 2024
    
    Remote Work Guidelines:
    - Employees may work remotely up to 3 days per week
    - Core collaboration hours are 10 AM - 3 PM in company timezone
    - All remote workers must have reliable internet connection
    - Home office setup stipend of $500 available annually
    - Weekly team check-ins required for remote workers
    - Equipment provided: laptop, monitor, webcam, headset
    
    Approval Process:
    - Manager approval required for remote work arrangements
    - HR notification within 48 hours of approval
    - Quarterly reviews of remote work effectiveness
    
    Productivity Expectations:
    - Same performance standards apply to remote workers
    - Response time expectations: email within 4 hours, messages within 1 hour
    - Participation in all scheduled meetings mandatory
    """
    
    vacation_policy = """
    Vacation and Time Off Policy
    
    Vacation Entitlement:
    - All full-time employees receive 25 vacation days per year
    - Vacation days accrue monthly at 2.08 days per month
    - Maximum carryover is 5 days into the following year
    - Vacation requests require 2 weeks advance notice for trips over 5 days
    
    Sick Leave:
    - 10 sick days per year
    - No doctor's note required for absences under 3 days
    - Sick leave does not roll over to following year
    
    Personal Days:
    - 3 personal days per year for personal matters
    - Can be used in half-day increments
    - 24 hours advance notice preferred
    """
    
    python_dev_jd = """
    Senior Python Developer - Job Description
    
    Required Skills:
    - 5+ years of Python development experience
    - Strong experience with Django or FastAPI frameworks
    - Proficiency in SQL databases (PostgreSQL, MySQL)
    - Experience with cloud platforms (AWS, GCP, or Azure)
    - Knowledge of containerization (Docker, Kubernetes)
    """
    
    agent.add_document("Work From Home Policy", wfh_policy, "HR Department")
    agent.add_document("Vacation and Time Off Policy", vacation_policy, "HR Department") 
    agent.add_document("Senior Python Developer JD", python_dev_jd, "HR Department")
    
    print("✅ Added sample company documents!")

if __name__ == "__main__":
    agent = EnhancedKnowledgeAgent()
    add_sample_job_descriptions(agent)
    
    print("🤖 Enhanced Knowledge Base Agent started!")
    print("📅 Calendar: 'What's my calendar for this week?'")
    print("📧 Email: 'Draft an email to John about the WFH policy'")
    print("📚 Knowledge: 'What's our vacation policy?'")
    
    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        response = agent.chat(user_input, chat_history)
        print(f"Agent: {response}\n")
        
        chat_history.append(("user", user_input))
        chat_history.append(("agent", response))