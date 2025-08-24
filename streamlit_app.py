import streamlit as st
import os
from pathlib import Path
import io
from knowledge_agent import KnowledgeBaseAgent

# Optional imports for document processing
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    st.warning("PyPDF2 not installed. PDF upload disabled.")

try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    st.warning("python-docx not installed. DOCX upload disabled.")

# Page configuration
st.set_page_config(
    page_title="AI Knowledge Base Agent",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = KnowledgeBaseAgent()
    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def extract_text_from_uploaded_file(uploaded_file):
    """Extract text from uploaded files"""
    if uploaded_file.type == "text/plain":
        return str(uploaded_file.read(), "utf-8")
    
    elif uploaded_file.type == "application/pdf" and PDF_SUPPORT:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" and DOCX_SUPPORT:
        doc = docx.Document(uploaded_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    else:
        st.error(f"Unsupported file type or missing dependencies: {uploaded_file.type}")
        return None

# Main App
def main():
    st.title("🤖 AI Knowledge Base Agent")
    st.markdown("Chat with your company's knowledge base powered by local AI!")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.radio(
            "Choose a page:",
            ["💬 Chat", "📁 Upload Documents", "📚 Knowledge Base"]
        )
        
        st.markdown("---")
        st.markdown("### 🔧 System Status")
        
        # Check Ollama status
        try:
            import requests
            response = requests.get("http://localhost:11434", timeout=2)
            st.success("✅ Ollama is running")
        except:
            st.error("❌ Ollama not running. Start with `ollama serve`")
        
        # Knowledge base stats
        try:
            import sqlite3
            conn = sqlite3.connect(st.session_state.agent.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT title) FROM documents")
            doc_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM documents")
            chunk_count = cursor.fetchone()[0]
            conn.close()
            
            st.info(f"📄 {doc_count} documents\n🧩 {chunk_count} chunks")
        except:
            st.info("📄 0 documents")

    # Chat Page
    if page == "💬 Chat":
        st.header("💬 Chat with Your Knowledge Base")
        
        # Create a container for the chat messages with fixed height
        chat_container = st.container()
        
        with chat_container:
            # Display chat history in a scrollable container
            if st.session_state.chat_history:
                for i, (role, message) in enumerate(st.session_state.chat_history):
                    if role == "user":
                        # User message - right aligned with blue background
                        st.markdown(
                            f"""
                            <div style="
                                display: flex; 
                                justify-content: flex-end; 
                                margin: 10px 0;
                            ">
                                <div style="
                                    background-color: #0066cc;
                                    color: white;
                                    padding: 10px 15px;
                                    border-radius: 18px;
                                    max-width: 70%;
                                    word-wrap: break-word;
                                ">
                                    <strong>You:</strong> {message}
                                </div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    else:
                        # Agent message - left aligned with gray background
                        st.markdown(
                            f"""
                            <div style="
                                display: flex; 
                                justify-content: flex-start; 
                                margin: 10px 0;
                            ">
                                <div style="
                                    background-color: #f0f0f0;
                                    color: #333;
                                    padding: 10px 15px;
                                    border-radius: 18px;
                                    max-width: 80%;
                                    word-wrap: break-word;
                                    border: 1px solid #ddd;
                                ">
                                    <strong>🤖 Agent:</strong> {message}
                                </div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
            else:
                # Welcome message
                st.markdown(
                    """
                    <div style="
                        text-align: center; 
                        padding: 40px; 
                        color: #666;
                        font-style: italic;
                    ">
                        👋 Welcome! I'm your AI knowledge base assistant.<br>
                        Ask me anything about your company documents.
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        # Add some spacing before the input area
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Example questions in a collapsible section
        with st.expander("💡 Example Questions", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 What's our remote work policy?", use_container_width=True):
                    st.session_state.example_query = "What's our remote work policy?"
                    st.rerun()
                if st.button("💰 What are our pricing plans?", use_container_width=True):
                    st.session_state.example_query = "What are our pricing plans?"
                    st.rerun()
            
            with col2:
                if st.button("🛠️ What technologies do we use?", use_container_width=True):
                    st.session_state.example_query = "What technologies do we use for backend development?"
                    st.rerun()
                if st.button("🏖️ How many vacation days do we get?", use_container_width=True):
                    st.session_state.example_query = "How many vacation days do employees get?"
                    st.rerun()
        
        # Fixed input area at the bottom
        st.markdown("---")
        
        # Create a form for better UX (Enter key submission)
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([6, 1])
            
            with col1:
                user_input = st.text_input(
                    "Message",
                    placeholder="Ask me anything about your company...",
                    label_visibility="collapsed"
                )
            
            with col2:
                send_button = st.form_submit_button("Send 📤", use_container_width=True, type="primary")
        
        # Process the input
        if send_button and user_input.strip():
            # Add user message to history
            st.session_state.chat_history.append(("user", user_input.strip()))
            
            # Get agent response
            with st.spinner("🤖 Thinking..."):
                try:
                    response = st.session_state.agent.chat(user_input.strip())
                    st.session_state.chat_history.append(("agent", response))
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.session_state.chat_history.append(("agent", error_msg))
            
            # Rerun to show new messages
            st.rerun()
        
        # Handle example questions
        if 'example_query' in st.session_state and st.session_state.example_query:
            query_to_process = st.session_state.example_query
            st.session_state.example_query = ''
            
            # Process the example query immediately
            st.session_state.chat_history.append(("user", query_to_process))
            with st.spinner("🤖 Thinking..."):
                try:
                    response = st.session_state.agent.chat(query_to_process)
                    st.session_state.chat_history.append(("agent", response))
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.session_state.chat_history.append(("agent", error_msg))
            st.rerun()
        
        # Clear chat button in sidebar
        if st.session_state.chat_history:
            st.sidebar.markdown("---")
            if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    # Upload Documents Page
    elif page == "📁 Upload Documents":
        st.header("📁 Upload Documents")
        st.markdown("Upload company documents to build your knowledge base.")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['txt', 'pdf', 'docx', 'md'],
            accept_multiple_files=True,
            help="Supported formats: TXT, PDF, DOCX, Markdown"
        )
        
        if uploaded_files:
            st.markdown(f"**Selected {len(uploaded_files)} file(s):**")
            for file in uploaded_files:
                st.markdown(f"- 📄 {file.name} ({file.size} bytes)")
        
        if st.button("📤 Upload to Knowledge Base", type="primary") and uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            uploaded_count = 0
            total_files = len(uploaded_files)
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                progress_bar.progress((i + 1) / total_files)
                
                try:
                    # Extract text
                    text_content = extract_text_from_uploaded_file(uploaded_file)
                    
                    if text_content and text_content.strip():
                        # Add to knowledge base
                        title = uploaded_file.name.rsplit('.', 1)[0]
                        st.session_state.agent.add_document(
                            title=title,
                            content=text_content,
                            source=f"Upload: {uploaded_file.name}"
                        )
                        uploaded_count += 1
                        st.success(f"✅ Uploaded: {uploaded_file.name}")
                    else:
                        st.error(f"❌ Could not extract text from: {uploaded_file.name}")
                
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            
            status_text.text("Upload complete!")
            st.success(f"🎉 Successfully uploaded {uploaded_count} out of {total_files} documents!")
            
            if uploaded_count > 0:
                st.balloons()

    # Knowledge Base Page
    elif page == "📚 Knowledge Base":
        st.header("📚 Knowledge Base Contents")
        
        try:
            import sqlite3
            conn = sqlite3.connect(st.session_state.agent.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT title, source, COUNT(*) as chunk_count, 
                       MIN(created_at) as created_at
                FROM documents 
                GROUP BY title, source 
                ORDER BY created_at DESC
            ''')
            
            documents = cursor.fetchall()
            conn.close()
            
            if documents:
                st.markdown(f"**Total Documents: {len(documents)}**")
                
                for doc in documents:
                    title, source, chunk_count, created_at = doc
                    
                    with st.expander(f" {title}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"**Source:** {source}")
                        with col2:
                            st.markdown(f"**Chunks:** {chunk_count}")
                        with col3:
                            st.markdown(f"**Added:** {created_at}")
                        
                        # Show first chunk as preview
                        conn = sqlite3.connect(st.session_state.agent.db_path)
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT content FROM documents WHERE title = ? LIMIT 1",
                            (title,)
                        )
                        preview = cursor.fetchone()
                        conn.close()
                        
                        if preview:
                            st.markdown("**Preview:**")
                            st.text(preview[0][:200] + "..." if len(preview[0]) > 200 else preview[0])
            else:
                st.info("📭 No documents in the knowledge base yet. Upload some documents to get started!")
                
        except Exception as e:
            st.error(f"Error loading knowledge base: {str(e)}")

if __name__ == "__main__":
    main()
