# AI Knowledge Base Agent

A local AI agent that can chat with your company's knowledge base using RAG (Retrieval-Augmented Generation), manage your Google Calendar, and create Gmail drafts. Built with Ollama for local LLM processing and Streamlit for the web interface.

## Features

- 🤖 **Local AI Processing** - Runs entirely on your machine using Ollama
- 📚 **Knowledge Base** - Upload and search through company documents
- 💬 **Chat Interface** - Clean, ChatGPT-like web interface with conversation context
- 📄 **Multi-format Support** - PDF, DOCX, TXT, and Markdown files
- 🔍 **Semantic Search** - AI-powered document search using vector embeddings
- 📅 **Google Calendar Integration** - View upcoming events and meetings
- 📧 **Gmail Draft Creation** - Create email drafts with automatic knowledge base context
- 🧠 **Conversation Context** - Remembers previous interactions for natural follow-up questions
- 💡 **Proactive Intelligence** - Automatically includes relevant company information
- 🔒 **Privacy First** - All data stays on your local machine


## Prerequisites

- Python 3.8 or higher
- At least 8GB RAM (for running local LLM)
- [Ollama](https://ollama.ai/) installed on your system
- Google account for Calendar and Gmail integration

## Quick Start

### 1. Install Ollama

Visit [ollama.ai](https://ollama.ai/) and install Ollama for your operating system.

After installation, pull the required model:
```bash
ollama pull llama3.2
```

### 2. Clone and Setup the Project

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-knowledge-agent.git
cd ai-knowledge-agent

# Create virtual environment
python -m venv agent-env

# Activate virtual environment
# On macOS/Linux:
source agent-env/bin/activate
# On Windows:
agent-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup Google Services (Optional but Recommended)

For Calendar and Gmail integration:

1. **Create Google Cloud Project**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project

2. **Enable APIs**
   - Enable "Google Calendar API"
   - Enable "Gmail API"

3. **Create Credentials**
   - Go to APIs & Services > Credentials
   - Create OAuth 2.0 Client ID for Desktop application
   - Download the credentials JSON file

4. **Add Credentials**
   - Rename downloaded file to `credentials.json`
   - Place in your project root directory

5. **OAuth Consent Screen**
   - Configure OAuth consent screen
   - Add scopes:
     - `https://www.googleapis.com/auth/calendar.readonly`
     - `https://www.googleapis.com/auth/gmail.compose`
   - Add your email as test user

### 4. Start Ollama Server

In a separate terminal, start Ollama:
```bash
ollama serve
```

Keep this terminal open while using the agent.

### 5. Run the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

### Upload Documents
1. Navigate to the "📁 Upload Documents" section
2. Drag and drop or select your company documents (PDF, DOCX, TXT)
3. Click "📤 Upload to Knowledge Base"

### Chat with Your Agent
1. Go to the "💬 Chat" section
2. Ask questions using natural language
3. The agent automatically determines what you want to do

### Google Integration Features
- **Calendar:** "What's my calendar this week?", "Show me tomorrow's meetings"
- **Email:** "Draft an email to John about the remote work policy"
- **Context:** Ask follow-up questions referencing previous responses

## Example Workflows

### Knowledge-Aware Email Creation
1. **You:** "What's our vacation policy?"
2. **Agent:** [Shows vacation policy from knowledge base]
3. **You:** "Draft an email to the team about this"
4. **Agent:** [Creates Gmail draft automatically including vacation policy details]

### Natural Calendar Interaction
1. **You:** "What's my schedule tomorrow?"
2. **Agent:** [Shows actual calendar events]
3. **You:** "Tell me more about the 2 PM meeting"
4. **Agent:** [Provides details about that specific meeting from context]

### Intelligent Document Search
1. **You:** "How many remote work days are allowed?"
2. **Agent:** [Searches and finds relevant policy information]
3. **You:** "Send this info to Sarah"
4. **Agent:** [Creates email draft with policy details to Sarah]

## How It Works

### RAG (Retrieval-Augmented Generation)
1. **Document Processing**: Uploaded files are split into chunks and stored
2. **Vector Embeddings**: Each chunk is converted to mathematical vectors
3. **Semantic Search**: User questions find relevant document chunks
4. **AI Response**: Local LLM generates answers using found context

### Conversation Context
- Maintains memory of recent interactions
- Enables natural follow-up questions
- References previous responses intelligently


## Architecture

```
Streamlit Interface ←→ Enhanced Agent ←→ Ollama (llama3.2)
                           ↓
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
Knowledge Base          Google Calendar        Gmail API
(FAISS + SQLite)       (Real Events)         (Draft Creation)
```

## Troubleshooting

### Ollama Not Running
```bash
# Check if Ollama is running
curl http://localhost:11434

# If not running, start it:
ollama serve
```

### Google Services Not Connected
- Check if `credentials.json` exists in project root
- Verify OAuth consent screen is properly configured
- Delete `token.pickle` and restart app to re-authenticate

### Slow Responses
- Responses take 3-6 seconds (normal for local LLM)
- Keep Ollama running to avoid cold starts
- Consider using smaller model: `ollama pull llama3.2:1b`

## Customization

### Using Different Models
Edit `knowledge_agent.py` and change:
```python
self.model = "llama3.2"  # Change to your preferred model
```

### Adjusting Context Window
Modify conversation context length:
```python
def build_conversation_context(self, chat_history, max_exchanges=3):  # Increase for longer memory
```

### Customizing Email Templates
The agent creates emails using natural language generation - no templates needed. It automatically adapts to different email types and includes relevant company information.

## Development Journey

Read about the complete development journey:
- [Part 1: Building the Foundation](link-to-part-1)
- [Part 2: From Chatbot to True Agent](link-to-part-2)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [Ollama](https://ollama.ai/) for local LLM processing
- Uses [Streamlit](https://streamlit.io/) for the web interface
- Powered by [FAISS](https://github.com/facebookresearch/faiss) for vector search
- Document processing with [Sentence Transformers](https://www.sbert.net/)
- Google Calendar and Gmail integration via Google APIs