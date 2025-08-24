# AI Knowledge Base Agent

A local AI agent that can chat with your company's knowledge base using RAG (Retrieval-Augmented Generation). Built with Ollama for local LLM processing and Streamlit for the web interface.

![Demo Screenshot](demo.png)

## Features

- 🤖 **Local AI Processing** - Runs entirely on your machine using Ollama
- 📚 **Knowledge Base** - Upload and search through company documents
- 💬 **Chat Interface** 
- 📄 **Multi-format Support** - PDF, DOCX, TXT, and Markdown files
- 🔍 **Semantic Search** - AI-powered document search using vector embeddings
- 🔒 **Privacy First** - All data stays on your local machine

## Prerequisites

- Python 3.8 or higher
- At least 8GB RAM (for running local LLM)
- [Ollama](https://ollama.ai/) installed on your system

## Quick Start

### 1. Install Ollama

Visit [ollama.ai](https://ollama.ai/) and install Ollama for your operating system.

After installation, pull the required model:
```bash
ollama pull llama3.2
