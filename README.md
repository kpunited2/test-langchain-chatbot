ThoughtfulAI Chatbot
A conversational AI chatbot built with LangChain, Gradio, and Mistral-7B that can provide template-based or generated responses based on semantic similarity.
Features

Uses Mistral-7B-Instruct LLM for natural language generation
Implements semantic search for matching queries to pre-defined templates
Maintains conversation history across chat sessions
User-friendly Gradio web interface
Customizable response templates via JSON configuration

Requirements

Python 3.11+
CUDA-compatible GPU (optional, for faster inference)

Installation

Clone this repository:
git clone https://github.com/kpunited2/test-langchain-chatbot.git
cd test-langchain-chatbot

Create and activate a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

Install the required dependencies:
pip install -r requirements.txt


Usage

Run the chatbot:
python main.py

Open your web browser and navigate to the URL displayed in the terminal (typically http://127.0.0.1:7860).
Start chatting with the bot through the Gradio interface.

How It Works

Template Matching: When a user sends a message, the system first calculates the semantic similarity between the query and predefined templates using sentence transformers.
Response Generation:

If a template match with similarity above the threshold (default: 0.5) is found, the pre-defined answer is returned.
Otherwise, the query is sent to the Mistral-7B LLM for generating a contextual response.


Conversation Memory: The chatbot maintains conversation history for each session, allowing for contextual responses.