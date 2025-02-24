import gradio as gr
from langchain_community.llms import LlamaCpp
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory.chat_message_histories import ChatMessageHistory
from typing import List, Dict
import json
from huggingface_hub import hf_hub_download
import numpy as np


class Chatbot:
    def __init__(self, llm, prompt, template_file):
        self.templates = self.load_template(template_file)
        self.messages_by_id: Dict[str, List] = {}
        self.chain_with_history = RunnableWithMessageHistory(
                                                            prompt | llm,
                                                            self.store_messages,
                                                            input_messages_key="input",
                                                            history_messages_key="history"
                                                        )
        self.embeddings = HuggingFaceEmbeddings(
                                                model_name="sentence-transformers/all-mpnet-base-v2"
                                            )
        self.similarity_threshold = 0.5
    
    def store_messages(self, session_id: str) -> ChatMessageHistory:
        """Return a message store for the given session ID."""
        if session_id not in self.messages_by_id:
            self.messages_by_id[session_id] = ChatMessageHistory()
        return self.messages_by_id[session_id]

    def load_template(self, template_file):
        """Load templates from JSON file."""
        with open(template_file, 'r') as f:
            return json.load(f)['questions']

    def get_most_similar_template(self, query: str):
        """
        Find the most similar template key and its corresponding value.
        Returns (key, value, similarity_score) tuple.
        """
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Calculate similarities for all template keys
        similarities = []
        for qa_pair in self.templates:
            key = qa_pair['question']
            answer = qa_pair['answer']
            key_embedding = self.embeddings.embed_query(key)
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, key_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(key_embedding)
            )
            similarities.append((key, answer, similarity))
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        # # Print all similarities for debugging
        # print("\nSimilarity Scores:")
        # for key, answer, score in similarities:
        #     print(f"Key: {key}")
        #     print(f"Score: {score:.3f}")
        #     print(f"Value: {answer}\n")
        
        # Return most similar if above threshold
        if similarities and similarities[0][2] > self.similarity_threshold:
            best_key = similarities[0][0]
            best_answer = similarities[0][1]
            return best_key, best_answer, similarities[0][2]
        
        return None, None, 0.0

    def chat(self, message: str, history: List[List[str]]) -> str:
        """
        Process a chat message with history.
        
        Args:
            message: The current message from the user
            history: List of [user_message, assistant_message] pairs from Gradio
        """
        # Convert Gradio history format to LangChain message format
        session_id = "chat_session"
        message_history = self.store_messages(session_id)
        
        # Clear existing messages and rebuild from history
        message_history.clear()
        
        # Print history for debugging
        print(message)
        print("History received:", history)
        
        # Safely handle history pairs
        for pair in history:
            if len(pair) >= 2:  # Make sure we have both user and assistant messages
                human_msg, ai_msg = pair[0], pair[1]
                if human_msg:  # Add user message if it exists
                    message_history.add_user_message(human_msg)
                if ai_msg:     # Add AI message if it exists
                    message_history.add_ai_message(ai_msg)
        
        # Search templates using semantic similarity
        template_key, template_value, similarity = self.get_most_similar_template(message)
        
        if template_value is not None:
            print(f"\nUsing template response:")
            print(f"Matched key: {template_key}")
            print(f"Similarity score: {similarity:.3f}")
            return template_value
        
        # Use chain with history for LLM response
        response = self.chain_with_history.invoke(
            {"input": message},
            config={"configurable": {"session_id": session_id}}
        )

        return response

    def calculate_similarity(self, query_embedding, doc_embedding):
        """Calculate cosine similarity between two embeddings."""
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        return float(similarity)

    def launch(self):
        iface = gr.ChatInterface(
            fn=self.chat,
            title="Chatbot for ThoughtfulAI",
            chatbot=gr.Chatbot()
        )
        iface.launch()

if __name__ == "__main__":
    #download the model and store 
    model_path = hf_hub_download(
            repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            repo_type="model",
            cache_dir="models"
        )

    #initialize the model
    llm = LlamaCpp(
        model_path="models/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/snapshots/3a6fbf4a41a1d52e415a4958cde6856d34b2db93/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_gpu_layers=-1,
        n_ctx=2048,
        temperature=0.7
    )

    # Create chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
    ])

    #set the template file
    template_file = "templates.json"

    chatbot = Chatbot(llm, prompt, template_file)
    chatbot.launch()