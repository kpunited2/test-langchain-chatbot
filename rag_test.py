import gradio as gr
from langchain_community.llms import LlamaCpp
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory.chat_message_histories import ChatMessageHistory
from typing import List, Dict, Optional
import json
from huggingface_hub import hf_hub_download
import numpy as np
import os
import tempfile
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


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
        self.document_chunks: List[Document] = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
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
        
        # Return most similar if above threshold
        if similarities and similarities[0][2] > self.similarity_threshold:
            best_key = similarities[0][0]
            best_answer = similarities[0][1]
            return best_key, best_answer, similarities[0][2]
        
        return None, None, 0.0

    def process_document(self, file_obj) -> List[Document]:
        """
        Process an uploaded document and convert it to text chunks for RAG.
        
        Args:
            file_obj: The uploaded file object from Gradio
            
        Returns:
            List of Document objects containing the processed text chunks
        """
        if file_obj is None:
            return []
            
        print(f"Processing document: {file_obj.name}")
        
        # Create a temporary file to save the uploaded file
        file_extension = os.path.splitext(file_obj.name)[1].lower()
        temp_path = None
        
        try:
            # Gradio's file upload gives us a path rather than a file object with read method
            if hasattr(file_obj, 'name'):
                # Copy the file to a temp location to ensure we have proper access
                with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                    temp_path = temp_file.name
                    
                    # If it's a NamedString (Gradio specific), we can access the path
                    if hasattr(file_obj, 'path'):
                        # Copy content from the original file to our temp file
                        with open(file_obj.path, 'rb') as src_file:
                            temp_file.write(src_file.read())
                    else:
                        # Fallback in case file_obj has a read method
                        try:
                            temp_file.write(file_obj.read())
                        except AttributeError:
                            print("File object doesn't have a read method, trying to access as path")
                            with open(file_obj, 'rb') as src_file:
                                temp_file.write(src_file.read())
            else:
                # If file_obj is just a path string
                temp_path = file_obj
            
            # Load document based on file type
            if file_extension == '.txt':
                loader = TextLoader(temp_path)
            elif file_extension == '.pdf':
                loader = PyPDFLoader(temp_path)
            elif file_extension in ['.doc', '.docx']:
                loader = Docx2txtLoader(temp_path)
            else:
                # Default to text loader for unknown types
                loader = TextLoader(temp_path)
                
            documents = loader.load()
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            print(f"Document processed into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            return []
        finally:
            # Clean up the temporary file if we created one
            if temp_path and temp_path != file_obj and os.path.exists(temp_path):
                os.remove(temp_path)
    
    def get_relevant_document_context(self, query: str) -> Optional[str]:
        """
        Retrieve relevant document chunks based on the query.
        
        Args:
            query: The user's query
            
        Returns:
            String containing the relevant document context, or None if no relevant chunks
        """
        if not self.document_chunks:
            return None
            
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Calculate similarities for all document chunks
        chunk_similarities = []
        for i, chunk in enumerate(self.document_chunks):
            chunk_embedding = self.embeddings.embed_query(chunk.page_content)
            similarity = self.calculate_similarity(query_embedding, chunk_embedding)
            chunk_similarities.append((i, similarity, chunk))
        
        # Sort by similarity
        chunk_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 3 most relevant chunks
        top_chunks = chunk_similarities[:3]
        
        # Only use chunks with similarity above threshold
        relevant_chunks = [chunk for _, sim, chunk in top_chunks if sim > 0.3]
        
        if not relevant_chunks:
            return None
            
        # Combine relevant chunks into a context string
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        return context

    def chat(self, message: str, history: List[List[str]], uploaded_file=None) -> List[List[str]]:
        """
        Process a chat message with history.
        
        Args:
            message: The current message from the user
            history: List of [user_message, assistant_message] pairs from Gradio
            uploaded_file: Optional uploaded document to reference
            
        Returns:
            Updated history with new message pair added
        """
        # Convert Gradio history format to LangChain message format
        session_id = "chat_session"
        message_history = self.store_messages(session_id)
        
        # Clear existing messages and rebuild from history
        message_history.clear()
        
        # Print information for debugging
        print(message)
        print("History received:", history)
        
        user_message = message
        
        # Get relevant document context based on user query
        document_context = self.get_relevant_document_context(message)
        
        # Safely handle history pairs
        for pair in history:
            if len(pair) >= 2:  # Make sure we have both user and assistant messages
                human_msg, ai_msg = pair[0], pair[1]
                if human_msg:  # Add user message if it exists
                    message_history.add_user_message(human_msg)
                if ai_msg:     # Add AI message if it exists
                    message_history.add_ai_message(ai_msg)
        
        # Search templates using semantic similarity
        template_key, template_value, similarity = self.get_most_similar_template(user_message)
        
        if template_value is not None:
            print(f"\nUsing template response:")
            print(f"Matched key: {template_key}")
            print(f"Similarity score: {similarity:.3f}")
            response = template_value
        else:
            # Enhance user message with document context if available
            if document_context:
                print("Adding document context to query")
                enhanced_message = (
                    f"{message}\n\n"
                    f"Here is relevant information from the uploaded document:\n"
                    f"{document_context}"
                )
                # Use chain with history for LLM response
                response = self.chain_with_history.invoke(
                    {"input": enhanced_message},
                    config={"configurable": {"session_id": session_id}}
                )
            else:
                # Use chain with history for LLM response without document context
                response = self.chain_with_history.invoke(
                    {"input": user_message},
                    config={"configurable": {"session_id": session_id}}
                )
        
        # Return the updated history with the new message pair
        history.append([user_message, response])
        return history

    def calculate_similarity(self, query_embedding, doc_embedding):
        """Calculate cosine similarity between two embeddings."""
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        return float(similarity)

    def launch(self):
        with gr.Blocks() as interface:
            gr.Markdown("# ThoughtfulAI Chatbot")
            
            with gr.Row():
                with gr.Column(scale=4):
                    # Horizontal scrolling marquee-like text
                    gr.Markdown(
                        """
                        <div style="overflow-x: scroll; white-space: nowrap; padding: 10px; background-color: ##525254; border-radius: 5px; margin-bottom: 10px;">
                        🤖 Welcome to ThoughtfulAI Chatbot! Ask questions about our healthcare automation agents like EVA, CAM, and PHIL. 
                        Our chatbot uses advanced semantic matching and LLM technology to provide helpful responses.
                        </div>
                        """,
                        elem_id="scrolling-text"
                    )
                    
                    chatbot = gr.Chatbot(height=450)
                    
                    with gr.Row():
                        message = gr.Textbox(
                            placeholder="Type your message here...",
                            show_label=False,
                            scale=9
                        )
                        submit = gr.Button("Send", scale=1)
                
                with gr.Column(scale=1):
                    gr.Markdown("### Upload Documents")
                    uploaded_file = gr.File(
                        label="Upload a document for reference",
                        file_types=[".pdf", ".txt", ".doc", ".docx"],
                        type="filepath",
                        file_count="single"
                    )
                    gr.Markdown(
                        """
                        #### Document Tips
                        - Upload relevant documents to enhance chat
                        - Supported file types: PDF, TXT, DOC, DOCX
                        - Max file size: 10MB
                        """
                    )
                    
                    # Add clear button for documents
                    clear_docs = gr.Button("Clear Document Context")
            
            # Event handlers
            def process_message(msg, chat_history, doc):
                if not msg.strip():
                    return chat_history
                return self.chat(msg, chat_history, doc)
            
            def handle_file_upload(file, chat_history):
                if file is None:
                    return chat_history
                
                # Process the file once
                self.document_chunks = self.process_document(file)
                
                if self.document_chunks:
                    chat_history.append([
                        f"I've uploaded {file.name}.", 
                        f"I've processed the document '{file.name}' and extracted {len(self.document_chunks)} chunks of content. You can now ask questions about it!"
                    ])
                else:
                    chat_history.append([
                        f"I've uploaded {file.name}.", 
                        f"I couldn't process the document '{file.name}'. Please check if the file format is supported (PDF, TXT, DOC, DOCX)."
                    ])
                
                return chat_history
            
            def clear_document_context(chat_history):
                self.document_chunks = []
                chat_history.append([None, "Document context has been cleared. The chatbot will no longer reference previously uploaded documents."])
                return chat_history
            
            # Set up event handling for file uploads separately
            uploaded_file.change(
                handle_file_upload,
                inputs=[uploaded_file, chatbot],
                outputs=[chatbot],
                queue=True
            )
            
            message.submit(
                process_message,
                inputs=[message, chatbot, None],  # Pass None instead of uploaded_file
                outputs=chatbot,
                queue=True
            ).then(
                lambda: "",
                None,
                message,
                queue=False
            )
            
            submit.click(
                process_message,
                inputs=[message, chatbot, None],  # Pass None instead of uploaded_file
                outputs=chatbot,
                queue=True
            ).then(
                lambda: "",
                None,
                message,
                queue=False
            )
            
            # Handle document clearing
            clear_docs.click(
                clear_document_context,
                inputs=[chatbot],
                outputs=[chatbot],
                queue=True
            )
                            
        interface.launch()

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

    # Create chat prompt template with enhanced RAG instructions
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. When provided with document context, use this information to inform your answers. If the question pertains to the document, base your response on the document content. If there's no relevant information in the document for a question, acknowledge this and provide a general response based on your knowledge."),
        ("human", "{input}"),
    ])

    #set the template file
    template_file = "templates.json"

    chatbot = Chatbot(llm, prompt, template_file)
    chatbot.launch()