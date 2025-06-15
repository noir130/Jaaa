import gradio as gr
from transformers import pipeline
import os

# Load a lightweight model (optimized for Render's free tier CPU)
model = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",  # Auto-detects CPU/GPU
    torch_dtype="auto"
)

def chat(message, history):
    # Generate response (truncate for faster replies)
    response = model(
        message,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7
    )[0]['generated_text']
    return response

# Build the interface
interface = gr.ChatInterface(
    chat,
    title="🤖 Your AI Assistant",
    theme="soft",
    examples=["Hello!", "Explain quantum computing"]
)

# Launch with production settings
interface.launch(
    server_name="0.0.0.0",
    server_port=int(os.getenv("PORT", 7860))
    )
