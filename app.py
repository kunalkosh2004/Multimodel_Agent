import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import pytesseract
from PIL import Image
import docx2txt
import PyPDF2
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

try:
    # Get available models
    available_models = [m.name for m in genai.list_models()]
    
    # Find appropriate models for text and multimodal
    if "models/gemini-1.5-pro" in available_models:
        text_model_name = "models/gemini-1.5-pro"
        vision_model_name = "models/gemini-1.5-pro"
    elif "models/gemini-1.0-pro" in available_models:
        text_model_name = "models/gemini-1.0-pro"
        vision_model_name = "models/gemini-1.0-pro-vision"
    else:
        # Fallback to basic models
        text_model_name = [m for m in available_models if "text" in m.lower()][0]
        vision_model_name = [m for m in available_models if "vision" in m.lower()][0]
    
    # Configure model settings
    generation_config = {
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    }
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    # Initialize models
    model = genai.GenerativeModel(model_name=vision_model_name, 
                                 generation_config=generation_config,
                                 safety_settings=safety_settings)
    
    text_model = genai.GenerativeModel(model_name=text_model_name,
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)
    
except Exception as e:
    st.error(f"Error initializing Gemini models: {str(e)}")
    st.info("Please make sure you have a valid API key and the models are accessible.")
    text_model = None
    model = None

st.set_page_config(page_title="Data Analyst Agent", layout="wide")

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'file_content' not in st.session_state:
    st.session_state.file_content = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'available_models' not in st.session_state:
    st.session_state.available_models = []

def extract_text_from_image(image):
    # Extract text from image using OCR
    return pytesseract.image_to_string(image)

def extract_text_from_pdf(file):
    # Extract text from PDF file
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    # Extract text from DOCX file
    return docx2txt.process(file)

def process_file(uploaded_file):
    # Process the uploaded file and extract content
    file_type = uploaded_file.type
    st.session_state.file_type = file_type
    
    if file_type.startswith('image'):
        # Process image file
        image = Image.open(uploaded_file)
        st.session_state.file_content = extract_text_from_image(image)
        st.session_state.image = image
        return "Image processed successfully."
        
    elif file_type == 'application/pdf':
        # Process PDF file
        text = extract_text_from_pdf(uploaded_file)
        st.session_state.file_content = text
        return "PDF processed successfully."
        
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        # Process DOCX file
        text = extract_text_from_docx(uploaded_file)
        st.session_state.file_content = text
        return "DOCX processed successfully."
        
    elif file_type == 'text/plain':
        # Process TXT file
        text = uploaded_file.getvalue().decode('utf-8')
        st.session_state.file_content = text
        return "Text file processed successfully."
        
    elif file_type == 'text/csv' or file_type == 'application/vnd.ms-excel' or file_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        # Process CSV or Excel file
        try:
            if file_type == 'text/csv':
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.data = df
            st.session_state.file_content = df.head(10).to_string()
            return "Data loaded successfully. Shape: " + str(df.shape)
        except Exception as e:
            return f"Error loading data: {str(e)}"
    
    else:
        return "Unsupported file type: " + file_type

def generate_plot(plot_type, df, x_column, y_column, color_column=None, title=""):
    # Generate plot based on user inputs
    if plot_type == "Bar Chart":
        fig = px.bar(df, x=x_column, y=y_column, color=color_column, title=title)
    elif plot_type == "Line Chart":
        fig = px.line(df, x=x_column, y=y_column, color=color_column, title=title)
    elif plot_type == "Scatter Plot":
        fig = px.scatter(df, x=x_column, y=y_column, color=color_column, title=title)
    elif plot_type == "Histogram":
        fig = px.histogram(df, x=x_column, title=title)
    elif plot_type == "Box Plot":
        fig = px.box(df, y=y_column, x=x_column, title=title)
    elif plot_type == "Heatmap":
        # Create correlation matrix
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='Viridis'))
        fig.update_layout(title=title)
    else:
        st.error("Plot type not supported")
        return None

    return fig

def answer_question(question):
    # Use Gemini to answer questions about the data
    if text_model is None or model is None:
        return "Error: Gemini models not properly initialized. Please check your API key and model availability."
        
    try:
        if st.session_state.data is not None:
            # Prepare context about the data
            data_info = f"Data shape: {st.session_state.data.shape}\n"
            data_info += f"Columns: {', '.join(st.session_state.data.columns)}\n"
            data_info += f"Data sample:\n{st.session_state.data.head(5).to_string()}\n"
            data_info += f"Data description:\n{st.session_state.data.describe().to_string()}\n"
            
            chat_context = "You are a data analyst assistant. Answer the following question about the data.\n\n"
            
            if st.session_state.chat_history:
                chat_context += "Previous conversation:\n"
                for q, a in st.session_state.chat_history:
                    chat_context += f"User: {q}\nAssistant: {a}\n\n"
            
            prompt = chat_context + data_info + f"\nQuestion: {question}\n\nProvide a clear, concise answer. If the question requires generating code for visualization, provide Python code using Plotly or Matplotlib that would work in a Streamlit app."
            
            response = text_model.generate_content(prompt)
            answer = response.text
            
        elif st.session_state.file_type and st.session_state.file_type.startswith('image'):
            # For image files, use the multimodal capabilities of Gemini
            prompt = f"Analyze this image and answer the following question: {question}"
            image_parts = [{"mime_type": "image/jpeg", "data": st.session_state.image}]
            response = model.generate_content([prompt, image_parts[0]])
            answer = response.text
            
        elif st.session_state.file_content:
            # For text content from non-tabular files
            prompt = f"Based on the following content, please answer this question: {question}\n\nContent:\n{st.session_state.file_content[:5000]}"  # Limit content size
            response = text_model.generate_content(prompt)
            answer = response.text
            
        else:
            answer = "Please upload a file first."
    
    except Exception as e:
        answer = f"Error generating response: {str(e)}"
    
    return answer

st.title("Data Analyst Agent with Gemini")

# Display model information
if 'available_models' in st.session_state and st.session_state.available_models:
    st.info(f"Using text model: {text_model_name}\nUsing vision model: {vision_model_name}")

with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt", "pdf", "docx", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        result = process_file(uploaded_file)
        st.write(result)
    
    if st.session_state.data is not None:
        st.header("Data Overview")
        st.write(f"Rows: {st.session_state.data.shape[0]}, Columns: {st.session_state.data.shape[1]}")
        st.write("Columns:", ", ".join(st.session_state.data.columns))
        
        # Visualization Tools
        st.header("Create Visualization")
        plot_type = st.selectbox("Select Plot Type", ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Heatmap"])
        
        if plot_type != "Heatmap":
            x_column = st.selectbox("Select X-axis Column", st.session_state.data.columns)
            
            if plot_type not in ["Histogram"]:
                y_column = st.selectbox("Select Y-axis Column", st.session_state.data.columns)
            else:
                y_column = None
            
            color_column = st.selectbox("Select Color Column (optional)", ["None"] + list(st.session_state.data.columns))
            if color_column == "None":
                color_column = None
        else:
            x_column = None
            y_column = None
            color_column = None
            
        plot_title = st.text_input("Plot Title", "")
        
        if st.button("Generate Plot"):
            if plot_type == "Heatmap":
                fig = generate_plot(plot_type, st.session_state.data, x_column, y_column, color_column, plot_title)
            elif plot_type == "Histogram":
                fig = generate_plot(plot_type, st.session_state.data, x_column, None, color_column, plot_title)
            else:
                fig = generate_plot(plot_type, st.session_state.data, x_column, y_column, color_column, plot_title)
                
            if fig:
                st.plotly_chart(fig, use_container_width=True)

# Main content area
if st.session_state.data is not None:
    st.header("Data Preview")
    st.dataframe(st.session_state.data.head(10))
elif st.session_state.file_content and not st.session_state.file_type.startswith('image'):
    st.header("File Content Preview")
    st.text(st.session_state.file_content[:1000] + "..." if len(st.session_state.file_content) > 1000 else st.session_state.file_content)
elif st.session_state.file_type and st.session_state.file_type.startswith('image'):
    st.header("Uploaded Image")
    st.image(st.session_state.image, use_column_width=True)

# Chat interface
st.header("Ask Questions About Your Data")
user_question = st.text_input("Enter your question:")
if st.button("Ask") and user_question:
    with st.spinner("Generating answer..."):
        answer = answer_question(user_question)
        st.session_state.chat_history.append((user_question, answer))
        
# Display chat history
if st.session_state.chat_history:
    st.header("Conversation History")
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        st.markdown(f"**Question {i+1}:** {question}")
        st.markdown(f"**Answer:** {answer}")
        st.markdown("---")