import streamlit as st
from googletrans import Translator
from transformers import BartForConditionalGeneration, BartTokenizer

# Initialize the translator
translator = Translator()

# Title and Introduction
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Marathi Summarizer <small>using Bart model</small></h1>", unsafe_allow_html=True)
st.markdown("---")

# Input text from user
st.markdown("<h3 style='color: #FF5722;'>Step 1: Enter Marathi Text</h3>", unsafe_allow_html=True)
marathi_text = st.text_area("", placeholder="Enter Marathi text here...", height=500)

# Add some space
st.markdown("<br>", unsafe_allow_html=True)

# Initialize the model and tokenizer outside the button click to avoid reloading every time
model_name = "facebook/bart-large-cnn"

@st.cache_data(show_spinner=False)
def load_model():
    try:
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

tokenizer, model = load_model()

# Check if model and tokenizer are loaded successfully
if tokenizer is None or model is None:
    st.stop()

# Translate Marathi text to English
if st.button("Translate to English"):
    if marathi_text:
        try:
            with st.spinner("Translating to English..."):
                translated_text = translator.translate(marathi_text, dest='en').text
                st.session_state.translated_text = translated_text
            st.markdown("<h3 style='color: #FF5722;'>Step 2: Translated Text (English)</h3>", unsafe_allow_html=True)
            st.write(translated_text)
        except Exception as e:
            st.error(f"Translation error: {e}")

# Summarize the translated text
if "translated_text" in st.session_state:
    translated_text = st.session_state.translated_text

    if st.button("Summarize Translated Text"):
        try:
            with st.spinner("Summarizing..."):
                inputs = tokenizer(translated_text, max_length=1024, return_tensors="pt", truncation=True)
                summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=600, min_length=200, length_penalty=2.0, early_stopping=True)
                summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                st.session_state.summary = summary_text
            st.markdown("<h3 style='color: #FF5722;'>Step 3: Summary</h3>", unsafe_allow_html=True)
            st.write(summary_text)
        except Exception as e:
            st.error(f"Summarization error: {e}")

# Translate the summary back to Marathi
if "summary" in st.session_state:
    summary_text = st.session_state.summary

    if st.button("Translate Summary to Marathi"):
        try:
            with st.spinner("Translating summary to Marathi..."):
                summarized_translated_text = translator.translate(summary_text, dest='mr').text
            st.markdown("<h3 style='color: #FF5722;'>Step 4: Summarized Text (Marathi)</h3>", unsafe_allow_html=True)
            st.write(summarized_translated_text)
        except Exception as e:
            st.error(f"Translation error: {e}")

# Add some custom CSS for styling
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        font-size: 16px;
    }
    .stTextArea textarea {
        font-size: 16px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
