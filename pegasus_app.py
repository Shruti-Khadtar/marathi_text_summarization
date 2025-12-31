import streamlit as st
from googletrans import Translator
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Initialize the translator
translator = Translator()

# Title and Introduction
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Marathi Summarizer <small>using Pegasus model</small></h1>", unsafe_allow_html=True)
st.markdown("---")

@st.cache(allow_output_mutation=True)
def load_model():
    model_name = "google/pegasus-xsum"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

# Load the model
tokenizer, model = load_model()

# Input text from user
st.markdown("<h3 style='color: #FF5722;'>Enter Marathi Text</h3>", unsafe_allow_html=True)
marathi_text = st.text_area("", placeholder="Enter Marathi text here...", height=500)

# Add some space
st.markdown("<br>", unsafe_allow_html=True)

# Translate Marathi text to English
if st.button("Translate to English"):
    if marathi_text:
        translated_text = translator.translate(marathi_text, dest='en').text
        st.session_state.translated_text = translated_text  # Store in session state
        st.markdown("<h3 style='color: #FF5722;'>Translated Text (English)</h3>", unsafe_allow_html=True)
        st.write(translated_text)

# Summarize the translated text
if "translated_text" in st.session_state:
    translated_text = st.session_state.translated_text

    if st.button("Summarize Translated Text"):
        # Pegasus model utilizes a subword tokenization technique called Byte Pair Encoding (BPE).
        tokens = tokenizer(translated_text, truncation=True, padding="longest", return_tensors="pt")
        
        # Adjusting parameters for longer summary
        summary_tokens = model.generate(
            **tokens,
            max_length=550,  # Increased max_length for longer summaries
            min_length=250,  # Increased min_length for more content
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        summary_text = tokenizer.decode(summary_tokens[0], skip_special_tokens=True)
        st.session_state.summary = summary_text  # Store in session state
        st.markdown("<h3 style='color: #FF5722;'>Summary</h3>", unsafe_allow_html=True)
        st.write(summary_text)

# Translate the summary back to Marathi
if "summary" in st.session_state:
    summary_text = st.session_state.summary

    if st.button("Translate Summary to Marathi"):
        summarized_translated_text = translator.translate(summary_text, dest='mr').text
        st.markdown("<h3 style='color: #FF5722;'>Summarized Text (Marathi)</h3>", unsafe_allow_html=True)
        st.write(summarized_translated_text)

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
