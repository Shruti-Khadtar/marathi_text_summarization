import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, T5ForConditionalGeneration, T5Tokenizer
import torch

# Device config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Language codes for mBART-50
LANG_CODES = {
    'English': 'en_XX',
    'Marathi': 'mr_IN'
}

@st.cache_resource
def load_translation_model():
    model_name = 'facebook/mbart-large-50-many-to-many-mmt'
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
    return tokenizer, model

@st.cache_resource
def load_summarizer():
    model_name = 't5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
    return tokenizer, model

# Translation function
def mbart_translate(text, src_lang, tgt_lang, tokenizer, model):
    tokenizer.src_lang = LANG_CODES[src_lang]
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[LANG_CODES[tgt_lang]],
        max_length=512
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Summarization function
def t5_summarize(text, tokenizer, model, min_ratio=0.5, max_ratio=0.7):
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    input_length = input_ids.shape[1]
    min_length = max(10, int(input_length * min_ratio))
    max_length = min(512, int(input_length * max_ratio))
    summary_ids = model.generate(
        input_ids,
        min_length=min_length,
        max_length=max_length,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# --- Streamlit UI ---
st.title("Marathi ↔ English Translation & Summarization (Offline)")
st.markdown("---")

# Load models
trans_tokenizer, trans_model = load_translation_model()
sum_tokenizer, sum_model = load_summarizer()

# User input
st.markdown("**Enter Marathi or English text:**")
input_text = st.text_area("Text", height=200)

src_lang = st.selectbox("Input language", ["Marathi", "English"])
tgt_lang = "English" if src_lang == "Marathi" else "Marathi"

if st.button(f"Translate to {tgt_lang}"):
    if input_text.strip():
        with st.spinner("Translating..."):
            translated = mbart_translate(input_text, src_lang, tgt_lang, trans_tokenizer, trans_model)
        st.markdown(f"**Translated Text ({tgt_lang}):**")
        st.write(translated)
        st.session_state.translated = translated
    else:
        st.warning("Please enter some text.")

# Summarization (only for English text)
if "translated" in st.session_state and tgt_lang == "English":
    if st.button("Summarize (English)"):
        with st.spinner("Summarizing..."):
            summary = t5_summarize(st.session_state.translated, sum_tokenizer, sum_model)
        st.markdown("**Summary (English):**")
        st.write(summary)
        st.session_state.summary = summary

# Translate summary back to Marathi
if "summary" in st.session_state and tgt_lang == "English":
    if st.button("Translate Summary to Marathi"):
        with st.spinner("Translating summary..."):
            summary_mr = mbart_translate(st.session_state.summary, "English", "Marathi", trans_tokenizer, trans_model)
        st.markdown("**Summarized Text (Marathi):**")
        st.write(summary_mr)
