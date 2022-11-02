from transformers import pipeline
import io
import streamlit as st
from PIL import Image
classifier = pipeline("sentiment-analysis", "blanchefort/rubert-base-cased-sentiment")



st.title('Тональность текста')
inp_text = st.text('Введите текст')

result = st.button('Определить тональность')

if result:
    c = classifier(inp_text)
    st.write(str(c))
	#st.write(str(c))
    
    