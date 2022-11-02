from transformers import pipeline
import io
import streamlit as st
from PIL import Image
classifier = pipeline("sentiment-analysis", "blanchefort/rubert-base-cased-sentiment")



st.title('Тональность текста')
inp_text = st.text_input('Введите текст', 'Обожаю питон')
st.write('',inp_text)

result = st.button('Определить тональность')

if result:
    c = classifier(str(inp_text))
    st.write(str(c))
	#st.write(str(c))
    
    