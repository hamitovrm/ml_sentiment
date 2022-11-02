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
    cl = classifier(str(inp_text))
    for i in cl:
        st.write((str(i["label"]),' с вероятностью '))
        #st.write((str(i["label"]),' с вероятностью ',str(100*float(i["score"])),'%')

    
    