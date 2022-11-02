from transformers import pipeline
import io
import streamlit as st
from transformers import MarianTokenizer, TFMarianMTModel

def translation(str_cl):
    batch = tokenizer([str_cl], return_tensors="tf")
    gen = model.generate(**batch)
    tr=tokenizer.batch_decode(gen, skip_special_tokens=True)
    st.write('Рус:', str(tr))


classifier = pipeline("sentiment-analysis", "blanchefort/rubert-base-cased-sentiment")

st.title('Тональность текста')
inp_text = st.text_input('Англ:', 'Обожаю питон')
st.write('',inp_text)

src = "en"  # source language
trg = "ru"  # target language
sample_text = "hello"
model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
model = TFMarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)
translation(sample_text)

result = st.button('Определить тональность')

if result:
    st.write('Англ: ',inp_text)
    translation(inp_text)
    cl = classifier(str(inp_text))
    for i in cl:
        st.write(str(i["label"]),' с вероятностью ',str(100*float(i["score"])),'%')

    
    