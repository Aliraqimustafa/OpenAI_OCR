import streamlit as st
import requests
import json
import pandas as pd
from my_openai import OpenAI  
def process_image(image_file, col_names):
    api_url = "https://api.ocr.space/parse/image"
    api_key = "api_key"
    
    payload = {
        "apikey": api_key,
        "language": "eng",
        "isOverlayRequired": True,
    }
    with open(image_file.name, "rb") as img_file:
        response = requests.post(api_url, data=payload, files={"file": img_file})
    
    json_response = response.json()
    parsed_text = json_response["ParsedResults"][0]["ParsedText"]
    
    prompt = f'''
    You are an artificial intelligence assistant.
    It is integrated with the OCR API so that data is extracted from images,
    knowing that the data will not be arranged (the data type is excel tables), so you must filter the data

    These are the column names: {col_names}

    data : """{parsed_text}"""
    Your answer must be like {{data : {{col1 :... , col2:...,...}}}}
    
    FINAL_JSON:
    '''

    llm = OpenAI() # Text Generation 
    result = llm(prompt)

    data_ = json.loads(result)

    df = pd.DataFrame(data_['data'])

    return df

def main():
    st.title("OCR and OpenAI with Streamlit")

    st.subheader("Upload Image:")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    st.subheader("Input Column Names:")
    col_names = st.text_area("Enter column names separated by commas (e.g., Product ID, Product Name, Price, Status)")

    if st.button("Run OCR and OpenAI"):
        if uploaded_file and col_names:
            col_names_list = [col.strip() for col in col_names.split(',')]
            result_df = process_image(uploaded_file, col_names_list)

            st.subheader("Result DataFrame:")
            st.write(result_df)
        else:
            st.warning("Please upload an image and provide column names.")

if __name__ == "__main__":
    main()

