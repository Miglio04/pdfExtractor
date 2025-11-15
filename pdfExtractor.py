import pypdf
from mistralai import Mistral
import re
import os

API_KEY = "DgFhIOq2TFGDDbUxXBMpSaEeMx7Ktps1"

PROMPT = """Interprete the following text extracted from a pdf invoice and return a json objects with the following structure containing all the informations inside the text i provided you: 
            {{   
                "invoice number": "",
                "date": "",
                "items": [
                    {{
                        "item": "",
                        "unit price": "",
                        "quantity": "",
                        "total": ""
                    }}
                ],
                "subtotal": "",
                "tax (in percentage)": "",
                "amount due": ""
            }}. 
            The content field should contain the informations required in the json with maximum accuracy. Don't include any other text outside of the json object exactly as I described it in your response.
            The text is the following: {}
        """
        
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

def process_pdf_text(pdf_stream):
    with Mistral(
        api_key=os.getenv("MISTRAL_API_KEY", API_KEY),
    ) as mistral:
        res = mistral.chat.complete(
            model="mistral-large-latest", 
            messages=[{
                "content": PROMPT.format(pdf_stream),
                "role": "user",
            }], 
            stream=False)
    content = res.choices[0].message.content
    content = re.search(r'\{.*\}', content, flags=re.DOTALL).group(0)
    return content    
    
def pdf_extractor(file_path: str) -> str:
    text = extract_text_from_pdf(file_path)
    result = process_pdf_text(text)
    return result

if __name__ == "__main__":
    file_path = "fattura1.pdf"
    extracted_data = pdf_extractor(file_path)
    print(extracted_data)