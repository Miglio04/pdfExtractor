import pypdf
from mistralai import Mistral
import re
import os

# Load API key from environment variable
API_KEY = os.getenv("API_KEY")

# Temporary file path for the PDF
BASE = os.path.dirname(__file__)
file_path = os.path.join(BASE, "fattura1.pdf")

# Define the prompt template
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
        
# Function to extract text from PDF
def extract_text_from_pdf(file_path: str) -> str:
    with open(file_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

# Function to process PDF text with Mistral API
def process_pdf_text(pdf_stream: str) -> str:
    with Mistral(
        api_key=os.getenv("MISTRAL_API_KEY", API_KEY),
    ) as mistral:
        res = mistral.chat.complete(
            model="mistral-small-latest", 
            messages=[{
                "content": PROMPT.format(pdf_stream),
                "role": "user",
            }], 
            stream=False)
    content = res.choices[0].message.content
    content = re.search(r'\{.*\}', content, flags=re.DOTALL).group(0)
    return content    
    
# Main function to be called
def main(context):
    text = extract_text_from_pdf(file_path)
    result = process_pdf_text(text)
    context.log(result)
    return context.res.empty()