import pypdf
import io
import re
import os
from mistralai import Mistral
from appwrite.client import Client
from appwrite.services.storage import Storage

# Load API key and Appwrite settings from environment variable
# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", default="DgFhIOq2TFGDDbUxXBMpSaEeMx7Ktps1")
# BUCKET_ID = os.getenv("BUCKET_ID", default="6919ad28000c5514222b")
# PROJECT_ID = os.getenv("PROJECT_ID", default="690b89fa001c2de87a3a")
# APPWRITE_ENDPOINT = os.getenv("APPWRITE_ENDPOINT", default="https://cloud.appwrite.io/v1")
# APPWRITE_API_KEY = os.getenv("API_KEY", default="standard_ee0427530eaee30e2db6920963710f4815a622c9118655b2e6ff37981a18d964d63d850a9e34d98a17178c99fe77d8a3227503408e343e9809ff7d4e8d9e78b1dc128190a6932a8f8937dfac5c39c00e12fb942cb7dca3a3a68c237a5dacc935a81fb902a12723965572f78b71cf29b1564f73e0b4e5f3977b08c4a820723f66")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
BUCKET_ID = os.getenv("BUCKET_ID")
PROJECT_ID = os.getenv("PROJECT_ID")
APPWRITE_ENDPOINT = os.getenv("APPWRITE_ENDPOINT")
APPWRITE_API_KEY = os.getenv("API_KEY")

client = Client()
if APPWRITE_ENDPOINT:
    client.set_endpoint(APPWRITE_ENDPOINT)
if APPWRITE_API_KEY:
    client.set_key(APPWRITE_API_KEY)
if PROJECT_ID:
    client.set_project(PROJECT_ID)
storage = Storage(client)

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
        
# Note: we no longer have a helper to download files. The code in `main`
# calls `storage.get_file_download(file_id)` directly and passes the result
# to `extract_text_from_pdf` (keeps main very short and simple).


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF provided as raw bytes.

    This function expects the bytes returned by `download_file_bytes` (i.e. a
    file downloaded from Appwrite Storage). It will raise a `TypeError` if the
    input is not bytes.
    """
    if not isinstance(file_bytes, (bytes, bytearray)):
        raise TypeError("extract_text_from_pdf expects raw bytes from Appwrite storage")

    stream = io.BytesIO(file_bytes)
    reader = pypdf.PdfReader(stream)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + '\n'
    return text

# Function to process PDF text with Mistral API
def process_pdf_text(pdf_stream: str) -> str:
    with Mistral(
        api_key=MISTRAL_API_KEY,
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
    # Static file id on Appwrite Storage (set it here)
    file_id = "6919ad42000f5377be92"
    pdf_bytes = storage.get_file_download(file_id)
    text = extract_text_from_pdf(pdf_bytes)
    result = process_pdf_text(text)
    print(result)
    return context.res.empty()

# if __name__ == "__main__":
#     file_id = "6919ad42000f5377be92"
#     print(BUCKET_ID, file_id)
#     pdf_bytes = storage.get_file_download(BUCKET_ID, file_id)
#     text = extract_text_from_pdf(pdf_bytes)
#     result = process_pdf_text(text)
#     print(result)