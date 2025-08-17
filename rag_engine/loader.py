import os
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

def search_onet_by_keyword(keyword: str):
    username=os.getenv("ONET_USERNAME")
    password=os.getenv("ONET_PASSWORD")
    url = f"https://services.onetcenter.org/ws/mnm/v1.0/keyword/{keyword}"

    res = requests.get(url, auth=HTTPBasicAuth(username, password))
    if res.status_code ==200:
        return res.text
    else:
        raise Exception(f"Error fetching data from O*NET: {res.status_code} - {res.text}")
    
def fetch_onet_career(occupation: str):
    username=os.getenv("ONET_USERNAME")
    password=os.getenv("ONET_PASSWORD")
    base_url="https://services.onetcenter.org/ws/online"

    docs=[]
    for endpoint in ["occupations","skills","projection"]:
        url=f"{base_url}/{endpoint}/{occupation_code}"
        res=requests.get(url, auth=HTTPBasicAuth(username,password))
        if res.status_code==200:
            docs.append(Document(
                page_content=res.text,
                metadata={"source":"onet","type":endpoint}
            ))
    
    return docs

def split_documents(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)