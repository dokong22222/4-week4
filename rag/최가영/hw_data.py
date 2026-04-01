# In[1]:


#get_ipython().system('pip install bs4')
#get_ipython().system('pip install requests')
#get_ipython().system('pip install pandas')

import subprocess
subprocess.run(["pip", "install", "bs4", "requests", "pandas"])

# In[2]:


from bs4 import BeautifulSoup
import requests
import os

url = f"https://www.scrapethissite.com/pages/simple/"
response=requests.get(url)
html=response.text
soup=BeautifulSoup(html, 'html.parser')
items=soup.select(".col-md-4.country")


data = []

for c in items:
    name = c.select_one(".country-name").get_text(strip=True)

    #수도
    capital=c.select_one(".country-capital").get_text(strip=True)

    #인구수
    population=c.select_one(".country-population").get_text(strip=True)

    #면적
    area=c.select_one(".country-area").get_text(strip=True)

    data.append({"name": name, "capital": capital, "population": population, "area": area})

print(data[:5])
print(len(data))

#.txt파일로 저장함
os.makedirs("./source_data", exist_ok=True)

# In[3]:
documents = []

for d in data:
    text = f"{d['name']}의 수도는 {d['capital']}이고 인구는 {d['population']}명이며 면적은 {d['area']}이다"
    filename = f"./source_data/{d['name'].replace(' ', '_')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

print("data .txt로 저장 완료")

#In[4]:

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# DB 초기화 함수
def init_db(db_path="./chroma_db"):
    dbclient = chromadb.PersistentClient(path=db_path)
    try:
        dbclient.delete_collection(name="rag_collection")
    except:
        pass
    collection = dbclient.create_collection(name="rag_collection")
    return dbclient, collection

# 텍스트 파일 로딩 함수
def load_text_files(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                docs.append((filename, text))
    return docs

# 임베딩 함수
def get_embedding(text, model="text-embedding-3-large"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


# In[5]:
#문서 로드 -> 임베딩 생성 -> DB 삽입.. 청킹은 필요없음! 너무 짧아서..
if __name__ == "__main__":
    dbclient, collection = init_db("./chroma_db")

    folder_path = "./source_data"
    docs = load_text_files(folder_path)

    for doc_id, (filename, text) in enumerate(docs):
        embedding = get_embedding(text)
        collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[{"filename": filename}],
            ids=[str(doc_id)]
        )

    print(f"{len(docs)}개 문서 저장 완료")

