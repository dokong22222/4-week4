# In[1]:
import os
import import_ipynb
import chromadb
from openai import OpenAI
from hw_data import get_embedding
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

dbclient = chromadb.PersistentClient(path="./chroma_db")
collection = dbclient.get_or_create_collection("rag_collection")


# In[2]:
# 문서를 검색한다
def retrieve(query, top_k=3):
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results


# In[3]:
# 답변 생성
def generate_answer_with_context(query, top_k=3):
    results = retrieve(query, top_k)
    found_docs = results["documents"][0]
    found_metadatas = results["metadatas"][0]

    context_texts = []
    for doc_text, meta in zip(found_docs, found_metadatas):
        context_texts.append(f"<<filename: {meta['filename']}>>\n{doc_text}")
    context_str = "\n\n".join(context_texts)

    system_prompt = """
    당신은 세계 국가들의 정보를 안내하는 FAQ 어시스턴트입니다. 다음 원칙을 엄격히 지키세요:

    1. 반드시 제공된 문서 내용에 근거해서만 답변하세요.
    2. 문서에 없는 내용은 추측하지 말고, "관련 정보를 찾지 못했습니다"라고 답변하세요.
    3. 간결하고 명확하게 답변하세요.
    4. 사용자가 한국어로 질문하면 한국어로, 영어로 질문하면 영어로 답변하세요.
    5. 국가명, 수도, 인구, 면적 등 수치 정보는 정확하게 전달하세요.
    """

    user_prompt = f"""아래는 검색된 국가 정보 문서들입니다:
    {context_str}

    질문: {query}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


# In[4]:
# 디버깅: DB 검색 확인
query = "미국의 수도는 어디야?"
query_embedding = get_embedding(query)
results = collection.query(query_embeddings=[query_embedding], n_results=3)

if results['documents'][0]:
    print("✅ DB에서 데이터를 찾았습니다!")
    for i, doc in enumerate(results['documents'][0]):
        print(f"[{i+1}] {doc[:100]}...")
else:
    print("❌ DB가 비어있거나 검색 결과가 없습니다. 인덱싱을 다시 하세요.")


# In[5]:
from tkinter import *
import tkinter.ttk as ttk

def reset_status():
    label_status.config(text="", foreground="black")

def process_query():
    query = text_input.get("1.0", END).strip()
    print(f"User Query: {query}")
    if query:
        label_status.config(text="질문 처리중...", foreground="blue")
        answer = generate_answer_with_context(query)
        print(f"Answer: {answer}")
        label_status.config(text="처리 완료", foreground="green")
        root.after(2000, reset_status)
        text_input.delete("1.0", "end-1c")
        text_output.config(state="normal")
        text_output.delete("1.0", END)
        text_output.insert(END, answer)
        text_output.config(state="disabled")
    else:
        label_status.config(text="질문을 입력해주세요.", foreground="red")

root = Tk()
root.title('국가 정보 FAQ 챗봇')
root.geometry('500x700')
root.resizable(False, False)
root.configure(bg="lavender")

frame_input = Frame(root, padx=10, pady=10)
frame_input.pack(fill="x")

label_input = ttk.Label(frame_input, text="질문 입력", font=("맑은 고딕", 12, "bold"))
label_input.pack(anchor="w")

text_input = Text(frame_input, height=6, font=("맑은 고딕", 11))
text_input.pack(pady=5)

btn = ttk.Button(frame_input, text="전송", command=process_query)
btn.pack(pady=5)

label_status = ttk.Label(frame_input, text="", font=("맑은 고딕", 10))
label_status.pack(anchor="w", pady=5)

separator = ttk.Separator(root, orient="horizontal")
separator.pack(fill="x", padx=10, pady=10)

frame_output = Frame(root, padx=10, pady=10)
frame_output.pack(fill="both", expand=True)

label_output = ttk.Label(frame_output, text="답변", font=("맑은 고딕", 12, "bold"))
label_output.pack(anchor="w")

text_output = Text(frame_output, wrap="word", font=("맑은 고딕", 11), state="disabled", height=20, bg="#f9f9f9")
text_output.pack(side="left", fill="both", expand=True)

scrollbar = ttk.Scrollbar(frame_output, command=text_output.yview)
scrollbar.pack(side="right", fill="y")
text_output.config(yscrollcommand=scrollbar.set)

root.mainloop()