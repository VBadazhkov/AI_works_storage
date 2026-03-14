!pip install llama-index llama-index-embeddings-huggingface
!pip install bitsandbytes
!unzip dataset.zip

import pandas as pd
books = pd.read_csv("/content/LR2.csv")['book']

book_mapping = {
    "Turgenev_Asya": "Ася",
    "Bulgakov_MasterIMargarita.txt": "Мастер и Маргарита",
    "Bulichev_Sto_let_tomu_vpered": "Сто лет тому вперёд",
    "Gogol_KakPossorilsyaIvanIvanovichSIvanomNikiforovichem": "Как поссорился Иван Иванович с Иваном Никифоровичем",
    "Gogol_TarasBulba": "Тарас Бульба",
    "Bulgakov_StalnoeGorlo": "Стальное горло",
    "Bulgakov_TmaEgipetskaya": "Тьма египетская",
    "Gorkiyi_DeloArtamonovyih": "Дело Артамоновых",
    "Bulichev_Gorod_bez_pamyati": "Город без памяти",
    "Gogol_StarosvetskiyePomeshchiki": "Старосветских помещиках",
    "Turgenev_Mumu": "Муму",
    "Bulichev_Lilovii_shar": "Лиловый шар",
    "Lermontov_KnyaginyaLigovskaya": "Княгиня Лиговская",
    "Bulgakov_PolotenceSPetuhom": "Полотенце с Петухом",
    "Tolstoy_VoynaIMir1": "Война и мир",
    "Tolstoy_VoynaIMir2": "Война и мир",
    "Turgenev_ZapiskiOhotnika": "Записки охотника",
    "Bulgakov_Morfiyi": "Морфий",
    "Bulgakov_RokovyeYayca": "Роковые яйца",
    "Bulgakov_ZvezdnayaSyp": "Звездная сыпь",
    "Gogol_MertvieDushi": "Мертвые души",
    "Bulgakov_KrescheniePovorotom": "Крещении поворотом",
    "Tolstoy_Decabristy": "Декабристы",
    "Bulgakov_PriklyucheniyaPokoynika": "Приключения покойника",
    "Gogol_Vii": "Вий"
}
len(book_mapping)

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        quantization_config=bnb_config,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

import zipfile
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

def chat(user_input):
    messages = [
        {"role": "system", "content": (
            "Ты — экспертная LLM, которая решает вопросы пошагово. "
            "Сначала коротко размышляй шаг за шагом (1–2 предложения на каждый вариант), "
            "потом выбери правильный вариант из списка (индексация начинается с единицы). "
            "Давай итоговый ответ строго в формате: 'Итоговый ответ: <индекс>'"
        )},
        {"role": "user", "content": user_input}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=3000, temperature=0.3, top_p=0.95, eos_token_id=tokenizer.eos_token_id)
    response_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    messages.append({"role": "assistant", "content": response})
    return response

# Чанкинг около 25 минут на bge
def load_books(dataset_dir: str):
    reader = SimpleDirectoryReader(
        input_dir=dataset_dir,
        recursive=True,
        required_exts=[".txt", ".TXT"]
    )

    documents = reader.load_data()

    filtered_docs = []
    for doc in documents:
        filename = Path(doc.metadata["file_path"]).stem
        if filename in book_mapping.keys():
            doc.metadata["book"] = book_mapping[filename]
            doc.metadata["author"] = Path(doc.metadata["file_path"]).parent.name
            filtered_docs.append(doc)

    splitter = SentenceSplitter(chunk_size=600, chunk_overlap=150, paragraph_separator="\n\n",)
    nodes = splitter.get_nodes_from_documents(filtered_docs)
    splitted_documents = [Document(text=node.text, metadata=node.metadata) for node in nodes]
    return splitted_documents

def extract_answer(text):
    pattern = r'Итоговый ответ:\s*([0-9,\s]+)'
    match = re.findall(pattern, text)
    if not match:
        clean = '0'
        return clean
    else:
        match = match[-1]
    clean = ','.join(re.findall(r'\d+', match))
    return clean

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pandas as pd
import re


def process_questions():
    print("Пошла аналитика")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        device="cuda"
    )

    print("Загружаем книжный корпус...")
    documents = load_books('/content/dataset')

    print("Создаем индекс...")
    index = VectorStoreIndex.from_documents(documents)

    print("Создаем ретриверы...")
    retriever = index.as_retriever(similarity_top_k=5)

    # Здесь напишем обработку вопросов датасета
    questions_df = pd.read_csv("/content/LR2.csv")
    answers = []

    print("Аналитика. Пожалуйста подождите")
    for i in range(len(questions_df)):
        questions = questions_df.iloc[i]
        formatted_question = f"""Вопрос: {questions['question']}\n
        Опция a: {questions['answer a']}\n
        Опция b: {questions['answer b']}\n
        Опция c: {questions['answer c']}\n
        Опция d: {questions['answer d']}\n
        Книга: {questions['book']}\n
        """
        target_book = questions['book']
        filters = MetadataFilters(filters=[
            ExactMatchFilter(key="book", value=target_book)
        ])

        filtered_retriever = index.as_retriever(similarity_top_k=5, filters=filters)
        nodes = filtered_retriever.retrieve(questions['question'])

        if not nodes:
            print("Поиск по всему корпусу")
            nodes = retriever.retrieve(questions['question'])

        retrieved_texts = "\n".join([node.text for node in nodes])
        prompt = f"""Вопрос и опции: {formatted_question}
        Представленная информация: {retrieved_texts}
        Основываясь на представленной информации, дай ответ на вопрос.
        В конце ответа обязательно напиши только одну строку в формате:
        Итоговый ответ: <индекс_правильного_ответа>
        """
        answer = chat(prompt)

        if "Итоговый ответ:" in answer:
            result = extract_answer(answer)
            answers.append(result)
            print(f"Ответ - {result}")
        else:
            result = '1'
            answers.append(result)
            print("Ответ записан вручную - 1")

        print(f"Обработано {i + 1} вопросов")

    return answers

res = process_questions()
res_copy = res

def save_csv(results):
    df = pd.DataFrame({
        'answer': results
    })
    final = df.to_csv("model_answers.csv", sep=',')

results = ["1" if i == "" else i for i in res]
save_csv(results)
print("Сохранено")