# -*- coding: utf-8 -*-

!pip install bitsandbytes

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-14B",
        quantization_config=bnb_config,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

def chat(user_input):
    messages = [
        {"role": "system", "content": (
            "Ты — экспертная LLM, которая решает вопросы пошагово. "
            "Сначала коротко размышляй шаг за шагом (1–2 предложения на каждый вариант), "
            "потом выбери правильный вариант из списка (индексация начинается с нуля). "
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

    outputs = model.generate(**inputs, max_new_tokens=1300, temperature=0.3, top_p=0.95, eos_token_id=tokenizer.eos_token_id)
    response_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    messages.append({"role": "assistant", "content": response})
    return response


import pandas as pd

def reflexion_solve(answers, questions, last_check=False):
    if last_check == False:
        prompt = f"""
        Тебе дан вопрос с вариантами ответов и твой выбранный ответ:

        Вопрос и опции:
        {answers}

        Первичный ответ модели: {questions}

        Проверь шаг за шагом:
        1) Коротко объясни, почему выбранный ответ может быть правильным или неправильным (два-три предложения на каждый шаг).
        2) Если считаешь его неверным, выбери другой правильный индекс.
        3) Итоговый ответ давай строго в формате:
           <Итоговый ответ: <индекс>>

        Индексация начинается с нуля!
        В конце ответа обязательно напиши только одну строку в формате:
        Итоговый ответ: <индекс_правильного_ответа>
        """
        solution = chat(prompt)

    else:
        prompt = f"""
        Тебе дан вопрос с вариантами ответов и твой выбранный ответ:

        Вопрос и опции:
        {answers}

        Первичный ответ модели: {questions}

        В ответе выведи только одну строчку - Итоговый ответ: <индекс_правильного_ответа>.
        Индексация начинается с нуля!
        Писать в ответе что либо кроме этой строчки нельзя
        """
        solution = chat(prompt)

    return solution

import re

def extract_answer(text):
    # Ищем "Итоговый ответ:" и за ним — последовательность из цифр, запятых и пробелов
    pattern = r'Итоговый ответ:\s*([0-9,\s]+)'
    match = re.findall(pattern, text)
    if not match:
        clean = '0'
        return clean
    else:
        match = match[-1]
    clean = ','.join(re.findall(r'\d+', match))
    return clean


def process_questions_separately():
    questions_df = pd.read_csv("/content/LR1.csv")
    batch_size = 1
    results = []

    for i in range(62, len(questions_df), batch_size):
        questions = questions_df.iloc[i:i+batch_size][['question', 'options']]
        formatted_questions = ""
        for idx, row in questions.iterrows():
            formatted_questions += f"Вопрос: {row['question']}\n"
            formatted_questions += f"Опции: {row['options']}\n\n"

        prompt = f"""
        Тебе дан вопрос с вариантами ответов. Твоя задача — выбрать правильный вариант.

        1) Сначала коротко размышляй, почему каждый вариант может быть правильным или неправильным (два-три предложения на каждый шаг).
        2) Не добавляй лишние детали, сосредоточься только на логике решения.
        3) После рассуждения выбери правильный вариант из списка.
        4) Итоговый ответ дай строго в формате (индексация начинается с нуля!):
        Итоговый ответ: <индекс>

        Пример: Итоговый ответ: 0

        Вопрос и опции:
        {formatted_questions}

        В конце ответа обязательно напиши только одну строку в формате:
        Итоговый ответ: <индекс_правильного_ответа>
        """

        result = chat(prompt)
        result = reflexion_solve(result, formatted_questions)

        if "Итоговый ответ:" in result:
            result = extract_answer(result)
            results.append(result)
            print(f"Ответ - {result}")
        else:
            print("Ошибка формата, пробуем еще раз с самокоррекцией...")
            result = reflexion_solve(result, formatted_questions, last_check=True)
            if "Итоговый ответ: " in result:
                result = extract_answer(result)
                results.append(result)
                print("Ответ поправлен")
                print(f"Ответ - {result}")
            else:
                result = '0'
                results.append(result)
                print("Ответ записан вручную - 0")

        print(f"Обработано {i + batch_size} вопросов")

    return results

results = process_questions_separately()
results_copy = results

for i, result in enumerate(results):
    print(f"Result {i}: {result}...")


import pandas as pd
def display_sample_from_dataframe():
    questions_df = pd.read_csv("/content/LR1.csv")
    print(questions_df.iloc[55:65][['question', 'options']])

display_sample_from_dataframe()


def save_csv(results):
    df = pd.DataFrame({
        'answer': results
    })
    final = df.to_csv("model_answers.csv", sep=',', index=False)

results = ["0" if i == "" else i for i in results]
save_csv(results)
print("Сохранено")