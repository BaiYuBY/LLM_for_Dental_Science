import os
os.chdir(r"/root/autodl-tmp")

import ast
import fire
import gradio as gr
from typing import Optional
import logging
import neo4j
import jsonlines
import csv
import pandas as pd
from py2neo import Graph, Node, Relationship
import re
import logging
from tqdm import tqdm
import networkx as nx
import glob
import pandas as pd
import matplotlib.pyplot as plt
import faiss
from transformers import ( 
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
import bitsandbytes
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict


# 配置日志记录
logging.basicConfig(level=logging.DEBUG)

def embedder_model_load():
    embedder_model = SentenceTransformer("./models/pubmedbert-base-embeddings")
    return embedder_model

def get_entity_embeddings(entities, embedder_model):
    embeddings = []
    for entity in tqdm(entities, desc="Processing entity embeddings"):
        output_embedding = embedder_model.encode(entity)
        embeddings.append(output_embedding)
    return np.vstack(embeddings)

def L2_distance_search(data, embedder_model, k=5, threshold=50):
    entity_embeddings = get_entity_embeddings(data, embedder_model)

    similar_entities = []
    for idx, entity_embedding in enumerate(tqdm(entity_embeddings, desc="Processing similarity search")):
        query_vector = np.array([entity_embedding], dtype=np.float32)
        distances, result_ids = index.search(query_vector, k=k)

        for i, dist in enumerate(distances[0]):
            if dist <= threshold:

                entity_name = data[idx]
                similar_entities.append((result_ids[0][i], entity_name, dist))
    return similar_entities

def normalize_embedder(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # 避免除以零
    return embeddings / norms

def cosine_distance_search(data, index, embedder_model, k=5, threshold=0.6):
    entity_embeddings = get_entity_embeddings(data, embedder_model)

    similar_entities = []
    for idx, entity_embedding in enumerate(tqdm(entity_embeddings, desc="Processing similarity search")):
        query_vector = np.array(normalize_embedder([entity_embedding]), dtype=np.float32)
        distances, result_ids = index.search(query_vector, k=k)
        
        for i, dist in enumerate(distances[0]):
            similarity_score = dist
            
            if similarity_score >= threshold:
                entity_name = data[idx]
                similar_entities.append((result_ids[0][i], entity_name, similarity_score))
                
    return similar_entities

def read_id_table():
    df = pd.read_csv('table.csv')
    return df

def get_type(graph, name):
    query = """
    MATCH (n)
    WHERE n.name IN """ + str(name) + """
    RETURN n.name AS name, labels(n) AS types"""
    result = graph.run(query)
    print(result)
    return result

def get_relation_list():
    df = pd.read_csv('./kg1.csv')

    type_relation_dict = defaultdict(set)

    for _, row in df.iterrows():
        subject_type = row['subject_type']
        relation = row['relation']
        type_relation_dict[subject_type].add(relation)

    for _, row in df.iterrows():
        object_type = row['object_type']
        relation = row['relation']
        type_relation_dict[object_type].add(relation)

    result_dict = {key: list(value) for key, value in type_relation_dict.items()}

    return result_dict

def build_query(entities, relationships):
    entity_conditions = []
    for entity in entities:
        name = entity['name']
        types = entity['types']
        type_conditions = ''

        type_conditions = " OR ".join([f"(n:`{t}` AND ({''' OR '''.join([f'r:`{r}`' for r in relationships[t]])}))" for t in types])
        entity_conditions.append(f"(n.name = '{name}' AND ({type_conditions}))")

    entity_query = " OR ".join(entity_conditions)

    # relationship_types = []
    # for typ in relationships:
    #     relationship_types.extend(relationships[typ])
    # relationship_query = " OR ".join([f"r:`{rel}`" for rel in relationship_types])
    
    query = f"""
    MATCH (n)-[r]-(m)
    WHERE ({entity_query})
    RETURN n.name AS entity_name, labels(n) AS entity_type, m.name AS related_entity_name, labels(m) AS related_entity_type, type(r) AS relationship_type
    """
    return query

def generate_prompt(data):
    SYSTEM_PROMPT = f"""You are given a list of entities, their types and their relationships."""
    prompt = ''
    for entity in data:
        prompt += f"The entity ({entity['entity_name']}, type: {entity['entity_type'][0]}) and its related entity ({entity['related_entity_name']}, type: {entity['related_entity_type'][0]}) are related by the relationship type: {entity['relationship_type']}.\n"
    return SYSTEM_PROMPT + prompt.strip('\n')

def get_important_entities(question, tokenizer, test_model):
    test_text ="""### Instruction: 
    Identify the important entities in the question that you need further extral knowledge to answer.

    ### System Prompt: 
    The following is an unstructured question, please extract the important entities in the text, and the output should follow the following format: [entity1, entity2, entity3]. No more note or explain is needed, only output a list.
    This extracted entities are used for searching further help and knowledge in an extal database. Thus, only find the entities that you are not sure or not familiar with. If you cannot find any entities that satisfy the above requirements, please output: []. No more note or explan is needed.

    ### Input:
    """ + question + """
    ### Response:
    """


    device = "cuda:0"
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    outputs = test_model.generate(**inputs, max_new_tokens=2000)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    extracted_text = response.split('### Response:')[-1].strip()
    print(extracted_text)
    formatted_text = extracted_text.replace('[', '').replace(']', '').split(',')
    return formatted_text
#     formatted_text = re.sub(r'(\w[\w\s]*\w)', r'"\1"', extracted_text)

#     # 输出处理后的文本格式
#     print("Formatted text:", formatted_text)

#     try:
#         # 使用 ast.literal_eval 解析格式化后的文本
#         return ast.literal_eval(formatted_text)
#     except (SyntaxError, ValueError) as e:
#         print(f"Error parsing response: {e}")
#         return []

def cypher_generate(graph, check_name):
    data = get_type(graph, check_name)

    results = [{"name": record["name"], "type": record["types"]} for record in data]

    aggregated_data = defaultdict(list)
    
    for record in results:
        name = record["name"]
        types = record["type"]
        aggregated_data[name].extend(types)
    
    results = [{"name": name, "types": list(set(types))} for name, types in aggregated_data.items()]
    return results

def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        './models/Llama3-Med42-8B',
        device_map={"":0},
        trust_remote_code=True,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained('./models/Llama3-Med42-8B')
    lora_config = LoraConfig.from_pretrained('./results/checkpoint-50')
    test_model = get_peft_model(model, lora_config)
    # tokenizer.pad_token = tokenizer.eos_token
    return model, test_model, tokenizer


def history_to_dialog_format(chat_history):
    dialog = []
    if len(chat_history) > 0:
        for idx, message in enumerate(chat_history[0]):
            role = "user" if idx % 2 == 0 else "assistant"
            dialog.append({
                "role": role,
                "content": message,
            })
    return dialog

def main():
    #cosine distnace
    df = read_id_table()
    embedder_model = embedder_model_load()

    loaded_embeddings = np.load('./combined_entity_embeddings.npy')
    normalize_embeddings = normalize_embedder(loaded_embeddings)
    entities_ids = np.array(list(df.to_dict()['name'].keys()))

    dimension = loaded_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index_with_ids = faiss.IndexIDMap(index)
    index_with_ids.add_with_ids(normalize_embeddings, entities_ids)

    index.ntotal

    uri = "neo4j://101.34.58.20:7687"
    username = "neo4j"
    password = "Tzmt541881"

    graph = Graph(uri, auth=(username, password))
    result_dict = get_relation_list()
    
    model, test_model, tokenizer = load_model_and_tokenizer()
    device = "cuda:0"
    
    
    def llama_response(question, history):
        dialog = history_to_dialog_format(history)
        dialog.append({"role": "user", "content": question})
        
        input_entity = get_important_entities(question, tokenizer, test_model)
        print(input_entity)
        results = cosine_distance_search(input_entity, index, embedder_model, k=10)
        check_name = []

        for i in results:
            check_name.append(df.to_dict()['name'][i[0]])
        print(check_name)
        if check_name != []:
            try:
                query = build_query(cypher_generate(graph, check_name), result_dict)
                query_results = graph.run(query).data()
                final_prompt = generate_prompt(query_results)
            except:
                print('Something wrong in the database.')
                final_prompt = 'There is no important information in this sentance.'
        else:
            final_prompt = 'There is no important information in this sentance.'
        
        inputs = tokenizer(f'You are a professional doctor and will be given a question and a list of useful information from the extral knowledge database. Use this information to answer the quesiton accurately and appropriately. Do not show the information directly in the response, just use it as a reference and judge the question with it. This information is: \n{final_prompt}\nThe question is: {question}\nResponse: \n', return_tensors="pt").to(device)
        outputs = test_model.generate(**inputs, max_new_tokens=500)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Response: " in generated_text:
            f_results = generated_text.split("Response: ", 1)[-1].strip()
        else:
            f_results = generated_text.strip()
        return f_results

    demo = gr.ChatInterface(
        llama_response, 
        title="Capstone",
        retry_btn=None,
        undo_btn=None,
        clear_btn=None,
    )

    demo.launch(share=True)


if __name__ == "__main__":
    fire.Fire(main)