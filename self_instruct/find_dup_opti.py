from sentence_transformers import SentenceTransformer, util
import json
import numpy as np

#리스트 total에서 1개인 finding과 겹치는 것이 있는지 확인하는 코드 겹친다면 True반환
def filter(total,finding,embedder,similarity_threshold=0.78):
    
    #corpus=[item['instruction'] for item in total]
    corpus=total
    query_embedding=embedder.encode(finding,convert_to_tensor=True)
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    return cos_scores
    '''
    cos_scores = cos_scores.cpu().numpy()
    print(cos_scores)
    duplicate_indices = np.where(cos_scores > similarity_threshold)[0]
    for  i in duplicate_indices:
        print(finding,' ',corpus[i],cos_scores[i])
        return True
    return False
    '''

'''
    embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")

    # Load jsonl data into a Python list of dictionaries
    data = []
    with open("../data/machine_generated_instructions.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))

    # Get the prompts from the data
    corpus = [item["instruction"] for item in data]

    # Compute the embeddings for the prompts
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    # Set a similarity threshold and initialize an empty set for duplicates
    similarity_threshold = 0.78 # Adjust this value according to your needs
    duplicates = set()
    duplicates_list = []

    # Compute pairwise cosine similarity scores, and find duplicates
    for i, query_embedding in enumerate(corpus_embeddings):
        if i not in duplicates:
            cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
            cos_scores = cos_scores.cpu().numpy()
            print(cos_scores)
            duplicate_indices = np.where(cos_scores > similarity_threshold)[0]
            for j in duplicate_indices:
                if i != j and j not in duplicates:
                    duplicates.add(j)
                    duplicates_list.append((i, j, cos_scores[j]))

    # Remove duplicates from the data based on the similarity threshold
    unique_data = [item for i, item in enumerate(data) if i not in duplicates]

    # Save unique data to a new jsonl file
    with open("unique_data.jsonl", "w", encoding="utf-8") as output_file:
        for item in unique_data:
            output_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Output duplicate sentence pairs and their similarity scores
    for i, j, score in duplicates_list:
        print(f"Line {i}: {corpus[i]}")
        print(f"Line {j}: {corpus[j]}")
        print(f"Similarity: {score:.2f}")
        print()

    print(f"\nremoved: {len(duplicates)}")
'''
if __name__ == "__main__":
    A=['배고파!!','밥 먹었어??','아껴쓰자','공부하자','비빔밥 먹고싶다']
    B='망했다'
    embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")
    
    filter(A,B,embedder)
    