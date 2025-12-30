import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ------------------------------
# 후처리용 유틸
# ------------------------------
def merge_subwords(token_label_list):
    """##로 시작하는 subword 토큰 합치기"""
    merged_tokens = []
    current_token, current_label = "", None

    for token, label in token_label_list:
        if token in ["[CLS]", "[SEP]"]:
            continue

        if token.startswith("##"):
            current_token += token[2:]
        else:
            if current_token:
                merged_tokens.append((current_token, current_label))
            current_token, current_label = token, label

    if current_token:
        merged_tokens.append((current_token, current_label))

    return merged_tokens

# ------------------------------
# NER 모델 불러오기
# ------------------------------
MODEL_NAME = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

# ------------------------------
# NER 추론 함수
# ------------------------------
def ner_predict(texts):
    results = []

    for text in texts:
        inputs = tokenizer(
            text.split(),
            return_tensors="pt",
            is_split_into_words=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=-1)[0]
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [model.config.id2label[p.item()] for p in predictions]

        merged = merge_subwords(list(zip(tokens, labels)))
        results.append(merged)

    return results

# ------------------------------
# main
# ------------------------------
if __name__ == "__main__":
    ocr_text = """김철수
테스트컴퍼니"""

    ner_input = [line.strip() for line in ocr_text.split("\n") if line.strip()]
    print("NER input:", ner_input)

    # NER 추론
    ner_results = ner_predict(ner_input)
    print("NER results:")
    for line, result in zip(ner_input, ner_results):
        print(f"- {line} → {result}")
