import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ------------------------------
# 1️⃣ Regex 정의
# ------------------------------
PHONE_REGEX = re.compile(r'(?:\+82[-.\s]?)?01[016789][-.\s]?\d{3,4}[-.\s]?\d{4}')
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

def extract_regex_entities(text):
    """텍스트에서 전화번호와 이메일 추출"""
    return {
        "phones": [m.group() for m in PHONE_REGEX.finditer(text)],
        "emails": [m.group() for m in EMAIL_REGEX.finditer(text)]
    }

# ------------------------------
# 2️⃣ 후처리용 유틸
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
# 3️⃣ NER 모델 불러오기
# ------------------------------
MODEL_NAME = "distilbert-base-multilingual-cased"  # 학습된 NER 모델로 바꾸면 됨
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

# ------------------------------
# 4️⃣ NER 추론 함수
# ------------------------------
def ner_predict(texts):
    results = []
    for text in texts:
        inputs = tokenizer(text.split(), return_tensors="pt", is_split_into_words=True)
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)[0]
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [model.config.id2label[p.item()] for p in predictions]
        merged = merge_subwords(list(zip(tokens, labels)))
        results.append(merged)
    return results

# ------------------------------
# 5️⃣ 메인 테스트
# ------------------------------
if __name__ == "__main__":
    ocr_text = """김철수
테스트컴퍼니
010-1234-5678
kimcs@test.com"""

    # 1) Regex로 이메일/전화번호 추출
    entities = extract_regex_entities(ocr_text)
    print("Regex:", entities)

    # 2) Regex 제거 후 NER 입력용 텍스트 준비
    ner_input = []
    for line in ocr_text.split("\n"):
        line = line.strip()
        if line not in entities["phones"] and line not in entities["emails"]:
            ner_input.append(line)
    print("NER input:", ner_input)

    # 3) NER 추론
    ner_results = ner_predict(ner_input)
    print("NER results:", ner_results)
