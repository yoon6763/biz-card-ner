import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# =========================
# 라벨 정의
# =========================
label_list = ["O", "NAME", "POSITION", "COMPANY"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# =========================
# 토크나이저
# =========================
MODEL_NAME = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    tokenized = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=16
    )
    tokenized["label"] = label2id[example["label"]]
    return tokenized

# =========================
# 학습 데이터 CSV 읽기
# =========================
df_name = pd.read_csv("train_data_set/name_train_data.csv", header=None, names=["text"])
df_name["label"] = "NAME"

df_company = pd.read_csv("train_data_set/company_train_data.csv", header=None, names=["text"])
df_company["label"] = "COMPANY"

df_position = pd.read_csv("train_data_set/position_train_data.csv", header=None, names=["text"])
df_position["label"] = "POSITION"

df_o = pd.read_csv("train_data_set/other_train_data.csv", header=None, names=["text"])
df_o["label"] = "O"

# 합치기
df_train = pd.concat([df_name, df_company, df_position, df_o], ignore_index=True)

# Dataset 변환
dataset = Dataset.from_pandas(df_train)

# tokenize 적용
dataset = dataset.map(tokenize, batched=False)

# text 컬럼 제거, label은 그대로
dataset = dataset.remove_columns(["text"])
dataset.set_format("torch")

# =========================
# 모델
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)
model.to(device)  # GPU로 이동

# =========================
# 학습 설정
# =========================
training_args = TrainingArguments(
    output_dir="./cls_model",
    per_device_train_batch_size=16,  # GPU 메모리 맞게 조절
    num_train_epochs=10,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    fp16=True  # GPU가 있으면 16-bit mixed precision 사용 가능
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# =========================
# 학습
# =========================
trainer.train()
print("학습 완료")

# =========================
# 예측 함수
# =========================
def predict(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=16
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}  # GPU로 이동

    model.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = torch.argmax(outputs.logits, dim=-1).item()
    return id2label[pred_id]

# =========================
# 테스트 실행
# =========================
name_test_set = pd.read_csv("test_data_set/name_test_data.csv", header=None, names=["text"])
name_test_set["label"] = "NAME"

company_test_set = pd.read_csv("test_data_set/company_test_data.csv", header=None, names=["text"])
company_test_set["label"] = "COMPANY"

position_test_set = pd.read_csv("test_data_set/position_test_data.csv", header=None, names=["text"])
position_test_set["label"] = "POSITION"

print("\n테스트 결과")
for df in [name_test_set, company_test_set, position_test_set]:
    for text, label in zip(df["text"], df["label"]):
        result = predict(text)
        correct = result == label
        print(f"{text} ({label}) → {result} | 맞음: {correct}")
