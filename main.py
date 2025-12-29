import re

# Regex patterns
PHONE_REGEX = re.compile(r'(?:\+82[-.\s]?)?01[016789][-.\s]?\d{3,4}[-.\s]?\d{4}')
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

def extract_regex_entities(text):
    return {
        "phones": [m.group() for m in PHONE_REGEX.finditer(text)],
        "emails": [m.group() for m in EMAIL_REGEX.finditer(text)]
    }


if __name__ == "__main__":
    ocr_text = """김철수
    테스트컴퍼니
    010-1234-5678
    kimcs@test.com"""
    entities = extract_regex_entities(ocr_text)
    print(entities)
