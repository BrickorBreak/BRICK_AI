from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

app = FastAPI()

MODEL_NAME = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=False) # 전처리
model = CLIPModel.from_pretrained(MODEL_NAME) # 모델
model.eval() #추론

FOOD_CHECK_TEXTS = [
    "a photo of food",
    "a photo of daily life"
]

CATEGORIES = [
    "a photo of Korean food",
    "a photo of Japanese food",
    "a photo of Chinese food",
    "a photo of Asian food",
    "a photo of Western food",
    "a photo of street food",
    "a photo of fast food",
    "a photo of dessert"
]

CATEGORY_TEXT_TO_KR = {
    "a photo of Korean food": "한식",
    "a photo of Japanese food": "일식",
    "a photo of Chinese food": "중식",
    "a photo of Asian food": "아시아",
    "a photo of Western food": "양식",
    "a photo of street food": "분식",
    "a photo of fast food": "패스트푸드",
    "a photo of dessert": "디저트"
}

CATEGORY_ID_MAP = {
    "한식": 1,
    "양식": 2,
    "일식": 3,
    "중식": 4,
    "아시아": 5,
    "분식": 6,
    "패스트푸드": 7,
    "디저트": 8,
    "기타": 9
}

@app.post("/analyze") #post api
async def analyze(image: UploadFile = File(...)):

    img_bytes = await image.read() # 업로드된 이미지는 바이트 상태
    image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB") # BytesIO → 메모리에서 파일처럼 취급 , Image.open → PIL 이미지 객체로 변환

    # 일상사진인지 음식인지 구분
    food_check_inputs = processor(
        text=FOOD_CHECK_TEXTS,
        images=image_pil,
        return_tensors="pt", # 텐서로 반환
        padding=True
    )

    with torch.no_grad(): # 추론시 기울기 계산 비활성화
        food_outputs = model(**food_check_inputs) #  이미지 , 텍스트 , 유사도 점수 계산
        food_probs = F.softmax(food_outputs.logits_per_image, dim=1) # 이미지 기준으로 각 텍스트와의 유사도 점수

    food_idx = food_probs.argmax(dim=1).item() # 확률이 가장 큰 인덱스 찾기
    food_confidence = food_probs[0][food_idx].item() # 해당 인덱스의 확률 값

    # 일상사진일 경우 바로 종료
    if FOOD_CHECK_TEXTS[food_idx] == "a photo of daily life" and food_confidence > 0.6:
        return {
            "type": "일상사진"
        }
    
    category_inputs = processor(
        text=CATEGORIES,
        images=image_pil,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        category_outputs = model(**category_inputs) # 이미지 1장 , 음식 카테고리 문장 여러 개
        category_probs = F.softmax(category_outputs.logits_per_image, dim=1) # 이미지 기준으로 각 텍스트와의 유사도 점수

    best_idx = category_probs.argmax(dim=1).item() # 확률이 제일 큰 인덱스
    confidence = category_probs[0][best_idx].item() # 해당 인덱스의 확률 값

    category_text = CATEGORIES[best_idx] # 확률이 가장 높은 카테고리 텍스트
    category_kr = CATEGORY_TEXT_TO_KR.get(category_text, "기타") # 한글 카테고리명

    if confidence < 0.4:
        category_kr = "기타" # 확률이 낮으면 기타로 분류

    category_id = CATEGORY_ID_MAP[category_kr] # 한글 카테고리명에 해당하는 ID

    return {
        "type": "음식",
        "categoryId": category_id,
        "categoryName": category_kr,
        "confidence": round(confidence, 3)
    }
