import os
import json
import requests
import PIL.Image
from io import BytesIO
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field

from google import genai


GEMINI_MODEL_NAME = "gemini-2.5-pro"

OCR_PROMPT = """이 식단표 이미지에서 요일별 메뉴를 JSON 형식으로 추출해줘.
실속일품코너는 같이 날짜 menus에 포함시켜주고, 샐러드바는 무시해.
기숙사 식당 식단표에서 구분이 할랄일 경우 restaurantType이 기숙사_식당_할랄이 되도록 해줘.
"""


class DailyMenu(BaseModel):
    date: str = Field(description="Date of the menu. Format: MM-DD.")
    day: str = Field(description="Day of the week. One of Mon, Tue, Wed, Thu, Fri, Sat, Sun.")
    menus: list[str] = Field(description="List of menu items for the day.")  # str -> list[str]로 변경
    calorie: int = Field(description="Total calories for the day's menu. If not available, set to 0.")
    time: str = Field(description="Meal time. One of breakfast, lunch, dinner.")


class WeeklyMenu(BaseModel):
    items: list[DailyMenu]
    restaurant_name: str = Field(description="Name of the restaurant. One of 기숙사_식당_한식, 기숙사_식당_할랄, 학생_식당, 교직원_식당.")
    start_date: str = Field(description="Start date of the week. Format: MM-DD.")
    end_date: str = Field(description="End date of the week. Format: MM-DD.")


# 3. 클라이언트 초기화 (API 키 환경변수 연동 확인)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# 이미지 다운로드
image_url = os.environ["IMAGE_URL"]
print(f"Downloading image from: {image_url}")
response = requests.get(image_url)
image_data = PIL.Image.open(BytesIO(response.content))

# 4. 생성 요청
response = client.models.generate_content(
    model=GEMINI_MODEL_NAME,
    contents=[OCR_PROMPT, image_data],
    config={
        "response_mime_type": "application/json",
        "response_schema": WeeklyMenu,
        # "thinking_level": "low",
    },
)

parsed_result = WeeklyMenu.model_validate_json(response.text)

year = datetime.now().year

restaurant_name = parsed_result.restaurant_name
start_date = parsed_result.start_date
end_date = parsed_result.end_date

save_file_name = Path(f"{restaurant_name}_{year}-{start_date}.json")
base_path = Path("menus")
final_path = base_path / save_file_name

assert not final_path.exists(), f"File {final_path} already exists!"


weekday_to_index = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
time_to_index = {"breakfast": 1, "lunch": 2, "dinner": 3}
# 저장할 때는 items 리스트만 깔끔하게 저장
with open(final_path, "w", encoding="utf-8") as f:
    json_data = parsed_result.model_dump(mode="json")

    output_for_save = []
    for daily_menu in json_data["items"]:
        if daily_menu["calorie"] is None:
            daily_menu["calorie"] = 0
        daily_menu["date"] = f"{year}-{daily_menu['date']}"
        daily_menu["restaurant_name"] = restaurant_name
        daily_menu["day"] = weekday_to_index[daily_menu["day"]]
        daily_menu["time"] = time_to_index[daily_menu["time"]]
        output_for_save.append(daily_menu)
    f.write(json.dumps(output_for_save, ensure_ascii=False, indent=2))


print("Menu extraction successful!")
