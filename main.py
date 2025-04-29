from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import re
import pandas as pd
from dotenv import load_dotenv
from openai import openai

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://reviewradar.vercel.app"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client (✅ no proxies anymore)
client = openai(
    base_url=os.getenv("OPENAI_BASE_URL", "https://models.inference.ai.azure.com"),
    api_key=os.getenv("OPENAI_API_KEY", "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"),
)

# Define Pydantic models
class ReviewRequest(BaseModel):
    review_text: str

class ReviewResponse(BaseModel):
    fakePercentage: int
    realPercentage: int
    summary: str
    raw_result: Optional[str] = None

class Feedback(BaseModel):
    feedback: str
    textFeedback: Optional[str] = None

# Helper Functions
def api_out(prompt):
    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise HTTPException(status_code=500, detail="Error calling AI service")

def data_extract(review_text, response):
    fake_percentage, real_percentage = 0, 0

    # Extract percentages
    fake_match = re.search(r"Fake Percentage:\s*(\d+)%", response)
    real_match = re.search(r"Real Percentage:\s*(\d+)%", response)

    if fake_match:
        fake_percentage = int(fake_match.group(1))
    if real_match:
        real_percentage = int(real_match.group(1))

    try:
        df = pd.read_csv("review.csv")
    except FileNotFoundError:
        # If CSV does not exist, create it
        df = pd.DataFrame(columns=["Review", "Fake%", "Real%", "Pred_Result", "User_Feedback", "Text_Feedback"])

    name_to_check = review_text.strip()
    if name_to_check not in df["Review"].values:
        pred_result = "fake" if fake_percentage > real_percentage else "real"
        new_data = {
            "Review": name_to_check,
            "Fake%": fake_percentage,
            "Real%": real_percentage,
            "Pred_Result": pred_result
        }
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        df.to_csv("review.csv", index=False)

def detect_fake_review(review_text):
    prompt = f"""
    Analyze the following customer review and determine whether it is potentially fake or real based on common patterns in online marketplace reviews from India.

    Consider Hinglish, local languages, broken grammar.

    Review:
    "{review_text}"

    Instructions:

    - If input is NOT a review, return Fake Percentage: 0%, Real Percentage: 0%.
    - Strictly return percentages in format:
        Fake Percentage: XX%
        Real Percentage: YY%
        accuracy: ZZ%

    - End with one-line summary.

    Response format:
    Fake Percentage: XX%
    Real Percentage: YY%
    accuracy: ZZ%
    Summary: one-line reason
    """

    response = api_out(prompt)
    print("OpenAI Response:", response)
    data_extract(review_text, response)
    return response

def parse_result(result):
    fake_percentage = int(re.search(r"Fake Percentage:\s*(\d+)%", result).group(1) or 0)
    real_percentage = int(re.search(r"Real Percentage:\s*(\d+)%", result).group(1) or 0)

    # Summary extraction
    lines = [line.strip() for line in result.split('\n') if line.strip()]
    summary = next((line for line in lines if "Percentage" not in line and "accuracy" not in line), "")

    return fake_percentage, real_percentage, summary

# Routes
@app.get("/")
def read_root():
    return {"message": "Fake Review Detector API is running ✅"}

@app.post("/api/analyze", response_model=ReviewResponse)
async def analyze_review(request: ReviewRequest):
    if not request.review_text.strip():
        raise HTTPException(status_code=400, detail="Review text is required")

    result = detect_fake_review(request.review_text)
    fake_percentage, real_percentage, summary = parse_result(result)

    return ReviewResponse(
        fakePercentage=fake_percentage,
        realPercentage=real_percentage,
        summary=summary,
        raw_result=result
    )

@app.get("/components/analytics")
async def get_fake_review_count():
    try:
        df = pd.read_csv('review.csv')
    except FileNotFoundError:
        return {"total_reviews": 0, "total_fake_reviews": 0}

    total_reviews = len(df)
    total_fake_reviews = len(df[df['Pred_Result'] == 'fake'])
    return {"total_reviews": total_reviews, "total_fake_reviews": total_fake_reviews}

@app.post("/api/feedback")
async def receive_feedback(feedback: Feedback):
    try:
        df = pd.read_csv("review.csv")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No reviews found to attach feedback to.")

    latest_index = df.index[-1]

    df.at[latest_index, "User_Feedback"] = feedback.feedback
    df.at[latest_index, "Text_Feedback"] = feedback.textFeedback or ""

    df.to_csv("review.csv", index=False)

    return {"message": "Feedback received successfully ✅"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
