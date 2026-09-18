from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os

app = FastAPI()

SECRET = os.environ.get("FLY_API_KEY", "")


class StepRequest(BaseModel):
    fly_id: str
    sensors: list[list[float]]


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Fly Neural Server"
    }


@app.post("/step")
def step(
    request: StepRequest,
    x_api_key: str | None = Header(default=None)
):
    if SECRET and x_api_key != SECRET:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    outputs = []

    for _ in request.sensors:
        outputs.append({
            "forward": 0.3,
            "turn": 0.0,
            "lift": 0.0,
            "brake": 0.0
        })

    return {
        "outputs": outputs
    }
