from fastapi import FastAPI

app = FastAPI(title="Smart Study Buddy API")


@app.get("/")
def read_root():
    return {"message": "Smart Study Buddy API is running!"}
