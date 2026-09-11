from fastapi import FastAPI
from app.routes import documents

app = FastAPI(title="Smart Study Buddy API")

# Register routers
app.include_router(documents.router)


@app.get("/")
def read_root():
    return {"message": "Smart Study Buddy API is running!"}
