import uvicorn
from fastapi import FastAPI
from api.logo_router import router as logo_router

app = FastAPI()
app.include_router(logo_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
