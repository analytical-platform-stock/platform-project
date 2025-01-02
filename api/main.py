import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from api_router import router, lifespan

app = FastAPI(
    title="analytic_platform_api",
    lifespan=lifespan
)

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint that returns a greeting message.
    """
    return {"message": "Привет!"}

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)