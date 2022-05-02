from fastapi import FastAPI

app = FastAPI()

STATUS = True

@app.get("/")
async def root():
    return {"message": "Buslight recognition API"}

@app.get("/status")
async def get_status():
    return {"status": STATUS}