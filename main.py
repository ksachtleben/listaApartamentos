import os
import uvicorn

from fastapi import FastAPI

from utils.DataManager import DataManager

from routes.selling_routes import sellingRoutes

app = FastAPI()
db  = DataManager()

app.include_router(sellingRoutes(db), prefix='/api/selling')

def normalizePort(port):
    if (port is None):
        return 8080

    if (type(port) is str):
        return int(port)

    return port

if __name__ == "__main__":
    uvicorn.run("main:app", port=normalizePort(os.environ.get("PORT") or os.environ.get("API_PORT")))
