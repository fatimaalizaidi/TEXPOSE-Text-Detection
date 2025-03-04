print("svc_texpose - starting")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from service import router as router

app = FastAPI(openapi_url="/api/v1/texpose/openapi.json", docs_url="/api/v1/texpose/docs")

# CORS (Cross-Origin Resource Sharing) configuration
origins = [
    "http://localhost:3000",
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)