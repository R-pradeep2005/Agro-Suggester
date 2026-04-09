from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import httpx
import time
import os

app = FastAPI(title="Agro-Suggester API Gateway")

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting settings
ip_request_counts = {}
RATE_LIMIT_DURATION = 60  # seconds
MAX_REQUESTS_PER_MINUTE = 60

@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    client_ip = request.client.host
    current_time = time.time()
    
    if client_ip not in ip_request_counts:
        ip_request_counts[client_ip] = []
        
    # Remove timestamps older than RATE_LIMIT_DURATION
    ip_request_counts[client_ip] = [t for t in ip_request_counts[client_ip] if current_time - t < RATE_LIMIT_DURATION]
    
    if len(ip_request_counts[client_ip]) >= MAX_REQUESTS_PER_MINUTE:
        return Response(
            content='{"detail": "Rate limit exceeded"}',
            media_type="application/json",
            status_code=429
        )
        
    ip_request_counts[client_ip].append(current_time)
    return await call_next(request)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "gateway"}

INPUT_PREP_URL = os.environ.get("INPUT_PREP_URL", "http://localhost:8001")
RECOMMENDATION_URL = os.environ.get("RECOMMENDATION_URL", "http://localhost:8002")

async def forward_request(url: str, request: Request):
    headers = dict(request.headers)
    # Remove host and content-length headers to let httpx handle them properly
    headers.pop("host", None)
    headers.pop("content-length", None)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=await request.body(),
            params=request.query_params
        )
        
    resp_headers = dict(response.headers)
    resp_headers.pop("content-length", None)
    resp_headers.pop("content-encoding", None)
    
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=resp_headers
    )

@app.api_route("/api/input_prep/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_input_prep(path: str, request: Request):
    print(f"\n[GATEWAY] ← Received {request.method} from Frontend: /api/input_prep/{path}")
    print(f"[GATEWAY] → Forwarding to Input Prep Service: {INPUT_PREP_URL}/{path}")
    response = await forward_request(f"{INPUT_PREP_URL}/{path}", request)
    print(f"[GATEWAY] ← Response from Input Prep: HTTP {response.status_code}")
    print(f"[GATEWAY] → Returning final result to Frontend\n")
    return response

@app.api_route("/api/recommendation/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_recommendation(path: str, request: Request):
    return await forward_request(f"{RECOMMENDATION_URL}/{path}", request)
