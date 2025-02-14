# github_app/main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import hmac
import hashlib
import boto3
import json
from knowlang.github_app.config import GitHubAppConfig

app = FastAPI()
settings = GitHubAppConfig()
sqs = boto3.client('sqs')

async def verify_webhook_signature(request: Request) -> bool:
    signature = request.headers.get('X-Hub-Signature-256')
    if not signature:
        return False
    
    # Verify webhook signature using GITHUB_WEBHOOK_SECRET
    body = await request.body()
    expected_signature = hmac.new(
        settings.GITHUB_WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected_signature}", signature)

@app.post("/webhook")
async def github_webhook(request: Request):
    if not await verify_webhook_signature(request):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    payload = await request.json()
    event_type = request.headers.get("X-GitHub-Event")
    
    if event_type == "push":
        # Enqueue parsing job to SQS
        sqs.send_message(
            QueueUrl=settings.SQS_QUEUE_URL,
            MessageBody=json.dumps({
                "repository": payload["repository"]["full_name"],
                "commit_sha": payload["after"],
                "branch": payload["ref"].split("/")[-1]
            })
        )
        return {"status": "Processing"}
    
    return {"status": "Ignored"}