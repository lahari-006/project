import boto3
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("myenv.env")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

# Create Bedrock client
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

MODEL_ID = "amazon.nova-micro-v1:0"

# Paragraph to summarize
paragraph = """
Artificial Intelligence is transforming industries across the world.
From healthcare and finance to education and transportation, AI systems
are helping automate tasks, improve decision-making, and enhance user
experiences. Machine learning models can analyze huge amounts of data
quickly and identify patterns that humans may miss. However, AI also
raises concerns regarding job displacement, privacy, and ethical use.
"""

# Prompt for summarization
prompt = f"""
Summarize the following paragraph in 3-4 lines:

{paragraph}
"""

# Request body
body = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "text": prompt
                }
            ]
        }
    ],
    "inferenceConfig": {
        "max_new_tokens": 150,
        "temperature": 0.3
    }
}

# Invoke model
response = bedrock_runtime.invoke_model(
    modelId=MODEL_ID,
    body=json.dumps(body),
    contentType="application/json",
    accept="application/json"
)

# Read response
response_body = json.loads(response["body"].read())

# Extract summary
summary = response_body["output"]["message"]["content"][0]["text"]

print("\nSummary:\n")
print(summary)