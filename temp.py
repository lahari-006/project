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

# Temperature values
temperatures = [0.0, 0.5, 1.0]

# PROMPTS
creative_prompt = "Write a short futuristic story about AI."

factual_prompt = "What are the main advantages of Artificial Intelligence?"

# FUNCTION
def run_experiment(prompt, task_name):

    print("\n" + "="*60)
    print(f"TASK: {task_name}")
    print("="*60)

    for temp in temperatures:

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
                "temperature": temp
            }
        }

        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(response["body"].read())

        output = response_body["output"]["message"]["content"][0]["text"]

        print("\n" + "-"*50)
        print(f"TEMPERATURE = {temp}")
        print("-"*50)

        print(output)


# RUN BOTH TASKS
run_experiment(creative_prompt, "Creative Writing")

run_experiment(factual_prompt, "Factual Task")