import os
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError, BotoCoreError

# =====================================================
# Load Environment Variables
# =====================================================
load_dotenv("myenv.env")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# =====================================================
# Create Bedrock Client
# =====================================================
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

MODEL_ID = "amazon.nova-micro-v1:0"

# =====================================================
# Conversation History
# =====================================================
conversation_history = []

# =====================================================
# Display History
# =====================================================
def display_history():

    if not conversation_history:
        print("\nNo conversation history found.\n")
        return

    print("\n" + "=" * 60)
    print("CONVERSATION HISTORY")
    print("=" * 60)

    for idx, message in enumerate(conversation_history, start=1):

        role = message["role"].upper()

        text = ""

        for item in message["content"]:
            if "text" in item:
                text += item["text"]

        print(f"\n[{idx}] {role}")
        print(text)

    print("\n" + "=" * 60)
    print(f"Total Messages: {len(conversation_history)}")
    print("=" * 60)

# =====================================================
# Get Response (Non-Streaming)
# =====================================================
def get_response():

    try:

        response = bedrock_client.converse(
            modelId=MODEL_ID,
            messages=conversation_history,
            inferenceConfig={
                "maxTokens": 1000,
                "temperature": 0.7,
                "topP": 0.9
            }
        )

        assistant_response = (
            response["output"]["message"]["content"][0]["text"]
        )

        print(f"\nAssistant: {assistant_response}\n")

        conversation_history.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "text": assistant_response
                    }
                ]
            }
        )

    except ClientError as error:
        print(f"\nAWS Client Error:\n{error}")

    except BotoCoreError as error:
        print(f"\nBoto3 Error:\n{error}")

    except Exception as error:
        print(f"\nUnexpected Error:\n{error}")

# =====================================================
# Main Chatbot
# =====================================================
def main():

    print("\n" + "=" * 60)
    print("AMAZON NOVA MICRO CHATBOT")
    print("=" * 60)

    print("\nCommands:")
    print("history -> Show chat history")
    print("clear   -> Clear history")
    print("exit    -> Exit chatbot\n")

    while True:

        try:

            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Exit
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nGoodbye!")
                break

            # History
            if user_input.lower() == "history":
                display_history()
                continue

            # Clear
            if user_input.lower() == "clear":
                conversation_history.clear()
                print("\nConversation history cleared.\n")
                continue

            # Store user message
            conversation_history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "text": user_input
                        }
                    ]
                }
            )

            # Get assistant response
            get_response()

            print(
                f"Conversation History Size: "
                f"{len(conversation_history)} messages\n"
            )

        except KeyboardInterrupt:
            print("\n\nChatbot terminated by user.")
            break

        except Exception as error:
            print(f"\nError: {error}")

# =====================================================
# Run
# =====================================================
if __name__ == "__main__":
    main()