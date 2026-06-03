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
AWS_REGION = os.getenv("AWS_REGION")

# Create Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)


#AWS_REGION = os.getenv("myenv.env", "us-east-1")

# =====================================================
# Create Bedrock Runtime Client
# =====================================================


MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

# =====================================================
# Conversation History
# =====================================================
conversation_history = []


# =====================================================
# Display Conversation History
# =====================================================
def display_history():
    """
    Display conversation history in a readable format.
    """

    if not conversation_history:
        print("\nNo conversation history found.\n")
        return

    print("\n" + "=" * 60)
    print("CONVERSATION HISTORY")
    print("=" * 60)

    for index, message in enumerate(conversation_history, start=1):

        role = message["role"].upper()

        text = ""

        for item in message["content"]:
            if "text" in item:
                text += item["text"]

        print(f"\n[{index}] {role}")
        print(text)

    print("\n" + "=" * 60)
    print(f"Total Messages: {len(conversation_history)}")
    print("=" * 60)


# =====================================================
# Stream Response from Bedrock
# =====================================================
def stream_response():
    """
    Streams assistant response token-by-token.
    """

    assistant_response = ""

    try:

        response = bedrock_client.converse_stream(
            modelId=MODEL_ID,
            messages=conversation_history,
            inferenceConfig={
                "maxTokens": 1000,
                "temperature": 0.7,
                "topP": 0.9
            }
        )

        print("\nAssistant: ", end="", flush=True)

        stream = response["stream"]

        for event in stream:

            if "contentBlockDelta" in event:

                delta = event["contentBlockDelta"]["delta"]

                if "text" in delta:

                    chunk = delta["text"]

                    print(chunk, end="", flush=True)

                    assistant_response += chunk

        print("\n")

        # Save assistant response into history
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

        return assistant_response

    except ClientError as error:
        print(f"\nAWS Client Error:\n{error}")

    except BotoCoreError as error:
        print(f"\nBoto3 Error:\n{error}")

    except Exception as error:
        print(f"\nUnexpected Error:\n{error}")

    return None


# =====================================================
# Main Chatbot Function
# =====================================================
def main():

    print("\n" + "=" * 60)
    print("AMAZON NOVA MICRO STREAMING CHATBOT")
    print("=" * 60)

    print("\nCommands:")
    print("history  -> Show conversation history")
    print("clear    -> Clear conversation history")
    print("exit     -> Exit chatbot")

    print("\nChatbot is ready!\n")

    while True:

        try:

            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Exit Commands
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nGoodbye!")
                break

            # Show History
            if user_input.lower() == "history":
                display_history()
                continue

            # Clear History
            if user_input.lower() == "clear":
                conversation_history.clear()
                print("\nConversation history cleared.\n")
                continue

            # Store User Message
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

            # Get Streaming Response
            stream_response()

            # Display Number of Messages
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
# Entry Point
# =====================================================
if __name__ == "__main__":
    main()