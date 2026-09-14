from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()  # ← this line reads the .env file and loads the key
client = Anthropic()  # ← this automatically grabs ANTHROPIC_API_KEY from what was loaded

message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)

print(message.content[0].text)