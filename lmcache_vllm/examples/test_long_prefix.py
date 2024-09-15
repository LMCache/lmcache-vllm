from openai import OpenAI
import time

openai_api_key = "EMPTY"
openai_api_base = f"http://localhost:8100/v1"

client = client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)
models = client.models.list()
model = models.data[0].id


messages = []
message = "Hello how are you?" *100
messages.append({"role": "user", "content": message})

chat_completion = client.chat.completions.create(
    messages=messages,
    model=model,
    temperature=0.0,
    stream=False,
    stop = "\n",
    max_tokens = 1,
)

print(chat_completion)

time.sleep(5)

messages = []
message = "Hello how are you?" *101
messages.append({"role": "user", "content": message})

chat_completion = client.chat.completions.create(
    messages=messages,
    model=model,
    temperature=0.0,
    stream=False,
    stop = "\n",
    max_tokens = 1,
)

print(chat_completion)