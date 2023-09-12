import time

from dotenv import load_dotenv
import os
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI")


def response(content, role="user"):
  """fetches the GPT 3.5 completion
    :type role: str
    :type content: str

    :returns :type JSON response: str
    """
  try:
    return openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      messages=[{
        "role": role,
        "content": content,
      }]
    )

  except openai.error.RateLimitError as e:
    retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
    print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
