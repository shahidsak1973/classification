import os
from openai import AzureOpenAI

endpoint = "https://shahidshaik-resource.cognitiveservices.azure.com/"
model_name = "gpt-4o"
deployment = "gpt-4o"

subscription_key = "4uE68jl4awJtdxtHtFzpmT0gVO77eCDF23MxiiCLb4pQ8zGa135lJQQJ99BLACHYHv6XJ3w3AAAAACOGMCMr"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=deployment
)

print(response.choices[0].message.content)