import base64
import re
from openai import OpenAI
import requests

def get_sambanova_response(messages, model="Meta-Llama-3.3-70B-Instruct", temperature=0.1, top_p=0.1):
    """
    Get response from SambaNova API
    
    Args:
        messages: List of message dictionaries
        model: Model name to use
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
    
    Returns:
        Response content as string
    """
    client = OpenAI(
        api_key="",
        base_url="https://api.sambanova.ai/v1",
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p
    )
    
    return response.choices[0].message.content

def append_message_to_list(messages, role, content):
    messages.append({"role":role, "content": content})

def encode_image_from_path(image_path: str) -> str:
    """Encode image from local file path to base64 string"""

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_from_url(image_url: str) -> str:
    """Download and encode image from URL to base64 string"""
    # Download the image
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()  # Raise an exception for bad status codes

    # Encode to base64
    image_base64 = base64.b64encode(response.content).decode('utf-8')
    return image_base64

def encode_image(image_source: str) -> str:
    """
    Universal function to encode image from either local path or URL

    Args:
        image_source: Either a local file path or HTTP/HTTPS URL

    Returns:
        Base64 encoded string of the image
    """
    if image_source.startswith(('http://', 'https://')):
        return encode_image_from_url(image_source)
    else:
        return encode_image_from_path(image_source)
    
def extract_image_info(query: str) -> dict:
    result = {
        "isImage": False,
        "imageSource": None
    }

    # Regex for URLs
    url_pattern = r'https?://[^\s]+(?:\.png|\.jpg|\.jpeg|\.gif|\.bmp|\.webp)?'

    # Regex for local or relative file paths
    local_pattern = r'(?:\.{0,2}/|[A-Za-z]:\\)[^\s]+\.(png|jpg|jpeg|gif|bmp|webp)'

    # Search for URL
    url_match = re.search(url_pattern, query)
    if url_match:
        result["isImage"] = True
        result["imageSource"] = url_match.group(0)
        return result

    # Search for local path
    local_match = re.search(local_pattern, query)
    if local_match:
        result["isImage"] = True
        result["imageSource"] = local_match.group(0)

    return result