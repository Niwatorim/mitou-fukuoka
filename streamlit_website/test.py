import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
# Print the first few characters of your key to verify it's loaded
api_key = os.getenv('GEMINI_API_KEY')
print(f"API key starts with: {api_key[:5]}...")

# Try to configure genai with your key
genai.configure(api_key=api_key)

# Test a simple model call
try:
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content('Hello')
    print(response.text)
except Exception as e:
    print(f"Error: {e}")