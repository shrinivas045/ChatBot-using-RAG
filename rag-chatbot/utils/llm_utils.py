import requests
import json
import streamlit as st

def query_llm(prompt, provider=None):
    try:
        api_key = st.secrets['GROQ_API_KEY']
        if not api_key:
            return {"error": "GROQ_API_KEY not found in Streamlit secrets"}
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }

        response = requests.post(url, headers=headers, json=payload)
        try:
            result = response.json()
            if 'error' in result:
                return {
                    "error": f"API Error: {result['error'].get('message', 'Unknown error')}",
                    "details": result['error']
                }
            return result
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON response: {response.text}"}
    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}
