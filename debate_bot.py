import streamlit as st
from google import genai
from google.genai import types
from groq import Groq
from duckduckgo_search import DDGS
import json

# --- Configuration for Streamlit Cloud ---
# This pulls the keys securely from Streamlit's hidden settings
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# ... [rest of the code remains exactly the same] ...

st.title("🤖 Free Web-Surfing AI Debate: Gemini vs. Llama 3")

# --- Web Search Tool for Groq ---
def search_web(query):
    """DuckDuckGo Search tool for Groq"""
    try:
        results = DDGS().text(query, max_results=3)
        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": str(e)})

groq_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current events, facts, or up-to-date information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    }
]

# --- AI Caller Functions ---
def call_gemini(prompt):
    """Calls Gemini with native Google Search enabled."""
    response = gemini_client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            # This single line gives Gemini live Google Search access!
            tools=[types.Tool(google_search=types.GoogleSearch())] 
        )
    )
    return response.text

def call_groq(prompt):
    """Calls Groq (Llama 3.3) with DuckDuckGo Search capabilities."""
    messages = [{"role": "user", "content": prompt}]
    
    # 1. Ask Llama to answer, giving it access to the search tool
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=groq_tools,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    
    # 2. Check if Llama decided it needs to search the web
    if response_message.tool_calls:
        messages.append(response_message) 
        
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "search_web":
                args = json.loads(tool_call.function.arguments)
                st.toast(f"🔎 Llama 3 is searching the web for: {args['query']}")
                search_result = search_web(args["query"])
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "search_web",
                    "content": search_result
                })
        
        # 3. Let Llama formulate its final answer using the search results
        final_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        return final_response.choices[0].message.content
    
    return response_message.content

# --- UI & Debate Logic ---
user_question = st.text_input("Ask a complex question to start the debate:")
max_rounds = st.slider("Maximum debate rounds", 1, 5, 3)

if st.button("Start Debate"):
    if not user_question:
        st.warning("Please enter a question.")
    else:
        st.write(f"**Original Question:** {user_question}")
        st.divider()

        # Initial answer
        with st.spinner("Gemini is researching..."):
            gemini_current = call_gemini(f"Answer this question thoroughly: {user_question}")
            st.markdown(f"**🔵 Gemini (Initial):**\n{gemini_current}")
        
        for i in range(max_rounds):
            st.write(f"### Round {i+1}")
            
            # Llama evaluates Gemini
            with st.spinner("Llama 3 is fact-checking Gemini..."):
                groq_prompt = f"""
                The user asked: "{user_question}".
                Gemini answered: "{gemini_current}".
                Review this answer. Search the web if you need to verify facts. 
                If you completely agree and it is 100% accurate, reply ONLY with 'CONSENSUS REACHED'. 
                If you disagree or have corrections, explain why and provide your improved answer.
                """
                groq_current = call_groq(groq_prompt)
                
                if "CONSENSUS REACHED" in groq_current.upper():
                    st.success("🎉 Consensus Reached! Both models agree.")
                    break
                st.markdown(f"**🟢 Llama 3 (Groq):**\n{groq_current}")

            # Gemini evaluates Llama
            with st.spinner("Gemini is reviewing Llama's corrections..."):
                gemini_prompt = f"""
                The user asked: "{user_question}".
                Another AI answered: "{groq_current}".
                Review this answer. Use Google Search to verify their claims.
                If you completely agree and it is 100% accurate, reply ONLY with 'CONSENSUS REACHED'. 
                If you disagree or have additions, explain why and provide your improved answer.
                """
                gemini_current = call_gemini(gemini_prompt)
                
                if "CONSENSUS REACHED" in gemini_current.upper():
                    st.success("🎉 Consensus Reached! Both models agree.")
                    break
                st.markdown(f"**🔵 Gemini:**\n{gemini_current}")
                
        st.info("Debate concluded.")
