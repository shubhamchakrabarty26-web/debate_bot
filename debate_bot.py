import streamlit as st
from google import genai
from groq import Groq
import traceback
import time

# --- Configuration & Secrets Handling ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("🚨 Missing API Keys! Please add GEMINI_API_KEY and GROQ_API_KEY to your Streamlit App settings under 'Secrets'.")
    st.stop()

try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"🚨 Failed to initialize API clients: {e}")
    st.stop()

st.title("🤖 Paced AI Debate: Gemini 2.5 Flash Lite vs. Llama 3.3")

# --- Helper Functions & Prompts ---
def wait_with_countdown(seconds):
    """Displays a visual countdown to prevent the user from thinking the app froze."""
    countdown_placeholder = st.empty()
    for i in range(seconds, 0, -1):
        countdown_placeholder.info(f"⏳ Pacing API calls: Waiting {i} seconds for the next AI's turn...")
        time.sleep(1)
    countdown_placeholder.empty()

# Strong directive added to force absolute truth and prevent hallucination/fluff
truth_directive = """
CRITICAL INSTRUCTION: Provide the absolute hard truth and highly relevant information. 
Do NOT give generic, safe, or fluffy advice. Do NOT hallucinate or make up facts. 
If you do not know the answer to something, state honestly that you do not know.
"""

def call_gemini(prompt):
    try:
        response = gemini_client.models.generate_content(
            model='gemma-3-27b-it',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        st.error(f"🚨 Gemini API Error: {e}")
        return None

def call_groq(prompt):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"🚨 Groq API Error: {e}")
        st.expander("Show detailed error log").code(traceback.format_exc())
        return None

# --- UI & Debate Logic ---
user_question = st.text_area("Ask a complex question to start the debate:", height=150)
max_rounds = st.number_input("Maximum debate rounds", min_value=1, max_value=8, value=8)

if st.button("Start Debate"):
    if not user_question:
        st.warning("Please enter a question.")
    else:
        st.write(f"**Original Question:** {user_question}")
        st.divider()

        debate_history = f"Original User Question: {user_question}\n\n"
        consensus_reached = False

        # --- Initial Answer by Gemini ---
        with st.spinner("Gemini is formulating the initial response..."):
            initial_prompt = f"{truth_directive}\n\nAnswer this question thoroughly and state your stance: {user_question}"
            gemini_current = call_gemini(initial_prompt)
            
        if gemini_current:
            st.markdown(f"**🔵 Gemini (Initial):**\n{gemini_current}")
            debate_history += f"Gemini's Initial Answer: {gemini_current}\n\n"
            current_reply = gemini_current
            
            # --- The Debate Loop ---
            for i in range(max_rounds):
                st.write(f"### Round {i+1}")
                
                wait_with_countdown(15)
                
                # 1. Llama's Turn
                with st.spinner("Llama 3 is analyzing the full history and debating..."):
                    groq_prompt = f"""
                    {truth_directive}
                    
                    Here is the ENTIRE debate history from the very beginning:
                    ---
                    {debate_history}
                    ---
                    
                    The other AI just made this specific point: "{current_reply}"
                    
                    Your task: Evaluate their latest point within the context of the entire history. 
                    - If you completely agree with everything said and have nothing to add or correct, reply EXACTLY and ONLY with 'CONSENSUS REACHED'.
                    - If you disagree, find flaws, detect generic fluff, or have a different perspective, explain why and provide your counter-argument.
                    """
                    groq_current = call_groq(groq_prompt)
                
                if not groq_current:
                    st.error("Debate stopped due to Llama 3 (Groq) error.")
                    break
                    
                if "CONSENSUS REACHED" in groq_current.upper():
                    st.success("🎉 Consensus Reached! Llama 3 agrees with Gemini.")
                    consensus_reached = True
                    break
                    
                st.markdown(f"**🟢 Llama 3 (Groq):**\n{groq_current}")
                debate_history += f"Llama 3's Rebuttal (Round {i+1}): {groq_current}\n\n"
                current_reply = groq_current

                wait_with_countdown(15)

                # 2. Gemini's Turn
                with st.spinner("Gemini is analyzing the full history and Llama's rebuttal..."):
                    gemini_prompt = f"""
                    {truth_directive}
                    
                    Here is the ENTIRE debate history from the very beginning:
                    ---
                    {debate_history}
                    ---
                    
                    The other AI just made this specific point: "{current_reply}"
                    
                    Your task: Evaluate their latest point within the context of the entire history. 
                    - If you completely agree with everything said and have nothing to add or correct, reply EXACTLY and ONLY with 'CONSENSUS REACHED'.
                    - If you disagree, find flaws, detect generic fluff, or have a different perspective, explain why and provide your counter-argument.
                    """
                    gemini_current = call_gemini(gemini_prompt)
                
                if not gemini_current:
                    st.error("Debate stopped due to Gemini error.")
                    break

                if "CONSENSUS REACHED" in gemini_current.upper():
                    st.success("🎉 Consensus Reached! Gemini agrees with Llama 3.")
                    consensus_reached = True
                    break
                    
                st.markdown(f"**🔵 Gemini:**\n{gemini_current}")
                debate_history += f"Gemini's Rebuttal (Round {i+1}): {gemini_current}\n\n"
                current_reply = gemini_current
                
            # --- Final Master Summary (Always Runs) ---
            st.divider()
            st.write("### Preparing Final Master Analysis...")
            wait_with_countdown(15)
            
            with st.spinner("Gemini is analyzing the entire debate to compile the master reply..."):
                
                # Adjust the summary prompt depending on how the debate ended
                if consensus_reached:
                    status_context = "The models eventually reached a 100% consensus."
                    disagreement_instructions = "* **Estimated Disagreement:** 0%\n* **Core Disagreements:** None. The models resolved their differences and reached a full consensus."
                else:
                    status_context = f"The models debated for {max_rounds} rounds but did NOT reach a full consensus."
                    disagreement_instructions = "* **Estimated Disagreement:** [Insert a percentage, e.g., 20%]\n* **Core Disagreements:** [List bullet points of the exact specific areas where they fundamentally disagreed at the end]"

                summary_prompt = f"""
                {truth_directive}
                
                Analyze the following debate history between two AI models regarding the user's original question:
                ---
                {debate_history}
                ---
                
                {status_context}
                
                Provide a final, authoritative summary formatted EXACTLY like this:
                
                ### Disagreement Analysis
                {disagreement_instructions}
                
                ### Master Reply
                [Provide a synthesized, highly accurate final response. Base this on the absolute truth extracted from the debate. Filter out any fluff, generic advice, or potential hallucinations. Give the user the ultimate, factual, and most realistic answer to their original question.]
                """
                final_summary = call_gemini(summary_prompt)
                
                if final_summary:
                    st.info("### 📊 Final Disagreement Summary & Master Reply")
                    st.markdown(final_summary)

        st.info("Debate concluded.")
