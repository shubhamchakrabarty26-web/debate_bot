import streamlit as st
from google import genai
from groq import Groq
import traceback

# --- Configuration & Secrets Handling ---
try:
    # Pulls keys securely from Streamlit Community Cloud Secrets
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("🚨 Missing API Keys! Please add GEMINI_API_KEY and GROQ_API_KEY to your Streamlit App settings under 'Secrets'.")
    st.stop()

# Initialize API clients
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"🚨 Failed to initialize API clients: {e}")
    st.stop()

st.title("🤖 AI Debate: Gemini 2.5 Flash Lite vs. Llama 3.3")

# --- AI Caller Functions with Error Handling ---
def call_gemini(prompt):
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash-lite',
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

        # We will store the entire conversation here so models can reference past arguments
        debate_history = f"Original User Question: {user_question}\n\n"
        consensus_reached = False

        # --- Initial Answer by Gemini ---
        with st.spinner("Gemini is formulating the initial response..."):
            gemini_current = call_gemini(f"Answer this question thoroughly and state your stance: {user_question}")
            
        if gemini_current:
            st.markdown(f"**🔵 Gemini (Initial):**\n{gemini_current}")
            debate_history += f"Gemini's Initial Answer: {gemini_current}\n\n"
            
            # The "current_reply" is what the next model is specifically reacting to
            current_reply = gemini_current
            
            # --- The Debate Loop ---
            for i in range(max_rounds):
                st.write(f"### Round {i+1}")
                
                # 1. Llama's Turn
                with st.spinner("Llama 3 is analyzing and debating..."):
                    groq_prompt = f"""
                    Here is the debate history so far:
                    {debate_history}
                    
                    The other AI just said: "{current_reply}"
                    
                    Your task: Evaluate their latest point regarding the original question. 
                    - If you completely agree with everything said and have nothing to add, reply EXACTLY and ONLY with 'CONSENSUS REACHED'.
                    - If you disagree, find flaws, or have a different perspective, explain why and provide your counter-argument.
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

                # 2. Gemini's Turn
                with st.spinner("Gemini is analyzing Llama's rebuttal..."):
                    gemini_prompt = f"""
                    Here is the debate history so far:
                    {debate_history}
                    
                    The other AI just said: "{current_reply}"
                    
                    Your task: Evaluate their latest point regarding the original question. 
                    - If you completely agree with everything said and have nothing to add, reply EXACTLY and ONLY with 'CONSENSUS REACHED'.
                    - If you disagree, find flaws, or have a different perspective, explain why and provide your counter-argument.
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
                
            # --- Final Summary (If Max Rounds Reached Without Consensus) ---
            if not consensus_reached:
                st.warning(f"Max limit of {max_rounds} rounds reached without full consensus. Generating disagreement summary...")
                with st.spinner("Calculating disagreement percentage and compiling master reply..."):
                    summary_prompt = f"""
                    Analyze the following debate history between two AI models regarding the user's original question:
                    
                    {debate_history}
                    
                    They did not reach a full consensus. Provide a final summary formatted exactly like this:
                    
                    ### Disagreement Analysis
                    * **Estimated Disagreement:** [Insert a percentage, e.g., 20%]
                    * **Core Disagreements:** [List bullet points of the exact specific areas where they fundamentally disagreed]
                    
                    ### Master Reply
                    [Provide a synthesized final response that takes the best, most accurate, and most realistic points from BOTH sides to give the user the ultimate answer to their original question.]
                    """
                    final_summary = call_gemini(summary_prompt)
                    
                    if final_summary:
                        st.info("### 📊 Final Disagreement Summary & Master Reply")
                        st.markdown(final_summary)

        st.info("Debate concluded.")
