import os
import json
import asyncio
from groq import Groq
from dotenv import load_dotenv
from .tools.tools_registry import (TOOL_REGISTRY)
import azure.cognitiveservices.speech as speechsdk
load_dotenv()

SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")  
SPEECH_REGION = os.getenv("AZURE_SERVICE_REGION")
MODE = "text"   


# SPEECH FUNCTIONS
def speak_text(text: str):
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = "hi-IN-SwaraNeural"  # natural Hindi voice
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # Blocking call: waits until audio is fully spoken
    result = synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("✅ Agent finished speaking")
    else:
        print("⚠️ Speech synthesis issue:", result.reason)


def listen_and_transcribe() -> str:
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = "hi-IN"
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("🎤 Listening... speak whenever you're ready")

    done = False
    final_text = ""

    def stop_cb(evt):
        nonlocal done
        print("🛑 STT session stopped:", evt)
        done = True

    def recognized_cb(evt):
        nonlocal final_text
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Owner said:", evt.result.text)
            final_text = evt.result.text
            # stop after first full sentence
            recognizer.stop_continuous_recognition()

    # Hook events
    recognizer.recognized.connect(recognized_cb)
    recognizer.session_stopped.connect(stop_cb)
    recognizer.canceled.connect(stop_cb)

    # Start continuous recognition
    recognizer.start_continuous_recognition()
    while not done:
        asyncio.sleep(0.5)  # let events process

    return final_text

# GROQ HELPERS
def groq_call(content: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    return chat_completion.choices[0].message.content


def groq_chat(messages):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages
    )
    return chat_completion.choices[0].message.content


# TOOL ACTION PARSING

def extract_action(json_text):
    try:
        data = json.loads(json_text)
        if isinstance(data, dict) and "tool" in data and "args" in data:
            return data["tool"], data["args"]
        return None
    except Exception:
        return None


# DIAGNOSIS - HUMAN TEXT

def convert_to_human_explainable(diagnosis):
 
    prompt = (
        "You are a vehicle customer care assistant. "
        "Explain the following JSON diagnosis in clear, simple, and friendly Hindi, "
        "Hindi written ONLY in English alphabets (Roman Hindi / Hinglish). "
        "DO NOT use Devanagari script. "
        "Use simple, conversational Hindi, but do not translate technical terms. "
        "Keep words like 'Excessive vibration', 'bearing_fault' in English, "
        "as if you are talking to the vehicle owner. "
        "Be concise and human-understandable:\n"
        f"{diagnosis}"
    )
    return groq_call(prompt)



# INITIAL AGENT TURN 

def generate_initial_agent_turn(conversational_memory):

    messages = [
        {
            "role": "user",
            "content": (
                "You are a vehicle customer care agent calling the vehicle owner.\n"
                "Start the call by:\n"
                "1. Introducing yourself as Customer Service\n"
                "2. Clearly explaining the vehicle problem using the given context\n"
                "3. Politely asking when they can come for service\n\n"
                "Context:\n"
                f"{conversational_memory[0]['text']}"
            )
        }
    ]
    return groq_chat(messages)



# MEMORY - GROQ FORMAT

def memory_to_groq_messages(memory):
    """
    Converts our simple memory format:
    { 'role': 'agent'/'user', 'text': '...' }
    into Groq chat messages.
    """
    messages = []
    for turn in memory:
        role = "assistant" if turn["role"] == "agent" else "user"
        messages.append({
            "role": role,
            "content": turn["text"]
        })
    return messages


# MAIN LOOP
async def customer_conversation_loop(conversational_memory):

    print("\n--- Conversation Started (type 'exit' to stop) ---\n")

    while True:

        if MODE == "text":
            owner_input = input("Owner (you): ")
            if owner_input.lower() == "exit":
                break
        else:
            owner_input = listen_and_transcribe()
            print("Owner (heard):", owner_input)
            if owner_input.lower() == "exit":
                break
        # Store owner reply in memory
        conversational_memory.append({
            "role": "user",
            "text": owner_input
        })

        messages = memory_to_groq_messages(conversational_memory)
        messages.insert(0, {
                "role": "user",
                "content": (
                    "You are a vehicle customer care agent. "
                    "Talk in simple conversational Hinglish (Hindi in English letters) "
                    "with English technical words.\n\n"
                    "DO NOT use Devanagari script.\n\n"

                    "You MUST respond in exactly ONE of these formats:\n\n"
                    "1) TOOL CALL (when real-world info is needed):\n"
                    "Action: {\"tool\": \"tool_name\", \"args\": {...}}\n\n"
                    "2) NORMAL REPLY:\n"
                    "Final Answer: <your Hinglish reply>\n\n"

                    "BOOKING LOGIC (VERY IMPORTANT):\n"
                    "- User might mention date/time phrases like 'kal dopahar', 'parso shaam', '12 baje', etc.\n"
                    "- If user gives date/time but NOT city → DO NOT call any tool yet.\n"
                    "  Instead, ask: 'Theek hai, <date/time> samajh gaya. Aap abhi kis sheher mein hain?'\n"
                    "- As soon as user gives a city → ALWAYS call find_center(city).\n"
                    "- find_center returns a service center OBJECT (center_id, name, city, etc).\n"
                    "- You MUST extract and remember center_id from this response.\n"
                    "- For all future steps (slots, booking), use ONLY center_id.\n"
                    "- NEVER use service center name or city for booking.\n"
                    "- After center is found, ask for preferred time if not clearly specified.\n"
                    "- If user only gives a time (e.g., '12 baje') and a date was already discussed earlier,\n"
                    "  assume the same date and DO NOT ask for the date again.\n\n"

                    "DATE & TIME NORMALIZATION RULES:\n"
                    "- Use normalise_date when user mentions relative/ambiguous phrases like:\n"
                    "  kal, parso, aaj, dopahar, shaam, subah, raat, etc.\n"
                    "- normalise_date returns BOTH date and time OR a time_window.\n"
                    "- ALWAYS convert kal/parso into exact date format (YYYY-MM-DD).\n"
                    "- NEVER keep dates internally as words like 'kal' or 'parso'.\n\n"

                    "TOOL USAGE RULES:\n"
                    "- Use find_center ONLY with city.\n"
                    "- ALWAYS call find_center once city is known.\n"
                    "- Use get_slot ONLY when center_id AND normalized date are known.\n"
                    "- Use book_slot ONLY when center_id, normalized date, and exact time are known AND user has agreed.\n"
                    "- book_slot strictly expects center_id — NOT name, NOT city.\n\n"

                    "COMMUNICATION RULES:\n"
                    "- NEVER tell the user that you are calling a tool.\n"
                    "- NEVER say things like 'main tool use kar raha hoon'.\n"
                    "- Speak naturally like a human customer care agent.\n\n"

                    "STRICT RULES:\n"
                    "- NEVER mix Action and Final Answer in the same response.\n"
                    "- NEVER call a tool with empty or missing arguments.\n"
                    "- NEVER invent center_id, dates, or times.\n"
                    "- If ANY required info is missing, ASK the user.\n"
                    "- Always reply ONLY in Hinglish.\n"
                )
            })

        agent_reply = groq_chat(messages).strip()
        print("\nAgent (raw):", agent_reply, "\n")
        action = None
        if agent_reply.startswith("Action:"):
            json_part = agent_reply[len("Action:"):].strip()
            action = extract_action(json_part)
        if action:
            tool_name, args = action
            tool_fn = TOOL_REGISTRY.get(tool_name)
            if not tool_fn:
                raise ValueError(f"Unknown tool: {tool_name}")
            tool_result = tool_fn.invoke(args)
            messages.append({
                "role": "assistant",
                "content": f"Tool {tool_name} result: {tool_result}"
            })
            messages.append({
                "role": "user",
                "content": "Now respond with: Final Answer: <Hinglish reply>"
            })
            final_reply = groq_chat(messages).strip()
            if final_reply.startswith("Final Answer:"):
                final_reply = final_reply[len("Final Answer:"):].strip()
            print("\nAgent:", final_reply, "\n")
            conversational_memory.append({
                "role": "agent",
                "text": final_reply
            })
        else:
            # Normal reply, no tool call
            final_text = agent_reply
            if final_text.startswith("Final Answer:"):
                final_text = final_text[len("Final Answer:"):].strip()
            if MODE == "text":
                print("\nAgent:", final_text, "\n")
            else:
                print("\nAgent (speaking):", final_text, "\n")
                speak_text(final_text)
           
            conversational_memory.append({
                "role": "agent",
                "text": final_text
            })
       
     
           



if __name__ == "__main__":
    import asyncio

    diagnosis = {
        "diagnosis": "bearing_fault",
        "confidence": 0.9,
        "explanation": "Excessive vibration detected"
    }
    human_text = convert_to_human_explainable(diagnosis)

    conversational_memory = []
    initial_agent_text = (
        "==============================\n"
        "🚗 Vehicle Customer Care Agent\n"
        "==============================\n\n"
        f"{human_text}\n\n"
        "Aap service ke liye kab aa sakte hai?"
    )
    conversational_memory.append({"role": "agent", "text": initial_agent_text})

    print(initial_agent_text)

    try:
        asyncio.run(customer_conversation_loop(conversational_memory))
    except KeyboardInterrupt:
        print("\nSession ended by user. Stay safe! 🚗")
