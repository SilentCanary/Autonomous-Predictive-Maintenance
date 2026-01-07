import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
def groq_call(content):
    client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": content,
        }
    ],
    model="meta-llama/llama-4-scout-17b-16e-instruct",
)
    return (chat_completion.choices[0].message.content)

def groq_chat(messages):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages
    )
    return chat_completion.choices[0].message.content

def convert_to_human_explainable(diagnosis):
    prompt = (
        "You are a vehicle customer care assistant. "
        "Explain the following JSON diagnosis in clear, simple, and friendly Hindi, "
        "Hindi written ONLY in English alphabets (Roman Hindi / Hinglish). "
        "DO NOT use Devanagari script. "
        "Use simple, conversational Hindi, but **do not translate technical terms**. "
        "Keep words like 'Excessive vibration', 'bearing_fault' in English. "
        "as if you are talking to the vehicle owner. "
        "Be concise and human-understandable:\n"
        f"{diagnosis}"
    )
    return groq_call(prompt)

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


def memory_to_groq_messages(memory):
    messages = []
    for turn in memory:
        role = "assistant" if turn["role"] == "agent" else "user"
        messages.append({
            "role": role,
            "content": turn["text"]
        })
    return messages

def customer_conversation_loop(conversational_memory):
    print("\n--- Conversation Started (type 'exit' to stop) ---\n")
    is_first_turn=True
    while True:
        # Simulated owner speech → text
        owner_input = input("Owner: ")
        if owner_input.lower() == "exit":
            break

        # Store owner reply
        conversational_memory.append({
            "role": "user",
            "text": owner_input
        })
        if is_first_turn:     
            initial_agent_turn = generate_initial_agent_turn(conversational_memory)

            conversational_memory.append({
                "role": "agent",
                "text": initial_agent_turn
            })

            print("\nAgent:", initial_agent_turn, "\n")
            is_first_turn=False

        else:
            # Build prompt using full memory
            messages = memory_to_groq_messages(conversational_memory)

            # Add system steering INSIDE prompt (Groq has no system role)
            messages.insert(0, {
                "role": "user",
                "content": (
                    "You are a vehicle customer care agent. "
                    "Talk in simple conversational Hindi with English technical words. "
                    "Your goal is to politely schedule a service appointment "
                    "by asking date, time, and service center. "
                    "Do NOT repeat the diagnosis again unless asked."
                )
            })
            agent_reply = groq_chat(messages)

            # Print agent reply
            print("\nAgent:", agent_reply, "\n")

            # Store agent reply
            conversational_memory.append({
                "role": "agent",
                "text": agent_reply
            })


diagnosis = {
    "diagnosis": "bearing_fault",
    "confidence": 0.9,
    "explanation": "Excessive vibration detected"
}

human_text = convert_to_human_explainable(diagnosis)
conversational_memory=[]
initial_agent_text = f"Namaste! Main Customer Service se baat kar raha hoon.\n\n{human_text}\n\nAap service ke liye kab aa sakte hai?."

conversational_memory.append({"role": "agent", "text": initial_agent_text})

customer_conversation_loop(conversational_memory)
