import asyncio
import json
import os
from groq import Groq
from dotenv import load_dotenv
from .mcp_client import MCPClient

load_dotenv()


def groq_call(content: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": content}],
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


def extract_action(json_text):
    try:
        data = json.loads(json_text)
        if isinstance(data, dict) and "tool" in data and "args" in data:
            return data["tool"], data["args"]
        return None
    except Exception:
        return None



def convert_to_human_explainable(diagnosis):
    prompt = (
        "You are a vehicle customer care assistant. "
        "Explain the following JSON diagnosis in clear, simple, and friendly Hindi, "
        "Hindi written ONLY in English alphabets (Roman Hindi / Hinglish). "
        "DO NOT use Devanagari script. "
        "Use simple, conversational Hindi, but do not translate technical terms. "
        "Keep words like 'Excessive vibration', 'bearing_fault' in English. "
        "Be concise:\n"
        f"{diagnosis}"
    )
    return groq_call(prompt)




def memory_to_groq_messages(memory):
    messages = []
    for turn in memory:
        role = "assistant" if turn["role"] == "agent" else "user"
        messages.append({"role": role, "content": turn["text"]})
    return messages


async def prepare_initial_conversation(diagnosis_json):
    human_text = convert_to_human_explainable(diagnosis_json)

    conversational_memory = []

    initial_agent_text = (
        "Namaste! Main Customer Service se baat kar raha hoon.\n\n"
        f"{human_text}\n\n"
        "Aap service ke liye kab aa sakte hai?"
    )

    conversational_memory.append({
        "role": "agent",
        "text": initial_agent_text
    })

    return {
        "memory": conversational_memory,
        "agent_message": initial_agent_text
    }


async def process_user_turn(conversational_memory, user_input):
    conversational_memory.append({"role": "user", "text": user_input})

    messages = memory_to_groq_messages(conversational_memory)

    # Insert system instructions
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

    # Ask Groq what to do
    agent_reply = groq_chat(messages).strip()

    # Detect tool call
    action = None
    if agent_reply.startswith("Action:"):
        json_part = agent_reply[len("Action:"):].strip()
        action = extract_action(json_part)

    if action:
        tool_name, args = action

        mcp = MCPClient()
        await mcp.connect("tools/tool.py")
        tool_result = await mcp.call_tool(tool_name, args)
        await mcp.cleanup()

        # Ask Groq for final answer
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

        conversational_memory.append({"role": "agent", "text": final_reply})

        return {
            "memory": conversational_memory,
            "agent_message": final_reply,
            "tool_used": tool_name,
            "tool_result": tool_result
        }

    final_text = agent_reply
    if final_text.startswith("Final Answer:"):
        final_text = final_text[len("Final Answer:"):].strip()

    conversational_memory.append({"role": "agent", "text": final_text})

    return {
        "memory": conversational_memory,
        "agent_message": final_text,
        "tool_used": None
    }
