import os
import asyncio
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

# ===============================
# 1. Gemini API Key Setup
# ===============================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ Gemini API key setup complete.")
else:
    print("🔑 GOOGLE_API_KEY not found. Please export or set it in PyCharm first.")
    exit(1)


# ===============================
# 2. Define Agents
# ===============================

LessonAgent = Agent(
    name="LessonAgent",
    model=Gemini(model="gemini-2.5-flash-lite"),
    instruction=(
        "You are an educational tutor who explains topics clearly and concisely. "
        "Provide examples and key takeaways when relevant."
    ),
    tools=[google_search],
    output_key="lesson_findings",
)
print("✅ LessonAgent created.")

quiz_agent = Agent(
    name="quiz_agent",
    model=Gemini(model="gemini-2.5-flash-lite"),
    instruction=(
        "You are an interactive Quiz Tutor. "
        "Use {lesson_findings} to generate 3 to 5 multiple-choice questions. "
        "Present one question at a time, grade the student’s answers, "
        "and summarize results at the end."
    ),
    output_key="quiz_summary",
)
print("✅ QuizAgent created.")

feedback_agent = Agent(
    name="feedback_agent",
    model=Gemini(model="gemini-2.5-flash-lite"),
    instruction=(
        "You are a learning feedback coach. "
        "Use {quiz_summary} to produce a report with:\n"
        "- Performance Summary\n"
        "- Strengths\n"
        "- Areas for Improvement\n"
        "- Recommended Difficulty\n"
        "- Next Topics\n"
        "- Final Message"
    ),
    output_key="feedback_summary",
)
print("✅ FeedbackAgent created.")


# ===============================
# 3. Sequential Pipeline
# ===============================

root_agent = SequentialAgent(
    name="AITutorPipeline",
    sub_agents=[LessonAgent, quiz_agent, feedback_agent],
)
print("✅ Sequential AI Tutor pipeline created.\n")


# ===============================
# 4. Async Main Function
# ===============================

APP_NAME = "AITutorApp"
USER_ID = "student_001"
SESSION_ID = "debug_session_id"
MODEL_NAME = "gemini-2.5-flash-lite"

session_service = InMemorySessionService()
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)


async def ai_tutor_main():
    """Async main to manage session and tutor conversation."""
    print("✅ Stateful AI Tutor initialized!")
    print(f"   - Application: {APP_NAME}")
    print(f"   - User: {USER_ID}")
    print(f"   - Model: {MODEL_NAME}\n")

    # ✅ Create async session safely
    try:
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
        )
    except Exception:
        await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
        )

    print("AI Tutor: Hi! What topic do you want to learn?")
    print("   (Type 'exit' or 'quit' to end the session)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Session ended.")
            break

        try:
            query = types.Content(role="user", parts=[types.Part(text=user_input)])

            async for event in runner.run_async(
                user_id=USER_ID,
                session_id=SESSION_ID,
                new_message=query,
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        text = getattr(part, "text", None)
                        if text and text.strip() and text != "None":
                            print(f"AI Tutor: {text.strip()}\n")

        except Exception as e:
            print(f"⚠️ Error during conversation: {e}")


# ===============================
# 5. Launch Async Loop
# ===============================
if __name__ == "__main__":
    asyncio.run(ai_tutor_main())
