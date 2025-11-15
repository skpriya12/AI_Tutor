import os
import asyncio
import gradio as gr
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

# ==========================================================
# 1️⃣ Gemini API Key Setup
# ==========================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "🔑 GOOGLE_API_KEY not found. Please export it or set it in your environment variables."
    )
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
print("✅ Gemini API key setup complete.")


# ==========================================================
# 2️⃣ Define Agents
# ==========================================================
LessonAgent = Agent(
    name="LessonAgent",
    model=Gemini(model="gemini-2.5-flash-lite"),
    instruction="""You are an educational tutor who explains topics clearly and concisely.
    Provide examples and key takeaways when relevant.""",
    tools=[google_search],
    output_key="lesson_findings",
)
print("✅ LessonAgent created.")

quiz_agent = Agent(
    name="quiz_agent",
    model=Gemini(model="gemini-2.5-flash-lite"),
    instruction="""You are an interactive Quiz Tutor.
    Use {lesson_findings} to generate 3 to 5 multiple-choice questions.
    Present one question at a time, grade the student’s answers, and summarize results at the end.""",
    output_key="quiz_summary",
)
print("✅ QuizAgent created.")

feedback_agent = Agent(
    name="feedback_agent",
    model=Gemini(model="gemini-2.5-flash-lite"),
    instruction="""You are a learning feedback coach.
    Use {quiz_summary} to produce a report with:
    - Performance Summary
    - Strengths
    - Areas for Improvement
    - Recommended Difficulty
    - Next Topics
    - Final Message""",
    output_key="feedback_summary",
)
print("✅ FeedbackAgent created.")


# ==========================================================
# 3️⃣ Sequential AI Tutor Pipeline
# ==========================================================
root_agent = SequentialAgent(
    name="AITutorPipeline",
    sub_agents=[LessonAgent, quiz_agent, feedback_agent],
)
print("✅ Sequential AI Tutor pipeline created.\n")


# ==========================================================
# 4️⃣ Session & Runner Setup
# ==========================================================
APP_NAME = "AITutorApp"
USER_ID = "student_001"
SESSION_ID = "gradio_session"
MODEL_NAME = "gemini-2.5-flash-lite"

session_service = InMemorySessionService()
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

# Proper async session setup
async def setup_session():
    try:
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID
        )
    except Exception:
        await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID
        )

# initialize session
asyncio.run(setup_session())
print("✅ Session initialized.")


# ==========================================================
# 5️⃣ Tutor Logic (Async)
# ==========================================================
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

async def ai_tutor_pipeline(user_input):
    """Async AI Tutor conversation handler."""
    query = types.Content(role="user", parts=[types.Part(text=user_input)])
    response_texts = []

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=query,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                text = getattr(part, "text", None)
                if text and text.strip() and text != "None":
                    response_texts.append(text.strip())

    return "\n\n".join(response_texts)


def run_tutor_sync(topic: str):
    """Safe synchronous wrapper for Gradio."""
    try:
        coro = ai_tutor_pipeline(topic)
        result = asyncio.run_coroutine_threadsafe(coro, loop).result()
        return result
    except Exception as e:
        return f"⚠️ Error during AI Tutor run: {e}"


# ==========================================================
# 6️⃣ Gradio Interface
# ==========================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 AI Tutor — Interactive Learning Assistant")
    gr.Markdown("Learn any topic with explanations, quizzes, and feedback — powered by Gemini 💡")

    topic_input = gr.Textbox(
        label="Enter a topic to learn (e.g., Newton’s Laws, Photosynthesis, Python Basics)",
        placeholder="Type your topic here...",
    )
    output_box = gr.Textbox(
        label="AI Tutor Output",
        lines=25,
        placeholder="Lesson, quiz, and feedback will appear here...",
    )

    learn_button = gr.Button("Start Learning 🚀")
    learn_button.click(fn=run_tutor_sync, inputs=topic_input, outputs=output_box)

# ==========================================================
# 7️⃣ Launch
# ==========================================================
if __name__ == "__main__":
    demo.queue()  # Enables async-safe handling
    demo.launch()
