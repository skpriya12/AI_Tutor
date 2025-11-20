import os
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import asyncio
import gradio as gr

from google.adk.agents import Agent
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
    print("🔑 GOOGLE_API_KEY not found. Please set it before running.")

MODEL_NAME = "gemini-2.5-flash"

# ===============================
# 2. Memory Service
# ===============================


class MemoryService:
    def __init__(self, base_dir="memory"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, student_id: str):
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", student_id)
        return os.path.join(self.base_dir, f"{safe}.json")

    def load(self, student_id: str):
        p = self._path(student_id)
        if not os.path.exists(p):
            return {
                "student_id": student_id,
                "difficulty": "medium",
                "topics": {},
                "history": [],
            }
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {
                "student_id": student_id,
                "difficulty": "medium",
                "topics": {},
                "history": [],
            }

    def save(self, student_id: str, data: Dict[str, Any]):
        p = self._path(student_id)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def update_after_quiz(self, student_id: str, topic: str, results: dict):
        mem = self.load(student_id)
        score = results.get("score", 0)
        total = results.get("total", 1)
        pct = results.get("percentage", 0)

        topics = mem.setdefault("topics", {})
        t = topics.get(
            topic,
            {
                "attempts": 0,
                "avg_score": 0.0,
                "last_score": 0,
                "last_percentage": 0.0,
            },
        )

        t["attempts"] += 1
        t["last_score"] = score
        t["last_percentage"] = pct

        prev_sum = t["avg_score"] * (t["attempts"] - 1)
        t["avg_score"] = (prev_sum + pct) / t["attempts"]

        topics[topic] = t
        mem["topics"] = topics

        # Adjust global difficulty
        if pct >= 85:
            mem["difficulty"] = "hard"
        elif pct >= 60:
            mem["difficulty"] = "medium"
        else:
            mem["difficulty"] = "easy"

        mem.setdefault("history", []).append(
            {
                "topic": topic,
                "score": score,
                "total": total,
                "percentage": pct,
            }
        )

        self.save(student_id, mem)

    def summarize_for_prompt(self, student_id: str, topic: Optional[str] = None):
        mem = self.load(student_id)
        difficulty = mem.get("difficulty", "medium")
        history = mem.get("history", [])
        topics = mem.get("topics", {})

        if not history:
            return f"Student is new. Default difficulty: {difficulty}. No quiz data yet."

        overall_avg = sum(h.get("percentage", 0) for h in history) / len(history)
        out = [
            f"Overall difficulty: {difficulty}.",
            f"Average quiz score: {overall_avg:.1f}%.",
            f"Quizzes taken: {len(history)}.",
        ]
        if topic and topic in topics:
            t = topics[topic]
            out.append(
                f"For topic '{topic}': attempts={t.get('attempts', 0)}, "
                f"avg={t.get('avg_score', 0.0):.1f}%, "
                f"last={t.get('last_percentage', 0.0):.1f}%."
            )
        elif topic:
            out.append(f"No prior quiz data for topic '{topic}'.")
        return " ".join(out)


memory_service = MemoryService()

# ===============================
# 3. Define Agents
# ===============================

LessonAgent = Agent(
    name="LessonAgent",
    model=Gemini(model=MODEL_NAME),
    instruction=(
        "You are an educational tutor. Provide:\n"
        "- Simple definition\n"
        "- Deeper explanation\n"
        "- One example\n"
        "- 3–5 key takeaways\n"
        "Adapt depth to student profile."
    ),
    tools=[google_search],
)
print("✅ LessonAgent created.")

quiz_agent = Agent(
    name="QuizAgent",
    model=Gemini(model=MODEL_NAME),
    instruction=(
        "You are a Quiz Generator.\n"
        "You receive a lesson text and student profile.\n"
        "Generate exactly 4 multiple-choice questions (MCQs) in pure JSON.\n"
        "NO backticks, NO markdown.\n"
        "Format:\n"
        "[\n"
        "  {\n"
        "    \"question\": \"...\",\n"
        "    \"options\": {\"A\":\"...\",\"B\":\"...\",\"C\":\"...\",\"D\":\"...\"},\n"
        "    \"correct_answer\": \"A\"\n"
        "  },\n"
        "  ... (4 questions total)\n"
        "]"
    ),
)
print("✅ QuizAgent created.")

grader_agent = Agent(
    name="GraderAgent",
    model=Gemini(model=MODEL_NAME),
    instruction=(
        "You are a Grader.\n"
        "Input: quiz JSON including user answers.\n"
        "Output ONLY JSON, no markdown, of the form:\n"
        "{\n"
        "  \"score\": 3,\n"
        "  \"total\": 4,\n"
        "  \"percentage\": 75,\n"
        "  \"per_question\": [\n"
        "    {\n"
        "      \"question\": \"...\",\n"
        "      \"correct_answer\": \"A\",\n"
        "      \"user_answer\": \"B\",\n"
        "      \"is_correct\": false\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}"
    ),
)
print("✅ GraderAgent created.")

feedback_agent = Agent(
    name="FeedbackAgent",
    model=Gemini(model=MODEL_NAME),
    instruction=(
        "You are a Feedback Coach.\n"
        "You get grading JSON and a student profile summary.\n"
        "Provide concise, friendly feedback:\n"
        "- Score summary (e.g., 'You scored 3/4 (75%).')\n"
        "- Strengths\n"
        "- Specific areas for improvement\n"
        "- 1–3 suggested next topics\n"
    ),
)
print("✅ FeedbackAgent created.")

# ===============================
# 4. Runners (ADK 1.1.1 – keyword args)
# ===============================

APP_NAME = "AITutorApp"
USER_ID = "student_001"
SESSION_ID = "gradio_session"

session_service = InMemorySessionService()

lesson_runner = Runner(
    agent=LessonAgent,
    app_name=APP_NAME,
    session_service=session_service,
)
quiz_runner = Runner(
    agent=quiz_agent,
    app_name=APP_NAME,
    session_service=session_service,
)
grader_runner = Runner(
    agent=grader_agent,
    app_name=APP_NAME,
    session_service=session_service,
)
feedback_runner = Runner(
    agent=feedback_agent,
    app_name=APP_NAME,
    session_service=session_service,
)


# Create session at startup
async def _init_session():
    try:
        await session_service.delete_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
    except Exception:
        pass
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )


asyncio.run(_init_session())
print("🟢 ADK session initialized:", SESSION_ID)

# ===============================
# 5. Helpers
# ===============================


async def run_agent_collect_text(runner: Runner, text: str) -> str:
    """Collects streaming text and ignores all non-text ADK events."""
    q = types.Content(role="user", parts=[types.Part(text=text)])
    collected: List[str] = []
    async for ev in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=q,
    ):
        if ev.content and ev.content.parts:
            for part in ev.content.parts:
                t = getattr(part, "text", "")
                if isinstance(t, str) and t.strip():
                    collected.append(t)
    return "".join(collected).strip()


def clean_json(text: str) -> str:
    return re.sub(r"```(?:json)?|```", "", text).strip()


def parse_json_safely(text: str):
    return json.loads(clean_json(text))


def format_question(q: Dict[str, Any], idx: int) -> str:
    question = q.get("question", "No question text.")
    options = q.get("options", {})
    out = [f"Question {idx}: {question}"]
    for opt in ["A", "B", "C", "D"]:
        if opt in options:
            out.append(f"{opt}) {options[opt]}")
    out.append("Your answer (A/B/C/D)?")
    return "\n".join(out)


# ===============================
# 6. TutorState
# ===============================


@dataclass
class TutorState:
    phase: str = "await_topic"  # "await_topic" | "quiz"
    topic: Optional[str] = None
    lesson: Optional[str] = None
    quiz: List[dict] = field(default_factory=list)
    q_index: int = 0


# ===============================
# 7. Chat Logic
# ===============================


async def chat_fn(message, history, state: Optional[TutorState]):

    user_msg = (message or "").strip()
    if state is None:
        state = TutorState()

    # ---------------------------
    # PHASE 1 — USER GIVES TOPIC
    # ---------------------------
    if state.phase == "await_topic":
        topic = user_msg
        state.topic = topic

        profile = memory_service.summarize_for_prompt(USER_ID, topic)

        # 1) LESSON
        lesson_prompt = f"Student profile:\n{profile}\n\nTeach topic: {topic}"
        lesson = await run_agent_collect_text(lesson_runner, lesson_prompt)
        state.lesson = lesson

        # 2) QUIZ
        quiz_prompt = (
            f"Student profile:\n{profile}\n\n"
            f"Lesson:\n{lesson}\n\n"
            "Generate a quiz now (4 MCQs, pure JSON)."
        )
        quiz_raw = await run_agent_collect_text(quiz_runner, quiz_prompt)
        try:
            quiz_items = parse_json_safely(quiz_raw)
            if not isinstance(quiz_items, list) or len(quiz_items) != 4:
                raise ValueError("QuizAgent did not return a list of 4 questions.")
        except Exception as e:
            assistant_reply = (
                "I had trouble generating a quiz for that lesson.\n"
                f"Technical details: {e}"
            )
            history.append([user_msg, assistant_reply])
            return history, state

        state.quiz = quiz_items
        state.q_index = 0
        state.phase = "quiz"

        first_q = format_question(state.quiz[0], 1)
        assistant_reply = (
            lesson
            + "\n\nNow let's try a short quiz:\n\n"
            + first_q
        )

        history.append([user_msg, assistant_reply])
        return history, state

    # ---------------------------
    # PHASE 2 — QUIZ
    # ---------------------------
    if state.phase == "quiz":
        ans = user_msg.upper()
        if ans not in ["A", "B", "C", "D"]:
            # Re-ask same question
            q = format_question(state.quiz[state.q_index], state.q_index + 1)
            history.append([user_msg, f"Please answer A/B/C/D.\n\n{q}"])
            return history, state

        # Record user answer
        state.quiz[state.q_index]["user_answer"] = ans

        # If more questions remain
        if state.q_index < len(state.quiz) - 1:
            state.q_index += 1
            next_q = format_question(state.quiz[state.q_index], state.q_index + 1)
            history.append([user_msg, "Got it!\n\n" + next_q])
            return history, state

        # Last question: grade with GraderAgent, but normalize ourselves
        grade_in = json.dumps(state.quiz, ensure_ascii=False)
        grade_raw = await run_agent_collect_text(grader_runner, grade_in)

        try:
            # Try to parse, but we will recompute per_question & core stats
            _ = parse_json_safely(grade_raw)
        except Exception:
            # If GraderAgent output is garbage, we still compute locally
            pass

        # 🔒 Robust grading: compute everything from our own quiz data
        per_question: List[Dict[str, Any]] = []
        score = 0
        total = len(state.quiz) if state.quiz else 0

        for item in state.quiz:
            q_text = item.get("question", "")
            correct = item.get("correct_answer")
            user_answer = item.get("user_answer")
            is_correct = bool(user_answer and correct and user_answer == correct)
            if is_correct:
                score += 1
            per_question.append(
                {
                    "question": q_text,
                    "correct_answer": correct,
                    "user_answer": user_answer,
                    "is_correct": is_correct,
                }
            )

        percentage = int(round(score * 100 / total)) if total > 0 else 0

        grade_norm = {
            "score": score,
            "total": total,
            "percentage": percentage,
            "per_question": per_question,
        }

        # Update memory based on normalized grade
        if state.topic:
            memory_service.update_after_quiz(USER_ID, state.topic, grade_norm)

        # Build score summary
        lines = [f"You scored {score}/{total} ({percentage}%).", ""]
        lines.append("Per-question results:")
        for i, pq in enumerate(per_question, start=1):
            ok = pq.get("is_correct", False)
            status = "✅ Correct" if ok else "❌ Incorrect"
            ua = pq.get("user_answer", "?")
            ca = pq.get("correct_answer", "?")
            q_text = pq.get("question", "")
            lines.append(f"Q{i}: {status}")
            lines.append(f"  Your answer: {ua}")
            lines.append(f"  Correct answer: {ca}")
            if q_text:
                lines.append(f"  Question: {q_text}")
            lines.append("")

        score_report = "\n".join(lines)

        # Feedback
        profile_updated = memory_service.summarize_for_prompt(USER_ID, state.topic)
        fb_in = (
            f"Grading JSON:\n{json.dumps(grade_norm, ensure_ascii=False)}\n\n"
            f"Student profile:\n{profile_updated}"
        )
        feedback = await run_agent_collect_text(feedback_runner, fb_in)

        assistant_reply = (
            score_report
            + "\n"
            + feedback
            + "\n\nAsk for another topic anytime!"
        )

        # Reset state
        new_state = TutorState()

        history.append([user_msg, assistant_reply])
        return history, new_state

    # ---------------------------
    # FALLBACK
    # ---------------------------
    history.append([user_msg, "Let's begin. What topic do you want to study?"])
    return history, TutorState()


# ===============================
# 8. Gradio UI (3.x style)
# ===============================

with gr.Blocks() as demo:
    gr.Markdown("#  AI Tutor with Memory")

    chatbot = gr.Chatbot(label="AI Tutor")  # expects list[list[user, assistant]]
    state = gr.State(TutorState())
    txt = gr.Textbox(label="Your message", lines=2)

    async def on_submit(msg, hist, st):
        return await chat_fn(msg, hist, st)

    txt.submit(on_submit, [txt, chatbot, state], [chatbot, state])
    send = gr.Button("Send")
    send.click(on_submit, [txt, chatbot, state], [chatbot, state])

    clear = gr.Button("Reset")
    clear.click(lambda: ([], TutorState()), outputs=[chatbot, state])

if __name__ == "__main__":
    demo.launch()
