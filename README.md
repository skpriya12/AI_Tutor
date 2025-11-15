"""
# 🎓 AI Tutor — Interactive Learning Assistant

An **AI-powered tutor** built with **Google Gemini** and **Google ADK (AI Developer Kit)**.
It explains topics, generates quizzes, and provides personalized feedback — all in one interactive session.

---

##  Features
✅ Lesson explanations with examples  
✅ Interactive quizzes  
✅ Personalized feedback reports  
✅ Powered by Gemini 2.5 Flash Lite  
✅ Async and stateful session handling  

---

##  Architecture
The system uses a SequentialAgent pipeline:
LessonAgent ➜ QuizAgent ➜ FeedbackAgent

| Agent | Function |
|--------|-----------|
| LessonAgent | Explains a given topic |
| QuizAgent | Generates and evaluates quizzes |
| FeedbackAgent | Summarizes performance and gives recommendations |

---

##  Installation

1️⃣ Clone this repo:
```bash
git clone https://github.com/your-username/ai-tutor.git
cd ai-tutor
