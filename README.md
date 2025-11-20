#  AI Tutor — Interactive Learning Assistant

A **multi-agent AI tutoring system** powered by **Google Gemini**, **Google ADK**, and **Gradio**.

This interactive tutor can:

✔ Explain any topic  
✔ Generate multiple-choice quizzes  
✔ Grade your answers  
✔ Provide personalized feedback  
✔ Remember your progress over time  

---

##  Features

###  Smart Lessons
- Simple definitions  
- Deep explanations  
- Real-world examples  
- Key takeaways  
- Adaptive depth based on student profile  

###  Interactive Quizzes
- Automatically generated from lesson text  
- Always **4 MCQs**  
- Adaptive difficulty using memory  

###  Personalized Feedback
- Strengths & weaknesses  
- Per-question breakdown  
- Recommended next topics  
- Feedback tailored using stored history  

###  Long-Term Memory
Stored per student in `memory/<id>.json`, tracking:

- Difficulty level  
- Topic attempts  
- Running averages  
- Last quiz performance  
- Full performance history  

### ⚡ Technology Stack
- **Google ADK 1.1.1**  
- **Gemini 2.5 Flash**  
- **Gradio 3.50.2**  
- Python 3.10+  
- Async ADK Runners  

---

## 🏗️ Architecture

This system uses **multiple independent agents** (not sequential).

### 🔹 Agents Overview

| Agent | Responsibility |
|-------|----------------|
| **LessonAgent** | Teaches the topic and generates structured lessons |
| **QuizAgent** | Creates 4 multiple-choice questions in pure JSON |
| **GraderAgent** | Grades user responses automatically |
| **FeedbackAgent** | Produces natural-language coaching & recommendations |



