🧠 Dyslexia Assistive Web App

An AI-powered assistive tool designed to support individuals with dyslexia in reading, writing, speaking, and spelling. 
This web application corrects grammar and dyslexia-specific errors, provides pronunciation support, and tracks user progress over time.

✨ Features

✅ Core Functionality
- Grammar & Spell Correction
  Uses both standard grammar correction and a curated list of dyslexia-specific word corrections.

- Text Input Methods
  - Typed text
  - Image-based text (via OCR using Tesseract & EasyOCR)
  - Audio-based input (via Whisper)

- Pronunciation Assistance
  - Text-to-speech (TTS) with rate control
  - Audio feedback for mispronounced/misspelled words
  - IPA and phonetic spelling planned for future versions

- Progress Tracking
  - Tracks performance in reading, writing, and speaking tasks
  - Calculates a capability score per user
  - Displays top corrected words and learning history

- Gamified Exercises
  - Spelling exercises
  - Fill-in-the-blank tasks
  - Multiple-choice questions

📊 Machine Learning
- Uses a TF-IDF + RandomForest model to detect dyslexia-like writing patterns
- Offers context-based paragraph generation using Gemini/GPT for reading practice

🛠️ Tech Stack

- Frontend: HTML (via Flask templates)
- Backend: Python, Flask
- AI/NLP Libraries:
  - [Whisper](https://github.com/openai/whisper) for audio transcription
  - [language_tool_python](https://github.com/languagetool-org/languageTool) for grammar correction
  - [pyttsx3](https://github.com/nateshmbhat/pyttsx3) for TTS
  - [pytesseract](https://github.com/madmaze/pytesseract) for OCR
  - [easyocr](https://github.com/JaidedAI/EasyOCR) for multi-lingual image reading
  - [Google Generative AI (Gemini)](https://ai.google.dev/) for adaptive content

- Database: MongoDB
- Model: `dyslexia_model.pkl` trained on dyslexia-like writing patterns

## 🧩 Folder Structure

