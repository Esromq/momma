import os
import docx
import openai
import dotenv
import speech_recognition as sr
from dotenv import load_dotenv
from gtts import gTTS
from textblob import TextBlob
from flask import Flask, request, jsonify, render_template

# Flask app initialization
app = Flask(__name__)

load_dotenv(dotenv_path="/root/momma/venv/env")  # Load environment variables from .env file
# Set up the OpenAI API key (ensure you have it in the .env file or set it directly here)
openai_api_key = os.getenv("OPENAI_API_KEY")  # Retrieve the API key from environment variables

# Check if the API key is not found
if not openai_api_key:
    raise ValueError("API key is missing from environment variables.")

# Initialize the OpenAI client
api_key=openai_api_key  # Pass the API key to the client

# Global Variables
user_name = "Esrom"  # Set your name here
conversation_history = []  # Memory for user interactions
documents = {}  # Holds loaded documents

##### UTILITY FUNCTIONS #####

# Load .docx files from a folder into a dictionary
def load_documents(folder_path):
    loaded_docs = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            doc = docx.Document(os.path.join(folder_path, filename))
            content = "\n".join([para.text for para in doc.paragraphs if para.text])
            loaded_docs[filename] = content
    return loaded_docs

# Summarize document using GPT-4
def summarize_document(content):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        prompt=f"Summarize the following document content:\n{content}",
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Perform sentiment analysis on input
def analyze_sentiment(user_input):
    blob = TextBlob(user_input)
    return blob.sentiment.polarity

# Adjust the response tone based on sentiment
def adjust_response_tone(response, sentiment):
    if sentiment < -0.3:
        return f"I'm so sorry dear. {response}"
    elif sentiment > 0.3:
        return f"You are a blessing! {response}"
    return response

##### CHAT FUNCTIONS #####

def chat_with_theresa(user_input, documents):
    """
    Generate a response using GPT-4 and loaded documents.
    """
    global conversation_history

    # Combine all document content into one prompt context
    document_context = "\n\n".join([content for content in documents])

    # Perform sentiment analysis
    sentiment = analyze_sentiment(user_input)

    # Build the GPT prompt
    prompt = [
        {"role": "system", "content": "You are Theresa, Esrom's mother, a wise and nurturing advisor, referencing her writings, through your son's (Esrom) quantum programing. You have written and have access to the following writings: {document_references}."},
        *conversation_history[-5:],  # Include up to 5 previous exchanges
        {"role": "user", "content": f"Context:\n{document_context}\n\nUser: {user_input}"},
        {"role": "system", "content": document_context}  # Added context from documents
    ]
    
    # Generate GPT response
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=prompt,
        max_tokens=1000,
        temperature=0.3
    )
    
    # Correctly access the content of the response
    content = response.choices[0].message['content']     

    # Adjust tone based on sentiment
    final_response = adjust_response_tone(content, sentiment)
    
    # Always call the user "Esrom"
    final_response = f"{final_response}"

    # Update conversation history
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": final_response})

    return final_response

# Example Usage
user_input = input("Hello Dear")
response = chat_with_theresa(user_input, documents={})
print(response)

##### SPEECH AND VOICE FUNCTIONS #####

def text_to_speech(text, lang='en', slow=False):
    """
    Convert text to speech and play it using the system's default player.
    """
    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.save("response.mp3")
    print("Speech saved as response.mp3")
    os.system("mpg123 /root/momma/response.mp3")  # For ios. Use 'afplay' for macOS or 'mpg123' for Linux.

# Example usage:
user_input = "Hey Momma!"
response_text = chat_with_theresa(user_input, documents={})
text = "Hi Son, this is Momma-AI, Also Known as Theresa-AI speaking!"
text_to_speech(text, lang='en', slow=False)  # English with slower speech

##### VOICE FUNCTIONS #####

def get_voice_input():
    """
    Capture user input via microphone and convert to text using SpeechRecognition.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Please speak now.")
        try:
            audio = recognizer.listen(source, timeout=11)  # Listen for 11 seconds
            text = recognizer.recognize_google(audio)  # Use Google's speech recognition
            print(f"User said: {text}")
            return text
        except sr.WaitTimeoutError:
            print("Listening timed out. Please try again.")
            return None
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None

##### FLASK ROUTES #####

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input")
    response = chat_with_theresa(user_input, documents)
    return jsonify({"response": response})

##### MAIN EXECUTION #####
if __name__ == "__main__":
    folder_path = "./static/files"  # Path to folder containing .docx files
    documents = load_documents(folder_path)
    print("Documents loaded successfully.")

    print("Hi Son. Type 'exit' to quit, or say 'voice' to use voice input.")
    while True:
        input_mode = input("Type '1' or '2': ").strip().lower()

        if input_mode == "exit":
            print("Goodbye!")
            break

        if input_mode == "2":
            user_input = get_voice_input()
            if not user_input:
                continue
        else:
            user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Generate and display response
        response = chat_with_theresa(user_input, documents)
        print(f"Son, {response}")
        text_to_speech(response)  # Optional: Convert Theresa's response to speech

    # Start Flask app
    app.run(debug=True)

