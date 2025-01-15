import streamlit as st
import requests


def main():
    # Initialize session state variables
    if 'video_url' not in st.session_state:
        st.session_state.video_url = ""
    if 'question' not in st.session_state:
        st.session_state.question = ""

    # Set the app title
    st.title("Ask YouTube Video")

    # Prompt the user for a YouTube video URL
    video_url = st.text_input("Enter YouTube Video URL:", key="video_url")

    # If the user provides a valid URL
    if video_url:
        # Extract the video ID from the URL
        try:
            if "youtube.com" in video_url:
                video_id = video_url.split("v=")[1].split("&")[0]
            elif "youtu.be" in video_url:
                video_id = video_url.split(".be/")[1]
            else:
                st.error("Invalid YouTube URL. Please enter a valid one.")
                return

            # Embed the YouTube video in the app
            st.video(f"https://www.youtube.com/embed/{video_id}")
            # Call the write-captions endpoint
            try:
                response = requests.post(
                    "http://localhost:8000/write-captions",
                    json={"video_url": video_url}
                )
                if response.status_code == 200:
                    st.write("Captions processed successfully")
                else:
                    st.error(f"Error processing captions: {response.text}")
            except Exception as e:
                st.error(f"Error calling write-captions endpoint: {str(e)}")
        except Exception as e:
            st.error(f"Error processing the URL: {e}")
    else:
        st.info("Please enter a YouTube video URL to see the video embedded here.")

    # Add a text input for questions
    question = st.text_input("Ask a question about the video:", key="question")

    # If user enters a question, make API call
    if question:
        try:
            response = requests.post(
                "http://localhost:8000/ask",
                json={"question": question}
            )
            if response.status_code == 200:
                st.write("Answer:", response.json())
            else:
                st.error(f"Error getting answer: {response.text}")
        except Exception as e:
            st.error(f"Error making API request: {str(e)}")

if __name__ == "__main__":
    main()
