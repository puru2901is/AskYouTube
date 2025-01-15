from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from phi.knowledge.text import TextKnowledgeBase
from phi.vectordb.pgvector import PgVector
from phi.embedder.google import GeminiEmbedder
from phi.document.chunking.fixed import FixedSizeChunking
from phi.document.chunking.document import DocumentChunking
import os
from phi.agent import Agent, RunResponse
from phi.model.google import Gemini
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()

class CaptionRequest(BaseModel):
    video_url: str
    languages: list[str] = None


@app.post("/write-captions")
async def write_captions(request: CaptionRequest):
    """FastAPI endpoint to write YouTube video captions to file.
    
    Args:
        request (CaptionRequest): Request body containing video URL and languages
        
    Returns:
        dict: Response containing status message
    """
    result = write_captions_to_file_api(request.video_url, request.languages)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

def get_youtube_video_id(url: str) -> str:
    """Extract the video ID from a YouTube URL."""
    try:
        if "youtube.com/watch?v=" in url:
            return url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("/")[-1]
        else:
            raise ValueError("Invalid YouTube URL format")
    except Exception as e:
        raise ValueError(f"Error extracting video ID: {str(e)}")

def get_youtube_video_captions(url: str, languages: list = None) -> str:
    """Fetch captions from a YouTube video.

    Args:
        url (str): The URL of the YouTube video.
        languages (list): Preferred languages for captions.

    Returns:
        str: Captions as a string.
    """
    if not url:
        return "No URL provided"

    try:
        video_id = get_youtube_video_id(url)
    except ValueError as e:
        return str(e)

    try:
        kwargs = {}
        if languages:
            kwargs["languages"] = languages

        captions = None
        try:
            captions = YouTubeTranscriptApi.get_transcript(video_id, **kwargs)
        except NoTranscriptFound:
            # Attempt automatic captions
            captions = YouTubeTranscriptApi.get_transcript(video_id, languages=["auto"])

        if not captions:
            return "No captions available for this video."
        return " ".join(line["text"] for line in captions)
    except TranscriptsDisabled:
        return "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "No transcript found or auto-generated captions are unavailable."
    except Exception as e:
        return f"An error occurred while fetching captions: {str(e)}"
  
def write_captions_to_file_api(video_url: str, languages: list = None) -> dict:
    """FastAPI endpoint wrapper for write_captions_to_file function.
    
    Args:
        video_url (str): The URL of the YouTube video
        languages (list): Preferred languages for captions
        
    Returns:
        dict: Response containing status message
    """
    try:
        result = write_captions_to_file(video_url, languages)
        return {"status": "success", "message": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def write_captions_to_file(video_url: str, languages: list = None) -> str:
    """Write YouTube video captions to a text file.
    
    Args:
        video_url (str): The URL of the YouTube video
        languages (list): Preferred languages for captions
        
    Returns:
        str: Status message indicating success or failure
    """
    try:
        # Get video ID and captions
        video_id = get_youtube_video_id(video_url)
        print("Video ID:")
        print(video_id)
        captions = get_youtube_video_captions(video_url, languages)
        print("Captions:")
        print(captions)
        
        if not isinstance(captions, str) or captions.startswith("Error") or captions.startswith("No"):
            return f"Could not write captions: {captions}"
            
        # Create data directory if it doesn't exist
        os.makedirs("data/txt_files", exist_ok=True)
        
        # Write captions to file
        filename = f"data/txt_files/{video_id}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(captions)
        
        create_knowledge_base_with_captions()
        return f"Successfully wrote captions to {filename}"
        
    except Exception as e:
        return f"Error writing captions to file: {str(e)}"


def create_knowledge_base_with_captions() -> str:
    """Create a text knowledge base and add YouTube video captions to it.
    
        
    Returns:
        str: Status message
    """
    try:
        knowledge_base = TextKnowledgeBase(
            path="data/txt_files",
            vector_db=PgVector(
                table_name="text_documents", 
                db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
                embedder=GeminiEmbedder(dimensions=768),
            ),
            chunking_strategy=FixedSizeChunking(),
        )
    
        
        # Add captions to knowledge base
        knowledge_base.load(recreate=False)

        print("Knowledge base loaded")

        
        agent = Agent(
            model=Gemini(id="gemini-2.0-flash-exp"),
            knowledge=knowledge_base,
            search_knowledge=True,
            show_tool_calls=True,
            markdown=True,
            debug_mode=True,
            instructions=["You are an RAG agent, use the knowledge base only to answer the question",
                          "You are an intelligent and concise assistant. When answering, strictly adhere to the following rules:",
                          "Direct Answers Only: Provide a direct and accurate answer to the question asked based on the available knowledge.",
                          "No Unnecessary Information: Do not list or mention the topics or information you have in your knowledge base unless directly relevant to the question.",
                          "Admit Uncertainty: If you do not know the answer or cannot find the information, respond only with: 'I don’t know the answer to that.'",
                          "Always ensure your responses are precise, to the point, and strictly follow these rules."],
        )
        
        return f"Successfully created knowledge base and added captions for video"
    
        
        
    except Exception as e:
        return f"Error creating knowledge base with captions: {str(e)}"


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
async def root():
    return {"message": "YouTube Video QA API"}


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        answer = get_answer(request.question)
        return {answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def get_answer(question: str) -> str:
    """Get answer from the agent based on the knowledge base.
    
    Args:
        question (str): Question to ask the agent
        
    Returns:
        str: Agent's response
    """
    try:
        knowledge_base = TextKnowledgeBase(
            path="data/txt_files", 
            vector_db=PgVector(
                table_name="text_documents",
                db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
                embedder=GeminiEmbedder(dimensions=768),
            ),
            chunking_strategy=FixedSizeChunking(),
        )

        agent = Agent(
            model=Gemini(id="gemini-2.0-flash-exp"),
            knowledge=knowledge_base,
            search_knowledge=True,
            #show_tool_calls=True,
            #markdown=True,
            debug_mode=True,
            instructions=["You are an RAG agent, use the knowledge base only to answer the question",
                        "You are an intelligent and concise assistant. When answering, strictly adhere to the following rules:",
                        "Direct Answers Only: Provide a direct and accurate answer to the question asked based on the available knowledge.",
                        "No Unnecessary Information: Do not list or mention the topics or information you have in your knowledge base unless directly relevant to the question.",
                        "Admit Uncertainty: If you do not know the answer or cannot find the information, respond only with: 'Video does not have the answer to that.'",
                        "Always ensure your responses are precise, to the point, and strictly follow these rules."],
        )
        
        response: RunResponse = agent.run(question, content_type="str",stream=False)
        return response.content
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting answer: {str(e)}")


    
    
