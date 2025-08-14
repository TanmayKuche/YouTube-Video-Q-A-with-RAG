import streamlit as st
from rag_chain_main import build_rag_chain, load_transcript_and_index
from youtube_transcript_api import TranscriptsDisabled

st.set_page_config(page_title="🎬 YouTube RAG QA", layout="centered")
st.title("🎥 YouTube Video Q&A with RAG")

youtube_url = st.text_input("Enter YouTube video URL")

if youtube_url:
    try:
        import re
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", youtube_url)
        video_id = match.group(1) if match else None

        if video_id:
            with st.spinner("Fetching and processing transcript..."):
                retriever, rag_chain = load_transcript_and_index(video_id)
                
            question = st.text_input("Ask a question about the video")

            if question:
                with st.spinner("Thinking..."):
                    answer = rag_chain.invoke(question)
                st.subheader("📢 Answer")
                st.write(answer)

        else:
            st.error("Invalid YouTube URL.")

    except TranscriptsDisabled:
        st.error("🚫 Captions are disabled for this video.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
