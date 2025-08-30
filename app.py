import streamlit as st
from search_engine import SearchEngine

@st.cache_resource
def load_search_engine():
    """Loads and caches the SearchEngine."""
    try:
        engine = SearchEngine()
        return engine
    except Exception as e:
        st.error(f"Failed to initialize the search engine: {e}")
        st.error("Please ensure you have run 'update_metadata.py', have a .env file, and all dependencies are installed.")
        return None

# --- Page Configuration ---
st.set_page_config(page_title="Semantic Movie Search", page_icon="🎬", layout="wide")

# --- Sidebar for Filters ---
st.sidebar.title("🔍 Filters")
genre = st.sidebar.text_input("Genre contains...", help="e.g., Comedy, Drama, Action")
runtime_filter = st.sidebar.slider("Maximum runtime (minutes)...", 0, 400, 400, 10)
director = st.sidebar.text_input("Director contains...", help="e.g., Christopher Nolan")


# --- Main UI ---
st.title("🎬 Semantic Movie Search")
st.markdown("Search for movies by meaning and filter the results.")

search_engine = load_search_engine()

if search_engine:
    search_query = st.text_input("Search for a movie...", placeholder="e.g., a psychological thriller on a remote island")

    if search_query:
        with st.spinner("Searching, filtering, and reranking..."):
            results = search_engine.search(
                query=search_query,
                top_k=5,
                genre_filter=genre if genre else None,
                runtime_filter=runtime_filter if runtime_filter < 400 else None,
                director_filter=director if director else None
            )

        # This check prevents the error if results is None
        if results is not None:
            st.subheader(f"Top {len(results)} Results for '{search_query}'")
            if not results:
                st.warning("No results found. Try a different query or loosen your filters!")
            else:
                for i, movie in enumerate(results):
                    with st.container(border=True):
                        title = movie.get('title', 'No Title')
                        release_year = int(movie.get('release_year', 0))
                        vote_avg = movie.get('vote_average', 0)
                        genres_display = movie.get('genres', 'N/A')
                        overview_display = movie.get('overview', 'No overview available.')
                        score = movie.get('rerank_score', 0)
                        tagline_display = movie.get('tagline', '')
                        runtime_display = int(movie.get('runtime', 0))
                        cast_display = movie.get('cast', 'N/A')
                        director_display = movie.get('director', 'N/A')

                        # Display Title (and year if valid)
                        title_text = f"#### {i+1}. {title}"
                        if release_year > 0:
                            title_text += f" ({release_year})"
                        st.markdown(title_text)

                        # Display Tagline if it exists
                        if tagline_display:
                            st.markdown(f"> *{tagline_display}*")
                        
                        # Display Director if available
                        if director_display and director_display != 'N/A':
                            st.markdown(f"**Director:** {director_display}")
                        
                        # Display Cast if it exists
                        if cast_display and cast_display != 'N/A':
                            st.markdown(f"**Cast:** {cast_display}")

                        # Display Info on new lines
                        st.markdown(f"**Genres:** {genres_display}")
                        if runtime_display > 0:
                            st.markdown(f"**Rating:** {vote_avg:.1f} | **Runtime:** {runtime_display} min")
                        else:
                            st.markdown(f"**Rating:** {vote_avg:.1f}")

                        st.write(overview_display)
        else:
            st.error("Search failed and returned an unexpected result. Please check the console logs.")