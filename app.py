import streamlit as st
from utils.extract import process_file_to_docmeta
from pipeline.rag import answer_query
from pipeline.index import save_index_to_disk, load_index_from_disk

st.set_page_config(page_title="Financial Document Q&A Assistant", layout="wide")
st.title("Financial Document Q&A Assistant — Soothsayer Internship")

st.markdown("""
Upload a PDF or Excel financial statement. The app will extract text & tables, build an index,
and allow you to ask questions about the document (uses local Ollama LLM).
""")

uploaded = st.file_uploader("Upload PDF / XLS / XLSX", type=["pdf", "xls", "xlsx"])

# simple session storage hooks
if "doc_meta" not in st.session_state:
    st.session_state.doc_meta = None

if uploaded:
    with st.spinner("Processing document — extracting text & tables..."):
        # process_file_to_docmeta returns a dict with keys: id, pages, tables, chunks, index_path (optional)
        doc_meta = process_file_to_docmeta(uploaded)
        st.session_state.doc_meta = doc_meta
    st.success("Document processed and indexed.")

if st.session_state.doc_meta:
    dm = st.session_state.doc_meta
    st.sidebar.header("Document summary")
    st.sidebar.write(f"File id: {dm.get('doc_id')}")
    st.sidebar.write(f"Pages: {len(dm.get('pages', []))}")
    st.sidebar.write(f"Chunks indexed: {len(dm.get('chunks', []))}")

    tabs = st.tabs(["Extracted Tables", "Raw Text", "Chat (Q&A)"])
    with tabs[0]:
        st.header("Extracted Tables (preview)")
        tables = dm.get("tables", [])
        if tables:
            for i, t in enumerate(tables, 1):
                st.subheader(f"Table {i}")
                st.dataframe(t)
        else:
            st.info("No tables extracted.")

    with tabs[1]:
        st.header("Raw Text (page previews)")
        for p in dm.get("pages", []):
            st.markdown(f"**Page {p.get('page')}**")
            st.write(p.get("text")[:2000])  # show first 2k chars

    with tabs[2]:
        st.header("Ask questions about the document")
        if "qa_history" not in st.session_state:
            st.session_state.qa_history = []

        q = st.text_input("Enter your question:", key="question_input")
        if st.button("Ask"):
            if not q.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving answer..."):
                    answer_obj = answer_query(q, dm)
                st.session_state.qa_history.append({"q": q, "a": answer_obj})
        # show history
        for entry in reversed(st.session_state.qa_history):
            st.markdown(f"**Q:** {entry['q']}")
            st.markdown(f"**A:** {entry['a'].get('answer')}")
            sources = entry['a'].get('sources', [])
            if sources:
                st.markdown("**Sources:**")
                for s in sources:
                    st.write(s)
            st.write("---")

st.markdown("---")
st.caption("Notes: This is a prototype. Ollama must be installed locally and a model pulled (e.g., `ollama pull mistral`).")
