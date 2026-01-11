
import streamlit as st
import os
import pandas as pd
import inspect
from agents.scraper import ScraperAgent
from agents.visualizer import VisualizerAgent

# Try importing agraph (optional dependency)
try:
    from streamlit_agraph import agraph, Node, Edge, Config
    HAS_AGRAPH = True
except ImportError:
    HAS_AGRAPH = False

# Page Config
st.set_page_config(page_title="SR Semantic Mapper", layout="wide")

# Initialize Agents
if 'scraper_agent' not in st.session_state:
    st.session_state.scraper_agent = ScraperAgent()

# Robust Agent Loading: Check for method signature updates
if 'visualizer_agent' not in st.session_state:
    st.session_state.visualizer_agent = VisualizerAgent()
else:
    # Check for stale code (missing 'dimensions' arg or 'set_config' method)
    # Also check for new search signature (max_score, ascending) and show_links, show_clusters, chat method, hybrid_search
    agent = st.session_state.visualizer_agent
    try:
        sig_map = inspect.signature(agent.create_semantic_map)
        sig_search = inspect.signature(agent.search)
        
        if ('dimensions' not in sig_map.parameters or 
            'show_links' not in sig_map.parameters or
            'show_clusters' not in sig_map.parameters or
            not hasattr(agent, 'set_config') or 
            not hasattr(agent, 'chat') or
            not hasattr(agent, 'hybrid_search') or
            'max_score' not in sig_search.parameters):
            
            st.session_state.visualizer_agent = VisualizerAgent()
    except Exception:
        st.session_state.visualizer_agent = VisualizerAgent()

def main():
    st.title("🕸️ Agentic Semantic Mapper")
    
    # Sidebar: Controls
    st.sidebar.header("Configuration")
    
    provider = st.sidebar.radio("Embedding Provider", ["OpenAI", "Gemini"])
    
    # Dynamic view dim options
    view_dim_options = ["2D", "3D"]
    if HAS_AGRAPH:
        view_dim_options.append("Interactive Network")
    view_dim = st.sidebar.radio("Visualization Mode", view_dim_options)
    
    if provider == "OpenAI":
        api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if api_key:
            st.session_state.visualizer_agent.set_config("openai", api_key)
    else:
        api_key = st.sidebar.text_input("Gemini API Key", type="password")
        if api_key:
            st.session_state.visualizer_agent.set_config("gemini", api_key)

    target_urls = st.sidebar.text_area("Target Websites (One per line)", 
                                     value="https://www.bajajfinservmarkets.in/")
    max_pages = st.sidebar.slider("Max Pages per Site", 10, 500, 50)
    
    if st.sidebar.button("🧹 Reset Agents"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if st.sidebar.button("🚀 Start Scraping Agent"):
        urls = [u.strip() for u in target_urls.split('\n') if u.strip()]
        if not urls:
            st.error("Please enter at least one URL.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(msg):
                status_text.text(msg)
                
            with st.spinner("Agent is working..."):
                results = st.session_state.scraper_agent.scrape(urls, max_pages, update_progress)
                
            st.success("Scraping Complete!")
            st.json(results)

    # File Upload Logic
    st.sidebar.markdown("---")
    st.sidebar.header("Import Data")
    uploaded_file = st.sidebar.file_uploader("Upload JSON", type=['json'])
    if uploaded_file is not None:
        save_path = os.path.join("data", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Saved: {uploaded_file.name}")

    # Main Area: Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Viewer", "🗺️ Semantic Map", "🔎 Semantic Search", "💬 Chat with Data"])
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    # --- Tab 1: Data Viewer ---
    with tab1:
        st.header("Inspect Extracted Data")
        if not files:
            st.info("No data found. Upload or scrape first.")
        else:
            selected_file = st.selectbox("Select Dataset", files)
            if selected_file:
                file_path = os.path.join(data_dir, selected_file)
                df = st.session_state.visualizer_agent.load_data(file_path)
                st.dataframe(df)
                st.write(f"Total Pages: {len(df)}")
                
                # Download Raw Data
                st.download_button(
                    label="Download Raw Data (JSON)",
                    data=df.to_json(orient='records', indent=2),
                    file_name=f"{selected_file.replace('.json', '')}_raw.json",
                    mime="application/json"
                )

    # --- Tab 2: Semantic Map ---
    with tab2:
        st.header("Visualize Closeness")
        if not files:
            st.info("No data found.")
        else:
            viz_files = st.multiselect("Select Dataset(s) to Visualize", files, default=[files[0]])
            
            col_controls, col_info = st.columns([2, 1])
            with col_controls:
                 if st.button("Generate Map") and viz_files:
                     generate = True
                 else:
                     generate = False
            with col_info:
                 show_links = st.checkbox("Show Network Links", value=False, help="Draw lines between pages that link to each other.")
                 show_clusters = st.checkbox("Cluster & Name Topics", value=False, help="Automatically group pages and name the topics using AI.")
            
            if generate:
                # Load and Combine
                combined_df = pd.DataFrame()
                
                for f in viz_files:
                    path = os.path.join(data_dir, f)
                    df_temp = st.session_state.visualizer_agent.load_data(path)
                    df_temp['source'] = f 
                    combined_df = pd.concat([combined_df, df_temp], ignore_index=True)

                if combined_df.empty:
                    st.warning("Selected datasets are empty.")
                else:
                    if view_dim == "Interactive Network":
                        with st.spinner("Building Interactive Graph..."):
                            # Trigger clustering if requested
                            if show_clusters and 'cluster_name' not in combined_df.columns:
                                # Run dummy map gen to get clusters
                                st.session_state.visualizer_agent.create_semantic_map(combined_df, dimensions=2, show_clusters=True)
                                # Reload modified df logic or just assume agent modified it in place?
                                # Agent modifies DF in place for cols.
                            
                            nodes_data, edges_data = st.session_state.visualizer_agent.get_graph_data(combined_df)
                            
                            if not nodes_data:
                                st.warning("No nodes found.")
                            else:
                                nodes = [Node(**n) for n in nodes_data]
                                edges = [Edge(**e) for e in edges_data]
                                
                                config = Config(
                                    width=750, height=750, 
                                    directed=True, 
                                    nodeHighlightBehavior=True, 
                                    highlightColor="#F7A7A6", 
                                    collapsible=False,
                                    physics={
                                        "enabled": True,
                                        "stabilization": {
                                            "enabled": True,
                                            "iterations": 200,
                                            "fit": True
                                        },
                                        "barnesHut": {
                                            "gravitationalConstant": -2000,
                                            "centralGravity": 0.1,
                                            "springLength": 150,
                                            "springConstant": 0.05,
                                            "damping": 0.3, # Increased damping to stop spinning
                                            "avoidOverlap": 0.5
                                        },
                                        "maxVelocity": 30,
                                        "minVelocity": 0.1,
                                        "solver": "barnesHut"
                                    }
                                )
                                
                                # Capture return value (selected node) though we might not use it yet
                                return_value = agraph(nodes=nodes, edges=edges, config=config, key="interactive_graph")
                    else:
                        with st.spinner(f"Embedding & Mapping {len(combined_df)} pages..."):
                            try:
                                dims = 3 if view_dim == "3D" else 2
                                fig, error, final_df = st.session_state.visualizer_agent.create_semantic_map(
                                    combined_df, 
                                    dimensions=dims, 
                                    show_links=show_links,
                                    show_clusters=show_clusters
                                )
                                
                                if error:
                                    st.error(error)
                                elif fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Save embeddings
                                    try:
                                        for f in viz_files:
                                            subset = final_df[final_df['source'] == f].copy()
                                            subset_to_save = subset.drop(columns=['x', 'y', 'z', 'source', 'score'], errors='ignore')
                                            subset_to_save.to_json(os.path.join(data_dir, f), orient='records', indent=2)
                                    except: pass

                                    st.download_button(
                                        label="Download Merged Data (JSON)",
                                        data=final_df.to_json(orient='records', indent=2),
                                        file_name="merged_embeddings.json",
                                        mime="application/json"
                                    )
                            except Exception as e:
                                st.error(f"Error: {e}")

    # --- Tab 3: Semantic Search ---
    with tab3:
        st.header("🔎 Semantic Search")
        if not files:
             st.info("No data found.")
        else:
            search_files = st.multiselect("Select Dataset(s) to Search", files, default=[files[0]], key='search_select')
            
            col1, col2 = st.columns([3, 1])
            with col1:
                query = st.text_input("Enter Search Query")
                sort_order = st.radio("Sort Order", ["Descending (Most Similar)", "Ascending (Least Similar)"], horizontal=True)
            with col2:
                top_n = st.number_input("Max Results", min_value=1, max_value=20, value=5)
                score_range = st.slider("Similarity Score Range", 0.0, 1.0, (0.0, 1.0))
                use_hybrid = st.checkbox("Enable Hybrid Search (BM25)", value=False, help="Combines Semantic Vector Search with Keyword Search for better accuracy.")
                
            if st.button("Search") and search_files:
                combined_df = pd.DataFrame()
                for f in search_files:
                    path = os.path.join(data_dir, f)
                    df_temp = st.session_state.visualizer_agent.load_data(path)
                    df_temp['source'] = f
                    combined_df = pd.concat([combined_df, df_temp], ignore_index=True)

                if combined_df.empty:
                    st.warning("Selected datasets are empty.")
                elif 'embedding' not in combined_df.columns:
                     st.warning("⚠️ One or more selected datasets have no embeddings. Please generate map first.")
                else:
                    with st.spinner("Searching..."):
                        try:
                            # Parse UI inputs
                            ascending = "Ascending" in sort_order
                            min_s, max_s = score_range
                            
                            if use_hybrid:
                                results = st.session_state.visualizer_agent.hybrid_search(
                                    combined_df, query, top_n
                                )
                                # Hybrid returns fixed cols (score, bm25, rrf). We just show them.
                            else:
                                results = st.session_state.visualizer_agent.search(
                                    combined_df, query, top_n, 
                                    min_score=min_s, 
                                    max_score=max_s, 
                                    ascending=ascending
                                )
                            
                            if results.empty:
                                st.info("No matches found.")
                            else:
                                for _, row in results.iterrows():
                                    source_label = row.get('source', 'Unknown')
                                    # Display Score (RRF or Cosine)
                                    score_display = f"RRF: {row['rrf_score']:.4f}" if use_hybrid else f"Score: {row['score']:.4f}"
                                    title_display = f"{score_display} | {source_label} | {row['title']}"
                                    
                                    with st.expander(title_display):
                                        st.write(f"**URL:** {row['url']}")
                                        if use_hybrid:
                                             st.caption(f"Vector Score: {row['score']:.4f} | BM25 Score: {row['bm25_score']:.4f}")
                                        st.write(row['content'][:500] + "...")
                        except Exception as e:
                            st.error(f"Search Error: {str(e)}")

    # --- Tab 4: Chat with Data ---
    with tab4:
        st.header("💬 Chat with Data (RAG)")
        if not files:
            st.info("No data found.")
        else:
            chat_files = st.multiselect("Select Dataset(s) to Chat With", files, default=[files[0]], key='chat_select')
            
            # Initialize Chat History
            if "messages" not in st.session_state:
                st.session_state.messages = []
                
            # Display History
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat Input
            if prompt := st.chat_input("Ask a question about your data..."):
                if not chat_files:
                    st.warning("Please select at least one dataset.")
                else:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                        
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        
                        # Load Data
                        combined_df = pd.DataFrame()
                        for f in chat_files:
                            path = os.path.join(data_dir, f)
                            df_temp = st.session_state.visualizer_agent.load_data(path)
                            combined_df = pd.concat([combined_df, df_temp], ignore_index=True)
                        
                        # Check embeddings
                        if 'embedding' not in combined_df.columns:
                            response = "⚠️ Selected datasets do not have embeddings. Please generate a map first."
                            context_df = pd.DataFrame()
                        else:
                            with st.spinner("Thinking..."):
                                ret = st.session_state.visualizer_agent.chat(combined_df, prompt)
                                if isinstance(ret, tuple):
                                    response, context_df = ret
                                else:
                                    response = ret
                                    context_df = pd.DataFrame()
                        
                        message_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Show Context Tools if data exists
                        if not context_df.empty:
                            with st.expander("📚 Sources & Tools"):
                                st.dataframe(context_df[['score', 'title', 'url']])
                                
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.download_button(
                                        "📥 Download Context",
                                        data=context_df.to_json(orient='records', indent=2),
                                        file_name="chat_context.json",
                                        mime="application/json"
                                    )
                                with c2:
                                    if st.button("🗺️ Visualize Context"):
                                        st.caption("Generating subset map...")
                                        # Use standard map gen but only for context rows
                                        # Force 2D for quick viz
                                        fig, _, _ = st.session_state.visualizer_agent.create_semantic_map(
                                            context_df, 
                                            dimensions=2,
                                            show_links=False, 
                                            show_clusters=False
                                        )
                                        if fig:
                                            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
