from agents.base import BaseAgent
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
from openai import OpenAI
import google.generativeai as genai

class VisualizerAgent(BaseAgent):
    def __init__(self, provider="openai", api_key=None):
        super().__init__("VisualizerAgent")
        self.provider = provider
        self.api_key = api_key
        self.client = None
        self.reducer = None
        
        self.configure_client()

    def set_config(self, provider, api_key):
        self.provider = provider
        self.api_key = api_key
        self.configure_client()

    def configure_client(self):
        if not self.api_key:
            self.client = None
            return

        if self.provider == "openai":
            self.client = OpenAI(api_key=self.api_key)
        elif self.provider == "gemini":
            try:
                genai.configure(api_key=self.api_key)
                self.client = genai
            except Exception as e:
                self.log(f"Gemini Config Error: {e}")
                self.client = None

    def load_data(self, json_path):
        if not os.path.exists(json_path):
            return pd.DataFrame()
        with open(json_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def _get_reducer(self):
        try:
            import umap
            if self.reducer is None:
                self.log("Initializing UMAP...")
                # Conservative defaults
                self.reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, metric='cosine', random_state=42)
            return self.reducer
        except ImportError:
            self.log("UMAP not installed")
            return None

    def _get_gemini_chat_model(self):
        """
        Dynamically find a supported Gemini model for generation.
        """
        try:
            # diligent search for a working model
            preferred_order = ["gems/gemini-1.5-flash", "models/gemini-1.5-flash", "models/gemini-pro", "models/gemini-1.0-pro"]
            
            available_models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            available_names = [m.name for m in available_models]
            
            # 1. Try preferred aliases
            for p in preferred_order:
                if p in available_names:
                     return genai.GenerativeModel(p)
            
            # 2. Search for any gemini model
            for m in available_models:
                if "gemini" in m.name.lower():
                    return genai.GenerativeModel(m.name)
            
            # 3. Fallback to first available generative model
            if available_models:
                return genai.GenerativeModel(available_models[0].name)
                
            return genai.GenerativeModel("models/gemini-pro") # Hard fallback with prefix
            
        except Exception as e:
            self.log(f"Model Discovery Error: {e}")
            return genai.GenerativeModel("models/gemini-pro")

    def cluster_and_label(self, df, n_clusters=5):
        try:
            from sklearn.cluster import KMeans
            matrix = np.vstack(df['embedding'].values)
            n_clusters = min(n_clusters, len(df))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            df['cluster'] = kmeans.fit_predict(matrix)
            
            # Label Clusters via LLM
            cluster_names = {}
            for i in range(n_clusters):
                subset = df[df['cluster'] == i]
                titles = subset['title'].head(5).tolist()
                prompt = f"Generate a short 2-3 word category name for these web pages: {titles}"
                
                try:
                    name = f"Cluster {i}"
                    if self.provider == "openai":
                        resp = self.client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        name = resp.choices[0].message.content.strip().replace('"', '')
                    elif self.provider == "gemini":
                        model = self._get_gemini_chat_model()
                        resp = model.generate_content(prompt)
                        name = resp.text.strip().replace('"', '')
                    cluster_names[i] = name
                except Exception as e:
                    self.log(f"Naming Error ({i}): {e}")
                    cluster_names[i] = f"Topic {i+1}"
            
            df['cluster_name'] = df['cluster'].map(cluster_names)
            return df
        except Exception as e:
            self.log(f"Clustering Error: {e}")
            return df

    def create_semantic_map(self, df, dimensions=2, show_links=False, show_clusters=False):
        """
        Takes a DataFrame with 'content' column.
        dimensions: 2 or 3
        show_links: bool, whether to draw lines for internal links
        show_clusters: bool, whether to cluster and color by topic
        Returns: (plotly_fig, error_message, df_with_embeddings)
        """
        if df.empty or 'content' not in df.columns:
            return None, "Data is empty or missing 'content' column.", df
            
        if not self.api_key:
            return None, f"API Key for {self.provider} is missing.", df

        # Embed
        self.log(f"Generating Embeddings via {self.provider}...")
        embeddings = []
        
        # Check if embeddings already exist
        if 'embedding' in df.columns and not df['embedding'].isnull().any():
             embeddings = df['embedding'].tolist()
        else:
            try:
                # Preprocess
                texts = df['content'].apply(lambda x: x[:8000].replace("\n", " ")).tolist()
                
                if self.provider == "openai":
                    batch_size = 20
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i+batch_size]
                        if not batch: continue
                        response = self.client.embeddings.create(input=batch, model="text-embedding-3-small")
                        embeddings.extend([d.embedding for d in response.data])
    
                elif self.provider == "gemini":
                    for text in texts:
                        result = genai.embed_content(
                            model="models/text-embedding-004",
                            content=text,
                            task_type="retrieval_document"
                        )
                        embeddings.append(result['embedding'])
    
            except Exception as e:
                return None, f"Embedding Error ({self.provider}): {str(e)}", df

        if len(embeddings) < 3:
             return None, "Not enough data points to visualize (Need at least 3 pages/vectors).", df

        # Store raw embeddings for export
        df['embedding'] = embeddings
        
        # Clustering
        if show_clusters:
            df = self.cluster_and_label(df)
        
        # Reduce
        try:
            import umap
            n_neighbors = min(15, len(embeddings) - 1)
            
            # Re-init if config changed (dimensions or neighbors)
            if (self.reducer is None or 
                self.reducer.n_neighbors > n_neighbors or 
                self.reducer.n_components != dimensions):
                
                self.log(f"Initializing UMAP (n_neighbors={n_neighbors}, dim={dimensions})...") 
                self.reducer = umap.UMAP(
                    n_neighbors=n_neighbors, 
                    n_components=dimensions,
                    min_dist=0.1, 
                    metric='cosine', 
                    random_state=42
                )
                
            embedding_reduced = self.reducer.fit_transform(embeddings)
            
            df['x'] = embedding_reduced[:, 0]
            df['y'] = embedding_reduced[:, 1]
            if dimensions == 3:
                df['z'] = embedding_reduced[:, 2]
                
        except ImportError:
             return None, "umap-learn is not installed.", df
        except Exception as e:
            return None, f"UMAP/Reduction Error: {str(e)}", df

        # Plot
        try:
            title = f'Semantic Map ({len(df)} Pages) - {self.provider.upper()} - {dimensions}D'
            
            # Determine color column
            if show_clusters and 'cluster_name' in df.columns:
                color_col = 'cluster_name'
            else:
                color_col = 'source' if 'source' in df.columns else None
            
            if dimensions == 3:
                fig = px.scatter_3d(
                    df, x='x', y='y', z='z',
                    color=color_col,
                    hover_data=['title', 'url'],
                    title=title,
                    template='plotly_dark'
                )
            else:
                fig = px.scatter(
                    df, x='x', y='y',
                    color=color_col,
                    hover_data=['title', 'url'],
                    title=title,
                    template='plotly_dark'
                )
            
            # Draw Links Overlay
            if show_links and 'links' in df.columns:
                # Create a lookup for URL -> (x, y, [z])
                url_map = {}
                for idx, row in df.iterrows():
                    coords = [row['x'], row['y']]
                    if dimensions == 3: coords.append(row['z'])
                    url_map[row['url']] = coords
                
                edge_x = []
                edge_y = []
                edge_z = [] # Only for 3D
                
                for idx, row in df.iterrows():
                    if not isinstance(row['links'], list): continue
                    
                    start_pos = url_map.get(row['url'])
                    if not start_pos: continue
                    
                    for link in row['links']:
                        end_pos = url_map.get(link)
                        if end_pos:
                            edge_x.extend([start_pos[0], end_pos[0], None])
                            edge_y.extend([start_pos[1], end_pos[1], None])
                            if dimensions == 3:
                                edge_z.extend([start_pos[2], end_pos[2], None])
                
                if edge_x:
                    import plotly.graph_objects as go
                    if dimensions == 3:
                        fig.add_trace(go.Scatter3d(
                            x=edge_x, y=edge_y, z=edge_z,
                            mode='lines',
                            line=dict(color='rgba(150,150,150,0.5)', width=1),
                            hoverinfo='none',
                            name='Links'
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=edge_x, y=edge_y,
                            mode='lines',
                            line=dict(color='rgba(150,150,150,0.5)', width=0.5),
                            hoverinfo='none',
                            name='Links'
                        ))

            fig.update_traces(marker=dict(size=5 if dimensions==3 else 8, opacity=0.8))
            return fig, None, df
        except Exception as e:
            return None, f"Plotting Error: {str(e)}", df

    def search(self, df, query, top_n=5, min_score=0.0, max_score=1.0, ascending=False):
        """
        Semantic search against the dataframe.
        Requires df['embedding'] to exist.
        """
        if df.empty or 'embedding' not in df.columns or not query:
            return pd.DataFrame()
            
        if not self.api_key:
            raise ValueError(f"API Key for {self.provider} is missing.")
            
        # Embed Query
        try:
            query_embedding = None
            if self.provider == "openai":
                response = self.client.embeddings.create(input=[query], model="text-embedding-3-small")
                query_embedding = response.data[0].embedding
            elif self.provider == "gemini":
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=query,
                    task_type="retrieval_query"
                )
                query_embedding = result['embedding']
        except Exception as e:
            raise Exception(f"Query Embedding Error: {str(e)}")

        # Calculate Similarity
        def cosine_similarity(v1, v2):
            if not v1 or not v2: return 0.0
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        df['score'] = df['embedding'].apply(lambda x: cosine_similarity(query_embedding, x))
        
        # Filter (Range) & Sort
        mask = (df['score'] >= min_score) & (df['score'] <= max_score)
        results = df[mask].sort_values(by='score', ascending=ascending).head(top_n)
        
        return results[['score', 'title', 'url', 'content']]
        
    def hybrid_search(self, df, query, top_n=5, alpha=0.5):
        """
        Performs Hybrid Search using RRF (Reciprocal Rank Fusion).
        combines Vector Search (Cosine) and Keyword Search (BM25).
        """
        if df.empty or 'content' not in df.columns:
            return pd.DataFrame()

        # 1. Vector Search (Get more candidates for fusion)
        # We assume 'score' column is populated by a call to search, 
        # but we need to run it here to be safe and get raw scores for all docs.
        try:
            self.search(df, query, top_n=len(df))
        except Exception:
            return pd.DataFrame() 
            
        # 2. BM25 Search
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            self.log("rank_bm25 not installed")
            return self.search(df, query, top_n)
            
        # Text Preprocessing (Simple)
        tokenized_corpus = [str(text).lower().split(" ") for text in df['content'].fillna("")]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        df['bm25_score'] = doc_scores
        
        # 3. Reciprocal Rank Fusion (RRF)
        # score = 1 / (k + rank_vector) + 1 / (k + rank_bm25)
        k = 60
        
        # Calculate ranks (Lower rank is better, so ascending=False gives highest score rank 1? No.)
        # rank() method: 1 is highest value if ascending=False.
        df['vector_rank'] = df['score'].rank(ascending=False)
        df['bm25_rank'] = df['bm25_score'].rank(ascending=False)
        
        df['rrf_score'] = (1 / (k + df['vector_rank'])) + (1 / (k + df['bm25_rank']))
        
        # Return sorted results
        results = df.sort_values('rrf_score', ascending=False).head(top_n)
        return results[['score', 'bm25_score', 'rrf_score', 'title', 'url', 'content']]
        
    def get_graph_data(self, df):
        """
        Extracts Nodes and Edges for Interactive Graph Visualization.
        Returns: (nodes_list, edges_list)
        """
        if df.empty: return [], []
        
        nodes = []
        edges = []
        
        # 1. Create Nodes
        # Use URL as ID.
        for idx, row in df.iterrows():
            # Determine group
            group = "Page"
            if 'cluster_name' in row:
                group = row['cluster_name']
            elif 'source' in row:
                group = str(row['source']).replace(".json", "")
                
            nodes.append({
                "id": row['url'],
                "label": row['title'][:30] + "...",
                "title": row['title'], # Tooltip
                "group": group,
                "val": 10 # Size
            })
            
        # 2. Create Edges from Internal Links
        if 'links' in df.columns:
            # Create set of valid URLs in this dataset to avoid dangling edges
            valid_urls = set(df['url'].unique())
            
            for idx, row in df.iterrows():
                if not isinstance(row['links'], list): continue
                
                for link in row['links']:
                    if link in valid_urls and link != row['url']:
                        edges.append({
                            "source": row['url'],
                            "target": link,
                            "type": "CURVE_SMOOTH"
                        })
        
        return nodes, edges
        
    def chat(self, df, query):
        """
        RAG Chat: Retrieval Augmented Generation.
        1. Search for relevant context.
        2. Ask LLM to answer.
        """
        if not self.api_key:
             return "I need an API Key to answer questions."
        
        # 1. Retrieve Context
        results = self.search(df, query, top_n=5)
        if results.empty:
            return "I couldn't find any relevant information in the loaded data.", pd.DataFrame()
            
        context_str = ""
        for idx, row in results.iterrows():
            context_str += f"\n--- Source: {row['url']} ---\n{row['content'][:1500]}\n"
            
        # 2. Generate Answer
        system_prompt = "You are a helpful assistant. Answer the user's question based ONLY on the provided context."
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
        
        try:
            response_text = ""
            if self.provider == "openai":
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                response_text = resp.choices[0].message.content
                
            elif self.provider == "gemini":
                model = self._get_gemini_chat_model()
                # Gemini chat structure
                chat = model.start_chat()
                resp = chat.send_message(f"{system_prompt}\n{user_prompt}")
                response_text = resp.text
            
            return response_text, results
                
        except Exception as e:
            return f"Error generating answer: {str(e)}", pd.DataFrame()
