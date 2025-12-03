# Installing procedure
1. Clone from Git
2. Using python 3.9.11, install libraries using
```
pip install -r requirements.txt
```
3. Add .env file as follows:
```
GENERAL_LLM_API_KEY=[gemini-key]
PSEUDOGRAPH_RELABELLING_API_KEY=[gemini-key]

KG_URI=[neo4j-uri]
KG_USERNAME=[neo4j-username]
KG_PASSWORD=[neo4j-password]
KG_NAME=[neo4j-name]

SPECIALIZED_LLM_TOKEN=[specialized_llm_token]

EMBEDDING_STORAGE_PATH=[store-path, default to the $PROJECT_ROOT/resources/embeddings]
```
4. Add files of existed kg's embedded vectors for relations and entities, into $PROJECT_ROOT/resources/embeddings
```
embeddings
|-kg_relations.npy
|-kg_relations.txt
|-kg_entities.npy
|-kg_entities.txt
```

5. Run server
```
# Run server
cd src/main_pipeline
python server.py

# Or with uvicorn
uvicorn server:app --host 0.0.0.0 --port 8000
```

6. Check the server at
```
http://$DOMAIN:8000/health
```

# Project structure
- src: contains all source and data
In src, we have
- embeddings: containing class of ```embedder.py``` for embedding text (entities/relations) and find the closest match
- kg_connector: containing class of ```KGConnector.py``` for connecting and retrieve/create data to Neo4j
- llm: containing classes of LLM model with finetuned model or through Google Gemini API.
- main_pipeline: containing the main pipeline of processing data end-2-end.
- middle: containing code for processes that process data between 2 LLM, these are mostly algorithm.
- notebooks: containing *.ipynb files where codes are being build on the developing stage (not to concern, these are drafts)
- relabelling: containing codes for relabelling FactKG data's hard categorized samples into triplets.
- resources: containing data resources (model, embeddings, trie, pickle files)
- 