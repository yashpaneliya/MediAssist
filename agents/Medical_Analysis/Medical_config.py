import os 

NEO4j_URI = os.getenv("NEO4J_URI", "neo4j+s://fecbe539.databases.neo4j.io")
NEO4j_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4j_PASSWORD = os.getenv("NEO4J_PASSWORD", "yaxMZJyX9a6Ph4SNdRncWJWbSWbmiNGS9Flah_UV100")
DRUG_EXTRACTOR_MODEL_NAME = os.getenv("DRUG_EXTRACTOR_MODEL_NAME", "Llama-4-Maverick-17B-128E-Instruct")
DRUG_INFO_MODEL_NAME = os.getenv("DRUG_INFO_MODEL_NAME", "Meta-Llama-3.3-70B-Instruct")