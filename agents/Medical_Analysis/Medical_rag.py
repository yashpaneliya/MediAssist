import json
import pandas as pd
from agents.Utils.common_methods import encode_image, get_sambanova_response
from neo4j import GraphDatabase
from itertools import combinations
from core.config import get_settings
from utils.logger import logger

settings = get_settings()

class MedicalAgent :
    def __init__(self):
        self.driver = GraphDatabase.driver( settings.NEO4J_URI , auth=(settings.NEO4J_USERNAME , settings.NEO4J_PASSWORD ) )


    def get_drug_info(self, tx, drug_names):
        query = """
        MATCH (d:Drug)
        WHERE d.name IN $drug_names
        RETURN d.drugbank_id AS id, d.name AS name, d.description AS description,
            d.indication AS indication, d.mechanism_of_action AS mechanism, d.toxicity AS toxicity, d.food_interactions as food_interactions
        """
        return list(tx.run(query, drug_names=drug_names))
    

    def get_interactions(self, tx, pairs):
        results = []
        for drug1, drug2 in pairs:
            # drug_1_id, drug_2_id = name_to_drugbank_dict[drug1], name_to_drugbank_dict[drug2]
            query = """
            MATCH (a:Drug {name: $drug1})-[r:INTERACTS_WITH]-(b:Drug {name: $drug2})
            RETURN a.name AS drug1, b.name AS drug2, r.description AS description
            """
            rec = tx.run(query, drug1=drug1, drug2=drug2).single()
            if rec:
                results.append(dict(rec))
        return results
    

    def get_drugInfo_userPrompt(self, drug_infos, interaction_infos):
        """
        Build a well-organized, markdown-formatted prompt for drug information analysis.

        Args:
            drug_infos (list): List of dictionaries containing drug information
            interaction_infos (list): List of dictionaries containing drug interaction information

        Returns:
            str: Formatted prompt string
        """

        prompt = """# Clinical Drug Analysis Assistant

    ## Instructions
    You are a clinical assistant tasked with analyzing drug information. Please follow these steps:

    1. **Summarize each drug** - Provide a clear, concise summary of each medication
    2. **Analyze combination risks** - List and explain the risks of combining any of these drugs
    3. **Base responses on provided data only** - Do not generate information that cannot be verified from the data below

    ---

    ## Drug Information Database

    """

        # Add individual drug information
        for i, drug in enumerate(drug_infos, 1):
            prompt += f"""### {i}. {drug['name']}

    **Description:** {drug['description']}

    **Clinical Details:**
    - **Indication:** {drug['indication']}
    - **Mechanism of Action:** {drug['mechanism']}
    - **Toxicity Profile:** {drug['toxicity']}
    - **Food Interactions:** {drug['food_interactions']}

    ---

    """

        # Add drug interaction information
        prompt += """## Drug-Drug Interaction Analysis

    """

        if interaction_infos and len(interaction_infos) > 0:
            prompt += """### Known Interactions

    """
            for i, interaction in enumerate(interaction_infos, 1):
                prompt += f"""**{i}. {interaction['drug1']} + {interaction['drug2']}**
    - Risk: {interaction['description']}

    """
        else:
            prompt += """### No Known Interactions
    No documented drug-drug interactions were found among the provided medications based on the available data.

    """

        prompt += """---

    ## Output Requirements

    Please structure your response as follows:

    ### Drug Summaries
    Provide a concise summary for each medication listed above.

    ### Combination Risk Assessment
    Analyze and explain any risks associated with combining these medications, based solely on the interaction data provided.

    ### Clinical Recommendations
    Offer appropriate guidance for patients or clinicians based on the available information. Include food-interaction recommendations as well if appropriate

    **Note:** All information should be clearly attributed to the provided data sources. Do not include speculative or unverified information."""

        return prompt
    

    def get_drugListExtractor_systemPrompt(self):
        prompt = """You are a drug name extraction system. Extract all drug names from the provided content and return ONLY a JSON response.

Extract:
- Generic drug names (acetaminophen, ibuprofen)
- Brand names (Tylenol, Advil)
- Prescription and OTC medications
- Supplements with specific drug names

Do NOT extract:
- General terms (medication, pills, tablets)
- Dosage information
- Medical conditions
- Non-drug substances

JSON format:
json
{
    "drug_names": ["drug1", "drug2"],
    "confidence": [0.95, 0.87]
}

If no drugs found:
json
{
    "drug_names": [],
    "confidence": []
}

Confidence scale: 1.0 (certain) to 0.6 (minimum threshold).

CRITICAL: Return ONLY the JSON. No explanations, no additional text, no reasoning. Just the JSON object.
"""
        return prompt
    
    def complete_graphrag_search(self, extracted_drug_names_list: list[str]):
        drug_list = extracted_drug_names_list
        if len(drug_list) == 1:
            drug_list.append(drug_list[0])
        pairs = list(combinations(drug_list, 2))
        logger.info(f"Pairs: {pairs}")
        try:
            with self.driver.session() as session:
                drug_infos = session.read_transaction(lambda tx: self.get_drug_info(tx, drug_list))
                interaction_infos = session.read_transaction(lambda tx: self.get_interactions(tx, pairs))
            logger.info(f"Drug Infos: {drug_infos}")
            return drug_infos, interaction_infos
        except Exception as e:
            logger.error(f"Error during graph search: {e}")
            return None, None
        
    def drug_extractor(self , isImage : bool , image_source : str = None , query : str = ""):

        extract_messages = [
        {
            "role": "system",
            "content": self.get_drugListExtractor_systemPrompt()
        }
        ]

        if(isImage):
            logger.info(f"Found image as source")
            image_base64 = encode_image(image_source)
            extract_messages += [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the drug-names from the image"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }
            ]
        else : 
             extract_messages += [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query}
                        ]
                    }
                ]
             
        extract_response = get_sambanova_response(extract_messages, model = settings.DRUG_EXTRACTOR_MODEL_NAME )
        logger.info(f"Extract Response: {extract_response}")
        extracted_drug_names = json.loads(extract_response[extract_response.find("{"): extract_response.rfind("}")+1])["drug_names"]
        return extracted_drug_names
    
    
    def get_responder_output(self, isImage : bool , image_source : str = None , query : str = ""):
        extracted_drug_names_list = self.drug_extractor(isImage, image_source, query)
        logger.info(f"Extracted drug names: {extracted_drug_names_list}")
        drug_infos, interaction_infos = self.complete_graphrag_search(extracted_drug_names_list)
        user_prompt = self.get_drugInfo_userPrompt(drug_infos, interaction_infos)
        logger.info(f"Drug Info User Prompt: {user_prompt}")
        messages=[
            {"role": "system", "content": "You are a helpful clinical assistant."},
            {"role": "user", "content": user_prompt}
        ]

        response = get_sambanova_response(messages, model = settings.DRUG_INFO_MODEL_NAME)
        logger.info(f"Drug Responder Output: {response}")
        return response
    
# medical_agent = MedicalAgent()
# # Example usage:
# response = medical_agent.get_responder_output(isImage=False, query="I have been prescribed Aspirin and Ibuprofen. Can you tell me about them?")
# print(response)