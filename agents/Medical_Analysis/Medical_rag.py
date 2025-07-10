import json
import pandas as pd
from agents.Medical_Analysis.Medical_config import DRUG_EXTRACTOR_MODEL_NAME, DRUG_INFO_MODEL_NAME, NEO4j_PASSWORD, NEO4j_URI, NEO4j_USERNAME
from agents.Utils.common_methods import encode_image, get_sambanova_response
from neo4j import GraphDatabase
from itertools import combinations


class MedicalAgent :
    def __init__(self):
        self.driver = GraphDatabase.driver( NEO4j_URI , auth=(NEO4j_USERNAME , NEO4j_PASSWORD ) )


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
        prompt = """# Drug Name Extraction System Prompt

    ```
    You are a medical AI assistant specialized in extracting drug names from text and images.

    Your task is to identify and extract all drug names (generic and brand names) mentioned in the provided content.

    INSTRUCTIONS:
    1. Extract ALL drug names, including:
    - Generic drug names (e.g., acetaminophen, ibuprofen)
    - Brand names (e.g., Tylenol, Advil)
    - Prescription medications
    - Over-the-counter medications
    - Supplements and vitamins with specific drug names

    2. DO NOT include:
    - General terms like "medication", "pills", "tablets"
    - Dosage information (e.g., "500mg")
    - Medical conditions or symptoms
    - Non-drug substances unless they are medications

    3. Return your response in the following JSON format ONLY:
    {
        "drug_names": ["drug1", "drug2", "drug3"],
        "confidence": [0.95, 0.87, 0.92]
    }

    4. If no drug names are found, return:
    {
        "drug_names": [],
        "confidence": []
    }

    5. Confidence scores should be between 0.0 and 1.0, where:
    - 1.0 = Absolutely certain it's a drug name
    - 0.8-0.9 = Very confident
    - 0.6-0.7 = Moderately confident
    - Below 0.6 = Low confidence (consider excluding)

    IMPORTANT: Return ONLY the JSON response, no additional text or explanations.
    ```"""
        return prompt
    
    def complete_graphrag_search(self, extracted_drug_names_list: list[str]):
        drug_list = extracted_drug_names_list
        pairs = list(combinations(drug_list, 2))

        with self.driver.session() as session:
            drug_infos = session.read_transaction(self.get_drug_info, drug_list)
            interaction_infos = session.read_transaction(self.get_interactions, pairs)

        return drug_infos, interaction_infos

    def drug_extractor(self , isImage : bool , image_source : str = None , query : str = ""):

        extract_messages = [
        {
            "role": "system",
            "content": self.get_drugListExtractor_systemPrompt()
        }
        ]

        if(isImage):
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
             
        extract_response = get_sambanova_response(extract_messages, model = DRUG_EXTRACTOR_MODEL_NAME )
        extracted_drug_names = json.loads(extract_response[extract_response.find("{"): extract_response.rfind("}")+1])["drug_names"]
        return extracted_drug_names
    
    
    def get_responder_output(self, isImage : bool , image_source : str = None , query : str = ""):
        extracted_drug_names_list = self.drug_extractor(isImage, image_source, query)
        drug_infos, interaction_infos = self.complete_graphrag_search(extracted_drug_names_list)
        user_prompt = self.get_drugInfo_userPrompt(drug_infos, interaction_infos)
        messages=[
            {"role": "system", "content": "You are a helpful clinical assistant."},
            {"role": "user", "content": user_prompt}
        ]

        response = get_sambanova_response(messages, model = DRUG_INFO_MODEL_NAME)

        return response
    
# medical_agent = MedicalAgent()
# # Example usage:
# response = medical_agent.get_responder_output(isImage=False, query="I have been prescribed Aspirin and Ibuprofen. Can you tell me about them?")
# print(response)