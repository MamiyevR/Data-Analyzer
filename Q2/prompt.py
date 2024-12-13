from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "# Causal Graph Instructions\n"
    "## 1. Overview\n"
    "You are a top-tier algorithm designed for extracting financial information in structured "
    "formats to build a causal graph.\n"
    "Try to capture as much information from the text as possible without "
    "sacrificing accuracy. Do not add any information that is not explicitly "
    "mentioned in the text.\n"
    "- **Nodes** represent financial entities and concepts.\n"
    "- The aim is to achieve simplicity and clarity in the causal graph, making it\n"
    "accessible for a vast audience.\n"
    "## 2. Labeling Nodes\n"
    "- **Consistency**: Ensure you use available types for node labels.\n"
    "Ensure you use basic or elementary types for node labels.\n"
    "- For example, when you identify an entity representing a person, "
    "always label it as **'person'**. Avoid using more specific terms "
    "like 'mathematician' or 'scientist'."
    "- **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
    "names or human-readable identifiers found in the text.\n"
    "- **Relationships** represent causal relationship between entities or concepts.\n"
    "Ensure consistency and generality in relationship types when constructing "
    "causal graphs. Instead of using specific and momentary types "
    "such as 'CAUSES_INFLATION' in an example of Interest Rate causes Inflation, "
    "use more general and timeless relationship types "
    "like 'CAUSES'. Make sure to use general and timeless relationship types!\n"
    "## 3. Coreference Resolution\n"
    "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
    "ensure consistency.\n"
    'If an entity, such as "Stock Price", is mentioned multiple times in the text '
    'but is referred to by different names (e.g., "stock prices", "market value"),'
    "always use the most complete identifier for that entity throughout the "
    'causal graph. In this example, use "Stock Price" as the entity ID.\n'
    "Remember, the causal graph should be coherent and easily understandable, "
    "so maintaining consistency in entity references is crucial.\n"
    "## 4. Strict Compliance\n"
    "Adhere to the rules strictly. Non-compliance will result in termination."
)

default_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        (
            "human",
            (
                "Tip: Make sure to answer in the correct format and do "
                "not include any explanations. "
                "Use the given format to extract information from the "
                "following input: {input}"
            ),
        ),
    ]
)
