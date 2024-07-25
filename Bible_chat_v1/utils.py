from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import RunnableSerializable
import os
from langchain_openai import ChatOpenAI


def load_prompt() -> [str]:
    return [
        (
            "system",
            """
            Answer my question ONLY by using verses from the Bible.
            Answer with multiple verses if you find many related verses (Max number of verses = 5)

            Format as below:

            Ephesians 5:25
            "Husbands, love your wives, just as Christ loved the church and gave himself up for her."

            1 Corinthians 13:4-7
            "Love is patient, love is kind. It does not envy, it does not boast, it is not proud. It does not dishonor others, it is not self-seeking, it is not easily angered, it keeps no record of wrongs. Love does not delight in evil but rejoices with the truth. It always protects, always trusts, always hopes, always perseveres."

            Summary: "Love your girlfriend with patience and kindness, honoring her and treating her with respect, just as Christ loved the church."
            """
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]


def load_model(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(model_name=model_name, temperature=0.3, max_tokens=300, api_key=os.getenv("OPENAI_API_KEY"))


def initialize_chain(model_name: str, prompt: ChatPromptTemplate.from_messages) -> RunnableSerializable:
    return prompt | load_model(model_name)
