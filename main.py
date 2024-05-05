import cohere
from cohere import ClassifyExample
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class DocumentClassifyManager(BaseModel):
    classify_model: str
    cohere_api_key: str
    query_text: str
    example_docs: list[dict]


@app.post("/v1/route")
def classify_docs(manager: DocumentClassifyManager) -> str:
    """
    Classifies documents using the specified classify model.

    Parameters:
        manager (DocumentClassifyManager): The manager object containing the classify model, Cohere API key, query text, and example documents.

    Returns:
        str: The prediction of the classify model for the query text.

    Raises:
        None
    """
    classify_model = manager.classify_model
    cohere_api_key = manager.cohere_api_key
    query_text = manager.query_text
    example_docs = manager.example_docs

    cohere_client = cohere.Client(api_key=cohere_api_key)
    response = cohere_client.classify(
        model=classify_model,
        inputs=[query_text],
        examples=[ClassifyExample(**d) for d in example_docs]
    )
    print('The confidence levels of the labels are: {}'.format(response.classifications))
    return response.classifications[0].prediction
