import os

import cohere
from cohere import ClassifyExample
from dotenv import find_dotenv, load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel

_ = load_dotenv(find_dotenv())

app = FastAPI()

COHERE_API_KEY = os.environ["COHERE_API_KEY"]
cohere_client = cohere.Client(api_key=COHERE_API_KEY)


class DocumentClassifyManager(BaseModel):
    classify_model: str
    query_text: str
    example_docs: list[dict]


@app.post("/classify/")
def classify_docs(manager: DocumentClassifyManager) -> str:
    classify_model = manager.classify_model
    query_text = manager.query_text
    example_docs = manager.example_docs

    response = cohere_client.classify(
        model=classify_model,
        inputs=[query_text],
        examples=[ClassifyExample(**d) for d in example_docs]
    )
    print('The confidence levels of the labels are: {}'.format(response.classifications))
    return response.classifications[0].prediction

        # examples=[ClassifyExample(
        #     text="I want to set up a recurring monthly transfer between my chequing and savings account.  How do I do this?",
        #     label="Savings accounts (chequing & savings)"), ClassifyExample(
        #     text="I would like to add my wife to my current chequing account so it\'s a joint account. What do I need to do?",
        #     label="Savings accounts (chequing & savings)"),
        #           ClassifyExample(text="Can I set up automated payment for my bills?",
        #                           label="Savings accounts (chequing & savings)"), ClassifyExample(
        #         text="Interest rates are going up - does this impact the interest rate in my savings account?",
        #         label="Savings accounts (chequing & savings)"),
        #           ClassifyExample(text="What is the best option for a student savings account?",
        #                           label="Savings accounts (chequing & savings)"), ClassifyExample(
        #         text="My family situation is changing and I need to update my risk profile for my equity investments",
        #         label="Investments"),
        #           ClassifyExample(text="Where can I see the YTD return in my investment account?", label="Investments"),
        #           ClassifyExample(text="How can I change my beneficiaries of my investment accounts?",
        #                           label="Investments"),
        #           ClassifyExample(text="Is crypto an option for my investment account?", label="Investments"),
        #           ClassifyExample(text="How often do you rebalance your investment portfolios?", label="Investments"),
        #           ClassifyExample(text="What is the monthly fee on the investment accounts?", label="Investments"),
        #           ClassifyExample(text="How can I withdraw funds from my investment account?", label="Investments"),
        #           ClassifyExample(text="Can I buy stocks and ETFs listed on non Canadian exchanges?",
        #                           label="Investments"),
        #           ClassifyExample(text="How can I minimize my tax exposure?", label="Taxes"), ClassifyExample(
        #         text="I\"m going to be late filing my ${currentYear - 1} tax returns. Is there a penalty?",
        #         label="Taxes"),
        #           ClassifyExample(text="I\'m going to have a baby in November - what happens to my taxes?",
        #                           label="Taxes"),
        #           ClassifyExample(text="How can I see my ${currentYear - 2} tax assessment?", label="Taxes"),
        #           ClassifyExample(text="When will I get my tax refund back?", label="Taxes"),
        #           ClassifyExample(text="How much does it cost to use your tax filing platform?", label="Taxes"),
        #           ClassifyExample(text="I\'d like to increase my monthly RRSP contributions to my RRSP", label="RRSP"),
        #           ClassifyExample(
        #               text="I want to take advantage of the First Time Home Buyers program and take money out of my RRSP.  How does the program work?",
        #               label="RRSP"), ClassifyExample(text="What is the ${currentYear} RRSP limit?", label="RRSP"),
        #           ClassifyExample(text="Does your system ensure I won\'t overcontribute to my RRSP?", label="RRSP"),
        #           ClassifyExample(text="How do I set up employer contributions to my RRSP", label="RRSP")])

    # if ranker_model == 'BAAI/bge-reranker-v2-minicpm-layerwise':
    #     cross = [(query_text, doc) for doc in unranked_docs]
    #     ce_scores = bge_reranker_v2_minicpm_layerwise.compute_score(cross, batch_size=1, cutoff_layers=[28])
    #     return ce_scores
    # elif ranker_model == 'BAAI/bge-reranker-v2-gemma':
    #     cross = [(query_text, doc) for doc in unranked_docs]
    #     ce_scores = bge_reranker_v2_gemma.compute_score(cross, batch_size=1)
    #     return ce_scores
    # elif ranker_model == 'BAAI/bge-reranker-v2-m3':
    #     cross = [(query_text, doc) for doc in unranked_docs]
    #     ce_scores = bge_reranker_v2_m3.compute_score(cross, batch_size=1)
    #     return ce_scores
    # else:
    #     cross = [(query_text, doc) for doc in unranked_docs]
    #     ce_scores = bge_reranker_v2_m3.compute_score(cross, batch_size=1, cutoff_layers=[28])
    #     return ce_scores
