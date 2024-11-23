from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas import RunConfig, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings

from typing import TypedDict

class RagasDataset(TypedDict):
    question: str
    contexts: list[str]
    answer: str

def init_ragas_metrics(metrics, llm, embedding):
    for metric in metrics:
        if isinstance(metric, MetricWithLLM):
            metric.llm = llm
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = embedding
        run_config = RunConfig()
        metric.init(run_config)


def eval_ragas(data: RagasDataset):
    context_precision_without_reference = LLMContextPrecisionWithoutReference()
    metrics = [ResponseRelevancy(), context_precision_without_reference, faithfulness]
    llm = ChatOpenAI()
    emb = OpenAIEmbeddings()
    init_ragas_metrics(
        metrics,
        llm=LangchainLLMWrapper(llm),
        embedding=LangchainEmbeddingsWrapper(emb),
    )
    scores = {}
    sample = SingleTurnSample(
        user_input=data["question"],
        retrieved_contexts=data["contexts"],
        response=data["answer"],
    )
    for m in metrics:
        print(f"calculating {m.name}")
        scores[m.name] = m.single_turn_score(
            sample
        )
    return scores