#!/usr/bin/env python3
"""LangChain integration demo.

Uses NeuroCognitiveLLM as a drop-in LangChain LLM. Every call is
routed through NEURO's 38 cognitive modules (memory, reasoning,
causal analysis) before hitting the underlying model.

Requires the langchain extra:
    pip install elo-agi[langchain]
"""

from neuro.integrations.langchain import NeuroCognitiveLLM

# Create a LangChain-compatible LLM backed by NEURO
llm = NeuroCognitiveLLM(provider="auto")

# Use it like any LangChain LLM
response = llm.invoke("Explain quantum entanglement in two sentences.")
print(response)

# Works with LangChain chains and prompts
from langchain_core.prompts import PromptTemplate  # noqa: E402

prompt = PromptTemplate.from_template("Summarize {topic} for a beginner.")
chain = prompt | llm
print(chain.invoke({"topic": "neural networks"}))
