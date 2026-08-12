import os
from dotenv import load_dotenv
import ollama

load_dotenv()

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field
from typing import List

# ---------------------------------------------------------------
# LLM CONFIG -- no OpenAI key required.
# Uses a local Ollama model by default (100% free, runs on your
# machine). Install: https://ollama.com/download, then:
#     ollama pull llama3.2
# Ollama runs automatically as a background service on port 11434.
#
# To use a hosted free-tier LLM instead (e.g. Groq -- very fast,
# generous free tier, just needs a free API key from console.groq.com),
# comment out the Ollama line below and uncomment the Groq one.
# ---------------------------------------------------------------

local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434",
)

# local_llm = LLM(model="groq/llama-3.3-70b-versatile")  # needs GROQ_API_KEY env var (free)

# ---------------------------------------------------------------
# 1. TOOLS
# Agents are only as good as the tools you give them. Here we use
# a built-in search tool. You can also write custom tools by
# subclassing BaseTool -- useful for hitting your own internal
# APIs (Jira, Confluence, an internal knowledge base, etc.)
# ---------------------------------------------------------------

class InternalDocsLookupTool(BaseTool):
    """Example of a custom tool -- swap this for a real internal API call."""
    name: str = "Internal Docs Lookup"
    description: str = (
        "Searches internal architecture docs / wikis for context on a topic. "
        "Input should be a short search phrase."
    )

    def _run(self, query: str) -> str:
        # In production: call your Confluence/SharePoint/vector DB here.
        return f"[stub] No internal docs indexed yet for: '{query}'"


internal_docs_tool = InternalDocsLookupTool()

# SerperDevTool needs a free SERPER_API_KEY (serper.dev, 2500 free
# searches/month, no credit card). If you don't have one yet, the
# researcher simply runs without web search -- it'll rely on the
# internal docs stub tool only. Add SERPER_API_KEY to your .env
# once you're ready to enable real web search.
if os.getenv("SERPER_API_KEY"):
    researcher_tools = [SerperDevTool(), internal_docs_tool]
else:
    print("[info] SERPER_API_KEY not set -- researcher will run without web search.")
    researcher_tools = [internal_docs_tool]


# ---------------------------------------------------------------
# 2. STRUCTURED OUTPUT SCHEMA
# Forcing structured output (instead of free-text) is what makes
# CrewAI usable in production pipelines -- you can pass this JSON
# straight into a downstream system.
# ---------------------------------------------------------------

class ResearchFinding(BaseModel):
    topic: str = Field(description="Sub-topic investigated")
    summary: str = Field(description="2-3 sentence summary of findings")
    source_confidence: str = Field(description="high | medium | low")


class MarketReport(BaseModel):
    executive_summary: str
    findings: List[ResearchFinding]
    recommendation: str


# ---------------------------------------------------------------
# 3. AGENTS
# role/goal/backstory aren't just flavor text -- they're injected
# directly into the system prompt CrewAI builds for the underlying
# LLM call. Precision here materially changes output quality.
# ---------------------------------------------------------------

researcher = Agent(
    role="Senior Market Research Analyst",
    goal="Find accurate, current, well-sourced information on {topic}",
    backstory=(
        "You are a meticulous analyst who cross-checks claims across "
        "multiple sources before reporting them. You flag low-confidence "
        "findings instead of guessing."
    ),
    tools=researcher_tools,
    llm=local_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=8,          # cap tool-call loops to control cost/latency
)

analyst = Agent(
    role="Principal Technology Analyst",
    goal="Synthesize raw research into structured, decision-ready findings",
    backstory=(
        "You've advised engineering leadership for 15 years. You cut "
        "noise, call out risks plainly, and never pad findings with "
        "generic filler."
    ),
    llm=local_llm,
    verbose=True,
    allow_delegation=False,
)

writer = Agent(
    role="Technical Report Writer",
    goal="Produce a concise, executive-ready report from analyzed findings",
    backstory=(
        "You write for busy engineering leaders. Every sentence earns "
        "its place. No fluff, no repetition of the brief."
    ),
    llm=local_llm,
    verbose=True,
    allow_delegation=False,
)


# ---------------------------------------------------------------
# 4. TASKS
# `context=[...]` is the key wiring mechanism: it pipes one task's
# output into the next task's prompt, without you manually
# stitching strings together.
# ---------------------------------------------------------------

research_task = Task(
    description=(
        "Research the current state of {topic}. Cover: adoption trends, "
        "key competitors/alternatives, and any recent (last 6 months) "
        "notable changes. Use the search tool -- do not rely on memory."
    ),
    expected_output="A raw bulleted list of findings with source notes.",
    agent=researcher,
)

analysis_task = Task(
    description=(
        "Take the raw research and identify the 3-5 findings that "
        "actually matter for an engineering leadership decision. "
        "Discard anything speculative or unsourced."
    ),
    expected_output="A structured list of ResearchFinding objects.",
    agent=analyst,
    context=[research_task],       # <-- depends on research_task's output
)

report_task = Task(
    description=(
        "Write the final market report for {topic}, aimed at a VP of "
        "Engineering deciding whether to adopt it. Include an explicit "
        "recommendation."
    ),
    expected_output="A MarketReport JSON object.",
    agent=writer,
    context=[analysis_task],
    output_pydantic=MarketReport,  # forces structured, parseable output
)

# ---------------------------------------------------------------
# 5. CREW
# Sequential process = task order below is execution order.
# Swap Process.sequential -> Process.hierarchical to instead have
# a manager LLM dynamically decide delegation (useful when task
# order can't be known ahead of time).
# ---------------------------------------------------------------

research_crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, report_task],
    process=Process.sequential,
    memory=True,          # enables short-term + entity memory across the run
    verbose=True,
)


if __name__ == "__main__":
    result = research_crew.kickoff(
        inputs={"topic": "AI agent orchestration frameworks for enterprise .NET shops"}
    )

    # result.pydantic gives you the typed MarketReport object directly
    report: MarketReport = result.pydantic
    print(report.executive_summary)
    for finding in report.findings:
        print(f"- [{finding.source_confidence}] {finding.topic}: {finding.summary}")
    print("\nRecommendation:", report.recommendation)