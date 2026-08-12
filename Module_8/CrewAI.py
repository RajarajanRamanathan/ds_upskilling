import os
from dotenv import load_dotenv
import ollama

load_dotenv()

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field
from typing import List

local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434",
)

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

if os.getenv("SERPER_API_KEY"):
    researcher_tools = [SerperDevTool(), internal_docs_tool]
else:
    print("[info] SERPER_API_KEY not set -- researcher will run without web search.")
    researcher_tools = [internal_docs_tool]

class ResearchFinding(BaseModel):
    topic: str = Field(description="Sub-topic investigated")
    summary: str = Field(description="2-3 sentence summary of findings")
    source_confidence: str = Field(description="high | medium | low")

class MarketReport(BaseModel):
    executive_summary: str
    findings: List[ResearchFinding]
    recommendation: str
    
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
