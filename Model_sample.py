# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 19:01:56 2024

@author: mathe
"""

import torch
import transformers
#from transformers import pipeline

from crewai import Agent, Task, Crew
from langchain.llms import Ollama
import os
#os.environ["OPENAI_API_KEY"] = "NA"

llm = Ollama(
    model = "llama2",
    base_url = "http://localhost:11434")



planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You're working on plannig a formal report "
              "about the topic: {topic}."
              "You collect information that helps the "
              "readers learn something "
              "and make insightful decisions. "
              "the content writer uses the insights you provide"
              "to write a report on this topic.",
    allow_delegation=False,
	verbose=True,
    llm = llm
)


# ### Agent: Writer

# In[ ]:


writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate "
         "opinion piece about the topic: {topic}",
    backstory="You're working on a writing "
              "a new opinion piece about the topic: {topic}. "
              "Your writing should be based on the work of "
              "the Content Planner, who provides an outline "
              "and relevant context about the topic. "
              "You follow the main objectives and "
              "direction of the outline, "
              "as provided by the Content Planner. "
              "You also provide objective and impartial insights "
              "and back them up with information "
              "provide by the Content Planner. "
              "You acknowledge in your opinion piece "
              "when your statements are opinions "
              "as opposed to objective statements.",
    allow_delegation=False,
    verbose=True,
    llm = llm
)


# ### Agent: Editor

# In[ ]:


editor = Agent(
    role="Editor",
    goal="Edit a given report to align with "
         "the writing style of the organization. ",
    backstory="You are an editor who receives a report "
              "from the Content Writer. "
              "Your goal is to review the report "
              "to ensure that it follows proper structure,"
              "provides balanced viewpoints and has standard writing "
              "when providing opinions or assertions, "
              "and also avoids major controversial topics "
              "or opinions when possible.",
    allow_delegation=False,
    verbose=True,
    llm = llm
)


# ## Creating Tasks
# 
# - Define your Tasks, and provide them a `description`, `expected_output` and `agent`.

# ### Task: Plan

# In[ ]:


plan = Task(
    description=(
        "1. Prioritize the most important factors, key points, "
            "and avoid the irrelevant elements on {topic}.\n"
        "2. Identify the target readers, considering "
            "their interests and weaknesses.\n"
        "3. Develop a detailed content outline including "
            "an introduction, a body, conclusion and references.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output="A comprehensive content plan document "
        "with an outline, important analysis, "
        "SEO keywords, and resources.",
    agent=planner,
    llm = llm
)


# ### Task: Write

# In[ ]:


write = Task(
    description=(
        "1. Use the content plan to craft a compelling "
            "report on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
		"3. Sections/Subtitles are properly named "
            "in an engaging manner.\n"
        "4. Ensure the report is structured with an "
            "engaging introduction, insightful body, "
            "a summarizing conclusion and proper factual references.\n"
        "5. Proofread for grammatical errors and "
            "alignment with the brand's voice.\n"
    ),
    expected_output="A well-written and structured report "
        "in markdown format, ready for publication, "
        "each section should have 2 or 3 paragraphs.",
    agent=writer,
    llm = llm 
)

edit = Task(
    description=("Proofread the given report for "
                 "grammatical errors and "
                 "alignment with the brand's voice."),
    expected_output="A well-written report in markdown format, "
                    "ready for publication, "
                    "each section should have 2 or 3 paragraphs.",
    agent=editor,
    llm = llm
)


# ### Task: Edit

# In[ ]:


crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=2
)

topic = ""
result = crew.kickoff(inputs={"topic": "topic"})

from IPython.display import Markdown
Markdown(result)



