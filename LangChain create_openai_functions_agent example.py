from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI

# Load API_KEY in .env
load_dotenv()

# System Assistant Purpose
instruction = "You are a helpful assistant"

# Pull Templates
base_prompt = hub.pull("langchain-ai/openai-functions-template")

# input_variables=['agent_scratchpad', 'input', 'instructions'] optional_variables=['chat_history'] ...
prompt = base_prompt.partial(instructions=instruction)

# Model, tools
llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))
# model = "gpt-4o",
# temperature = 0,
# max_tokens = None,
# timeout = None,
# max_retries = 2,
# api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
# base_url="...",
# organization="...",
# other params...

tool = TavilySearchResults(
    tavily_api_key=os.getenv("TAVILY_API_KEY"), include_domains = ["nyt.com", "fp.com"]
)
# max_results= 5
# search_depth = "advanced"
# include_domains = []
# exclude_domains = []
# include_answer = False
# include_raw_content = False
# include_images = False

tools = [tool]

# Agent using the predefined model and tools
agent = create_openai_functions_agent(llm, tools, prompt)

# Execute the Agent
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

result = agent_executor.invoke({"input": "What happened to president Park in South Korea?"})
print(result)
"""
> Entering new AgentExecutor chain...

Invoking: `tavily_search_results_json` with `{'query': 'President Park South Korea news'}`


[]President Park Geun-hye of South Korea was impeached in December 2016 and subsequently removed from office in March 2017. She was accused of corruption and abuse of power, particularly in connection with a scandal involving her close confidante, Choi Soon-sil. Following her removal, Park was tried and convicted on multiple charges, including bribery and abuse of power, and was sentenced to prison. In December 2021, her sentence was pardoned by President Moon Jae-in, allowing her to be released from prison after serving nearly four years. 

If you need more recent updates or specific details, please let me know!

> Finished chain.
{'input': 'What happened to president Park in South Korea?', 'output': 'President Park Geun-hye of South Korea was impeached in December 2016 and subsequently removed from office in March 2017. She was accused of corruption and abuse of power, particularly in connection with a scandal involving her close confidante, Choi Soon-sil. Following her removal, Park was tried and convicted on multiple charges, including bribery and abuse of power, and was sentenced to prison. In December 2021, her sentence was pardoned by President Moon Jae-in, allowing her to be released from prison after serving nearly four years. \n\nIf you need more recent updates or specific details, please let me know!'}
"""