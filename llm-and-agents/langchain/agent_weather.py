from load_env import load_env_file
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
import random
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from dataclasses import dataclass
from langchain.agents import create_agent

load_env_file()

# Step 1: System Prompt - The Agent's initial instructions or personality.
system_prompt = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean whereever they are, use the get_user_location tool to find their location."""

# Step 2: Create tools - tools are functions that can be called, they interact with external data to get stuff done.
def get_weather_for_location(city: str) -> str:
    '''Get weather for a given city'''
    conditions = random.choice(['sunny', 'rainy', 'cloudy'])
    return f'It is {conditions} in {city}'


# A lookup table for demo purposes
USER_LOCATION = {
    "1":"Florida",
    "2":"SF"
}

'''
@tool decorator turns Python callables into LangChain `StructuredTool` objects
that the agent can discover and invoke. It can then use LangChain's tool metadata 
like names, descriptions, config injections.
'''
@tool
def get_user_location(config: RunnableConfig) -> str:
    '''Retrieve user information'''
    user_id = config.get("configurable", {}).get("user_id")
    return USER_LOCATION[user_id]

# Step 3: Configure the model
model = init_chat_model(
    "openai:gpt-4o-mini",
    temperature=0,
)
# Step 4: Define response format
@dataclass
class WeatherResponse:
    conditions: str
    punny_response: str

# Step 5: Add memory for the agent to remember conversation history
checkpointer = InMemorySaver()

# Step 6: Bring it all together
agent = create_agent(
    model=model,
    prompt=system_prompt,
    tools=[get_user_location, get_weather_for_location],
    response_format=WeatherResponse,
    checkpointer=checkpointer
)

# config = {"configurable": {"thread_id": "1"}}
# context = {"user_id": "1"}

'''
`config` is the run metadata shared across every runnable (models, tools, graphs).
`config` has reserved keys like "configurable", "run_name", "tags", "metadata", "callbacks".
"configurable" is a catch-all for values we want to read back inside the graph or tools.
"thread_id" must be supplied when using `InMemorySaver` or any checkpointer - it decides which conversation thread to load.
'''

config = {"configurable": {"thread_id": "1", "user_id": "2"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
)

print(response['structured_response'])

response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
)

print(response['structured_response'])