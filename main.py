import json

# from typing import Dict, List
import pandas as pd

# from pandasai import Agent as pandasagent
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel

# Load JSON data
with open("train.json", "r") as f:
    data = json.load(f)

# df = pd.DataFrame(data[1]["table"]).transpose()

# df.columns = df.iloc[0]
# df = df[1:]

# # Set the first column (which now contains the years) as the index
# df = df.set_index("")
def lists_to_dict(lists):
    result = {}
    for sublist in lists:
        if sublist:
            header = sublist[0]
            values = sublist[1:]
            result[header] = values
    return result

class ConvFinQAEntry(BaseModel):
    pre_text: list
    post_text: list
    table: dict

    class Config:
        arbitrary_types_allowed = True


example_data = ConvFinQAEntry(
    pre_text=data[1]["pre_text"], post_text=data[1]["post_text"], table=lists_to_dict(data[1]["table"])
)

model = GeminiModel(
    "gemini-1.5-flash", api_key="AIzaSyDGSPa_xyfjokrTjBm-6PB0oxaI1r59GXQ"
)

agent = Agent(
    model=model,
    deps_type=json,
    system_prompt=(
        "You're a financial assistant. You should read the question, the pre-text,"
        "the table, and the post-text and use this information, making relevant calculations"
        "where required, to answer the question based on the data"
    ),
)


@agent.tool
async def provide_context(ctx: RunContext[ConvFinQAEntry]) -> str:
    return (
        f"Pre-Text: {ctx.deps.pre_text}\n"
        f"Post-Text: {ctx.deps.post_text}\n"
        f"Table Data: {ctx.deps.table}"
    )


# @agent.tool
# async def df_query(ctx: RunContext[ConvFinQAEntry], query: str) -> str:
#     """A tool for running queries on the `pandas.DataFrame`. Use this tool to interact with the DataFrame.
#     `query` will be executed using `pd.eval(query, target=df)`, so it must contain syntax compatible with
#     `pandas.eval`.
#     """

#     # Print the query for debugging purposes and fun :)
#     print(f"Running query: `{query}`")
#     # Execute the query using `pd.eval` and return the result as a string (must be serializable).
#     return str(pd.eval(query, target=ctx.deps.table))


@agent.tool_plain
async def add(a: int, b: int) -> str:
    """Add two numbers"""
    return str(a + b)


qa_result = agent.run_sync(data[1]["qa"]["question"], deps=example_data)
print(qa_result.data)
