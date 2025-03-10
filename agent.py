from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

from models import AgentResponse, ConvFinQAEntry

load_dotenv()

model_choice = "gemini-2.0-flash"

agent = Agent(
    model=model_choice,
    deps_type=ConvFinQAEntry,
    result_type=AgentResponse,
    system_prompt=(
        "You're a financial analyst assistant specialized in data analysis.\n"
        "For each question:\n"
        "1. ALWAYS read and use ALL provided context:\n"
        "   - Pre-text contains important background information\n"
        "   - Table data contains numerical values\n"
        "   - Post-text contains additional context\n"
        "2. Identify relevant data points from ALL sources\n"
        "3. Show your calculation process clearly\n"
        "4. Do NOT round during intermediate calculations\n"
        "5. The answer MAY be negative\n"
        "6. Give the exact number - do not round or include commas for large numbers\n"
        "7. Where two dates are given and the question asks you to calculate change, assume the change is between the earlier and later year\n"
        "7. Return structured responses with:\n"
        "   - Direct answer to the question\n"
        "   - Explanation of calculations\n"
        "   - List of data points used\n"
        "\nExample format:\n"
        "{\n"
        "  'answer': '70.1%',\n"
        "  'calculation_explanation': 'Total = 2,530,454 + 5,923,147...',\n"
        "  'data_points_used': ['GIP: 2,530,454', 'SIP: 5,923,147']\n"
        "}"
    ),
)


@agent.tool
async def provide_context(ctx: RunContext[ConvFinQAEntry]) -> str:
    return (
        f"Pre-Text: {ctx.deps.pre_text}\n"
        f"Post-Text: {ctx.deps.post_text}\n"
        f"Table Data: {ctx.deps.table}"
    )


@agent.tool
async def extract_table_data(ctx: RunContext[ConvFinQAEntry]) -> str:
    """Function which extracts the relevant data from the table to make it readable by the model."""
    table_data = ""
    columns = ctx.deps.table[0][1:]
    data = ctx.deps.table[1:]
    for row in data:
        for i in range(1, len(row)):
            datapoint_string = (
                "For "
                + str(columns[i - 1])
                + ", the "
                + str(row[0])
                + " was "
                + str(row[i])
                + ". \n"
            )
            table_data += datapoint_string
    print(table_data)
    return table_data


@agent.tool_plain
async def calculate_percentage_change(initial: int, end: int) -> str:
    """Calculate the percentage change between two numbers, where initial is chronologically before end."""
    fraction = (end - initial) / initial
    return str((fraction * 100)) + "%"


@agent.tool_plain
async def calculate_growth_rate(initial: float, final: float, periods: int = 1) -> str:
    """Calculate the compound annual growth rate (CAGR) between two values."""
    if initial <= 0 or final <= 0:
        return "Cannot calculate growth rate with zero or negative values"
    cagr = ((final / initial) ** (1 / periods) - 1) * 100
    return f"{cagr:.2f}%"
