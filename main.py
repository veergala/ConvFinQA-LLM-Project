import json

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

from models import ConvFinQAEntry, FinancialResponse

load_dotenv()


agent = Agent(
    model="gemini-1.5-flash",
    deps_type=ConvFinQAEntry,
    result_type=FinancialResponse,
    system_prompt=(
        "You're a financial analyst assistant specialized in data analysis.\n"
        "For each question:\n"
        "1. ALWAYS read and use ALL provided context:\n"
        "   - Pre-text contains important background information\n"
        "   - Table data contains numerical values\n"
        "   - Post-text contains additional context\n"
        "2. Identify relevant data points from ALL sources\n"
        "3. Show your calculation process clearly\n"
        "4. Format numbers with appropriate precision\n"
        "5. Return structured responses with:\n"
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
                + ". /n"
            )
            table_data += datapoint_string
    print(table_data)
    return table_data


@agent.tool_plain
async def calculate_percentage_change(initial: int, end: int) -> str:
    """Calculate the percentage change between two numbers labelled initial and end."""
    fraction = (initial - end) / initial
    return str((fraction * 100)) + "%"


@agent.tool_plain
async def calculate_growth_rate(initial: float, final: float, periods: int = 1) -> str:
    """Calculate the compound annual growth rate (CAGR) between two values."""
    if initial <= 0 or final <= 0:
        return "Cannot calculate growth rate with zero or negative values"
    cagr = ((final / initial) ** (1 / periods) - 1) * 100
    return f"{cagr:.2f}%"


@agent.tool_plain
async def calculate_financial_ratio(numerator: float, denominator: float) -> str:
    """Calculate financial ratios with proper formatting."""
    if denominator == 0:
        return "Cannot divide by zero"
    ratio = numerator / denominator
    return f"{ratio:.3f}"


@agent.tool_plain
async def format_currency(amount: float, decimals: int = 2) -> str:
    """Format numbers as currency with proper comma separation."""
    return f"${amount:,.{decimals}f}"


def run_qa(row):
    """Run QA evaluation for a given row in the training data.

    Handles both single QA format (data[row]["qa"]) and
    multiple QA format (data[row]["qa_0"], data[row]["qa_1"], etc).
    """
    # Create example data object
    example_data = ConvFinQAEntry(
        pre_text=data[row]["pre_text"],
        post_text=data[row]["post_text"],
        table=data[row]["table"],
    )

    # Check for QA format
    if "qa" in data[row]:
        # Single QA format
        qa_pairs = [("qa", data[row]["qa"])]
    else:
        # Multiple QA format - gather all qa_N keys
        qa_pairs = [
            (f"qa_{i}", data[row][f"qa_{i}"])
            for i in range(10)  # Check up to qa_9
            if f"qa_{i}" in data[row]
        ]

    # Process each QA pair
    for qa_key, qa_data in qa_pairs:
        print(f"\n{'=' * 50}")
        print(f"Processing {qa_key}:")
        print(f"Question: {qa_data['question']}")

        # Run the agent
        result = agent.run_sync(qa_data["question"], deps=example_data)

        # Print results
        print("\nAgent Response:")
        print(f"Answer: {result.data.answer}")
        print(f"Calculation: {result.data.calculation_explanation}")
        print(f"Data Points Used: {result.data.data_points_used}")
        print(f"\nExpected Answer: {qa_data['answer']}")

        # Calculate accuracy (if answers are numeric)
        try:
            actual = float(result.data.answer.rstrip("%"))
            expected = float(qa_data["answer"].rstrip("%"))
            error = abs(actual - expected)
            print(f"Absolute Error: {error:.1f}%")
        except (ValueError, AttributeError):
            # Non-numeric answers
            print("Note: Non-numeric answer comparison")
    return qa_data["answer"]


if __name__ == "__main__":
    # Load JSON data
    with open("train.json", "r") as f:
        data = json.load(f)

    data_size = 0
    error = 0

    for i in range(len(data)):
        data_size += 1
        run_qa(i)
