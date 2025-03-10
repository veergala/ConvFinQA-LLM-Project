import json
import os

from agent import agent, model_choice
from models import ConvFinQAEntry, ResponseMetadata


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
        # Double QA format - gather qa_0 and qa_1
        qa_pairs = [
            (f"qa_{i}", data[row][f"qa_{i}"])
            for i in range(2)
            if f"qa_{i}" in data[row]
        ]

    # Process each QA pair
    response_metadata_list = []
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
            error = abs((actual - expected) / expected) * 100
            print(actual, expected, error)
            print(f"Error: {error}")
        except (ValueError, AttributeError):
            # Non-numeric answers
            print("Note: Non-numeric answer comparison")
            error = None

        # Create ResponseMetadata object
        if error is not None:
            response_metadata = ResponseMetadata(
                answer=result.data.answer,
                calculation_explanation=result.data.calculation_explanation,
                data_points_used=result.data.data_points_used,
                expected_answer=qa_data["answer"],
                percentage_error=error,
                question=qa_data["question"],
                model_choice = model_choice
            )
            response_metadata_list.append(response_metadata)

    return response_metadata_list


if __name__ == "__main__":
    # Load JSON data
    with open("train.json", "r") as f:
        data = json.load(f)

    if os.path.exists("qa_results.json"):
        with open("qa_results.json", "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    # Process the data and collect results
    for i in range(5):
        results = run_qa(i)
        for result in results:
            all_results.append(result.dict())

    # Write the accumulated results to the file, appending to the existing data
    with open("qa_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
