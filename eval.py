import json

import matplotlib.pyplot as plt


def generate_ape_histogram(model, results_json="qa_results.json"):
    with open(results_json, "r") as f:
        results = json.load(f)

    percentage_errors = [
        entry["percentage_error"]
        for entry in results
        if entry.get("model_choice") == model
    ]
    print(max(percentage_errors))
    plt.hist(
        percentage_errors,
        bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600],
        edgecolor="black",
        alpha=0.7,
    )

    # Labels
    plt.xlabel("Absolute Percentage Error")
    plt.ylabel("Frequency")
    plt.title("Distribution of Absolute Percentage Errors")

    plt.savefig("ape_dist_histogram.png", dpi=300, bbox_inches="tight")
    return


def eval_model_ape(model, results_json="qa_results.json"):
    with open(results_json, "r") as f:
        results = json.load(f)

    percentage_errors = [
        entry["percentage_error"]
        for entry in results
        if entry.get("model_choice") == model
    ]

    return sum(percentage_errors) / len(percentage_errors)


print(eval_model_ape("gemini-2.0-flash"))


def eval_model_ape_threshold(model, threshold, results_json="qa_results.json"):
    with open(results_json, "r") as f:
        results = json.load(f)

    percentage_errors = [
        entry["percentage_error"]
        for entry in results
        if entry.get("model_choice") == model
    ]

    count_below_ten = sum(1 for x in percentage_errors if x < threshold)

    return count_below_ten / len(percentage_errors)


print(eval_model_ape_threshold("gemini-2.0-flash", threshold=10))

generate_ape_histogram("gemini-2.0-flash")
