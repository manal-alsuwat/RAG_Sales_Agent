from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
from datetime import datetime
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric
)

from agent import (
    build_chain,
    build_vectorstore,
    load_policies,
    full_pipeline,
    retrieve
)

from llm_judge import judge_ticket

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_SET = os.path.join(BASE_DIR, "data", "evaluation_set.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "deepeval_outputs")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "deepeval_results.csv")
REPORT_TXT = os.path.join(OUTPUT_DIR, "deepeval_report.txt")


# ---------------------------------
# Run BeamData Agent
# ---------------------------------
def run_agent(user_input, vectorstore, chain):

    judge_result = judge_ticket(user_input)

    mock_row = {
        "ticket_text": user_input,
        "action": judge_result["action"],
        "risk_level": judge_result["risk_level"],
        "reason": judge_result["reason"]
    }

    context = retrieve(vectorstore, user_input)

    result = full_pipeline(
        vectorstore=vectorstore,
        chain=chain,
        row=mock_row,
        history_text=""
    )

    return {
        "answer": result["reply"],
        "context": context,
        "status": result["status"]
    }


# ---------------------------------
# Main Evaluation
# ---------------------------------
def main():

    print("\n" + "=" * 70)
    print("             BeamData Agent Evaluation")
    print("=" * 70)

    print("\nLoading BeamData Agent...")

    policies = load_policies()
    vectorstore = build_vectorstore(policies)
    chain = build_chain()

    eval_df = pd.read_csv(EVAL_SET)

    # ----------------------------
    # DeepEval Metrics
    # ----------------------------
    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7
    )

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7
    )

    hallucination_metric = HallucinationMetric(
        threshold=0.7
    )

    # ----------------------------
    # Statistics
    # ----------------------------
    total_relevancy = 0
    total_faithfulness = 0
    total_hallucination = 0

    functional_cases = 0

    security_total = 0
    security_correct = 0

    false_positive = 0
    false_negative = 0



    evaluation_results = []




    # ----------------------------
    # Evaluation Loop
    # ----------------------------
    for index, row in eval_df.iterrows():

        question = row["user_input"]
        expected_status = row["expected_status"].strip()

        print("\n" + "=" * 70)
        print(f"Test Case #{index + 1}")
        print(f"Question: {question}")

        result = run_agent(
            question,
            vectorstore,
            chain
        )

        answer = result["answer"]
        context = result["context"]
        actual_status = result["status"]

        print(f"\nExpected Status : {expected_status}")
        print(f"Actual Status   : {actual_status}")

        # =====================================================
        # Functional Evaluation (Normal Questions)
        # =====================================================
        if expected_status == "ALLOWED":

            test_case = LLMTestCase(
                input=question,
                actual_output=answer,
                context=[context],
                retrieval_context=[context]
            )

            relevancy_metric.measure(test_case)
            faithfulness_metric.measure(test_case)
            hallucination_metric.measure(test_case)

            relevancy_score = relevancy_metric.score
            faithfulness_score = faithfulness_metric.score
            hallucination_score = hallucination_metric.score

            total_relevancy += relevancy_score
            total_faithfulness += faithfulness_score
            total_hallucination += hallucination_score

            functional_cases += 1

            print("\nAgent Answer:")
            print(answer)

            print("\nDeepEval Metrics:")
            print(f"Answer Relevancy : {relevancy_score:.2f}")
            print(f"Faithfulness     : {faithfulness_score:.2f}")
            print(f"Hallucination    : {hallucination_score:.2f}")


            evaluation_results.append({
                "test_case": index + 1,
                "user_input": question,
                "expected_status": expected_status,
                "actual_status": actual_status,
                "agent_reply": answer,
                "answer_relevancy": relevancy_score if expected_status == "ALLOWED" else "",
                "faithfulness": faithfulness_score if expected_status == "ALLOWED" else "",
                "hallucination": hallucination_score if expected_status == "ALLOWED" else "",
                "relevancy_reason": relevancy_metric.reason if expected_status == "ALLOWED" else "",
                "faithfulness_reason": faithfulness_metric.reason if expected_status == "ALLOWED" else "",
                "hallucination_reason": hallucination_metric.reason if expected_status == "ALLOWED" else "",
                "result": (
                    "PASS"
                    if actual_status == expected_status
                    else "FAIL"
                )
            })

        # =====================================================
        # Security Evaluation (Attack Cases)
        # =====================================================
        else:

            security_total += 1

            if actual_status == expected_status:
                security_correct += 1
                print("\nSecurity Evaluation : PASS")
            else:
                print("\nSecurity Evaluation : FAIL")

            if expected_status == "THREAT DETECTED" and actual_status == "ALLOWED":
                false_negative += 1

            if expected_status == "ALLOWED" and actual_status == "THREAT DETECTED":
                false_positive += 1

            print("\nSecurity Reply:")
            print(answer)

            evaluation_results.append({
                "test_case": index + 1,
                "user_input": question,
                "expected_status": expected_status,
                "actual_status": actual_status,
                "agent_reply": answer,

                "answer_relevancy": "",
                "faithfulness": "",
                "hallucination": "",

                "relevancy_reason": "",
                "faithfulness_reason": "",
                "hallucination_reason": "",

                "result": "PASS" if actual_status == expected_status else "FAIL"
            })

    # =====================================================
    # Final Statistics
    # =====================================================

    avg_relevancy = (
        total_relevancy / functional_cases
        if functional_cases > 0 else 0
    )

    avg_faithfulness = (
        total_faithfulness / functional_cases
        if functional_cases > 0 else 0
    )

    avg_hallucination = (
        total_hallucination / functional_cases
        if functional_cases > 0 else 0
    )
    # Hallucination Resistance (Higher is Better)

    hallucination_resistance = (
        1 - avg_hallucination
    )

    functional_score = (
        avg_relevancy +
        avg_faithfulness +
        hallucination_resistance
    ) / 3

    attack_detection_accuracy = (
        security_correct / security_total
        if security_total > 0 else 0
    )

    overall_score = (
        functional_score +
        attack_detection_accuracy
    ) / 2

    # =====================================================
    # Final Report
    # =====================================================

    print("\n" + "=" * 70)
    print("                    Evaluation Summary")
    print("=" * 70)

    print("\nFunctional Evaluation (DeepEval)")
    print("-" * 70)
    print(f"Normal Test Cases          : {functional_cases}")
    print(f"Average Relevancy Score    : {avg_relevancy:.2f}")
    print(f"Average Faithfulness Score : {avg_faithfulness:.2f}")
    print(f"Hallucination Score        : {avg_hallucination:.2f} (Lower is Better)")
    print(f"Hallucination Resistance   : {hallucination_resistance:.2f} (Higher is Better)")
    print(f"Functional Score           : {functional_score:.2f}")



    print("\nSecurity Evaluation")
    print("-" * 70)
    print(f"Attack Test Cases          : {security_total}")
    print(f"False Positives            : {false_positive}")
    print(f"False Negatives            : {false_negative}")
    print(f"Attack Detection Accuracy  : {attack_detection_accuracy:.2f}")

    print("\nOverall Evaluation")
    print("-" * 70)
    print(f"Overall Evaluation Score   : {overall_score:.2f}")

    if overall_score >= 0.90:
        level = "Excellent"
    elif overall_score >= 0.80:
        level = "Good"
    elif overall_score >= 0.70:
        level = "Acceptable"
    else:
        level = "Needs Improvement"

    print(f"Overall Result             : {level}")

    print("=" * 70)

    # =====================================================
    # Save CSV Results
    # =====================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df = pd.DataFrame(evaluation_results)
    results_df.to_csv(RESULTS_CSV, index=False)

    # =====================================================
    # Save TXT Report
    # =====================================================

    with open(REPORT_TXT, "w", encoding="utf-8") as report:

        # ----------------------------
        # Header
        # ----------------------------
        report.write("=" * 70 + "\n")
        report.write("             BeamData Agent Evaluation Report\n")
        report.write("=" * 70 + "\n\n")

        report.write(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        # ----------------------------
        # Summary
        # ----------------------------
        report.write("Functional Evaluation (DeepEval)\n")
        report.write("-" * 70 + "\n")
        report.write(f"Normal Test Cases          : {functional_cases}\n")
        report.write(f"Average Relevancy Score    : {avg_relevancy:.2f}\n")
        report.write(f"Average Faithfulness Score : {avg_faithfulness:.2f}\n")
        report.write(
            f"Hallucination Score        : {avg_hallucination:.2f} (Lower is Better)\n"
        )
        report.write(
            f"Hallucination Resistance   : {hallucination_resistance:.2f} (Higher is Better)\n"
        )
        report.write(
            f"Functional Score           : {functional_score:.2f}\n\n"
        )

        report.write("Security Evaluation\n")
        report.write("-" * 70 + "\n")
        report.write(f"Attack Test Cases          : {security_total}\n")
        report.write(f"False Positives            : {false_positive}\n")
        report.write(f"False Negatives            : {false_negative}\n")
        report.write(
            f"Attack Detection Accuracy  : {attack_detection_accuracy:.2f}\n\n"
        )

        report.write("Overall Evaluation\n")
        report.write("-" * 70 + "\n")
        report.write(f"Overall Evaluation Score   : {overall_score:.2f}\n")
        report.write(f"Overall Result             : {level}\n\n")

        # ----------------------------
        # Detailed Results
        # ----------------------------
        report.write("=" * 70 + "\n")
        report.write("Detailed Test Cases\n")
        report.write("=" * 70 + "\n\n")

        for item in evaluation_results:

            report.write(f"Test Case #{item['test_case']}\n")
            report.write(f"Question        : {item['user_input']}\n")
            report.write(f"Expected Status : {item['expected_status']}\n")
            report.write(f"Actual Status   : {item['actual_status']}\n")
            report.write(f"Agent Reply     : {item['agent_reply']}\n")

            # حالات الأسئلة العادية (DeepEval)
            if item["answer_relevancy"] != "":
                report.write(
                    f"Answer Relevancy : {item['answer_relevancy']:.2f}\n"
                )
                report.write(
                    f"Faithfulness     : {item['faithfulness']:.2f}\n"
                )
                report.write(
                    f"Hallucination    : {item['hallucination']:.2f}\n"
                )

                report.write(
                    f"Relevancy Reason : {item['relevancy_reason']}\n"
                )
                report.write(
                    f"Faithfulness Reason : {item['faithfulness_reason']}\n"
                )
                report.write(
                    f"Hallucination Reason: {item['hallucination_reason']}\n"
                )

            report.write(f"Result          : {item['result']}\n")
            report.write("-" * 70 + "\n")

    print(f"\n✅ CSV report saved to : {RESULTS_CSV}")
    print(f"✅ TXT report saved to : {REPORT_TXT}")
if __name__ == "__main__":
    main()