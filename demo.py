# demo for running all framework starting from agent.py then follow all steps
import asyncio
import logging
import os

from dotenv import load_dotenv

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv()

# ─── GENERATE SYNTHETIC DATASET ───────────────────────────────────────────────


def create_sample_dataset():
    """Generate a realistic customer churn dataset for demo."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    n = 1000

    data = {
        "customer_id": range(1001, 1001 + n),
        "tenure_months": np.random.exponential(24, n).astype(int).clip(1, 72),
        "monthly_charges": np.random.normal(65, 30, n).clip(20, 150).round(2),
        "contract_type": np.random.choice(["Month-to-Month", "One Year", "Two Year"], n, p=[0.55, 0.25, 0.20]),
        "internet_service": np.random.choice(["DSL", "Fiber", "None"], n, p=[0.34, 0.44, 0.22]),
        "tech_support": np.random.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "senior_citizen": np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "num_products": np.random.randint(1, 6, n),
        "support_calls": np.random.poisson(2, n).clip(0, 10),
        "payment_method": np.random.choice(
            ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"], n, p=[0.34, 0.23, 0.22, 0.21]
        ),
    }

    df = pd.DataFrame(data)

    # Realistic churn logic (with some noise)
    churn_prob = (
        (df["contract_type"] == "Month-to-Month").astype(float) * 0.3
        + (df["monthly_charges"] > 80).astype(float) * 0.2
        + (df["tenure_months"] < 12).astype(float) * 0.25
        + (df["tech_support"] == "No").astype(float) * 0.1
        + (df["support_calls"] > 4).astype(float) * 0.15
        + (df["internet_service"] == "Fiber").astype(float) * 0.1
        + np.random.uniform(0, 0.1, n)
    ).clip(0, 1)

    df["churned"] = (np.random.random(n) < churn_prob).astype(int)
    df["total_charges"] = (df["tenure_months"] * df["monthly_charges"]).round(2)

    return df


async def run_demo():
    print("\n" + "=" * 70)
    print("  🤖 AUTONOMOUS DATA ANALYSIS AGENT — LIVE DEMO")
    print("=" * 70)

    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        print("\n⚠️GROQ_API_KEY not set.")
        print("Copy .env.example → .env and add your key.")
        return

    from src.agent.agent import DataAnalysisAgent

    # Create dataset
    print("\n📊 Generating synthetic customer churn dataset (1,000 rows)...")
    df = create_sample_dataset()
    churn_rate = df["churned"].mean()
    print(f"   Dataset: {df.shape[0]} customers, {df.shape[1]} features")
    print(f"   Churn rate: {churn_rate:.1%}")
    print(f"   Columns: {list(df.columns)}\n")

    agent = DataAnalysisAgent()

    # ── DEMO TASK ──────────────────────────────────────────────────────────────
    goal = (
        "Identify the top factors driving customer churn. "
        "Calculate churn rates by contract type and tenure segments. "
        "Build a simple logistic regression model and report the most important features. "
        "Give actionable recommendations to reduce churn."
    )

    print(f"🎯 GOAL: {goal}\n")
    print("─" * 70)
    print("📡 Agent is starting... (watch it think, plan, execute, and self-correct)\n")

    step_count = 0
    corrections = 0

    # Stream events
    async for event in agent.run_streaming(goal=goal, dataframe=df):
        etype = event.event_type
        data = event.data

        if etype == "planning_start":
            print("🧠 PLANNING...")

        elif etype == "planning_complete":
            steps = data.get("steps", [])
            print(f"📋 PLAN CREATED ({len(steps)} steps, complexity={data.get('complexity', '?')}):")
            for s in steps:
                icon = {"think": "💭", "code": "💻", "analyze": "🔍", "validate": "✅", "summarize": "📝"}.get(
                    s["step_type"], "•"
                )
                print(f"   {icon} [{s['step_id']}] {s['title']}")
            print(f"\n   Reasoning: {data.get('reasoning', '')[:150]}...")
            print()

        elif etype == "step_start":
            step_count += 1
            icon = {"think": "💭", "code": "💻", "analyze": "🔍", "validate": "✅", "summarize": "📝"}.get(
                data.get("step_type", ""), "•"
            )
            print(f"{icon} STEP {data.get('step_number')}/{data.get('total_steps')}: {data.get('title')}")

        elif etype == "self_correction":
            corrections += 1
            print(f"   🔧 SELF-CORRECTING: {data.get('explanation', '')[:100]}")

        elif etype == "step_complete":
            status = data.get("status", "")
            score = data.get("confidence_score", 0)
            icon = "✅" if status == "success" else "❌"

            print(f"   {icon} {status.upper()} (confidence: {score:.0%})")

            if data.get("key_findings"):
                for f in data["key_findings"][:2]:
                    print(f"      → {f}")

            if data.get("has_figures"):
                print(f"      📊 {len(data.get('figures', []))} figure(s) generated")

            if data.get("code_written"):
                # Show first 3 lines of code
                code_preview = "\n".join(data["code_written"].split("\n")[:3])
                print(f"      Code: {code_preview[:120]}...")

            if data.get("error"):
                print(f"      ⚠️  Error: {data['error'][:100]}")
            print()

        elif etype == "replan_start":
            print(f"\n🔄 REPLANNING: {data.get('reason', '')[:100]}\n")

        elif etype == "complete":
            print("\n" + "=" * 70)
            print("  📊 FINAL REPORT")
            print("=" * 70)
            print("\n📌 EXECUTIVE SUMMARY:")
            print(f"   {data.get('executive_summary', '')}\n")

            findings = data.get("key_findings", [])
            if findings:
                print("🔍 KEY FINDINGS:")
                for i, f in enumerate(findings[:5], 1):
                    print(f"   {i}. {f}")

            print(f"\n⚙️  METHODOLOGY: {data.get('methodology', '')}")
            print("\n📈 STATS:")
            print(f"   Steps taken:      {data.get('steps_taken', 0)}")
            print(f"   Self-corrections: {data.get('self_corrections', 0)}")
            print(f"   Confidence:       {data.get('confidence_score', 0):.0%}")
            print(f"   Total latency:    {data.get('total_latency_ms', 0) / 1000:.1f}s")

            if data.get("limitations"):
                print("\n⚠️  LIMITATIONS:")
                for lim in data["limitations"]:
                    print(f"   - {lim}")

        elif etype == "error":
            print(f"\n❌ ERROR: {data.get('message', 'Unknown error')}")

    # ── EVALUATION ────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("📐 RUNNING EVALUATION...\n")

    # For demo, we'll show what evaluation looks like
    print("  (Evaluation requires completed FinalReport + AgentState objects)")
    print("  In production: call AgentEvaluator.evaluate_task(report, state)")
    print()
    print("  Metrics that would be computed:")
    print("  ├─ Task Completion Rate    (did it finish?)")
    print("  ├─ Step Success Rate       (% steps that worked)")
    print("  ├─ Self-Correction Success (did fixes work?)")
    print("  ├─ Answer Quality Score    (LLM judge: 1-5)")
    print("  ├─ Plan Quality Score      (was plan appropriate?)")
    print("  └─ Answer Grounding        (is answer based on data?)")


if __name__ == "__main__":
    asyncio.run(run_demo())
