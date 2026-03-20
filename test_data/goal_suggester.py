"""
goal_suggester.py
------------------
Autonomous Goal Suggester for DataWright Agent.
Uses Groq's FREE API (llama-3.1-70b) — no cost at all.

SETUP:
1. Sign up free at console.groq.com
2. Create an API key (takes 30 seconds)
3. pip install groq pandas gradio
4. python goal_suggester.py

FREE TIER LIMITS (more than enough):
- 14,400 requests / day
- 6,000 tokens / minute
- Models: llama-3.1-70b, mixtral-8x7b, gemma2-9b

HOW IT WORKS:
1. You upload a CSV (or paste CSV text)
2. Script profiles the dataset: column names, types, sample values, stats
3. Sends that profile to Groq LLM
4. LLM generates 8 smart, specific analysis goals
5. You pick one and copy it to use with the agent
"""

import json
import os
import re
from pathlib import Path

import pandas as pd

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Groq free models (pick one):
# - "llama-3.1-70b-versatile"  ← best quality, recommended
# - "llama-3.1-8b-instant"     ← faster, still good
# - "mixtral-8x7b-32768"       ← good for long datasets
# - "gemma2-9b-it"             ← Google's model, also free
GROQ_MODEL = "llama-3.1-8b-instant"


GOAL_CATEGORIES = {
    "exploratory": {"icon": "🔭", "desc": "Understand distributions, correlations, patterns"},
    "predictive": {"icon": "🎯", "desc": "Build models, forecast, classify"},
    "diagnostic": {"icon": "🔬", "desc": "Find root causes, anomalies, segments"},
    "prescriptive": {"icon": "💡", "desc": "Actionable recommendations, optimization"},
    "comparative": {"icon": "⚖️", "desc": "Compare groups, time periods, segments"},
}


# ─── DATASET PROFILER ─────────────────────────────────────────────────────────


class DatasetProfiler:
    """
    Profiles a pandas DataFrame into a rich text description
    that the LLM can use to generate smart, specific goals.

    The richer the profile, the better the goals.
    """

    def profile(self, df: pd.DataFrame, filename: str = "dataset") -> dict:
        """Generate a complete profile of the dataset."""
        profile = {
            "filename": filename,
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": self._profile_columns(df),
            "sample": df.head(3).to_string(),
            "missing_values": self._get_missing(df),
            "potential_target": self._guess_target(df),
            "domain": self._guess_domain(df),
        }
        return profile

    def _profile_columns(self, df: pd.DataFrame) -> list:
        """Profile each column: type, unique values, range, etc."""
        cols = []
        for col in df.columns:
            series = df[col].dropna()
            col_info = {"name": col, "dtype": str(df[col].dtype)}

            # Numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info["type"] = "numeric"
                col_info["min"] = round(float(series.min()), 2)
                col_info["max"] = round(float(series.max()), 2)
                col_info["mean"] = round(float(series.mean()), 2)
                col_info["std"] = round(float(series.std()), 2)

            # Categorical / low-cardinality
            elif series.nunique() <= 10:
                col_info["type"] = "categorical"
                col_info["unique_values"] = series.unique().tolist()
                col_info["value_counts"] = series.value_counts().head(5).to_dict()

            # Date columns
            elif "date" in col.lower() or "time" in col.lower():
                col_info["type"] = "datetime"
                try:
                    parsed = pd.to_datetime(series, errors="coerce")
                    col_info["date_range"] = f"{parsed.min()} to {parsed.max()}"
                except Exception:
                    pass

            # High-cardinality text
            else:
                col_info["type"] = "text"
                col_info["unique_count"] = int(series.nunique())
                col_info["sample_values"] = series.head(3).tolist()

            cols.append(col_info)
        return cols

    def _get_missing(self, df: pd.DataFrame) -> dict:
        """Find columns with missing values."""
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        return {col: int(cnt) for col, cnt in missing.items()}

    def _guess_target(self, df: pd.DataFrame) -> str:
        """
        Try to guess the target/label column.
        Looks for common names like 'churn', 'target', 'label', 'survived', etc.
        """
        target_keywords = [
            "churn",
            "target",
            "label",
            "class",
            "survived",
            "outcome",
            "result",
            "fraud",
            "default",
            "purchased",
            "clicked",
            "converted",
            "exited",
            "attrition",
        ]
        for col in df.columns:
            if any(kw in col.lower() for kw in target_keywords):
                return col
        # Binary columns are often targets
        for col in df.columns:
            if df[col].nunique() == 2:
                return f"{col} (binary — likely target)"
        return "Not identified"

    def _guess_domain(self, df: pd.DataFrame) -> str:
        """Guess the business domain from column names."""
        col_text = " ".join(df.columns).lower()
        domains = {
            "customer/churn": ["churn", "tenure", "contract", "subscription"],
            "sales/retail": ["sales", "revenue", "profit", "order", "product", "category"],
            "finance/banking": ["loan", "credit", "fraud", "balance", "transaction", "default"],
            "healthcare": ["patient", "diagnosis", "disease", "age", "bmi", "glucose"],
            "hr/people": ["employee", "salary", "department", "attrition", "performance"],
            "marketing": ["campaign", "click", "conversion", "impressions", "spend"],
            "logistics": ["delivery", "shipment", "warehouse", "shipping", "freight"],
        }
        for domain, keywords in domains.items():
            if any(kw in col_text for kw in keywords):
                return domain
        return "General"

    def to_prompt_text(self, profile: dict) -> str:
        """Convert profile dict to a clean text description for the LLM."""
        lines = [
            f"Dataset: {profile['filename']}",
            f"Size: {profile['shape']['rows']:,} rows × {profile['shape']['columns']} columns",
            f"Domain: {profile['domain']}",
            f"Likely target column: {profile['potential_target']}",
            "",
            "COLUMNS:",
        ]

        for col in profile["columns"]:
            if col["type"] == "numeric":
                lines.append(
                    f"  - {col['name']} (numeric): "
                    f"min={col['min']}, max={col['max']}, "
                    f"mean={col['mean']}, std={col['std']}"
                )
            elif col["type"] == "categorical":
                vals = ", ".join(str(v) for v in col.get("unique_values", [])[:6])
                lines.append(f"  - {col['name']} (categorical): [{vals}]")
            elif col["type"] == "datetime":
                lines.append(f"  - {col['name']} (datetime): {col.get('date_range', 'unknown range')}")
            else:
                lines.append(f"  - {col['name']} (text): {col.get('unique_count', '?')} unique values")

        if profile["missing_values"]:
            lines.append(f"\nMISSING VALUES: {profile['missing_values']}")

        lines.append(f"\nSAMPLE DATA (first 3 rows):\n{profile['sample']}")

        return "\n".join(lines)


# ─── GOAL GENERATOR ───────────────────────────────────────────────────────────


class GoalGenerator:
    """
    Uses Groq's free LLM API to generate analysis goals
    from a dataset profile.
    """

    def __init__(self, api_key: str, model: str = GROQ_MODEL):
        try:
            from groq import Groq

            self.client = Groq(api_key=api_key)
            self.model = model
            print(f"✅ Connected to Groq | Model: {model}")
        except ImportError:
            raise ImportError("Run: pip install groq")

    def generate(self, profile_text: str, n_goals: int = 8) -> list:
        """
        Generate analysis goals from dataset profile.

        Args:
            profile_text: Text description of the dataset
            n_goals: How many goals to generate (default 8)

        Returns:
            List of goal dicts with goal, category, complexity, why, steps
        """
        system_prompt = f"""You are a senior data scientist who generates precise, actionable analysis goals.

Given a dataset profile, generate exactly {n_goals} analysis goals covering these categories:
- exploratory (2): understand the data — distributions, correlations, outliers
- predictive (2): build models — classification, regression, forecasting
- diagnostic (2): find root causes — what drives X, segment analysis, anomalies
- prescriptive (1): recommendations — what actions to take based on data
- comparative (1): compare groups — by category, time period, or segment

RULES:
1. Use the ACTUAL column names from the dataset — not generic names
2. Each goal must be specific enough that an agent knows exactly what to do
3. Goals should vary in approach — don't repeat the same technique
4. Write goals as if giving instructions to a junior data scientist
5. Include specific metrics or techniques where relevant

OUTPUT: Respond ONLY with a valid JSON array, no other text:
[
  {{
    "goal": "Full goal text using actual column names. Be specific about what to analyze and what output is expected.",
    "category": "exploratory|predictive|diagnostic|prescriptive|comparative",
    "complexity": "simple|medium|advanced",
    "estimated_time": "5 min|15 min|30 min|1 hour",
    "why": "One sentence: what business decision this analysis supports",
    "key_techniques": ["technique1", "technique2"]
  }}
]"""

        user_message = f"""Here is the dataset profile:

{profile_text}

Generate {n_goals} analysis goals:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.7,  # Some creativity for varied goals
                max_tokens=2000,
            )

            raw = response.choices[0].message.content.strip()

            # Parse JSON from response
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if json_match:
                goals = json.loads(json_match.group())
                return goals
            else:
                print(f"⚠️  Could not parse JSON from response:\n{raw[:200]}")
                return []

        except Exception as e:
            print(f"❌ Groq API error: {e}")
            return []


# ─── DISPLAY ──────────────────────────────────────────────────────────────────


def display_goals(goals: list, profile: dict):
    """Pretty print goals to terminal."""
    cat_colors = {
        "exploratory": "\033[34m",  # Blue
        "predictive": "\033[35m",  # Purple
        "diagnostic": "\033[31m",  # Red
        "prescriptive": "\033[32m",  # Green
        "comparative": "\033[33m",  # Yellow
    }
    reset = "\033[0m"
    bold = "\033[1m"

    print(f"\n{'═'*65}")
    print(f"  📊 ANALYSIS GOALS FOR: {profile['filename'].upper()}")
    print(f"  {profile['shape']['rows']:,} rows · {profile['shape']['columns']} columns · Domain: {profile['domain']}")
    print(f"{'═'*65}\n")

    for i, goal in enumerate(goals, 1):
        cat = goal.get("category", "exploratory")
        icon = GOAL_CATEGORIES.get(cat, {}).get("icon", "•")
        color = cat_colors.get(cat, "")

        print(
            f"{bold}[{i}] {icon} {color}{cat.upper()}{reset}{bold} — {goal.get('complexity', 'medium')} · {goal.get('estimated_time', '?')}{reset}"
        )
        print(f"{goal['goal']}")
        print(f"{'033[90m'}Why: {goal.get('why', '')}{reset}")

        techniques = goal.get("key_techniques", [])
        if techniques:
            print(f"{'033[90m'}Techniques: {', '.join(techniques)}{reset}")
        print()


def pick_goal(goals: list) -> str:
    """Interactive goal selection in terminal."""
    while True:
        try:
            choice = input(f"Pick a goal [1-{len(goals)}] or 0 to use all: ").strip()
            num = int(choice)
            if num == 0:
                return "\n\n".join(g["goal"] for g in goals)
            if 1 <= num <= len(goals):
                return goals[num - 1]["goal"]
            print(f"Please enter a number between 0 and {len(goals)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting.")
            return ""


def save_goals(goals: list, profile: dict, output_path: str = "goals_output.json"):
    """Save goals to JSON file."""
    output = {
        "dataset": profile["filename"],
        "generated_at": pd.Timestamp.now().isoformat(),
        "domain": profile["domain"],
        "goals": goals,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"💾 Goals saved to {output_path}")


# ─── GRADIO UI (optional) ─────────────────────────────────────────────────────


def launch_gradio_ui(api_key: str):
    """
    Launch a simple Gradio web UI for the goal suggester.
    Run: python goal_suggester.py --ui
    """
    try:
        import gradio as gr
    except ImportError:
        print("Run: pip install gradio")
        return

    profiler = DatasetProfiler()
    generator = GoalGenerator(api_key=api_key)

    def process(file, model_choice):
        if file is None:
            return "Please upload a CSV file.", ""

        try:
            df = pd.read_csv(file.name)
            filename = Path(file.name).name
        except Exception as e:
            return f"Error reading file: {e}", ""

        # Profile
        profile = profiler.profile(df, filename)
        profile_text = profiler.to_prompt_text(profile)

        # Update model if changed
        generator.model = model_choice

        # Generate
        goals = generator.generate(profile_text)

        if not goals:
            return "Failed to generate goals. Check your API key.", ""

        # Format output
        output_lines = [
            f"## Analysis Goals for {filename}",
            f"**{profile['shape']['rows']:,} rows · {profile['shape']['columns']} columns · Domain: {profile['domain']}**",
            f"**Likely target column:** {profile['potential_target']}",
            "---",
        ]

        for i, goal in enumerate(goals, 1):
            cat = goal.get("category", "exploratory")
            icon = GOAL_CATEGORIES.get(cat, {}).get("icon", "•")
            output_lines.extend(
                [
                    f"### {icon} Goal {i} — {cat.title()} ({goal.get('complexity', 'medium')})",
                    f"**{goal['goal']}**",
                    f"*Why: {goal.get('why', '')}*",
                    f"Techniques: {', '.join(goal.get('key_techniques', []))}",
                    f"Estimated time: {goal.get('estimated_time', '?')}",
                    "---",
                ]
            )

        # Goals as dropdown choices
        goal_choices = [f"[{i+1}] {g['goal'][:80]}..." for i, g in enumerate(goals)]

        return "\n".join(output_lines), gr.Dropdown(choices=goal_choices, label="Select a goal to copy")

    with gr.Blocks(title="DataWright Goal Suggester", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
        # ⚡ DataWright — Goal Suggester
        Upload a CSV dataset and get AI-powered analysis goals using **Groq's free API**.
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Upload CSV", file_types=[".csv"])
                model_select = gr.Dropdown(
                    choices=[
                        "llama-3.1-70b-versatile",
                        "llama-3.1-8b-instant",
                        "mixtral-8x7b-32768",
                        "gemma2-9b-it",
                    ],
                    value="llama-3.1-70b-versatile",
                    label="Model (all free)",
                )
                generate_btn = gr.Button("✨ Generate Goals", variant="primary")

            with gr.Column(scale=2):
                output_md = gr.Markdown(label="Generated Goals")
                goal_dropdown = gr.Dropdown(label="Pick a goal to use with agent", visible=False)
                copy_box = gr.Textbox(label="Selected goal (copy this)", interactive=False)

        generate_btn.click(
            fn=process,
            inputs=[file_input, model_select],
            outputs=[output_md, goal_dropdown],
        )

        goal_dropdown.change(
            fn=lambda x: x.split("] ", 1)[-1] if x else "",
            inputs=[goal_dropdown],
            outputs=[copy_box],
        )

        gr.Markdown(
            """
        ---
        **Free tier limits:** 14,400 requests/day · No credit card needed
        Get your free API key at [console.groq.com](https://console.groq.com)
        """
        )

    demo.launch(share=False)


# ─── MAIN ─────────────────────────────────────────────────────────────────────


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="DataWright Goal Suggester — Generate analysis goals from any CSV")
    parser.add_argument("--csv", type=str, help="Path to CSV file")
    parser.add_argument("--ui", action="store_true", help="Launch Gradio web UI")
    parser.add_argument("--model", type=str, default=GROQ_MODEL, help="Groq model name")
    parser.add_argument("--save", type=str, default="", help="Save goals to JSON file")
    parser.add_argument("--n", type=int, default=8, help="Number of goals to generate")
    args = parser.parse_args()

    # Get API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\n⚠️  GROQ_API_KEY not set.")
        print("   1. Sign up free at console.groq.com")
        print("   2. Create an API key")
        print("   3. Set it: export GROQ_API_KEY=your_key_here")
        print("   Or add it to your .env file\n")
        api_key = input("Paste your Groq API key here (or press Enter to exit): ").strip()
        if not api_key:
            sys.exit(0)

    # Launch UI mode
    if args.ui:
        print("🌐 Launching Gradio UI...")
        launch_gradio_ui(api_key)
        return

    # CLI mode
    profiler = DatasetProfiler()
    generator = GoalGenerator(api_key=api_key, model=args.model)

    # Load CSV
    csv_path = args.csv
    if not csv_path:
        csv_path = input("\nEnter path to your CSV file: ").strip()

    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)

    print(f"\n📂 Loading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        sys.exit(1)

    print(f"✅ Loaded: {len(df):,} rows × {len(df.columns)} columns")

    # Profile
    print("🔍 Profiling dataset...")
    profile = profiler.profile(df, Path(csv_path).name)
    profile_text = profiler.to_prompt_text(profile)

    print("\n📊 Dataset Profile:")
    print(f"   Domain: {profile['domain']}")
    print(f"   Likely target: {profile['potential_target']}")
    print(f"   Columns: {', '.join(df.columns.tolist())}")
    if profile["missing_values"]:
        print(f"   Missing values: {profile['missing_values']}")

    # Generate
    print(f"\n✨ Generating {args.n} analysis goals using {args.model}...")
    goals = generator.generate(profile_text, n_goals=args.n)

    if not goals:
        print("❌ No goals generated. Check your API key and try again.")
        sys.exit(1)

    # Display
    display_goals(goals, profile)

    # Save if requested
    if args.save:
        save_goals(goals, profile, args.save)

    # Interactive selection
    print("─" * 65)
    selected = pick_goal(goals)
    if selected:
        print("\n✅ SELECTED GOAL:")
        print(f"{'─'*65}")
        print(selected)
        print(f"{'─'*65}")
        print("\n→ Use this as the 'goal' parameter in your DataWright agent\n")

        # Copy to clipboard if pyperclip available
        try:
            import pyperclip

            pyperclip.copy(selected)
            print("📋 Copied to clipboard!")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
