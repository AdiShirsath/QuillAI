import asyncio
import logging

from src.agent.agent import DataAnalysisAgent

logging.basicConfig(level=logging.INFO)

# Suppress noisy third-party logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("groq._base_client").setLevel(logging.WARNING)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

# Clean name mapping for screenshot-friendly log prefixes
LOG_LABELS = {
    "src.agent.agent": "🤖 AGENT   ",
    "src.agent.planner": "🗺  PLANNER ",
    "src.agent.executor": "⚙️  EXECUTOR",
    "src.tools.memory_manager": "🧠 MEMORY  ",
}


class CleanFormatter(logging.Formatter):
    def format(self, record):
        label = LOG_LABELS.get(record.name, f"   {record.name}")
        time = self.formatTime(record, "%H:%M:%S")
        return f"{time}  {label}  │  {record.getMessage()}"


handler = logging.StreamHandler()
handler.setFormatter(CleanFormatter())
root = logging.getLogger()
root.handlers.clear()
root.addHandler(handler)

agent = DataAnalysisAgent()

TASKS = []

RAW_TASKS = [
    {
        "dataset": "breast_cancer.csv",
        "generated_at": "2026-03-20T12:21:27.796363",
        "domain": "General",
        "goals": [
            {
                "goal": "Analyze the correlation between various radiometric features (mean radius, radius error, worst radius) and the target variable to understand the relationship between these features and cancer diagnosis.",
                "category": "exploratory",
                "complexity": "medium",
                "estimated_time": "30 min",
                "why": "This analysis will help identify the most informative radiometric features for cancer diagnosis.",
                "key_techniques": ["correlation analysis", "heatmap"],
            },
            {
                "goal": "Develop a logistic regression model using the 10 most informative features to predict the target variable (malignant or benign tumor).",
                "category": "predictive",
                "complexity": "medium",
                "estimated_time": "1 hour",
                "why": "This model will enable us to identify the most critical features for predicting cancer diagnosis and inform treatment decisions.",
                "key_techniques": ["logistic regression", "feature selection"],
            },
            {
                "goal": "Investigate the distribution of the worst fractal dimension feature across different target categories to identify potential outliers and anomalies.",
                "category": "diagnostic",
                "complexity": "simple",
                "estimated_time": "15 min",
                "why": "This analysis will help identify potential issues with the data and ensure that the model is robust to outliers.",
                "key_techniques": ["boxplot", "density plot"],
            },
            {
                "goal": "Segment the data into malignant and benign tumor groups based on the mean compactness feature and analyze the distribution of other features within each segment.",
                "category": "diagnostic",
                "complexity": "medium",
                "estimated_time": "45 min",
                "why": "This analysis will help identify the most informative features for distinguishing between malignant and benign tumors.",
                "key_techniques": ["k-means clustering", "feature comparison"],
            },
            {
                "goal": "Develop a decision tree model to predict the target variable based on the entire set of features and evaluate its performance using metrics such as accuracy and F1-score.",
                "category": "predictive",
                "complexity": "medium",
                "estimated_time": "1.5 hours",
                "why": "This model will enable us to identify the most critical features for predicting cancer diagnosis and inform treatment decisions.",
                "key_techniques": ["decision tree", "cross-validation"],
            },
            {
                "goal": "Analyze the relationship between the mean area feature and the target variable using a scatter plot to understand how the area of the tumor is related to cancer diagnosis.",
                "category": "exploratory",
                "complexity": "simple",
                "estimated_time": "10 min",
                "why": "This analysis will help identify the most informative features for predicting cancer diagnosis.",
                "key_techniques": ["scatter plot", "correlation analysis"],
            },
            {
                "goal": "Compare the distribution of the mean texture feature across different age groups to identify potential trends and correlations.",
                "category": "comparative",
                "complexity": "medium",
                "estimated_time": "45 min",
                "why": "This analysis will help identify potential relationships between the texture of the tumor and age.",
                "key_techniques": ["boxplot", "density plot"],
            },
            {
                "goal": "Develop a recommendation system to identify the top three features that are most informative for predicting cancer diagnosis and provide actionable insights for clinicians.",
                "category": "prescriptive",
                "complexity": "advanced",
                "estimated_time": "2 hours",
                "why": "This system will enable clinicians to make informed decisions about treatment and diagnosis.",
                "key_techniques": ["feature importance", "model interpretability"],
            },
        ],
    },
    {
        "dataset": "telco_churn.csv",
        "generated_at": "2026-03-20T12:21:13.459923",
        "domain": "customer/churn",
        "goals": [
            {
                "goal": "Analyze the distribution of tenure and monthly charges, and identify any outliers in these columns.",
                "category": "exploratory",
                "complexity": "medium",
                "estimated_time": "15 min",
                "why": "Understand the typical customer tenure and monthly charges to inform churn prevention strategies.",
                "key_techniques": ["histogram", "box plot", "z-score"],
            },
            {
                "goal": "Explore the relationship between senior citizen status, tenure, and monthly charges using correlation analysis.",
                "category": "exploratory",
                "complexity": "medium",
                "estimated_time": "15 min",
                "why": "Identify any patterns or correlations that may inform customer retention strategies.",
                "key_techniques": ["correlation matrix", "scatter plot"],
            },
            {
                "goal": "Build a logistic regression model to predict customer churn based on demographic and service-related variables.",
                "category": "predictive",
                "complexity": "advanced",
                "estimated_time": "1 hour",
                "why": "Develop a model that can accurately predict which customers are likely to churn, enabling proactive retention efforts.",
                "key_techniques": ["logistic regression", "feature engineering"],
            },
            {
                "goal": "Use decision trees to identify the most important factors contributing to customer churn, and visualize the decision-making process.",
                "category": "diagnostic",
                "complexity": "medium",
                "estimated_time": "30 min",
                "why": "Gain insights into the key drivers of churn and develop targeted retention strategies.",
                "key_techniques": ["decision tree", "feature importance"],
            },
            {
                "goal": "Analyze the distribution of total charges and identify any anomalies or irregularities.",
                "category": "diagnostic",
                "complexity": "medium",
                "estimated_time": "15 min",
                "why": "Verify the accuracy of total charges data and detect any potential errors or discrepancies.",
                "key_techniques": ["histogram", "box plot", "z-score"],
            },
            {
                "goal": "Develop a recommendation engine to suggest the most effective retention strategies based on customer demographics and service usage.",
                "category": "prescriptive",
                "complexity": "advanced",
                "estimated_time": "1 hour",
                "why": "Provide actionable insights to customer service representatives to improve retention rates.",
                "key_techniques": ["collaborative filtering", "content-based filtering"],
            },
            {
                "goal": "Compare the demographics and service usage of customers who have churned versus those who have not, to identify key differences.",
                "category": "comparative",
                "complexity": "medium",
                "estimated_time": "30 min",
                "why": "Understand the profiles of customers who are most likely to churn and develop targeted retention strategies.",
                "key_techniques": ["t-test", "ANOVA"],
            },
            {
                "goal": "Build a survival analysis model to forecast the likelihood of customer churn over time, based on service usage and demographic variables.",
                "category": "predictive",
                "complexity": "advanced",
                "estimated_time": "1 hour",
                "why": "Develop a model that can accurately predict the likelihood of churn at different time intervals, enabling proactive retention efforts.",
                "key_techniques": ["survival analysis", "hazards function"],
            },
        ],
    },
    {
        "dataset": "tips.csv",
        "generated_at": "2026-03-20T12:21:17.457730",
        "domain": "General",
        "goals": [
            {
                "goal": "Analyze the distribution of total_bill and tip amounts to understand the data range and central tendency. Plot histograms and report the mean, median, and standard deviation of both columns.",
                "category": "exploratory",
                "complexity": "simple",
                "estimated_time": "15 min",
                "why": "To understand the data distribution and identify any potential outliers.",
                "key_techniques": ["histogram", "summary statistics"],
            },
            {
                "goal": "Examine the correlation between total_bill, tip, and size to identify any relationships. Use a correlation matrix to visualize the relationships and report the correlation coefficients.",
                "category": "exploratory",
                "complexity": "medium",
                "estimated_time": "30 min",
                "why": "To identify any potential predictors of total_bill and tip amounts.",
                "key_techniques": ["correlation matrix", "correlation coefficients"],
            },
            {
                "goal": "Develop a classification model to predict the likelihood of a customer leaving a larger tip based on their sex, smoker status, day of the week, time of day, and total_bill amount.",
                "category": "predictive",
                "complexity": "medium",
                "estimated_time": "1 hour",
                "why": "To identify the key factors that influence a customer's tipping behavior.",
                "key_techniques": ["logistic regression", "cross-validation"],
            },
            {
                "goal": "Build a regression model to predict total_bill amount based on tip, size, sex, smoker status, day of the week, and time of day.",
                "category": "predictive",
                "complexity": "medium",
                "estimated_time": "1 hour",
                "why": "To identify the key factors that influence a customer's total_bill amount.",
                "key_techniques": ["linear regression", "cross-validation"],
            },
            {
                "goal": "Identify the root cause of the variation in total_bill amounts across different days of the week using segment analysis.",
                "category": "diagnostic",
                "complexity": "medium",
                "estimated_time": "30 min",
                "why": "To understand the impact of day of the week on total_bill amounts.",
                "key_techniques": ["segment analysis", "box plots"],
            },
            {
                "goal": "Analyze the relationship between tip amount and total_bill amount to identify any anomalies using a scatter plot.",
                "category": "diagnostic",
                "complexity": "simple",
                "estimated_time": "15 min",
                "why": "To identify any unusual patterns in the data.",
                "key_techniques": ["scatter plot", "outlier detection"],
            },
            {
                "goal": "Develop a recommendation system to suggest the optimal tip amount based on the customer's total_bill amount, size, sex, smoker status, day of the week, and time of day.",
                "category": "prescriptive",
                "complexity": "advanced",
                "estimated_time": "2 hours",
                "why": "To provide personalized recommendations to customers.",
                "key_techniques": ["decision tree", "feature engineering"],
            },
            {
                "goal": "Compare the average tip amounts across different sex categories to identify any differences using a t-test.",
                "category": "comparative",
                "complexity": "medium",
                "estimated_time": "30 min",
                "why": "To understand the impact of sex on tip amounts.",
                "key_techniques": ["t-test", "confidence intervals"],
            },
        ],
    },
    {
        "dataset": "titanic.csv",
        "generated_at": "2026-03-20T12:21:24.010633",
        "domain": "healthcare",
        "goals": [
            {
                "goal": "Analyze the distribution of the 'Survived' variable and identify any outliers using box plots and IQR method.",
                "category": "exploratory",
                "complexity": "medium",
                "estimated_time": "15 min",
                "why": "Understand the baseline distribution of the target variable to inform subsequent analysis and model selection.",
                "key_techniques": ["box plot", "IQR"],
            },
            {
                "goal": "Examine the correlation between 'Age', 'Fare', and 'Survived' variables using Pearson's correlation coefficient and visualize the results using a heatmap.",
                "category": "exploratory",
                "complexity": "medium",
                "estimated_time": "30 min",
                "why": "Identify potential relationships between variables that can inform feature engineering and model selection.",
                "key_techniques": ["Pearson's correlation coefficient", "heatmap"],
            },
            {
                "goal": "Build a logistic regression model to predict the probability of survival based on 'Pclass', 'Sex', and 'Age' variables and evaluate its performance using AUC-ROC.",
                "category": "predictive",
                "complexity": "medium",
                "estimated_time": "30 min",
                "why": "Develop a predictive model to estimate the likelihood of survival.",
                "key_techniques": ["logistic regression", "AUC-ROC"],
            },
            {
                "goal": "Use decision trees to identify the most influential features for predicting survival and visualize the decision tree.",
                "category": "predictive",
                "complexity": "medium",
                "estimated_time": "30 min",
                "why": "Gain insights into the relative importance of different features.",
                "key_techniques": ["decision trees", "feature importance"],
            },
            {
                "goal": "Analyze the distribution of 'Fare' variable across different 'Pclass' categories and identify any patterns or anomalies using box plots and statistical tests.",
                "category": "diagnostic",
                "complexity": "medium",
                "estimated_time": "15 min",
                "why": "Understand the relationship between fare and class.",
                "key_techniques": ["box plot", "ANOVA"],
            },
            {
                "goal": "Segment passengers into different groups based on 'Age', 'SibSp', and 'Parch' and analyze the distribution of 'Survived' within each group.",
                "category": "diagnostic",
                "complexity": "advanced",
                "estimated_time": "1 hour",
                "why": "Gain insights into the relationships between demographic variables and survival outcomes.",
                "key_techniques": ["k-means clustering", "histograms", "ANOVA"],
            },
            {
                "goal": "Develop a recommendation engine to suggest the most suitable cabin accommodations based on passenger demographics and fare ranges.",
                "category": "prescriptive",
                "complexity": "advanced",
                "estimated_time": "1 hour",
                "why": "Provide personalized recommendations and optimize revenue management.",
                "key_techniques": ["collaborative filtering", "matrix factorization"],
            },
            {
                "goal": "Compare the survival rates of passengers across different 'Embarked' ports and analyze the distribution of 'Survived' within each port using bar plots and statistical tests.",
                "category": "comparative",
                "complexity": "medium",
                "estimated_time": "15 min",
                "why": "Identify any patterns or differences in survival rates across embarkation ports.",
                "key_techniques": ["bar plot", "ANOVA"],
            },
        ],
    },
]

# Flatten RAW_TASKS into (dataset, goal) pairs that match run_all() expectations
for task_group in RAW_TASKS:
    dataset = task_group["dataset"]
    dataset_name = dataset.replace(".csv", "")
    for goal_obj in task_group["goals"]:
        TASKS.append(
            {
                "name": f"{dataset_name} — {goal_obj['category']} — {goal_obj['complexity']}",
                "file": f"data/datasets/{dataset}",
                "goal": goal_obj["goal"],
                "category": goal_obj["category"],
                "complexity": goal_obj["complexity"],
                "estimated_time": goal_obj["estimated_time"],
                "key_techniques": goal_obj["key_techniques"],
            }
        )


async def run_all():
    for task in TASKS:
        print(f"\n{'━'*60}")
        print(f"  🧪 {task['name']}")
        print(f"  📁 {task['file']}  │  🏷  {task['category']}  │  🎯 {task['complexity']}")
        print(f"{'━'*60}")

        report = await agent.run(
            goal=task["goal"],
            file_path=task["file"],
        )

        print(f"\n┌─ RESULTS {'─'*50}")
        print(f"│  📌 Summary     │ {report.executive_summary[:120]}...")
        print(f"│  🔍 Findings    │ {len(report.key_findings)}")
        print(f"│  📊 Figures     │ {len(report.figures)}")
        print(f"│  🔧 Corrections │ {report.self_corrections}")
        print(f"│  ⏱  Latency     │ {report.total_latency_ms/1000:.1f}s")
        print(f"│  💯 Confidence  │ {report.confidence_score:.0%}")
        print(f"└{'─'*60}")

        # Save figures
        import base64
        import os

        os.makedirs(f"data/outputs/{task['name'].replace(' ', '_')}", exist_ok=True)
        for i, fig_b64 in enumerate(report.figures):
            with open(f"data/outputs/{task['name'].replace(' ', '_')}/figure_{i+1}.png", "wb") as f:
                f.write(base64.b64decode(fig_b64))
            print(f"   💾 Saved figure_{i+1}.png")

        # Save report
        with open(f"data/outputs/{task['name'].replace(' ', '_')}/report.txt", "w") as f:
            f.write(f"GOAL: {task['goal']}\n\n")
            f.write(f"SUMMARY: {report.executive_summary}\n\n")
            f.write("KEY FINDINGS:\n")
            for finding in report.key_findings:
                f.write(f"  - {finding}\n")
            f.write(f"\nDETAILED ANALYSIS:\n{report.detailed_analysis}\n")
            f.write(f"\nMETHODOLOGY: {report.methodology}\n")


asyncio.run(run_all())
