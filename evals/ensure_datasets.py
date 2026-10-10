from dotenv import load_dotenv
from langsmith import Client

from evals.final_report.dataset import ensure_final_report_dataset
from evals.research.dataset import ensure_research_dataset
from evals.scoping.dataset import ensure_brief_dataset, ensure_plan_dataset
from evals.supervisor.dataset import ensure_supervisor_dataset


def main() -> None:
    load_dotenv()
    ls_client = Client()

    brief_dataset = ensure_brief_dataset(ls_client)
    print(f"Brief dataset updated: {brief_dataset.name}")

    plan_dataset = ensure_plan_dataset(ls_client)
    print(f"Plan dataset updated: {plan_dataset.name}")

    research_dataset = ensure_research_dataset(ls_client)
    print(f"Research dataset: {research_dataset.name}")

    supervisor_dataset = ensure_supervisor_dataset(ls_client)
    print(f"Supervisor dataset: {supervisor_dataset.name}")

    final_report_dataset = ensure_final_report_dataset(ls_client)
    print(f"Final report dataset: {final_report_dataset.name}")


if __name__ == "__main__":
    main()
