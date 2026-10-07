from dotenv import load_dotenv
from langsmith import Client

from evals.final_report.dataset import ensure_final_report_dataset
from evals.research.dataset import ensure_research_dataset
from evals.scoping.dataset import ensure_scoping_dataset
from evals.supervisor.dataset import ensure_supervisor_dataset


def main() -> None:
    load_dotenv()
    ls_client = Client()

    scoping_dataset = ensure_scoping_dataset(ls_client)
    print(f"Scoping dataset: {scoping_dataset.name}")

    research_dataset = ensure_research_dataset(ls_client)
    print(f"Research dataset: {research_dataset.name}")

    supervisor_dataset = ensure_supervisor_dataset(ls_client)
    print(f"Supervisor dataset: {supervisor_dataset.name}")

    final_report_dataset = ensure_final_report_dataset(ls_client)
    print(f"Final report dataset: {final_report_dataset.name}")


if __name__ == "__main__":
    main()
