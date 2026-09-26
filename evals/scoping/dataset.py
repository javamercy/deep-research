from langchain_core.messages import AIMessage, HumanMessage
from langsmith import Client

CONVERSATIONS = [
    [
        HumanMessage(content="What's the best way to invest $50,000 for retirement?"),
        AIMessage(
            content="Could you please provide some additional information to tailor the investment advice for your $50,000 retirement goal? Specifically:\n Your current age or desired retirement age\n Your risk tolerance (low, medium, high)\n Any preferences for investment types (e.g., stocks, bonds, mutual funds, real estate)\n Whether you are investing through a tax-advantaged account (e.g., IRA, 401(k)) or a regular brokerage account\n This will help me provide more personalized and relevant suggestions."),
        HumanMessage(
            content="I'm 25 and I want to retire by 45. My risk tolerance is high right now but I think will decrease over time. I have heard that stocks and ETFs are a good choice, but I'm open to anything. And I already have a 401k, but this would just be through a regular brokerage account."),
    ],
    [
        HumanMessage(content="I am looking for an apartment in NYC, can you help me?"),
        AIMessage(
            content="Could you please specify your apartment preferences? For example:\n Desired neighborhoods or boroughs\n Number of bedrooms/bathrooms\n Budget range (monthly rent)\n Any amenities or must-have features\n Preferred move-in date\n This information will help me provide the most relevant apartment options in NYC."),
        HumanMessage(
            content="I'd prefer to live in Chelsea, Flatiron, or West Village. I'm looking for a 2 bed 2 bath, and I am looking for monthly rent below 7k. I'd like this to be a doorman building and have an in unit washer and dryer, but it's okay if there's no washer dryer. It's a plus if the building has a gym. And I'd like to move in in September 2026."),
    ],
    [
        HumanMessage(
            content="I'm looking for a new laptop for work and personal use. Can you help me find one?"
        ),
        AIMessage(
            content="Of course! Could you tell me a little more about what you're looking for? Specifically:\n"
                    "Your budget\n"
                    "What you'll primarily use the laptop for\n"
                    "Any operating system preference (Windows or macOS)\n"
                    "Preferred screen size\n"
                    "Any important features or requirements, such as battery life, memory, or performance\n"
                    "This will help me narrow down the options."
        ),
        HumanMessage(
            content="I can spend up to $1,500. I'll mainly use it for Python programming and everyday tasks, with occasional photo editing. I'd prefer Windows, but I'm open to a Mac. A 13- or 14-inch screen would be ideal, though I'm flexible. I need at least 32GB RAM, and battery life matters more to me than gaming performance."
        ),
    ]
]

CRITERIAS = [
    [
        "User wants to invest $50,000 for retirement",
        "Current age is 25",
        "Desired retirement age is 45",
        "Current risk tolerance is high but may decrease over time",
        "Interested in investing in stocks and ETFs",
        "Open to forms of investment beyond stocks and ETFs",
        "Investment account is a regular brokerage account",
    ],
    [
        "Looking for a 2 bed 2 bath apartment in Chelsea, Flatiron, or West Village",
        "Monthly rent below 7k",
        "Should be in a doorman building",
        "Ideally have an in unit washer and dryer but not strict",
        "Ideally have a gym but not strict",
        "Desired move-in date is September 2026"
    ],
    [
        "Looking for a laptop for work and personal use",
        "Budget is up to $1,500",
        "Primary uses are Python programming and everyday tasks, with occasional photo editing",
        "Prefers Windows but is open to macOS",
        "Ideally a 13- or 14-inch screen, but flexible",
        "Requires at least 32GB RAM",
        "Battery life is more important than gaming performance",
    ]
]


def ensure_scoping_dataset(langsmith_client: Client, dataset_name: str) -> None:
    """Ensure that the scoping dataset exists in LangSmith, and create it if it doesn't"""

    if not langsmith_client.has_dataset(dataset_name=dataset_name):
        dataset = langsmith_client.create_dataset(
            dataset_name,
            description="A dataset that measures the quality of research briefs generated from an input conversation",
        )

        langsmith_client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "inputs": {"messages": conversation},
                    "outputs": {"criteria": criteria},
                }
                for i, (conversation, criteria) in enumerate(zip(CONVERSATIONS, CRITERIAS, strict=True))
            ],
        )
