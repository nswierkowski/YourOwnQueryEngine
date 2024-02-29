from typing import Optional
from index_engine import get_index


def __get_user_input() -> Optional[str]:
    return user_prompt if (user_prompt := input("Enter your question (or q to exit): ")) != "q" else None


def run() -> None:
    query_engine = get_index().as_query_engine()
    while user_prompt := __get_user_input():
        print(query_engine.query(user_prompt))
    print("Farewell!")
