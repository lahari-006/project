"""
Interactive CLI for the Internal Knowledge Navigator.

    python -m app.main
"""
from app.rag_pipeline import answer_query


BANNER = """
============================================================
 Internal Knowledge Navigator  (Nimbus Cloud Systems wiki)
 Type a question, or 'quit' to exit.
============================================================
"""


def main():
    print(BANNER)
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        result = answer_query(query)

        print("\n--- Answer " + "-" * 50)
        print(result["answer"])

        if result["sources"]:
            print("\n--- Sources " + "-" * 49)
            for s in result["sources"]:
                flag = "" if s["status"] == "current" else f"  [{s['status'].upper()}]"
                print(f"  [{s['page_id']}] {s['title']} > {s['heading']}{flag}")

        print(f"\n(latency: {result['latency_seconds']}s | "
              f"est. cost: ${result['cost_usd']:.5f} | blocked: {result['blocked']})")


if __name__ == "__main__":
    main()
