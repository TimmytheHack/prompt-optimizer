from ga import run_ga

if __name__ == "__main__":
    best = run_ga()
    print("\n== Best prompt ==")
    print("".join(best))
