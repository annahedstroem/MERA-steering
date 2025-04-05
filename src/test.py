import pickle
import os

def main() -> None:
    """Load and print 10 examples from completions_str.pkl."""
    # Update this path as needed
    save_dir = "../runs/mmlu_high_school/Llama-3.2-1B/"
    file_name = "3000_completions_str.pkl"
    file_path = os.path.join(save_dir, file_name)

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    completions_str = data.get("completions_str", [])
    for i, comp in enumerate(completions_str[:10]):
        print(f"[{i}] {comp}\n")

if __name__ == "__main__":
    main()
