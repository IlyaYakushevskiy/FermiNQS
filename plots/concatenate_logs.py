import json

def concatenate_logs(log1_path, log2_path, output_path, split_index=950):
    """
    Concatenates two VMC optimization logs.
    Takes the first `split_index` entries from log1 and appends all entries from log2.
    """
    # Load the JSON files
    with open(log1_path, 'r') as f1, open(log2_path, 'r') as f2:
        log1 = json.load(f1)
        log2 = json.load(f2)

    combined = {
        "acceptance": {},
        "Energy": {}
    }

    # --- 1. Combine Acceptance Data ---
    # Slice the first log up to split_index, then append the entirety of the second log
    acc_vals = log1["acceptance"]["value"][:split_index] + log2["acceptance"]["value"]
    
    # Rebuild 'iters' so the x-axis remains continuous for plotting
    total_iters = len(acc_vals)
    combined["acceptance"]["iters"] = list(range(total_iters))
    combined["acceptance"]["value"] = acc_vals

    # --- 2. Combine Energy Data ---
    en1 = log1["Energy"]
    en2 = log2["Energy"]
    
    combined["Energy"]["iters"] = list(range(total_iters))
    
    # Handle the nested 'Mean' dictionary containing 'real' and 'imag' lists
    combined["Energy"]["Mean"] = {
        "real": en1["Mean"]["real"][:split_index] + en2["Mean"]["real"],
        "imag": en1["Mean"]["imag"][:split_index] + en2["Mean"]["imag"]
    }
    
    # Handle the remaining flat lists in the Energy dictionary
    metrics = ["Variance", "Sigma", "R_hat", "TauCorr"]
    for metric in metrics:
        combined["Energy"][metric] = en1[metric][:split_index] + en2[metric]

    # --- 3. Save the Combined Data ---
    with open(output_path, 'w') as out_file:
        # Avoid adding unnecessary whitespace to keep the file size compact
        json.dump(combined, out_file, separators=(',', ':'))
        
    print(f"Successfully concatenated logs. Total iterations: {total_iters}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    # Specify your file paths here
    FILE_1 = "/home/ilya/FermiNQS/outputs/2026-05-05/19-48-09/optimization_results.log"  # Replace with the path to your first log
    FILE_2 = "/home/ilya/FermiNQS/outputs/2026-05-06/07-38-22/optimization_results.log"  # Replace with the path to your second log
    OUTPUT_FILE = "/home/ilya/FermiNQS/outputs/2026-05-05/19-48-09/combined_optimization_results.log"
    
    concatenate_logs(FILE_1, FILE_2, OUTPUT_FILE, split_index=950)