import json


def convert_glove_to_feat(glove_input_path, feat_output_path, vector_dim=100):
    with open(glove_input_path, "r", encoding="utf-8") as fin, open(
        feat_output_path, "w", encoding="utf-8"
    ) as fout:
        for line_num, line in enumerate(fin, 1):
            parts = line.strip().split()
            if len(parts) != vector_dim + 1:
                print(f"Line {line_num}: Unexpected number of dimensions. Skipping.")
                continue
            word = parts[0]
            vector = parts[1:]
            vector_dict = {str(i): float(val) for i, val in enumerate(vector)}
            vector_json = json.dumps(vector_dict)
            fout.write(f"{word}\t{vector_json}\n")
    print(f"Conversion completed. Output saved to '{feat_output_path}'.")


# Example usage:
convert_glove_to_feat(
    glove_input_path="glove.6B.100d.txt", feat_output_path="train.feat", vector_dim=100
)
