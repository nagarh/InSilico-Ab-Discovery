def process_sequences(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        raw_number = 1
        for line in infile:
            line = line.strip().replace('<', '').replace('>', '')

            if '|' not in line:
                continue

            parts = line.split('|')
            if len(parts) != 2:
                continue

            vh, vl = parts

            if 110 <= len(vh) <= 140 and 105 <= len(vl) <= 120:
                outfile.write(f">{raw_number}_H\n{vh}\n")
                outfile.write(f">{raw_number}_L\n{vl}\n")

            raw_number += 1


input_file = "./data/generated_sequences.txt"
output_file = "./data/filtered_sequences.fasta"

process_sequences(input_file, output_file)