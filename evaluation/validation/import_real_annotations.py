import os
import json

def convert_annotation(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Warning: {input_path} not found.")
        return
    
    gt_map = {}
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                # Assuming comma separated x,y,w,h
                parts = [float(p) for p in line.split(',')]
                if len(parts) == 4:
                    x, y, w, h = parts
                    # Standardize to [x1, y1, x2, y2]
                    gt_map[str(i)] = [x, y, x + w, y + h]
            except ValueError:
                continue
    
    with open(output_path, 'w') as f:
        json.dump(gt_map, f, indent=2)
    print(f"Converted {input_path} to {output_path}")

def main():
    base_dir = "data/inputs"
    target_dir = "data/annotations"
    os.makedirs(target_dir, exist_ok=True)

    # Map inputs to scenario names
    mappings = [
        ("Car1/annotation.txt", "car2.json"),
        ("Car2/annotation.txt", "car3.json"),
        ("bike/annotation.txt", "motorcycle.json")
    ]

    for input_rel, output_name in mappings:
        input_path = os.path.join(base_dir, input_rel)
        output_path = os.path.join(target_dir, output_name)
        convert_annotation(input_path, output_path)

if __name__ == "__main__":
    main()
