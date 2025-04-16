import os

def reconstruct_pkl(output_file="label_encoders.pkl", parts_dir="models/encoder_parts", prefix="label_encoders.part_"):
    parts = sorted([f for f in os.listdir(parts_dir) if f.startswith(prefix)])
    with open(output_file, 'wb') as out_file:
        for part in parts:
            with open(os.path.join(parts_dir, part), 'rb') as part_file:
                out_file.write(part_file.read())
    print(f"✅ Reconstructed file saved as: {output_file}")

if __name__ == "__main__":
    reconstruct_pkl()
