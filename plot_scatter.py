import csv
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

def main(csv_file, out_img):
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    scores = []
    mos = []
    refs = []
    dists = []
    for r in data:
        scores.append(float(r['score']))
        mos.append(float(r['mos_label']))
        refs.append(r['ref_img_path'])
        dists.append(r['dis_img_path'])

    scores = np.array(scores)
    mos = np.array(mos)

    # Normalize to 0-1 for outlier detection
    s_norm = (scores - scores.min()) / (scores.max() - scores.min())
    m_norm = (mos - mos.min()) / (mos.max() - mos.min())
    
    # Simple linear fit for visualization
    a, b = np.polyfit(scores, mos, 1)

    plt.figure(figsize=(8, 6))
    plt.scatter(scores, mos, alpha=0.5, edgecolors='none')
    plt.plot(scores, a * scores + b, color='red', linestyle='--')
    plt.title('Predicted Score vs MOS')
    plt.xlabel('Predicted Score')
    plt.ylabel('MOS')
    plt.savefig(out_img)
    print(f"Scatterplot saved to {out_img}")

    # Find outliers
    # We define outlier as max absolute difference between normalized prediction and MOS
    diffs = np.abs(s_norm - m_norm)
    top_indices = np.argsort(diffs)[-5:][::-1]

    print("\nTOP 5 OUTLIERS:")
    for idx in top_indices:
        print(f"Ref: {refs[idx]} | Dist: {dists[idx]} | Pred (norm): {s_norm[idx]:.3f} | MOS (norm): {m_norm[idx]:.3f} | Diff: {diffs[idx]:.3f}")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
