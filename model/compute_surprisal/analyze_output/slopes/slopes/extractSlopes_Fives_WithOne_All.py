import subprocess
with open(f"output/{__file__}.tsv", "w") as outFile:
    print("\t".join([str(q) for q in ["delta", "lambda", "item", "intercept", "embedding", "depth", "embBias", "embBias_One", "embBias_Two", "embBias_Three", "compatible", "compatible_Two", "compatible_Three", "compatible:EmbBias"]]), file=outFile)
for delta_ in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    for lambda_ in [0, 0.25, 0.5, 0.75, 1]:
        subprocess.call(["python3", "extractSlopes_Fives_WithOne.py", str(delta_), str(lambda_)])
