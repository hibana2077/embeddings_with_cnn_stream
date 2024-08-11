import matplotlib.pyplot as plt
import json

# Load data from JSON files
models = ['convnext_atto', 'densenet201', 'ours', 'resnet18', 'resnet152']
highest_accs = []

for model in models:
    with open(f'{model}.json', 'r') as f:
        data = json.load(f)
        highest_accs.append(max(data['test_acc']))

# Create the plot
plt.figure(figsize=(10, 6))
plt.bar(models, highest_accs)
plt.title('Highest Test Accuracy for Each Model')
plt.xlabel('Models')
plt.ylabel('Highest Test Accuracy (%)')
plt.ylim(0, 100)  # Set y-axis limit from 0 to 100%

# Add value labels on top of each bar
for i, v in enumerate(highest_accs):
    plt.text(i, v + 0.5, f'{v:.2f}', ha='center')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('highest_test_accuracies.png')
plt.show()