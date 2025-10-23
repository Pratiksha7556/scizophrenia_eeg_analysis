import os
import numpy as np
import matplotlib.pyplot as plt
import csv

# === Paths ===
base_dir = r'C:\Users\Pratiksha\OneDrive\Desktop\scizophrenia\data\organized'
output_plot_dir = r'C:\Users\Pratiksha\OneDrive\Desktop\scizophrenia\outputs\plots_all_channels'
os.makedirs(output_plot_dir, exist_ok=True)
log_file = os.path.join(os.path.dirname(output_plot_dir), 'qc_log.csv')

# === Constants ===
num_channels = 16
num_samples = 7680
sample_rate = 128  # Hz
plot_duration_sec = 5
samples_to_plot = sample_rate * plot_duration_sec

channel_labels = ["F7", "F3", "F4", "F8",
                  "T3", "C3", "Cz", "C4",
                  "T4", "T5", "P3", "Pz",
                  "P4", "T6", "O1", "O2"]

# === Start QC Log ===
with open(log_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Filename', 'Class', 'Flat Channels', 'Noisy Channels', 'Errors'])

    # === Loop through both classes ===
    for label in ['healthy', 'schizophrenia']:
        class_dir = os.path.join(base_dir, label)
        if not os.path.exists(class_dir):
            print(f"❌ Missing folder: {class_dir}")
            continue

        for file in os.listdir(class_dir):
            if not file.lower().endswith('.eea'):
                continue

            file_path = os.path.join(class_dir, file)
            errors = []
            flat_channels = []
            noisy_channels = []

            try:
                # === Load and reshape ===
                eeg_flat = np.loadtxt(file_path)
                if eeg_flat.shape[0] != num_channels * num_samples:
                    raise ValueError(f"Expected {num_channels*num_samples}, got {eeg_flat.shape[0]}")

                eeg = eeg_flat.reshape(num_channels, num_samples)

                # === Check flat / noisy ===
                for ch in range(num_channels):
                    std = np.std(eeg[ch])
                    if std < 1e-2:
                        flat_channels.append(channel_labels[ch])
                    elif std > 1000:  # arbitrary large noise threshold
                        noisy_channels.append(channel_labels[ch])

                # === Plot first 5 seconds ===
                fig, axs = plt.subplots(4, 4, figsize=(14, 10))
                fig.suptitle(f'{file} - First 5 Seconds', fontsize=16)

                for i in range(16):
                    row, col = divmod(i, 4)
                    axs[row, col].plot(eeg[i][:samples_to_plot], linewidth=0.8)
                    axs[row, col].set_title(channel_labels[i], fontsize=10)
                    axs[row, col].set_xticks([])
                    axs[row, col].set_yticks([])
                    axs[row, col].grid(True)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plot_path = os.path.join(output_plot_dir, file.replace('.eea', '_all_channels.png'))
                plt.savefig(plot_path)
                plt.close()

                print(f"✅ QC done for: {file}")

            except Exception as e:
                errors.append(str(e))
                print(f"❌ Error in {file}: {e}")

            # === Log results ===
            writer.writerow([
                file,
                label,
                ';'.join(flat_channels) if flat_channels else 'None',
                ';'.join(noisy_channels) if noisy_channels else 'None',
                ';'.join(errors) if errors else 'None'
            ])
