
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

class PhaseTransitionTrigger:
    """
    Expert final decision layer for ultra-early bearing fault detection.
    Monitors SI scores and triggers alarm at the 'Phase Transition' point.
    """
    def __init__(self, contamination=0.01, window_size=15):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.window_size = window_size

    def detect_transition(self, si_timeseries):
        """
        Detects the index of phase transition using an Isolation Forest boundary.
        Input: 1D array of SI scores [time_steps, 1].
        """
        si = si_timeseries.flatten()
        
        # 1. Sliding Window Smoothing to prevent salt-and-pepper noise triggers
        if len(si) < self.window_size:
            return None
            
        si_smoothed = np.convolve(si, np.ones(self.window_size)/self.window_size, mode='valid').reshape(-1, 1)
        
        # 2. Monitor for Phase Transition Anomaly
        # Isolation Forest flags outliers as -1
        self.model.fit(si_smoothed)
        outlier_labels = self.model.predict(si_smoothed)
        
        # 3. Pinpoint the first consistent breach
        # We look for where the score fundamentally shifts from the baseline distribution
        anomaly_indices = np.where(outlier_labels == -1)[0]
        
        if len(anomaly_indices) > 0:
            # Map back to original index (offset by half window for center or full for leading)
            trigger_idx = anomaly_indices[0] + (self.window_size - 1)
            return int(trigger_idx)
        
        return None

    def plot_transition(self, si_timeseries, trigger_idx, dataset_name="General"):
        """
        Visualizes the SI time-series and the triggered transition point.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(si_timeseries, color='#1f77b4', label='Instability Score (SI)', linewidth=2)
        
        if trigger_idx is not None:
            plt.axvline(x=trigger_idx, color='red', linestyle='--', linewidth=2, 
                        label=f'Phase Transition Triggered (Idx: {trigger_idx})')
            plt.scatter(trigger_idx, si_timeseries[trigger_idx], color='red', zorder=5)
            
        plt.title(f"Bearing Health Phase Transition Monitoring - {dataset_name}", fontweight='bold')
        plt.xlabel("Operational Time Steps")
        plt.ylabel("SI Score (Anomaly Intensity)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Usage Example for Q1 Manuscript
    t = np.linspace(0, 100, 100)
    # Healthy baseline (stable) -> Incipient Fault (Rising dynamics)
    si_series = 0.1 + 0.05 * np.random.randn(100)
    si_series[70:] += np.linspace(0, 0.8, 30) # Phase transition birth at 70
    
    trigger = PhaseTransitionTrigger(window_size=10)
    idx = trigger.detect_transition(si_series)
    print(f"Phase Transition Detected at: {idx}")
    # trigger.plot_transition(si_series, idx)
