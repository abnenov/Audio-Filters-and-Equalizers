# Analysis of different filter behaviors and their influence on input signals

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.io.wavfile
from scipy.fftpack import fft, fftfreq, ifft
from scipy import signal
import matplotlib.patches as patches

# Plot settings
plt.rcParams['font.size'] = 12

# ==========================================
# 1. What is a signal? Time domain vs Frequency domain
# ==========================================

def demonstrate_signal_basics():
    """
    Demonstrates basic concepts of signals, time domain and frequency domain
    """
    print("=== 1. SIGNAL BASICS ===")
    print("Signal: a function that carries information - usually changes over time")
    print("Time domain: representation of signal as a function of time")
    print("Frequency domain: representation of signal as a function of frequency")
    
    # Create example signal - mixture of sinusoids
    t = np.linspace(0, 2, 1000)  # 2 seconds
    freq1, freq2, freq3 = 5, 15, 30  # Hz
    
    signal1 = np.sin(2 * np.pi * freq1 * t)
    signal2 = 0.5 * np.sin(2 * np.pi * freq2 * t)
    signal3 = 0.3 * np.sin(2 * np.pi * freq3 * t)
    combined_signal = signal1 + signal2 + signal3
    
    # Add noise
    noise = 0.2 * np.random.randn(len(t))
    noisy_signal = combined_signal + noise
    
    # Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time domain
    ax1.plot(t[:200], combined_signal[:200], 'b-', linewidth=2)
    ax1.set_title('Clean Signal (Time Domain)')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    ax2.plot(t[:200], noisy_signal[:200], 'r-', linewidth=1)
    ax2.set_title('Noisy Signal (Time Domain)')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    
    # Frequency domain
    sampling_rate = len(t) / (t[-1] - t[0])
    frequencies = fftfreq(len(combined_signal), 1/sampling_rate)
    fft_clean = fft(combined_signal)
    fft_noisy = fft(noisy_signal)
    
    # Show only positive frequencies
    pos_mask = frequencies >= 0
    frequencies_pos = frequencies[pos_mask]
    fft_clean_pos = np.abs(fft_clean[pos_mask])
    fft_noisy_pos = np.abs(fft_noisy[pos_mask])
    
    ax3.plot(frequencies_pos[:100], fft_clean_pos[:100], 'b-', linewidth=2)
    ax3.set_title('Clean Signal (Frequency Domain)')
    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel('Amplitude')
    ax3.grid(True)
    
    ax4.plot(frequencies_pos[:100], fft_noisy_pos[:100], 'r-', linewidth=1)
    ax4.set_title('Noisy Signal (Frequency Domain)')
    ax4.set_xlabel('Frequency [Hz]')
    ax4.set_ylabel('Amplitude')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return t, noisy_signal, sampling_rate

# ==========================================
# 2. How does Fourier Transform work
# ==========================================

def demonstrate_fourier_transform():
    """
    Demonstrates how Fourier Transform works with examples
    """
    print("\n=== 2. FOURIER TRANSFORM ===")
    print("Fourier Transform converts a signal from time domain to frequency domain")
    print("Shows which frequencies are present in the signal and their strength")
    
    # Examples with different functions
    t = np.linspace(-2, 2, 1000)
    
    # 1. Sine wave
    sine_wave = np.sin(2 * np.pi * 5 * t)
    
    # 2. Square pulse
    pulse = np.where(np.abs(t) < 0.5, 1, 0)
    
    # 3. Gaussian function
    gaussian = np.exp(-t**2)
    
    # 4. Sinc function
    sinc_func = np.sinc(2 * t)  # sinc(x) = sin(πx)/(πx)
    
    functions = [sine_wave, pulse, gaussian, sinc_func]
    titles = ['Sine Wave (5 Hz)', 'Square Pulse', 'Gaussian Function', 'Sinc Function']
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    
    for i, (func, title) in enumerate(zip(functions, titles)):
        # Time domain
        axes[i, 0].plot(t, func, 'b-', linewidth=2)
        axes[i, 0].set_title(f'{title} (Time Domain)')
        axes[i, 0].set_xlabel('Time [s]')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True)
        
        # Frequency domain
        fft_func = fft(func)
        freqs = fftfreq(len(func), t[1] - t[0])
        
        # Center frequencies for better visualization
        fft_shifted = np.fft.fftshift(fft_func)
        freqs_shifted = np.fft.fftshift(freqs)
        
        axes[i, 1].plot(freqs_shifted, np.abs(fft_shifted), 'r-', linewidth=2)
        axes[i, 1].set_title(f'{title} (Frequency Domain)')
        axes[i, 1].set_xlabel('Frequency [Hz]')
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].grid(True)
        axes[i, 1].set_xlim(-20, 20)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. What is a filter and how different types work
# ==========================================

def create_filter_examples():
    """
    Creates examples for different types of filters
    """
    print("\n=== 3. FILTER TYPES ===")
    print("Filter: a system that changes the amplitude of different frequencies in a signal")
    print("Main types:")
    print("- Low-pass: passes low frequencies, blocks high frequencies")
    print("- High-pass: passes high frequencies, blocks low frequencies") 
    print("- Band-pass: passes frequencies in a specific range")
    print("- Band-stop: blocks frequencies in a specific range")
    
    # Parameters
    sampling_rate = 1000  # Hz
    t = np.linspace(0, 2, 2 * sampling_rate)
    
    # Create test signal with different frequencies
    frequencies = [5, 20, 50, 100, 200]
    test_signal = np.zeros_like(t)
    for freq in frequencies:
        test_signal += np.sin(2 * np.pi * freq * t)
    
    # Add noise
    noise = 0.5 * np.random.randn(len(t))
    noisy_signal = test_signal + noise
    
    return t, noisy_signal, sampling_rate

def apply_filters(t, signal_data, sampling_rate):
    """
    Applies different types of filters to the signal
    """
    nyquist = sampling_rate / 2
    
    # Define filters
    filters = {}
    
    # Low-pass filter (cuts frequencies above 30 Hz)
    low_cutoff = 30 / nyquist
    filters['Low-pass'] = signal.butter(4, low_cutoff, btype='low')
    
    # High-pass filter (cuts frequencies below 40 Hz)  
    high_cutoff = 40 / nyquist
    filters['High-pass'] = signal.butter(4, high_cutoff, btype='high')
    
    # Band-pass filter (passes 15-60 Hz)
    band_low = 15 / nyquist
    band_high = 60 / nyquist
    filters['Band-pass'] = signal.butter(4, [band_low, band_high], btype='band')
    
    # Band-stop filter (blocks 45-55 Hz)
    stop_low = 45 / nyquist
    stop_high = 55 / nyquist
    filters['Band-stop'] = signal.butter(4, [stop_low, stop_high], btype='bandstop')
    
    # Apply filters
    filtered_signals = {}
    filtered_signals['Original'] = signal_data
    
    for filter_name, (b, a) in filters.items():
        filtered_signals[filter_name] = signal.filtfilt(b, a, signal_data)
    
    # Visualization
    fig, axes = plt.subplots(len(filtered_signals), 2, figsize=(15, 20))
    
    for i, (name, filtered_signal) in enumerate(filtered_signals.items()):
        # Time domain
        axes[i, 0].plot(t[:500], filtered_signal[:500], linewidth=2)
        axes[i, 0].set_title(f'{name} - Time Domain')
        axes[i, 0].set_xlabel('Time [s]')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True)
        
        # Frequency domain
        fft_filtered = fft(filtered_signal)
        freqs = fftfreq(len(filtered_signal), 1/sampling_rate)
        pos_mask = freqs >= 0
        
        axes[i, 1].plot(freqs[pos_mask][:200], np.abs(fft_filtered[pos_mask][:200]), linewidth=2)
        axes[i, 1].set_title(f'{name} - Frequency Domain')
        axes[i, 1].set_xlabel('Frequency [Hz]')
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].grid(True)
        
        # Add vertical lines for filter frequencies
        if name == 'Low-pass':
            axes[i, 1].axvline(30, color='red', linestyle='--', alpha=0.7, label='Cutoff')
        elif name == 'High-pass':
            axes[i, 1].axvline(40, color='red', linestyle='--', alpha=0.7, label='Cutoff')
        elif name == 'Band-pass':
            axes[i, 1].axvline(15, color='red', linestyle='--', alpha=0.7, label='Low cutoff')
            axes[i, 1].axvline(60, color='red', linestyle='--', alpha=0.7, label='High cutoff')
        elif name == 'Band-stop':
            axes[i, 1].axvspan(45, 55, color='red', alpha=0.3, label='Stop band')
        
        if name != 'Original':
            axes[i, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return filtered_signals

# ==========================================
# 4. Combining filters
# ==========================================

def demonstrate_filter_combination():
    """
    Demonstrates how to combine filters and why
    """
    print("\n=== 4. FILTER COMBINATION ===")
    print("Filters can be combined for:")
    print("- Sharper cutoffs (steeper transitions)")
    print("- Complex frequency characteristics") 
    print("- Improved filtering quality")
    
    # Create test signal
    sampling_rate = 1000
    t = np.linspace(0, 2, 2 * sampling_rate)
    
    # Signal with many frequencies
    frequencies = np.arange(10, 200, 10)
    complex_signal = np.zeros_like(t)
    for freq in frequencies:
        amplitude = 1 / (1 + (freq/50)**2)  # Amplitude decreases with frequency
        complex_signal += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Apply sequential filters
    nyquist = sampling_rate / 2
    
    # First filter: Low-pass at 80 Hz
    b1, a1 = signal.butter(2, 80/nyquist, btype='low')
    filtered_once = signal.filtfilt(b1, a1, complex_signal)
    
    # Second filter: High-pass at 20 Hz
    b2, a2 = signal.butter(2, 20/nyquist, btype='high')
    filtered_twice = signal.filtfilt(b2, a2, filtered_once)
    
    # Equivalent band-pass filter
    b3, a3 = signal.butter(4, [20/nyquist, 80/nyquist], btype='band')
    filtered_equivalent = signal.filtfilt(b3, a3, complex_signal)
    
    # Visualize results
    signals = {
        'Original': complex_signal,
        'After Low-pass (80 Hz)': filtered_once,
        'After High-pass (20 Hz)': filtered_twice,
        'Equivalent Band-pass': filtered_equivalent
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, sig) in enumerate(signals.items()):
        fft_sig = fft(sig)
        freqs = fftfreq(len(sig), 1/sampling_rate)
        pos_mask = freqs >= 0
        
        axes[i].plot(freqs[pos_mask][:150], np.abs(fft_sig[pos_mask][:150]), linewidth=2)
        axes[i].set_title(name)
        axes[i].set_xlabel('Frequency [Hz]')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True)
        
        if i >= 1:  # Add reference lines
            axes[i].axvline(20, color='red', linestyle='--', alpha=0.7)
            axes[i].axvline(80, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. What is an equalizer
# ==========================================

class SimpleEqualizer:
    """
    Simple equalizer class for demonstration
    """
    def __init__(self, sampling_rate, num_bands=10):
        self.sampling_rate = sampling_rate
        self.num_bands = num_bands
        self.nyquist = sampling_rate / 2
        
        # Define frequency bands (logarithmic distribution)
        self.freq_bands = np.logspace(np.log10(20), np.log10(self.nyquist), num_bands + 1)
        self.gains = np.ones(num_bands)  # Initially no change
        
    def set_band_gain(self, band_index, gain_db):
        """Sets the gain for a specific band in dB"""
        if 0 <= band_index < self.num_bands:
            self.gains[band_index] = 10**(gain_db / 20)  # Convert from dB
    
    def apply_eq(self, signal_data):
        """Applies equalization to the signal"""
        # FFT of signal
        fft_signal = fft(signal_data)
        freqs = fftfreq(len(signal_data), 1/self.sampling_rate)
        
        # Create frequency response
        freq_response = np.ones_like(freqs, dtype=complex)
        
        for i in range(self.num_bands):
            # Find frequencies in current band
            low_freq = self.freq_bands[i]
            high_freq = self.freq_bands[i + 1]
            
            band_mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) < high_freq)
            freq_response[band_mask] *= self.gains[i]
        
        # Apply equalization
        equalized_fft = fft_signal * freq_response
        
        # Return to time domain
        equalized_signal = np.real(ifft(equalized_fft))
        
        return equalized_signal
    
    def plot_frequency_response(self):
        """Shows the frequency response of the equalizer"""
        freqs = np.logspace(np.log10(20), np.log10(self.nyquist), 1000)
        response = np.ones_like(freqs)
        
        for i in range(self.num_bands):
            low_freq = self.freq_bands[i]
            high_freq = self.freq_bands[i + 1]
            band_mask = (freqs >= low_freq) & (freqs < high_freq)
            response[band_mask] *= self.gains[i]
        
        plt.figure(figsize=(12, 6))
        plt.semilogx(freqs, 20 * np.log10(response))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Gain [dB]')
        plt.title('Equalizer Frequency Response')
        plt.grid(True, which="both", ls="-", alpha=0.3)
        
        # Add vertical lines for band boundaries
        for freq in self.freq_bands:
            plt.axvline(freq, color='red', linestyle='--', alpha=0.5)
        
        plt.show()

def demonstrate_equalizer():
    """
    Demonstrates equalizer operation
    """
    print("\n=== 5. EQUALIZER ===")
    print("Equalizer: device that allows independent control of")
    print("gain in different frequency bands")
    
    # Create test signal
    sampling_rate = 44100  # CD quality
    duration = 2  # seconds
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Create musical signal (simulation)
    # Bass (20-250 Hz)
    bass = 0.8 * np.sin(2 * np.pi * 80 * t) + 0.6 * np.sin(2 * np.pi * 120 * t)
    
    # Mid frequencies (250-4000 Hz) - vocals and instruments
    mids = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t) + 0.2 * np.sin(2 * np.pi * 1760 * t)
    
    # High frequencies (4000+ Hz) - details
    highs = 0.2 * np.sin(2 * np.pi * 8000 * t) + 0.1 * np.sin(2 * np.pi * 12000 * t)
    
    # Combine
    original_signal = bass + mids + highs
    
    # Create equalizer
    eq = SimpleEqualizer(sampling_rate, num_bands=10)
    
    # Configure equalizer - typical "V-shape" EQ
    eq.set_band_gain(0, 3)    # Boost bass
    eq.set_band_gain(1, 2)    # Moderate boost low
    eq.set_band_gain(2, 0)    # Neutral
    eq.set_band_gain(3, -2)   # Small cut low-mid
    eq.set_band_gain(4, -3)   # Cut mid frequencies
    eq.set_band_gain(5, -2)   # Small cut high-mid
    eq.set_band_gain(6, 0)    # Neutral
    eq.set_band_gain(7, 2)    # Boost highs
    eq.set_band_gain(8, 3)    # More boost highs
    eq.set_band_gain(9, 2)    # Moderate boost highest
    
    # Apply equalization
    equalized_signal = eq.apply_eq(original_signal)
    
    # Show frequency response
    eq.plot_frequency_response()
    
    # Before and after comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time domain comparison
    time_slice = slice(0, int(0.1 * sampling_rate))  # First 0.1 seconds
    ax1.plot(t[time_slice], original_signal[time_slice], 'b-', label='Original', linewidth=2)
    ax1.set_title('Original Signal (Time Domain)')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    ax2.plot(t[time_slice], equalized_signal[time_slice], 'r-', label='Equalized', linewidth=2)
    ax2.set_title('Equalized Signal (Time Domain)')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    
    # Frequency domain comparison
    fft_orig = fft(original_signal)
    fft_eq = fft(equalized_signal)
    freqs = fftfreq(len(original_signal), 1/sampling_rate)
    pos_mask = freqs > 0
    
    ax3.semilogx(freqs[pos_mask], 20 * np.log10(np.abs(fft_orig[pos_mask])), 'b-', linewidth=2)
    ax3.set_title('Original Signal (Frequency Domain)')
    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel('Amplitude [dB]')
    ax3.grid(True, which="both", ls="-", alpha=0.3)
    ax3.set_xlim(20, sampling_rate/2)
    
    ax4.semilogx(freqs[pos_mask], 20 * np.log10(np.abs(fft_eq[pos_mask])), 'r-', linewidth=2)
    ax4.set_title('Equalized Signal (Frequency Domain)')
    ax4.set_xlabel('Frequency [Hz]')
    ax4.set_ylabel('Amplitude [dB]')
    ax4.grid(True, which="both", ls="-", alpha=0.3)
    ax4.set_xlim(20, sampling_rate/2)
    
    plt.tight_layout()
    plt.show()
    
    return original_signal, equalized_signal, eq

# ==========================================
# Main demonstration function
# ==========================================

def main():
    """
    Main function that executes all demonstrations
    """
    print("AUDIO FILTERS AND EQUALIZERS ANALYSIS")
    print("=" * 50)
    
    # 1. Signal basics
    t, noisy_signal, sampling_rate = demonstrate_signal_basics()
    
    # 2. Fourier Transform demonstration
    demonstrate_fourier_transform()
    
    # 3. Filter demonstration
    t, test_signal, sr = create_filter_examples()
    filtered_signals = apply_filters(t, test_signal, sr)
    
    # 4. Filter combination
    demonstrate_filter_combination()
    
    # 5. Equalizer demonstration
    original, equalized, equalizer = demonstrate_equalizer()
    
    print("\n=== CONCLUSION ===")
    print("Filters and equalizers are powerful tools for:")
    print("- Removing unwanted frequencies (noise, interference)")
    print("- Emphasizing important frequencies")
    print("- Correcting acoustic problems")
    print("- Creative sound shaping")
    print("\nIn real applications, more complex algorithms are used")
    print("such as IIR and FIR filters, adaptive filters, etc.")

# Run the demonstration
if __name__ == "__main__":
    main()
