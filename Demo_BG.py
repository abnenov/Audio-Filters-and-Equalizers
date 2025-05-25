# Анализ на поведението на различни филтри и тяхното влияние върху входния сигнал

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.io.wavfile
from scipy.fftpack import fft, fftfreq, ifft
from scipy import signal
import matplotlib.patches as patches

# Настройки за българска локализация на графиките
plt.rcParams['font.size'] = 12

# ==========================================
# 1. Какво е сигнал? Time domain vs Frequency domain
# ==========================================

def demonstrate_signal_basics():
    """
    Демонстрира основните концепции за сигнали, time domain и frequency domain
    """
    print("=== 1. ОСНОВИ НА СИГНАЛИТЕ ===")
    print("Сигнал: функция, която носи информация - обикновено се променя във времето")
    print("Time domain: представяне на сигнала като функция от времето")
    print("Frequency domain: представяне на сигнала като функция от честотата")
    
    # Създаване на примерен сигнал - смес от синусоиди
    t = np.linspace(0, 2, 1000)  # 2 секунди
    freq1, freq2, freq3 = 5, 15, 30  # Hz
    
    signal1 = np.sin(2 * np.pi * freq1 * t)
    signal2 = 0.5 * np.sin(2 * np.pi * freq2 * t)
    signal3 = 0.3 * np.sin(2 * np.pi * freq3 * t)
    combined_signal = signal1 + signal2 + signal3
    
    # Добавяне на шум
    noise = 0.2 * np.random.randn(len(t))
    noisy_signal = combined_signal + noise
    
    # Визуализация
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time domain
    ax1.plot(t[:200], combined_signal[:200], 'b-', linewidth=2)
    ax1.set_title('Чист сигнал (Time Domain)')
    ax1.set_xlabel('Време [s]')
    ax1.set_ylabel('Амплитуда')
    ax1.grid(True)
    
    ax2.plot(t[:200], noisy_signal[:200], 'r-', linewidth=1)
    ax2.set_title('Сигнал с шум (Time Domain)')
    ax2.set_xlabel('Време [s]')
    ax2.set_ylabel('Амплитуда')
    ax2.grid(True)
    
    # Frequency domain
    sampling_rate = len(t) / (t[-1] - t[0])
    frequencies = fftfreq(len(combined_signal), 1/sampling_rate)
    fft_clean = fft(combined_signal)
    fft_noisy = fft(noisy_signal)
    
    # Показваме само положителните честоти
    pos_mask = frequencies >= 0
    frequencies_pos = frequencies[pos_mask]
    fft_clean_pos = np.abs(fft_clean[pos_mask])
    fft_noisy_pos = np.abs(fft_noisy[pos_mask])
    
    ax3.plot(frequencies_pos[:100], fft_clean_pos[:100], 'b-', linewidth=2)
    ax3.set_title('Чист сигнал (Frequency Domain)')
    ax3.set_xlabel('Честота [Hz]')
    ax3.set_ylabel('Амплитуда')
    ax3.grid(True)
    
    ax4.plot(frequencies_pos[:100], fft_noisy_pos[:100], 'r-', linewidth=1)
    ax4.set_title('Сигнал с шум (Frequency Domain)')
    ax4.set_xlabel('Честота [Hz]')
    ax4.set_ylabel('Амплитуда')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return t, noisy_signal, sampling_rate

# ==========================================
# 2. Как работи Fourier Transform
# ==========================================

def demonstrate_fourier_transform():
    """
    Демонстрира как работи Fourier Transform с примери
    """
    print("\n=== 2. FOURIER TRANSFORM ===")
    print("Fourier Transform преобразува сигнал от time domain към frequency domain")
    print("Показва кои честоти са присъстващи в сигнала и с каква сила")
    
    # Примери с различни функции
    t = np.linspace(-2, 2, 1000)
    
    # 1. Синусоида
    sine_wave = np.sin(2 * np.pi * 5 * t)
    
    # 2. Правоъгълен импулс
    pulse = np.where(np.abs(t) < 0.5, 1, 0)
    
    # 3. Гаусова функция
    gaussian = np.exp(-t**2)
    
    # 4. Sinc функция
    sinc_func = np.sinc(2 * t)  # sinc(x) = sin(πx)/(πx)
    
    functions = [sine_wave, pulse, gaussian, sinc_func]
    titles = ['Синусоида (5 Hz)', 'Правоъгълен импулс', 'Гаусова функция', 'Sinc функция']
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    
    for i, (func, title) in enumerate(zip(functions, titles)):
        # Time domain
        axes[i, 0].plot(t, func, 'b-', linewidth=2)
        axes[i, 0].set_title(f'{title} (Time Domain)')
        axes[i, 0].set_xlabel('Време [s]')
        axes[i, 0].set_ylabel('Амплитуда')
        axes[i, 0].grid(True)
        
        # Frequency domain
        fft_func = fft(func)
        freqs = fftfreq(len(func), t[1] - t[0])
        
        # Центриране на честотите за по-добра визуализация
        fft_shifted = np.fft.fftshift(fft_func)
        freqs_shifted = np.fft.fftshift(freqs)
        
        axes[i, 1].plot(freqs_shifted, np.abs(fft_shifted), 'r-', linewidth=2)
        axes[i, 1].set_title(f'{title} (Frequency Domain)')
        axes[i, 1].set_xlabel('Честота [Hz]')
        axes[i, 1].set_ylabel('Амплитуда')
        axes[i, 1].grid(True)
        axes[i, 1].set_xlim(-20, 20)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. Какво е филтър и как работят различните типове
# ==========================================

def create_filter_examples():
    """
    Създава примери за различните типове филтри
    """
    print("\n=== 3. ТИПОВЕ ФИЛТРИ ===")
    print("Филтър: система която променя амплитудата на различни честоти в сигнала")
    print("Основни типове:")
    print("- Low-pass: пропуска ниски честоти, блокира високи")
    print("- High-pass: пропуска високи честоти, блокира ниски") 
    print("- Band-pass: пропуска честоти в определен диапазон")
    print("- Band-stop: блокира честоти в определен диапазон")
    
    # Параметри
    sampling_rate = 1000  # Hz
    t = np.linspace(0, 2, 2 * sampling_rate)
    
    # Създаване на тестов сигнал с различни честоти
    frequencies = [5, 20, 50, 100, 200]
    test_signal = np.zeros_like(t)
    for freq in frequencies:
        test_signal += np.sin(2 * np.pi * freq * t)
    
    # Добавяне на шум
    noise = 0.5 * np.random.randn(len(t))
    noisy_signal = test_signal + noise
    
    return t, noisy_signal, sampling_rate

def apply_filters(t, signal_data, sampling_rate):
    """
    Прилага различни типове филтри върху сигнала
    """
    nyquist = sampling_rate / 2
    
    # Дефиниране на филтрите
    filters = {}
    
    # Low-pass филтър (отрязва честоти над 30 Hz)
    low_cutoff = 30 / nyquist
    filters['Low-pass'] = signal.butter(4, low_cutoff, btype='low')
    
    # High-pass филтър (отрязва честоти под 40 Hz)  
    high_cutoff = 40 / nyquist
    filters['High-pass'] = signal.butter(4, high_cutoff, btype='high')
    
    # Band-pass филтър (пропуска 15-60 Hz)
    band_low = 15 / nyquist
    band_high = 60 / nyquist
    filters['Band-pass'] = signal.butter(4, [band_low, band_high], btype='band')
    
    # Band-stop филтър (блокира 45-55 Hz)
    stop_low = 45 / nyquist
    stop_high = 55 / nyquist
    filters['Band-stop'] = signal.butter(4, [stop_low, stop_high], btype='bandstop')
    
    # Прилагане на филтрите
    filtered_signals = {}
    filtered_signals['Original'] = signal_data
    
    for filter_name, (b, a) in filters.items():
        filtered_signals[filter_name] = signal.filtfilt(b, a, signal_data)
    
    # Визуализация
    fig, axes = plt.subplots(len(filtered_signals), 2, figsize=(15, 20))
    
    for i, (name, filtered_signal) in enumerate(filtered_signals.items()):
        # Time domain
        axes[i, 0].plot(t[:500], filtered_signal[:500], linewidth=2)
        axes[i, 0].set_title(f'{name} - Time Domain')
        axes[i, 0].set_xlabel('Време [s]')
        axes[i, 0].set_ylabel('Амплитуда')
        axes[i, 0].grid(True)
        
        # Frequency domain
        fft_filtered = fft(filtered_signal)
        freqs = fftfreq(len(filtered_signal), 1/sampling_rate)
        pos_mask = freqs >= 0
        
        axes[i, 1].plot(freqs[pos_mask][:200], np.abs(fft_filtered[pos_mask][:200]), linewidth=2)
        axes[i, 1].set_title(f'{name} - Frequency Domain')
        axes[i, 1].set_xlabel('Честота [Hz]')
        axes[i, 1].set_ylabel('Амплитуда')
        axes[i, 1].grid(True)
        
        # Добавяне на вертикални линии за честотите на филтрите
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
# 4. Комбиниране на филтри
# ==========================================

def demonstrate_filter_combination():
    """
    Демонстрира как се комбинират филтри и защо
    """
    print("\n=== 4. КОМБИНИРАНЕ НА ФИЛТРИ ===")
    print("Филтрите могат да се комбинират за:")
    print("- По-остри срезове (по-стръмни преходи)")
    print("- Комплексни честотни характеристики") 
    print("- Подобряване на качеството на филтрирането")
    
    # Създаване на тестов сигнал
    sampling_rate = 1000
    t = np.linspace(0, 2, 2 * sampling_rate)
    
    # Сигнал с много честоти
    frequencies = np.arange(10, 200, 10)
    complex_signal = np.zeros_like(t)
    for freq in frequencies:
        amplitude = 1 / (1 + (freq/50)**2)  # Амплитудата намалява с честотата
        complex_signal += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Прилагане на последователни филтри
    nyquist = sampling_rate / 2
    
    # Първи филтър: Low-pass на 80 Hz
    b1, a1 = signal.butter(2, 80/nyquist, btype='low')
    filtered_once = signal.filtfilt(b1, a1, complex_signal)
    
    # Втори филтър: High-pass на 20 Hz
    b2, a2 = signal.butter(2, 20/nyquist, btype='high')
    filtered_twice = signal.filtfilt(b2, a2, filtered_once)
    
    # Еквивалентен band-pass филтър
    b3, a3 = signal.butter(4, [20/nyquist, 80/nyquist], btype='band')
    filtered_equivalent = signal.filtfilt(b3, a3, complex_signal)
    
    # Визуализация на резултатите
    signals = {
        'Оригинален': complex_signal,
        'След Low-pass (80 Hz)': filtered_once,
        'След High-pass (20 Hz)': filtered_twice,
        'Еквивалентен Band-pass': filtered_equivalent
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, sig) in enumerate(signals.items()):
        fft_sig = fft(sig)
        freqs = fftfreq(len(sig), 1/sampling_rate)
        pos_mask = freqs >= 0
        
        axes[i].plot(freqs[pos_mask][:150], np.abs(fft_sig[pos_mask][:150]), linewidth=2)
        axes[i].set_title(name)
        axes[i].set_xlabel('Честота [Hz]')
        axes[i].set_ylabel('Амплитуда')
        axes[i].grid(True)
        
        if i >= 1:  # Добавяне на референтни линии
            axes[i].axvline(20, color='red', linestyle='--', alpha=0.7)
            axes[i].axvline(80, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. Какво е еквалайзер
# ==========================================

class SimpleEqualizer:
    """
    Прост еквалайзер клас за демонстрация
    """
    def __init__(self, sampling_rate, num_bands=10):
        self.sampling_rate = sampling_rate
        self.num_bands = num_bands
        self.nyquist = sampling_rate / 2
        
        # Дефиниране на честотните ленти (логаритмично разпределение)
        self.freq_bands = np.logspace(np.log10(20), np.log10(self.nyquist), num_bands + 1)
        self.gains = np.ones(num_bands)  # Първоначално без промяна
        
    def set_band_gain(self, band_index, gain_db):
        """Задава усилването за определена лента в dB"""
        if 0 <= band_index < self.num_bands:
            self.gains[band_index] = 10**(gain_db / 20)  # Конвертиране от dB
    
    def apply_eq(self, signal_data):
        """Прилага еквализацията върху сигнала"""
        # FFT на сигнала
        fft_signal = fft(signal_data)
        freqs = fftfreq(len(signal_data), 1/self.sampling_rate)
        
        # Създаване на честотна характеристика
        freq_response = np.ones_like(freqs, dtype=complex)
        
        for i in range(self.num_bands):
            # Намиране на честотите в текущата лента
            low_freq = self.freq_bands[i]
            high_freq = self.freq_bands[i + 1]
            
            band_mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) < high_freq)
            freq_response[band_mask] *= self.gains[i]
        
        # Прилагане на еквализацията
        equalized_fft = fft_signal * freq_response
        
        # Връщане в time domain
        equalized_signal = np.real(ifft(equalized_fft))
        
        return equalized_signal
    
    def plot_frequency_response(self):
        """Показва честотната характеристика на еквалайзера"""
        freqs = np.logspace(np.log10(20), np.log10(self.nyquist), 1000)
        response = np.ones_like(freqs)
        
        for i in range(self.num_bands):
            low_freq = self.freq_bands[i]
            high_freq = self.freq_bands[i + 1]
            band_mask = (freqs >= low_freq) & (freqs < high_freq)
            response[band_mask] *= self.gains[i]
        
        plt.figure(figsize=(12, 6))
        plt.semilogx(freqs, 20 * np.log10(response))
        plt.xlabel('Честота [Hz]')
        plt.ylabel('Усилване [dB]')
        plt.title('Честотна характеристика на еквалайзера')
        plt.grid(True, which="both", ls="-", alpha=0.3)
        
        # Добавяне на вертикални линии за границите на лентите
        for freq in self.freq_bands:
            plt.axvline(freq, color='red', linestyle='--', alpha=0.5)
        
        plt.show()

def demonstrate_equalizer():
    """
    Демонстрира работата на еквалайзера
    """
    print("\n=== 5. ЕКВАЛАЙЗЕР ===")
    print("Еквалайзер: устройство което позволява независимо управление на")
    print("усилването в различни честотни ленти")
    
    # Създаване на тестов сигнал
    sampling_rate = 44100  # CD качество
    duration = 2  # секунди
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Създаване на музикален сигнал (симулация)
    # Бас (20-250 Hz)
    bass = 0.8 * np.sin(2 * np.pi * 80 * t) + 0.6 * np.sin(2 * np.pi * 120 * t)
    
    # Средни честоти (250-4000 Hz) - вокали и инструменти
    mids = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t) + 0.2 * np.sin(2 * np.pi * 1760 * t)
    
    # Високи честоти (4000+ Hz) - детайли
    highs = 0.2 * np.sin(2 * np.pi * 8000 * t) + 0.1 * np.sin(2 * np.pi * 12000 * t)
    
    # Комбиниране
    original_signal = bass + mids + highs
    
    # Създаване на еквалайзер
    eq = SimpleEqualizer(sampling_rate, num_bands=10)
    
    # Настройка на еквалайзера - типичен "V-shape" EQ
    eq.set_band_gain(0, 3)    # Boost басите
    eq.set_band_gain(1, 2)    # Умерено boost на ниските
    eq.set_band_gain(2, 0)    # Неутрално
    eq.set_band_gain(3, -2)   # Малко cut на долните средни
    eq.set_band_gain(4, -3)   # Cut на средните честоти
    eq.set_band_gain(5, -2)   # Малко cut на горните средни
    eq.set_band_gain(6, 0)    # Неутрално
    eq.set_band_gain(7, 2)    # Boost на високите
    eq.set_band_gain(8, 3)    # По-голям boost на високите
    eq.set_band_gain(9, 2)    # Умерено boost на най-високите
    
    # Прилагане на еквализацията
    equalized_signal = eq.apply_eq(original_signal)
    
    # Показване на честотната характеристика
    eq.plot_frequency_response()
    
    # Сравнение преди и след
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time domain сравнение
    time_slice = slice(0, int(0.1 * sampling_rate))  # Първите 0.1 секунди
    ax1.plot(t[time_slice], original_signal[time_slice], 'b-', label='Оригинал', linewidth=2)
    ax1.set_title('Оригинален сигнал (Time Domain)')
    ax1.set_xlabel('Време [s]')
    ax1.set_ylabel('Амплитуда')
    ax1.grid(True)
    
    ax2.plot(t[time_slice], equalized_signal[time_slice], 'r-', label='Еквализиран', linewidth=2)
    ax2.set_title('Еквализиран сигнал (Time Domain)')
    ax2.set_xlabel('Време [s]')
    ax2.set_ylabel('Амплитуда')
    ax2.grid(True)
    
    # Frequency domain сравнение
    fft_orig = fft(original_signal)
    fft_eq = fft(equalized_signal)
    freqs = fftfreq(len(original_signal), 1/sampling_rate)
    pos_mask = freqs > 0
    
    ax3.semilogx(freqs[pos_mask], 20 * np.log10(np.abs(fft_orig[pos_mask])), 'b-', linewidth=2)
    ax3.set_title('Оригинален сигнал (Frequency Domain)')
    ax3.set_xlabel('Честота [Hz]')
    ax3.set_ylabel('Амплитуда [dB]')
    ax3.grid(True, which="both", ls="-", alpha=0.3)
    ax3.set_xlim(20, sampling_rate/2)
    
    ax4.semilogx(freqs[pos_mask], 20 * np.log10(np.abs(fft_eq[pos_mask])), 'r-', linewidth=2)
    ax4.set_title('Еквализиран сигнал (Frequency Domain)')
    ax4.set_xlabel('Честота [Hz]')
    ax4.set_ylabel('Амплитуда [dB]')
    ax4.grid(True, which="both", ls="-", alpha=0.3)
    ax4.set_xlim(20, sampling_rate/2)
    
    plt.tight_layout()
    plt.show()
    
    return original_signal, equalized_signal, eq

# ==========================================
# Главна функция за демонстрация
# ==========================================

def main():
    """
    Главна функция която изпълнява всички демонстрации
    """
    print("АНАЛИЗ НА АУДИО ФИЛТРИ И ЕКВАЛАЙЗЕРИ")
    print("=" * 50)
    
    # 1. Основи на сигналите
    t, noisy_signal, sampling_rate = demonstrate_signal_basics()
    
    # 2. Fourier Transform демонстрация
    demonstrate_fourier_transform()
    
    # 3. Демонстрация на филтри
    t, test_signal, sr = create_filter_examples()
    filtered_signals = apply_filters(t, test_signal, sr)
    
    # 4. Комбиниране на филтри
    demonstrate_filter_combination()
    
    # 5. Еквалайзер демонстрация
    original, equalized, equalizer = demonstrate_equalizer()
    
    print("\n=== ЗАКЛЮЧЕНИЕ ===")
    print("Филтрите и еквалайзерите са мощни инструменти за:")
    print("- Премахване на нежелани честоти (шум, интерференции)")
    print("- Подчертаване на важни честоти")
    print("- Корекция на акустични проблеми")
    print("- Творческо оформяне на звука")
    print("\nВ реални приложения се използват по-сложни алгоритми")
    print("като IIR и FIR филтри, адаптивни филтри, и др.")

# Изпълнение на демонстрацията
if __name__ == "__main__":
    main()
