import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================
# 1. ЗАГРУЗКА АУДИО (п. 3.1.2)
# ========================================== 
audio_file = 'C:\\Users\\kormu\\OneDrive\\Desktop\\FrMethods\\2_lab\\Accord(10).mp3' 
print(f"Загрузка файла: {audio_file}...")

# librosa.load автоматически конвертирует в моно (один канал) и возвращает sr (частоту дискретизации)
y, sr = librosa.load(audio_file, sr=None, mono=True) 

# Для ускорения численного интегрирования возьмем короткий отрезок (например, 1 секунду)
# Численное интегрирование всего трека займет очень много времени
duration_sec = 2.5
samples_count = int(sr * duration_sec)
y = y[:samples_count]

# Создаем массив времени t (п. 3.1.3)
t = np.arange(len(y)) / sr

print(f"Длительность анализируемого отрезка: {duration_sec} сек")
print(f"Количество отсчетов: {len(y)}")

# ==========================================
# 2. ПОСТРОЕНИЕ ГРАФИКА f(t) (п. 3.1.3)
# ==========================================
plt.figure(figsize=(12, 4))
plt.plot(t, y)
plt.title(f'Функция f(t)', size = 15)
plt.xlabel('Время t (с)', size = 15)
plt.ylabel('Амплитуда', size = 15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# 3. ЧИСЛЕННОЕ ИНТЕГРИРОВАНИЕ (п. 3.1.4)
# ==========================================
# Задание требует использовать trapz вместо fft
# Формула из задания: Y(k) = ∫ y(t) * exp(-i * 2 * pi * v(k) * t) dt

# Выбираем диапазон частот для анализа (человеческий слух до 20 кГц, но ноты аккорда обычно ниже 2-3 кГц)
v_max = 2000  # Гц
dv = 1.0      # Шаг частоты (Гц). Чем меньше шаг, тем точнее, но дольше считать.
v = np.arange(0, v_max, dv)

print("Вычисление Фурье-образа методом трапеций (это может занять время)...")

F_hat = np.zeros(len(v), dtype=complex)

# Цикл для вычисления интеграла для каждой частоты
# В MATLAB примере был цикл, здесь делаем аналогично для наглядности выполнения задания
for k, freq in enumerate(v):
    # Подынтегральное выражение: y(t) * e^(-i * 2 * pi * freq * t)
    integrand = y * np.exp(-1j * 2 * np.pi * freq * t)
    # Численное интегрирование методом трапеций (аналог trapz)
    F_hat[k] = np.trapezoid (integrand, t)

# Примечание: В заголовке_lab_работы сказано про "унитарное преобразование".
# Унитарный коэффициент обычно 1/sqrt(2*pi). Для поиска нот модуль не важен, 
# но для строгого соответствия можно умножить на коэффициент.
# Здесь оставляем как в примере кода из задания (без нормировки), так как важен спектр.

# ==========================================
# 4. ГРАФИК МОДУЛЯ |f^(v)| (п. 3.1.5)
# ==========================================
plt.figure(figsize=(12, 4))
plt.plot(v, np.abs(F_hat))
plt.title(f'Модуль Фурье-образа |F(v)|', size = 15)
plt.xlabel('Частота v (Гц)', size = 15)
plt.ylabel('|F(v)|', size = 15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# 5. АНАЛИЗ И НОТЫ (п. 3.1.6)
# ==========================================

# Функция для перевода частоты в ноту
def freq_to_note(freq):
    if freq <= 0: return None
    # A4 = 440 Гц. Номер ноты MIDI (A4 = 69)
    # formula: n = 12 * log2(f / 440) + 69
    midi_num = 12 * np.log2(freq / 440.0) + 69
    midi_num = int(np.round(midi_num))
    
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_name = notes[midi_num % 12]
    octave = (midi_num // 12) - 1
    return f"{note_name}{octave}", midi_num

# Поиск пиков в спектре
# Простой алгоритм: находим локальные максимумы выше порога
magnitude = np.abs(F_hat)
threshold = np.max(magnitude) * 0.1  # Порог 10% от максимума
peaks = []

for i in range(1, len(magnitude) - 1):
    if magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]:
        if magnitude[i] > threshold:
            peaks.append(v[i])

# Убираем дубликаты частот, которые слишком близки (в пределах несколько Гц)
clean_peaks = []
if peaks:
    clean_peaks.append(peaks[0])
    for p in peaks[1:]:
        if p - clean_peaks[-1] > 5: # Если разница больше 5 Гц, считаем новой нотой
            clean_peaks.append(p)

print("\n--- Результаты анализа аккорда ---")
print(f"Найдено основных частот (пиков): {len(clean_peaks)}")
for freq in clean_peaks:
    note, midi = freq_to_note(freq)
    print(f"Частота: {freq:.2f} Гц -> Нота: {note}")

print("\nВывод: Аккорд предположительно состоит из нот, указанных выше.")