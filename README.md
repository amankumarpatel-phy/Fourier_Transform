# Fourier Transform Visualization Lab

An interactive, research-oriented Streamlit laboratory for understanding Fourier analysis through visualization.

## What this project demonstrates

- Fourier series reconstruction of square, triangle and sawtooth waves
- Time-domain ↔ frequency-domain transformation using FFT
- Harmonic decomposition and spectral anatomy
- Noise generation, FFT diagnosis and digital filtering
- Convolution theorem
- 2D Fourier transform of images and inverse reconstruction
- Periodic lattices, diffraction and reciprocal-space structure factors
- Connections to quantum mechanics, wave physics, solid-state physics and communication
- Sampling, aliasing, spectral leakage, windowing and time-shift experiments

## Educational progression

**Oscillations → Fourier Series → Fourier Transform → Spectrum → Filtering → Convolution → 2D Fourier Transform → Reciprocal Space → Solid State Physics**

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

The app is designed for Streamlit Community Cloud. Select `app.py` as the entry point.

## Core equations

Fourier transform:

[
X(f)=\int_{-\infty}^{\infty}x(t)e^{-i2\pi ft}\,dt
]

Inverse transform:

[
x(t)=\int_{-\infty}^{\infty}X(f)e^{i2\pi ft}\,df
]

Convolution theorem:

[
\mathcal{F}\{x*h\}=X(f)H(f)
]

Time-shift theorem:

[
x(t-t_0)\leftrightarrow X(f)e^{-i2\pi ft_0}
]

## Project identity

**Aman Edge Physics**  
Computational Physics & Scientific Visualization
