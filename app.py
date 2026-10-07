import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from PIL import Image
import io

st.set_page_config(
    page_title="Fourier Transform Visualization Lab",
    page_icon="∿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title {font-size: 2.6rem; font-weight: 750; margin-bottom: 0.2rem;}
.subtitle {font-size: 1.05rem; opacity: .75; margin-bottom: 1.2rem;}
.section {font-size: 1.55rem; font-weight: 700; margin-top: 1rem;}
.formula {padding: 0.8rem 1rem; border-left: 4px solid #4c78a8; background: rgba(76,120,168,.08); border-radius: 5px; font-family: serif;}
.small {font-size: .9rem; opacity: .72;}
</style>
""", unsafe_allow_html=True)

def fft_data(x, fs):
    n = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(n, 1/fs)
    Xs = np.fft.fftshift(X)
    fsx = np.fft.fftshift(f)
    amp = np.abs(Xs) / n
    return fsx, amp, Xs

def spectrum_positive(x, fs):
    n = len(x)
    f = np.fft.rfftfreq(n, 1/fs)
    X = np.fft.rfft(x)
    amp = 2*np.abs(X)/n
    if n > 1:
        amp[0] /= 2
    return f, amp

def make_signal(t, components, noise_std=0, impulse=False):
    x = np.zeros_like(t)
    for amp, freq, phase in components:
        x += amp*np.sin(2*np.pi*freq*t + phase)
    if noise_std > 0:
        rng = np.random.default_rng(42)
        x += rng.normal(0, noise_std, len(t))
    if impulse:
        idx = len(t)//2
        width = max(2, len(t)//200)
        x[idx-width:idx+width] += 2.5
    return x

def line_fig(x, y, name, xlabel, ylabel, title, height=430):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
    fig.update_layout(template="plotly_white", title=title, height=height,
                      xaxis_title=xlabel, yaxis_title=ylabel,
                      margin=dict(l=55,r=25,t=55,b=50))
    return fig

def metric(label, value, help_text=None):
    st.metric(label, value, help=help_text)

st.sidebar.title("Fourier Transform Lab")
st.sidebar.caption("Aman Edge Physics")
page = st.sidebar.radio(
    "Choose a module",
    [
        "Home & Concept",
        "1 · Fourier Series",
        "2 · Time ↔ Frequency",
        "3 · Harmonics & Spectrum",
        "4 · Noise & Filtering",
        "5 · Convolution",
        "6 · 2D Image Fourier Transform",
        "7 · Diffraction & Reciprocal Space",
        "8 · Physics Applications",
        "9 · Interactive Experiments",
        "10 · STFT Spectrogram",
        "11 · DFT vs FFT Benchmark",
        "12 · Fourier Optics",
        "13 · 2D Frequency Filtering",
        "14 · 2D Reciprocal Lattice",
    ],
)
st.sidebar.divider()
st.sidebar.markdown("**Core idea**")
st.sidebar.latex(r"x(t) \xleftrightarrow{\mathcal F} X(f)")
st.sidebar.caption("Complex signals become understandable as collections of frequencies.")

if page == "Home & Concept":
    st.markdown('<div class="main-title">Fourier Transform Visualization Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">From oscillations and sound to filtering, images, diffraction, reciprocal space and quantum mechanics.</div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        metric("Domains", "Time / Frequency")
    with c2:
        metric("Dimensions", "1D / 2D")
    with c3:
        metric("Applications", "Signal → Physics")

    st.markdown('<div class="section">The central question</div>', unsafe_allow_html=True)
    st.write("A complicated waveform can look impossible to interpret in time. Fourier analysis asks a different question: **which elementary oscillations are hidden inside it?**")
    st.markdown('<div class="formula">A signal can be represented as a superposition of sinusoidal components with different amplitudes, frequencies and phases.</div>', unsafe_allow_html=True)

    t = np.linspace(0, 1, 2000, endpoint=False)
    x = (1.0*np.sin(2*np.pi*5*t) + 0.55*np.sin(2*np.pi*13*t+0.5)
         + 0.25*np.sin(2*np.pi*31*t+1.1))
    f, a = spectrum_positive(x, 2000)
    col1,col2 = st.columns(2)
    with col1:
        st.plotly_chart(line_fig(t, x, "Composite", "Time", "Amplitude",
                                 "A complicated signal in the time domain"), use_container_width=True)
    with col2:
        st.plotly_chart(line_fig(f[f<60], a[f<60], "Spectrum", "Frequency (Hz)", "Amplitude",
                                 "The hidden frequency components"), use_container_width=True)

    st.markdown('<div class="section">Why this matters</div>', unsafe_allow_html=True)
    items = [
        ("Voice & music", "Separate pitch, harmonics, noise and spectral content."),
        ("Communication", "Analyze carriers, bandwidth and multicarrier systems such as OFDM."),
        ("Imaging", "Manipulate spatial frequencies for filtering and reconstruction."),
        ("Medical imaging", "Fourier reconstruction is central to MRI image formation."),
        ("Machine diagnostics", "Faults often appear as characteristic vibration-frequency peaks."),
        ("Solid state", "Reciprocal space, diffraction, Bloch waves and Brillouin zones rely on Fourier ideas."),
        ("Quantum mechanics", "Position and momentum representations are Fourier-transform pairs."),
    ]
    for title, desc in items:
        st.markdown(f"**{title}** — {desc}")

elif page == "1 · Fourier Series":
    st.markdown('<div class="main-title">1 · Fourier Series</div>', unsafe_allow_html=True)
    st.write("Build a periodic waveform from harmonics. Increase the number of harmonics and watch the approximation converge.")
    c1,c2,c3 = st.columns(3)
    with c1: harmonics = st.slider("Number of odd harmonics", 1, 25, 5)
    with c2: fundamental = st.slider("Fundamental frequency (Hz)", 1.0, 20.0, 2.0)
    with c3: waveform = st.selectbox("Target waveform", ["Square wave", "Sawtooth", "Triangle wave"])

    t = np.linspace(0, 2/fundamental, 3000)
    if waveform == "Square wave":
        target = signal.square(2*np.pi*fundamental*t)
        approx = sum(4/np.pi/k*np.sin(2*np.pi*k*fundamental*t) for k in range(1,2*harmonics,2))
    elif waveform == "Triangle wave":
        target = signal.sawtooth(2*np.pi*fundamental*t, width=.5)
        approx = sum((8/np.pi**2)*((-1)**((k-1)//2))/(k**2)*np.sin(2*np.pi*k*fundamental*t)
                     for k in range(1,2*harmonics,2))
    else:
        target = signal.sawtooth(2*np.pi*fundamental*t)
        approx = sum((-2/np.pi)*(1/k)*np.sin(2*np.pi*k*fundamental*t) for k in range(1,harmonics+1))
        approx += 0.5

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t,y=target,name="Target",line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=t,y=approx,name=f"Fourier approximation ({harmonics} terms)"))
    fig.update_layout(template="plotly_white",height=500,xaxis_title="Time (s)",yaxis_title="Amplitude")
    st.plotly_chart(fig,use_container_width=True)
    st.latex(r"f(t)=\frac{a_0}{2}+\sum_{n=1}^{\infty}\left[a_n\cos(n\omega_0t)+b_n\sin(n\omega_0t)\right]")
    st.info("Notice the Gibbs phenomenon near discontinuities: increasing the number of terms makes the transition narrower, but the overshoot does not disappear completely.")

elif page == "2 · Time ↔ Frequency":
    st.markdown('<div class="main-title">2 · Time ↔ Frequency Domain</div>', unsafe_allow_html=True)
    st.write("Construct a signal and observe how its Fourier transform exposes the frequencies that created it.")
    fs = st.slider("Sampling frequency (Hz)", 100, 5000, 1000, step=100)
    duration = st.slider("Duration (s)", .25, 5.0, 1.0, step=.25)
    n = int(fs*duration)
    t = np.arange(n)/fs
    c1,c2,c3 = st.columns(3)
    with c1: f1 = st.slider("Frequency 1 (Hz)", 1, 200, 10)
    with c2: f2 = st.slider("Frequency 2 (Hz)", 1, 200, 40)
    with c3: f3 = st.slider("Frequency 3 (Hz)", 1, 200, 90)
    x = make_signal(t, [(1,f1,0),(0.6,f2,.4),(0.3,f3,1.0)])
    f, a = spectrum_positive(x, fs)
    left,right=st.columns(2)
    with left:
        st.plotly_chart(line_fig(t[:min(len(t),3000)],x[:min(len(x),3000)],"x(t)","Time (s)","Amplitude","Time-domain signal"),use_container_width=True)
    with right:
        mask=f<=min(250,fs/2)
        st.plotly_chart(line_fig(f[mask],a[mask],"|X(f)|","Frequency (Hz)","Magnitude","Frequency-domain spectrum"),use_container_width=True)
    st.latex(r"X(f)=\int_{-\infty}^{\infty}x(t)e^{-i2\pi ft}\,dt")
    st.success(f"Expected peaks: {f1} Hz, {f2} Hz and {f3} Hz. The FFT estimates these from the sampled signal.")

elif page == "3 · Harmonics & Spectrum":
    st.markdown('<div class="main-title">3 · Harmonics & Spectral Anatomy</div>', unsafe_allow_html=True)
    st.write("Explore how amplitude, phase and harmonic content change a waveform.")
    fs=2000
    t=np.arange(0,1,1/fs)
    amps=[]
    comps=[]
    cols=st.columns(4)
    for i,(freq,default_amp) in enumerate([(5,1.0),(10,.5),(15,.33),(25,.2)]):
        with cols[i]:
            amp=st.slider(f"A{i+1} @ {freq} Hz",0.0,1.5,float(default_amp),.05)
            phase=st.slider(f"Phase {freq} Hz",0.0,2*np.pi,0.0,0.1)
        comps.append((amp,freq,phase))
    x=make_signal(t,comps)
    f,a=spectrum_positive(x,fs)
    fig=make_subplots(rows=2,cols=1,shared_xaxes=False,vertical_spacing=.12,
                      subplot_titles=("Composite waveform","Amplitude spectrum"))
    fig.add_trace(go.Scatter(x=t[:1000],y=x[:1000],name="x(t)"),row=1,col=1)
    mask=f<100
    fig.add_trace(go.Bar(x=f[mask],y=a[mask],name="|X(f)|"),row=2,col=1)
    fig.update_layout(template="plotly_white",height=700)
    fig.update_xaxes(title_text="Time (s)",row=1,col=1)
    fig.update_xaxes(title_text="Frequency (Hz)",row=2,col=1)
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Spectral interpretation")
    st.write("A sharp spectral line indicates a coherent sinusoidal component. Multiple lines indicate multiple oscillatory components. Phase changes the waveform's alignment in time but does not, for an isolated sinusoid, change the magnitude-spectrum peak location.")

elif page == "4 · Noise & Filtering":
    st.markdown('<div class="main-title">4 · Noise, FFT and Filtering</div>', unsafe_allow_html=True)
    st.write("See how unwanted frequency components can be identified and suppressed.")
    fs=1000
    t=np.arange(0,2,1/fs)
    clean=np.sin(2*np.pi*12*t)+.55*np.sin(2*np.pi*35*t)
    noise_level=st.slider("Noise level",0.0,1.5,.45,.05)
    noise_freq=st.slider("Interference frequency (Hz)",50,250,120)
    noisy=clean+noise_level*np.random.default_rng(7).normal(size=len(t))+.7*np.sin(2*np.pi*noise_freq*t)
    cutoff=st.slider("Low-pass cutoff (Hz)",10,250,60)
    b,a=signal.butter(4,cutoff/(fs/2),btype="low")
    filtered=signal.filtfilt(b,a,noisy)
    f1,a1=spectrum_positive(noisy,fs); f2,a2=spectrum_positive(filtered,fs)
    fig=make_subplots(rows=3,cols=1,shared_xaxes=False,subplot_titles=("Clean vs noisy","Filtered signal","Spectra"))
    fig.add_trace(go.Scatter(x=t[:1500],y=clean[:1500],name="Clean"),row=1,col=1)
    fig.add_trace(go.Scatter(x=t[:1500],y=noisy[:1500],name="Noisy",opacity=.65),row=1,col=1)
    fig.add_trace(go.Scatter(x=t[:1500],y=filtered[:1500],name="Filtered"),row=2,col=1)
    m=f1<300
    fig.add_trace(go.Scatter(x=f1[m],y=a1[m],name="Noisy spectrum"),row=3,col=1)
    fig.add_trace(go.Scatter(x=f2[m],y=a2[m],name="Filtered spectrum"),row=3,col=1)
    fig.update_layout(template="plotly_white",height=850)
    st.plotly_chart(fig,use_container_width=True)
    st.info(f"Fourth-order Butterworth low-pass filter with cutoff {cutoff} Hz. The interference at {noise_freq} Hz is strongly attenuated when it lies above the cutoff.")

elif page == "5 · Convolution":
    st.markdown('<div class="main-title">5 · Convolution & Fourier Multiplication</div>', unsafe_allow_html=True)
    st.write("The convolution theorem is one of the most useful bridges between signal processing and Fourier analysis.")
    n=600
    x=np.zeros(n); x[220:380]=1
    sigma=18
    k=np.arange(-70,71)
    kernel=np.exp(-k**2/(2*sigma**2)); kernel/=kernel.sum()
    y=np.convolve(x,kernel,mode="same")
    X=np.fft.rfft(x); K=np.fft.rfft(kernel,n=n); Y=np.fft.rfft(y)
    c1,c2=st.columns(2)
    with c1:
        fig=go.Figure()
        fig.add_trace(go.Scatter(y=x,name="Input x[n]"))
        fig.add_trace(go.Scatter(y=kernel/kernel.max(),name="Kernel (scaled)"))
        fig.add_trace(go.Scatter(y=y,name="Convolution y[n]"))
        fig.update_layout(template="plotly_white",height=420,title="Time/domain view",xaxis_title="Sample")
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        f=np.arange(len(X))
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=f,y=np.abs(X)/np.max(np.abs(X)),name="|FFT{x}|"))
        fig.add_trace(go.Scatter(x=f,y=np.abs(K)/np.max(np.abs(K)),name="|FFT{h}|"))
        fig.add_trace(go.Scatter(x=f,y=np.abs(Y)/np.max(np.abs(Y)),name="|FFT{x*h}|"))
        fig.update_layout(template="plotly_white",height=420,title="Frequency-domain view",xaxis_title="Frequency bin")
        st.plotly_chart(fig,use_container_width=True)
    st.latex(r"\mathcal F\{x*h\}=X(f)H(f)")
    st.success("Convolution in the time domain corresponds to multiplication in the frequency domain. This is the mathematical foundation of many filters and linear systems.")

elif page == "6 · 2D Image Fourier Transform":
    st.markdown('<div class="main-title">6 · 2D Fourier Transform of Images</div>', unsafe_allow_html=True)
    st.write("Spatial frequency analysis: large-scale structures correspond to low spatial frequencies; fine textures and edges generate higher spatial frequencies.")
    uploaded=st.file_uploader("Upload an image (optional)",type=["png","jpg","jpeg"])
    if uploaded:
        img=Image.open(uploaded).convert("L")
    else:
        n=256
        yy,xx=np.mgrid[-1:1:complex(n),-1:1:complex(n)]
        img=np.exp(-((xx**2+yy**2)/.12))*255
        img=(img + 45*np.sin(2*np.pi*18*xx)+25*np.sin(2*np.pi*27*yy)).clip(0,255).astype(np.uint8)
        img=Image.fromarray(img)
    arr=np.asarray(img,dtype=float)
    F=np.fft.fftshift(np.fft.fft2(arr))
    logmag=np.log1p(np.abs(F))
    reconstructed=np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    c1,c2,c3=st.columns(3)
    with c1: st.image(img,caption="Input / spatial domain",use_container_width=True)
    with c2: st.image(logmag,caption="log(1 + |F(kx,ky)|)",use_container_width=True,clamp=True)
    with c3: st.image(np.clip(reconstructed,0,255).astype(np.uint8),caption="Inverse FFT reconstruction",use_container_width=True)
    st.latex(r"F(k_x,k_y)=\iint I(x,y)e^{-i(k_xx+k_yy)}\,dx\,dy")
    st.info("The centered spectrum places the zero spatial-frequency component at the image center. Bright off-center structures indicate dominant periodic or directional features.")

elif page == "7 · Diffraction & Reciprocal Space":
    st.markdown('<div class="main-title">7 · Diffraction, Reciprocal Space & Crystals</div>', unsafe_allow_html=True)
    st.write("A visual bridge from Fourier analysis to solid-state physics: periodic structures generate discrete reciprocal-space features.")
    N=st.slider("Number of lattice points",5,30,13)
    a=st.slider("Lattice spacing a",0.5,2.5,1.0,.1)
    x=np.arange(-N,N+1)*a
    density=np.zeros_like(x,dtype=float)
    density[:] = 1
    q=np.linspace(-12/a,12/a,2500)
    # finite lattice structure factor
    S=np.abs(np.sum(np.exp(-1j*np.outer(q,x)),axis=1))**2
    S/=S.max()
    c1,c2=st.columns(2)
    with c1:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=x,y=density,mode="markers",marker=dict(size=9),name="Lattice sites"))
        fig.update_layout(template="plotly_white",height=420,title="1D periodic lattice",xaxis_title="Position x",yaxis_title="Site occupancy")
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=q,y=S,name="Structure factor"))
        for m in range(-4,5):
            if m != 0:
                fig.add_vline(x=2*np.pi*m/a,line_dash="dot",opacity=.35)
        fig.update_layout(template="plotly_white",height=420,title="Reciprocal-space intensity",xaxis_title="Wavevector q",yaxis_title="Normalized intensity")
        st.plotly_chart(fig,use_container_width=True)
    st.latex(r"\rho(x)=\sum_n\delta(x-na)\quad\Longrightarrow\quad \rho(q)\propto\sum_G\delta(q-G)")
    st.write("The reciprocal lattice vectors are integer multiples of 2π/a in this 1D example. This same Fourier-space language generalizes to 2D and 3D crystals, diffraction and Brillouin zones.")

elif page == "8 · Physics Applications":
    st.markdown('<div class="main-title">8 · Fourier Transform Across Physics</div>', unsafe_allow_html=True)
    tabs=st.tabs(["Quantum Mechanics","Wave Physics","Solid State","Communication"])
    with tabs[0]:
        st.markdown("### Position ↔ momentum")
        st.latex(r"\phi(p)=\frac{1}{\sqrt{2\pi\hbar}}\int\psi(x)e^{-ipx/\hbar}\,dx")
        x=np.linspace(-8,8,1600)
        sigma=1.0
        psi=np.exp(-x**2/(4*sigma**2))*np.exp(1j*3*x)
        p=np.linspace(-8,8,1600)
        phi=np.exp(-sigma**2*(p-3)**2)
        fig=make_subplots(rows=1,cols=2,subplot_titles=("Position-space wavefunction","Momentum-space distribution"))
        fig.add_trace(go.Scatter(x=x,y=np.real(psi),name="Re ψ(x)"),row=1,col=1)
        fig.add_trace(go.Scatter(x=p,y=np.abs(phi)**2,name="|φ(p)|²"),row=1,col=2)
        fig.update_layout(template="plotly_white",height=430)
        st.plotly_chart(fig,use_container_width=True)
    with tabs[1]:
        st.write("For waves, Fourier decomposition separates a complicated field into plane-wave components characterized by wavevector k and frequency ω.")
        st.latex(r"e^{i(kx-\omega t)}")
        st.info("This viewpoint connects dispersion relations, wave packets, diffraction and interference.")
    with tabs[2]:
        st.write("Periodic potentials motivate reciprocal space. Fourier components of a crystal potential couple states whose wavevectors differ by reciprocal lattice vectors.")
        st.latex(r"V(\mathbf r)=\sum_{\mathbf G}V_{\mathbf G}e^{i\mathbf G\cdot\mathbf r}")
        st.info("This is a key mathematical step toward Bloch's theorem and band-structure calculations.")
    with tabs[3]:
        st.write("Digital communication uses frequency-selective processing and multicarrier representations. OFDM is a major example where the FFT/IFFT provides computationally efficient modulation and demodulation.")

elif page == "9 · Interactive Experiments":
    st.markdown('<div class="main-title">9 · Interactive Fourier Experiments</div>', unsafe_allow_html=True)
    st.write("Use these experiments to develop intuition rather than memorizing formulas.")
    exp=st.selectbox("Experiment",[
        "Uncertainty: pulse width vs bandwidth",
        "Sampling and aliasing",
        "Spectral leakage and windowing",
        "Time shift and phase",
    ])
    if exp=="Uncertainty: pulse width vs bandwidth":
        width=st.slider("Gaussian pulse σ",.05,1.5,.35,.05)
        fs=1000
        t=np.linspace(-4,4,4000)
        x=np.exp(-t**2/(2*width**2))
        f,a=spectrum_positive(x,fs)
        fig=make_subplots(rows=1,cols=2,subplot_titles=("Narrow/wide pulse","Frequency spectrum"))
        fig.add_trace(go.Scatter(x=t,y=x,name="pulse"),row=1,col=1)
        m=f<15
        fig.add_trace(go.Scatter(x=f[m],y=a[m],name="spectrum"),row=1,col=2)
        fig.update_layout(template="plotly_white",height=430)
        st.plotly_chart(fig,use_container_width=True)
        st.latex(r"\Delta x\,\Delta k\gtrsim \frac{1}{2}")
        st.write("A narrower pulse in one domain requires a broader distribution of spatial/temporal frequencies in the conjugate domain.")
    elif exp=="Sampling and aliasing":
        fs=st.slider("Sampling frequency",20,200,60)
        f0=st.slider("Signal frequency",5,120,45)
        duration=1
        t=np.linspace(0,duration,1000,endpoint=False)
        ts=np.arange(0,duration,1/fs)
        x=np.sin(2*np.pi*f0*t)
        xs=np.sin(2*np.pi*f0*ts)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=t,y=x,name="Continuous-like signal"))
        fig.add_trace(go.Scatter(x=ts,y=xs,mode="markers",name="Samples"))
        fig.update_layout(template="plotly_white",height=500,xaxis_title="Time (s)",yaxis_title="Amplitude")
        st.plotly_chart(fig,use_container_width=True)
        st.warning(f"Nyquist frequency = {fs/2:.1f} Hz. If f₀ exceeds it, aliasing occurs.")
    elif exp=="Spectral leakage and windowing":
        cycles=st.slider("Number of cycles in observation",1.0,20.0,5.5,.5)
        N=1024
        t=np.arange(N)/N
        x=np.sin(2*np.pi*cycles*t)
        windows={"Rectangular":np.ones(N),"Hann":np.hanning(N),"Hamming":np.hamming(N),"Blackman":np.blackman(N)}
        fig=go.Figure()
        for name,w in windows.items():
            f=np.fft.rfftfreq(N,1/N)
            A=np.abs(np.fft.rfft(x*w)); A/=A.max()
            fig.add_trace(go.Scatter(x=f[:100],y=A[:100],name=name))
        fig.update_layout(template="plotly_white",height=500,title="Window comparison",xaxis_title="Frequency bin",yaxis_title="Normalized magnitude")
        st.plotly_chart(fig,use_container_width=True)
        st.write("When the observation interval does not contain an integer number of cycles, energy spreads into neighboring frequency bins. Window functions trade main-lobe width against sidelobe suppression.")
    else:
        shift=st.slider("Time shift",0.0,1.0,.25,.01)
        f0=5
        t=np.linspace(0,1,1000,endpoint=False)
        x=np.sin(2*np.pi*f0*t)
        xs=np.sin(2*np.pi*f0*(t-shift))
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=t,y=x,name="Original"))
        fig.add_trace(go.Scatter(x=t,y=xs,name="Shifted"))
        fig.update_layout(template="plotly_white",height=500,xaxis_title="Time",yaxis_title="Amplitude")
        st.plotly_chart(fig,use_container_width=True)
        st.latex(r"x(t-t_0)\xleftrightarrow{\mathcal F}X(f)e^{-i2\pi f t_0}")
        st.write("A shift in time changes phase in the frequency domain while leaving the magnitude spectrum unchanged.")

elif page == "10 · STFT Spectrogram":
    st.markdown('<div class="main-title">10 · Short-Time Fourier Transform & Spectrogram</div>', unsafe_allow_html=True)
    fs=1000; duration=5; t=np.arange(0,duration,1/fs)
    f0=st.slider("Start frequency (Hz)",5,100,10)
    rate=st.slider("Chirp rate (Hz/s)",1,80,20)
    nperseg=st.select_slider("STFT window",options=[64,128,256,512,1024],value=256)
    x=signal.chirp(t,f0,duration,f0+rate*duration,method="linear")
    f,tt,Z=signal.stft(x,fs=fs,nperseg=nperseg,noverlap=int(.75*nperseg))
    P=20*np.log10(np.abs(Z)+1e-7)
    fig=go.Figure(go.Heatmap(x=tt,y=f,z=P,colorscale="Viridis",colorbar_title="dB"))
    fig.update_layout(template="plotly_white",height=560,title="Time-frequency spectrogram",xaxis_title="Time (s)",yaxis_title="Frequency (Hz)")
    st.plotly_chart(fig,use_container_width=True)
    st.latex(r"X(\\tau,f)=\\int x(t)w(t-\\tau)e^{-i2\\pi ft}dt")
    st.info("The global FFT tells you what frequencies exist; the STFT also tells you when they exist.")

elif page == "11 · DFT vs FFT Benchmark":
    st.markdown('<div class="main-title">11 · Direct DFT vs FFT</div>', unsafe_allow_html=True)
    N=st.select_slider("Samples",options=[64,128,256,512,1024,2048],value=512)
    fs=1000; t=np.arange(N)/fs; x=np.sin(2*np.pi*73*t)+.4*np.sin(2*np.pi*181*t)
    import time
    t0=time.perf_counter(); Xd=np.array([np.sum(x*np.exp(-2j*np.pi*k*np.arange(N)/N)) for k in range(N)]); td=time.perf_counter()-t0
    t0=time.perf_counter(); Xf=np.fft.fft(x); tf=time.perf_counter()-t0
    c=st.columns(4)
    c[0].metric("Direct DFT",f"{td*1000:.2f} ms"); c[1].metric("FFT",f"{tf*1000:.2f} ms"); c[2].metric("Speed-up",f"{td/tf:.1f}×"); c[3].metric("Max error",f"{np.max(np.abs(Xd-Xf)):.2e}")
    f=np.fft.rfftfreq(N,1/fs); fig=go.Figure()
    fig.add_trace(go.Scatter(x=f,y=np.abs(Xd[:N//2+1])/N,name="DFT"))
    fig.add_trace(go.Scatter(x=f,y=np.abs(Xf[:N//2+1])/N,name="FFT",line=dict(dash="dash")))
    fig.update_layout(template="plotly_white",height=500,title="Numerical equivalence of DFT and FFT",xaxis_title="Frequency (Hz)",yaxis_title="Magnitude")
    st.plotly_chart(fig,use_container_width=True)
    st.latex(r"O(N^2)\\quad\\rightarrow\\quad O(N\\log N)")

elif page == "12 · Fourier Optics":
    st.markdown('<div class="main-title">12 · Fourier Optics & Fraunhofer Diffraction</div>', unsafe_allow_html=True)
    N=384; L=10; x=np.linspace(-L/2,L/2,N); X,Y=np.meshgrid(x,x)
    aperture=st.selectbox("Aperture",["Single slit","Double slit","Circular","Square"])
    width=st.slider("Aperture size",.2,4.,1.,.1)
    if aperture=="Single slit": A=(np.abs(X)<width/2).astype(float)
    elif aperture=="Double slit": A=((np.abs(X-width)<width*.15)|(np.abs(X+width)<width*.15)).astype(float)
    elif aperture=="Circular": A=((X*X+Y*Y)<(width/2)**2).astype(float)
    else: A=((np.abs(X)<width/2)&(np.abs(Y)<width/2)).astype(float)
    F=np.fft.fftshift(np.fft.fft2(A)); I=np.abs(F)**2; I/=I.max()
    fig=make_subplots(rows=1,cols=2,subplot_titles=("Aperture","Far-field intensity"))
    fig.add_trace(go.Heatmap(x=x,y=x,z=A,colorscale="Gray",showscale=False),row=1,col=1)
    fig.add_trace(go.Heatmap(x=x,y=x,z=np.log10(I+1e-8),colorscale="Viridis"),row=1,col=2)
    fig.update_layout(template="plotly_white",height=540)
    st.plotly_chart(fig,use_container_width=True)
    st.latex(r"U(k_x,k_y)\\propto\\mathcal{F}\\{A(x,y)\\}")

elif page == "13 · 2D Frequency Filtering":
    st.markdown('<div class="main-title">13 · 2D Frequency-Domain Image Filtering</div>', unsafe_allow_html=True)
    up=st.file_uploader("Upload image",type=["png","jpg","jpeg"],key="advanced_filter")
    if up: I=np.asarray(Image.open(up).convert("L").resize((384,384)),float)
    else:
        yy,xx=np.mgrid[:384,:384]; I=120+65*np.sin(xx/7)+35*np.sin(yy/17)+20*np.random.default_rng(3).normal(size=(384,384)); I=ndimage.gaussian_filter(I,1)
    F=np.fft.fftshift(np.fft.fft2(I)); yy,xx=np.mgrid[:I.shape[0],:I.shape[1]]; cy,cx=np.array(I.shape)//2; R=np.sqrt((xx-cx)**2+(yy-cy)**2)
    kind=st.selectbox("Filter",["Low-pass","High-pass","Band-pass","Gaussian low-pass"])
    r1=st.slider("Inner radius",2,150,30); r2=st.slider("Outer radius",10,220,90)
    if kind=="Low-pass": H=R<=r2
    elif kind=="High-pass": H=R>=r1
    elif kind=="Band-pass": H=(R>=r1)&(R<=r2)
    else: H=np.exp(-(R**2)/(2*r2**2))
    out=np.real(np.fft.ifft2(np.fft.ifftshift(F*H)))
    c=st.columns(4); c[0].image(np.clip(I,0,255).astype(np.uint8),caption="Original"); c[1].image(H.astype(float),caption="Frequency mask"); c[2].image(np.log1p(np.abs(F*H)),caption="Filtered spectrum"); c[3].image(np.clip(out,0,255).astype(np.uint8),caption="Reconstruction")
    st.info("The mask acts directly in spatial-frequency space. Low frequencies represent broad structure; high frequencies carry fine detail and edges.")

elif page == "14 · 2D Reciprocal Lattice":
    st.markdown('<div class="main-title">14 · 2D Bravais Lattice → Reciprocal Space</div>', unsafe_allow_html=True)
    lattice=st.selectbox("Direct lattice",["Square","Rectangular","Triangular"]); a=st.slider("Lattice constant a",.5,2.,1.,.1); b=st.slider("b",.5,2.,1.3,.1); M=8
    pts=[]
    for i in range(-M,M+1):
        for j in range(-M,M+1):
            if lattice=="Square": p=(i*a,j*a)
            elif lattice=="Rectangular": p=(i*a,j*b)
            else: p=(i*a+j*a/2,j*np.sqrt(3)*a/2)
            pts.append(p)
    pts=np.array(pts); q=np.linspace(-10,10,220); QX,QY=np.meshgrid(q,q); S=np.zeros_like(QX,dtype=complex)
    for p in pts[::max(1,len(pts)//600)]: S += np.exp(-1j*(QX*p[0]+QY*p[1]))
    I=np.abs(S)**2; I/=I.max()
    fig=make_subplots(rows=1,cols=2,subplot_titles=("Direct lattice","Reciprocal-space intensity"))
    fig.add_trace(go.Scatter(x=pts[:,0],y=pts[:,1],mode="markers",name="Sites"),row=1,col=1)
    fig.add_trace(go.Heatmap(x=q,y=q,z=np.log10(I+1e-7),colorscale="Viridis"),row=1,col=2)
    fig.update_layout(template="plotly_white",height=560)
    st.plotly_chart(fig,use_container_width=True)
    st.latex(r"\\mathbf a_i\\cdot\\mathbf b_j=2\\pi\\delta_{ij}")
    st.write("This is the computational bridge from periodic real-space structure to reciprocal lattice, diffraction and eventually Brillouin zones.")

st.divider()
st.caption("Aman Edge Physics · Fourier Transform Visualization Lab · Built with Python, NumPy, SciPy, Plotly and Streamlit")
