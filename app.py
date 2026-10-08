import io
import wave
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy import signal
from scipy.spatial import HalfspaceIntersection, ConvexHull
from PIL import Image

st.set_page_config(
    page_title="Fourier Applications Laboratory",
    page_icon="∿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- STYLE -----------------------------
st.markdown("""
<style>
:root{--accent:#ff4b4b;--ink:#20232d;--muted:#667085;--panel:#f7f8fa}
.hero{padding:22px 24px;border:1px solid rgba(30,40,60,.12);border-radius:18px;
background:linear-gradient(135deg,#ffffff,#f6f7fa);margin-bottom:18px}
.kicker{font-size:.72rem;font-weight:850;letter-spacing:.14em;color:#ff4b4b}
.title{font-size:2.55rem;font-weight:850;letter-spacing:-.045em;color:#20232d;margin:2px 0}
.sub{font-size:1rem;color:#667085;line-height:1.55}
.flow{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin-top:14px}
.flow span{background:#fff;border:1px solid rgba(30,40,60,.12);padding:6px 10px;border-radius:999px;
font-size:.70rem;font-weight:750}.flow b{color:#ff4b4b}
.section{font-size:1.35rem;font-weight:800;color:#20232d;margin:25px 0 8px}
.explain{padding:15px 17px;border-left:4px solid #ff4b4b;background:rgba(255,75,75,.045);
border-radius:9px;line-height:1.65}
.mathcard{padding:14px 16px;background:#fafafa;border:1px solid rgba(30,40,60,.10);border-radius:12px;margin:10px 0}
.metric-card{padding:10px;border:1px solid rgba(30,40,60,.10);border-radius:12px;background:white}
.stRadio>div{gap:6px}
.main .block-container{padding-bottom:72px}
.fixed-footer{position:fixed;left:0;bottom:0;width:100%;height:42px;z-index:999999;
display:flex;align-items:center;justify-content:center;background:rgba(20,20,24,.97);
color:#f5f5f5;border-top:1px solid rgba(255,255,255,.16);font-size:13px}
</style>
""", unsafe_allow_html=True)

# ----------------------------- CORE -----------------------------
def fft1(x, dt):
    return np.fft.fftshift(np.fft.fft(np.asarray(x))) * dt

def ifft1(X, dt):
    return np.fft.ifft(np.fft.ifftshift(X)) / dt

def fft2_image(img):
    return np.fft.fftshift(np.fft.fft2(img))

def ifft2_image(F):
    return np.real(np.fft.ifft2(np.fft.ifftshift(F)))

def spectrum_axis(n, dt):
    return np.fft.fftshift(np.fft.fftfreq(n, d=dt))

def fig_lines(traces, title, xlabel, ylabel, height=390, log_y=False):
    fig = go.Figure()
    for x, y, name, extra in traces:
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode=extra.get("mode","lines"),
                                 line=extra.get("line"), marker=extra.get("marker")))
    fig.update_layout(template="plotly_white", height=height, title=title,
                      xaxis_title=xlabel, yaxis_title=ylabel,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    if log_y:
        fig.update_yaxes(type="log")
    return fig

def heatmap(z, title, x=None, y=None, scale="Turbo", height=430, aspect="auto"):
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=scale, colorbar=dict(len=.82)))
    fig.update_layout(template="plotly_white", height=height, title=title,
                      xaxis_title="", yaxis_title="")
    return fig

def header(title, subtitle, module):
    st.markdown(
        f'<div class="hero"><div class="kicker">{module}</div>'
        f'<div class="title">{title}</div><div class="sub">{subtitle}</div>'
        f'<div class="flow"><span>01 · INPUT</span><b>→</b><span>02 · FOURIER SPACE</span>'
        f'<b>→</b><span>03 · VISUALIZATION</span><b>→</b><span>04 · NUMERICAL ANALYSIS</span>'
        f'<b>→</b><span>05 · MATHEMATICS</span></div></div>', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)

def explain(text):
    st.markdown(f'<div class="explain">{text}</div>', unsafe_allow_html=True)

def math_section(items):
    section("Complete Mathematics")
    for label, equation, explanation in items:
        st.markdown(f'<div class="mathcard"><b>{label}</b></div>', unsafe_allow_html=True)
        st.latex(equation)
        if explanation:
            st.caption(explanation)

def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)

# ----------------------------- AUDIO -----------------------------
def read_wav(upload):
    with wave.open(io.BytesIO(upload.getvalue()), "rb") as w:
        fs = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(float)
        scale = 32768.0
    elif sw == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(float) - 128
        scale = 128.0
    else:
        raise ValueError("Only 8-bit and 16-bit WAV files are supported.")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data / max(scale, 1.0), fs

def wav_bytes(x, fs):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(fs))
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()

# ----------------------------- VOICE -----------------------------
def voice_page():
    header("Fourier Transform of Our Voice",
           "Use a real WAV recording and watch speech move from acoustic pressure into frequency and time-frequency space.",
           "01 · OUR VOICE")
    upload = st.file_uploader("Upload your voice sample (.wav)", type=["wav"], key="voice_file")
    if upload:
        x, fs = read_wav(upload)
        source = "Uploaded recording"
    else:
        fs = 16000
        t0 = np.arange(0, 2.4, 1/fs)
        env = np.exp(-((t0-1.2)/.85)**8)
        x = sum(np.sin(2*np.pi*135*m*t0)/(m**0.72) for m in range(1,18)) * env
        x /= np.max(np.abs(x))
        source = "Built-in voiced-signal model"
        st.info("Upload a WAV recording to replace the built-in model.")

    duration = len(x)/fs
    metric_row([("Source", source), ("Sampling rate", f"{fs/1000:.1f} kHz"),
                ("Duration", f"{duration:.2f} s"), ("Samples", f"{len(x):,}")])

    max_samples = min(len(x), 65536)
    n = st.slider("FFT analysis window", 1024, max_samples, min(16384, max_samples), 1024)
    start = st.slider("Window start", 0, max(0, len(x)-n), 0, max(1,n//4))
    seg = x[start:start+n]
    t = np.arange(len(seg))/fs
    window = signal.windows.hann(len(seg))
    X = np.fft.rfft(seg*window)
    f = np.fft.rfftfreq(len(seg), 1/fs)
    mag = np.abs(X)
    mag /= max(mag.max(), 1e-15)

    section("1. What the Fourier transform sees")
    c1,c2 = st.columns(2)
    c1.plotly_chart(fig_lines([(t,seg,"pressure waveform",{})],
                              "Acoustic pressure in time","time (s)","normalized amplitude"),
                    use_container_width=True)
    c2.plotly_chart(fig_lines([(f,mag,"|X(f)|",{})],
                              "Single-sided Fourier magnitude","frequency (Hz)","normalized magnitude"),
                    use_container_width=True)

    peaks,_ = signal.find_peaks(mag, prominence=.015, distance=max(1,int(40/(f[1]-f[0]))))
    order = peaks[np.argsort(mag[peaks])[-10:]] if len(peaks) else np.array([],dtype=int)
    dominant = f[np.argmax(mag[1:])+1]
    metric_row([("Dominant frequency", f"{dominant:.1f} Hz"),
                ("Spectral centroid", f"{np.sum(f*mag)/(np.sum(mag)+1e-15):.1f} Hz"),
                ("Detected peaks", str(len(order))),
                ("Window", f"{n/fs*1000:.1f} ms")])

    section("2. Time-frequency anatomy of speech")
    nper = min(1024, len(x))
    fsp,tsp,Z = signal.stft(x, fs=fs, window="hann", nperseg=nper, noverlap=int(.75*nper))
    st.plotly_chart(heatmap(20*np.log10(np.maximum(np.abs(Z),1e-7)),
                            "Spectrogram: where the frequencies occur in time",tsp,fsp,"Turbo",500),
                    use_container_width=True)
    explain("A voice is not stationary. The spectrum changes as phonemes, pitch, harmonics and vocal-tract resonances change. The Fourier transform supplies the frequency decomposition; the short-time form preserves when those components occur.")

    section("3. Numerical verification")
    # Parseval consistency for the selected segment.
    lhs = np.sum(np.abs(seg*window)**2)
    rhs = np.sum(np.abs(np.fft.fft(seg*window))**2)/len(seg)
    rel = abs(lhs-rhs)/(abs(lhs)+1e-15)
    metric_row([("Parseval relative error", f"{rel:.3e}"),
                ("RMS amplitude", f"{np.sqrt(np.mean(seg**2)):.5f}"),
                ("Peak amplitude", f"{np.max(np.abs(seg)):.5f}")])
    if upload:
        st.audio(upload.getvalue(), format="audio/wav")

    math_section([
        ("Continuous Fourier transform", r"X(f)=\int_{-\infty}^{\infty}x(t)e^{-i2\pi ft}\,dt", "Decomposes the signal into complex sinusoidal components."),
        ("Inverse transform", r"x(t)=\int_{-\infty}^{\infty}X(f)e^{i2\pi ft}\,df", "Reconstructs the original waveform from its spectrum."),
        ("Windowed measurement", r"x_w(t)=x(t)w(t)", "A finite recording is analyzed through a window."),
        ("Short-time Fourier transform", r"X(\tau,f)=\int x(t)w(t-\tau)e^{-i2\pi ft}\,dt", "Provides the time-frequency representation used for changing speech."),
        ("Vocal-tract model", r"S(f)=E(f)H_{\mathrm{vocal\ tract}}(f)", "Speech spectrum can be viewed as excitation shaped by the vocal tract."),
        ("Parseval relation", r"\int |x(t)|^2dt=\int |X(f)|^2df", "Energy is preserved under the properly normalized Fourier transform.")
    ])

# ----------------------------- NOISE -----------------------------
def noise_page():
    header("Fourier Noise Cancellation",
           "Separate wanted and unwanted components in frequency space, apply a controllable filter, reconstruct the audio, and quantify the result.",
           "02 · NOISE CANCELLATION")
    upload = st.file_uploader("Upload noisy WAV audio", type=["wav"], key="noise_file")
    if upload:
        x,fs = read_wav(upload); source="Uploaded noisy recording"
    else:
        fs=8000;t=np.arange(0,3,1/fs);rng=np.random.default_rng(12)
        clean=np.sin(2*np.pi*280*t)+.35*np.sin(2*np.pi*560*t)
        x=clean+.45*np.sin(2*np.pi*1550*t)+.12*rng.normal(size=len(t))
        source="Synthetic noisy signal"
        st.info("Upload your own noisy WAV to perform the experiment on real data.")

    metric_row([("Source",source),("Sampling rate",f"{fs/1000:.1f} kHz"),("Duration",f"{len(x)/fs:.2f} s")])
    n=min(len(x),65536)
    x=x[:n];t=np.arange(n)/fs
    X=np.fft.rfft(x);f=np.fft.rfftfreq(n,1/fs)

    filter_type=st.selectbox("Frequency-domain operation",["Low-pass","High-pass","Band-pass","Notch"])
    if filter_type=="Low-pass":
        fc=st.slider("Cutoff frequency",20.,fs/2-20.,min(800.,fs/3),10.)
        H=(f<=fc).astype(float)
    elif filter_type=="High-pass":
        fc=st.slider("Cutoff frequency",20.,fs/2-20.,min(500.,fs/4),10.)
        H=(f>=fc).astype(float)
    elif filter_type=="Band-pass":
        lo,hi=st.slider("Pass band",20.,fs/2-20.,(200.,min(1200.,fs/2-20)),10.)
        H=((f>=lo)&(f<=hi)).astype(float)
    else:
        fn=st.slider("Notch center",20.,fs/2-20.,min(1550.,fs/3),10.)
        bw=st.slider("Notch width",5.,500.,80.,5.)
        H=(np.abs(f-fn)>bw/2).astype(float)

    Y=X*H
    y=np.fft.irfft(Y,n=n)
    residual=x-y
    input_power=np.mean(x*x); residual_power=np.mean(residual*residual)
    energy_ret=np.sum(np.abs(Y)**2)/(np.sum(np.abs(X)**2)+1e-15)
    c1,c2=st.columns(2)
    c1.plotly_chart(fig_lines([(t,x,"input",{}),(t,y,"filtered",{"line":dict(dash="dash")})],
                              "Time-domain comparison","time (s)","amplitude"),
                    use_container_width=True)
    c2.plotly_chart(fig_lines([(f,np.abs(X),"input spectrum",{}),(f,np.abs(Y),"filtered spectrum",{})],
                              "Frequency-domain filtering","frequency (Hz)","|X(f)|"),
                    use_container_width=True)

    section("Numerical analysis")
    metric_row([("Spectral energy retained",f"{100*energy_ret:.2f}%"),
                ("Residual RMS",f"{np.sqrt(residual_power):.5f}"),
                ("Input RMS",f"{np.sqrt(input_power):.5f}"),
                ("Output/input RMS",f"{np.sqrt(np.mean(y*y))/(np.sqrt(input_power)+1e-15):.3f}")])
    st.audio(wav_bytes(y,fs),format="audio/wav")
    explain("The filter does not 'know' what noise is. It applies a mathematical rule H(f). The physical assumption is that useful and unwanted components occupy distinguishable spectral regions.")

    math_section([
        ("Additive-noise model", r"y(t)=s(t)+n(t)", "The measured signal is the sum of desired signal and noise."),
        ("Frequency-domain mixture", r"Y(f)=S(f)+N(f)", "Linearity makes separation possible in spectral space."),
        ("Linear filtering", r"\widehat S(f)=H(f)Y(f)", "The filter weights or removes selected frequencies."),
        ("Inverse reconstruction", r"\widehat s(t)=\mathcal F^{-1}\{\widehat S(f)\}", "Returns the cleaned signal to the time domain."),
        ("Ideal low-pass filter", r"H(f)=\begin{cases}1,&|f|\le f_c\\0,&|f|>f_c\end{cases}", "An idealized frequency selector."),
        ("Output residual", r"e(t)=y(t)-\widehat s(t)", "Quantifies the component removed by the filter.")
    ])

# ----------------------------- COMMUNICATION -----------------------------
def communication_page():
    header("Mobile Phone & Wi‑Fi",
           "See why communication systems use frequency space: modulation moves information around a carrier, while OFDM packs orthogonal subcarriers into a controlled bandwidth.",
           "03 · MOBILE PHONE & WI-FI")
    mode=st.radio("Experiment",["Carrier modulation","OFDM"],horizontal=True)
    fs=24000
    duration=.08
    t=np.arange(0,duration,1/fs)

    if mode=="Carrier modulation":
        fm=st.slider("Message frequency fₘ (Hz)",50,1200,300,10)
        fc=st.slider("Carrier frequency f꜀ (Hz)",2000,9000,5000,100)
        mu=st.slider("Modulation index μ",0.1,1.0,.6,.05)
        message=np.cos(2*np.pi*fm*t)
        x=(1+mu*message)*np.cos(2*np.pi*fc*t)
        X=np.fft.rfft(x);f=np.fft.rfftfreq(len(x),1/fs)
        theory=np.array([fc-fm,fc,fc+fm])
        section("Physical experiment")
        c1,c2=st.columns(2)
        c1.plotly_chart(fig_lines([(t[:1800],x[:1800],"AM waveform",{})],
                                  "Amplitude-modulated carrier","time (s)","amplitude"),
                        use_container_width=True)
        c2.plotly_chart(fig_lines([(f,np.abs(X),"measured spectrum",{})],
                                  "Carrier + sidebands","frequency (Hz)","magnitude"),
                        use_container_width=True)
        metric_row([("Carrier",f"{fc:.0f} Hz"),("Message",f"{fm:.0f} Hz"),
                    ("Upper sideband",f"{fc+fm:.0f} Hz"),("Lower sideband",f"{fc-fm:.0f} Hz")])
        explain("The message frequency does not disappear. Modulation translates its spectral content to frequencies around the carrier. This is the central Fourier-space idea behind radio communication.")
        math_items=[
            ("AM signal",r"x(t)=[1+mu m(t)]cos(2pi f_ct)","Information is placed on a high-frequency carrier."),
            ("Cosine spectrum",r"mathcal F{cos(2pi f_ct)}=rac12[delta(f-f_c)+delta(f+f_c)]","A sinusoidal carrier is localized at ±f꜀."),
            ("Frequency translation",r"mathcal F{m(t)cos(2pi f_ct)}=rac12[M(f-f_c)+M(f+f_c)]","Multiplication in time creates shifted copies in frequency."),
            ("AM bandwidth",r"B_{mathrm{AM}}=2f_{m,max}","Both sidebands occupy the message bandwidth around the carrier.")
        ]
    else:
        N=st.slider("Number of subcarriers N",4,64,16,4)
        df=st.slider("Subcarrier spacing Δf (Hz)",25,500,100,25)
        fc=st.slider("Center frequency (Hz)",3000,8000,5500,100)
        carriers=fc+(np.arange(N)-(N-1)/2)*df
        phases=np.linspace(0,np.pi,N)
        x=np.sum([np.cos(2*np.pi*q*t+p) for q,p in zip(carriers,phases)],axis=0)/N
        X=np.fft.rfft(x);f=np.fft.rfftfreq(len(x),1/fs)
        c1,c2=st.columns(2)
        c1.plotly_chart(fig_lines([(t[:1800],x[:1800],"OFDM waveform",{})],
                                  "Time-domain OFDM signal","time (s)","amplitude"),
                        use_container_width=True)
        c2.plotly_chart(fig_lines([(f,np.abs(X),"OFDM spectrum",{})],
                                  "Orthogonal subcarrier spectrum","frequency (Hz)","magnitude"),
                        use_container_width=True)
        fig=go.Figure(go.Scatter(x=carriers,y=np.ones(N),mode="markers",marker=dict(size=10)))
        fig.update_layout(template="plotly_white",height=300,title="Subcarrier grid",
                          xaxis_title="frequency (Hz)",yaxis_title="relative amplitude")
        st.plotly_chart(fig,use_container_width=True)
        metric_row([("Subcarriers",str(N)),("Spacing",f"{df:.0f} Hz"),
                    ("Symbol interval",f"{1/df*1000:.2f} ms"),("Occupied span",f"{N*df/1000:.2f} kHz")])
        explain("OFDM chooses subcarrier spacing so neighboring complex exponentials are orthogonal over one symbol interval. Their spectra may overlap while remaining mathematically separable.")
        math_items=[
            ("OFDM signal",r"x(t)=sum_{k=0}^{N-1}X_k e^{i2pi kDelta f t}","Each subcarrier carries an independent complex coefficient."),
            ("Orthogonality",r"int_0^T e^{i2pi(k-l)Delta f t},dt=0,quad k
e l","The subcarriers do not interfere under ideal synchronization."),
            ("Spacing",r"Delta f=rac1T","The spacing is tied to the useful symbol duration."),
            ("Bandwidth",r"Bapprox NDelta f","Increasing subcarrier count or spacing expands occupied spectrum.")
        ]
    math_section(math_items)

# ----------------------------- IMAGE PROCESSING -----------------------------
def load_gray(upload):
    if upload:
        arr=np.asarray(Image.open(upload).convert("L"),dtype=float)/255.
        return arr
    return None

def image_page():
    header("Fourier Transform of Images",
           "Upload a real image and watch spatial detail become spatial-frequency content. The same Fourier representation drives denoising, sharpening, edges, microscopy, pattern recognition and reconstruction.",
           "04 · IMAGES")
    upload=st.file_uploader("Upload an image",type=["png","jpg","jpeg","bmp","tif","tiff"],key="image_file")
    img=load_gray(upload)
    if img is None:
        N=320;y,x=np.mgrid[0:N,0:N]
        img=.45+.25*np.sin(2*np.pi*x/22)+.16*np.sin(2*np.pi*y/37)
        img+=.10*np.exp(-((x-90)**2+(y-210)**2)/(2*22**2))
        img+=.08*np.random.default_rng(3).normal(size=(N,N))
        img=np.clip(img,0,1)
        st.info("No image uploaded — using a built-in microscopy/texture test object.")
    maxdim=st.slider("Working image size",128,512,min(384,max(img.shape)),32)
    if max(img.shape)>maxdim:
        img=np.asarray(Image.fromarray((img*255).astype(np.uint8)).resize((maxdim,maxdim)),dtype=float)/255.
    else:
        # keep the aspect ratio while bounding the largest dimension
        if max(img.shape)>512:
            scale=512/max(img.shape)
            img=np.asarray(Image.fromarray((img*255).astype(np.uint8)).resize((int(img.shape[1]*scale),int(img.shape[0]*scale))),dtype=float)/255.

    ny,nx=img.shape
    dx=st.number_input("Pixel pitch Δx (arbitrary physical units)",0.01,100.0,1.0,.01)
    dy=st.number_input("Pixel pitch Δy (arbitrary physical units)",0.01,100.0,1.0,.01)
    F=fft2_image(img)
    kx=np.fft.fftshift(np.fft.fftfreq(nx,d=dx))
    ky=np.fft.fftshift(np.fft.fftfreq(ny,d=dy))
    KX,KY=np.meshgrid(kx,ky)
    KR=np.sqrt(KX**2+KY**2)
    krmax=max(KR.max(),1e-12)

    operation=st.selectbox("Choose the experiment",
                           ["Image sharpening","Image denoising","Edge detection",
                            "Microscopy / texture","Pattern recognition","Image reconstruction"])
    frac=st.slider("Radial spatial-frequency control",0.02,.95,.25,.01)

    low=(KR<=frac*krmax).astype(float)
    high=1-low
    if operation=="Image denoising":
        H=low
        explanation="Low spatial frequencies preserve broad structures while suppressing fine-scale fluctuations."
    elif operation=="Image sharpening":
        strength=st.slider("High-frequency boost",0.0,4.0,1.5,.1)
        H=1+strength*high
        explanation="Sharpening increases the contribution of high spatial frequencies, which encode rapid spatial changes."
    elif operation=="Edge detection":
        H=high
        explanation="Edges are rapid spatial transitions, so a high-pass Fourier filter emphasizes them."
    elif operation=="Microscopy / texture":
        H=((KR>=.08*krmax)&(KR<=.55*krmax)).astype(float)
        explanation="Texture often occupies intermediate spatial-frequency bands rather than only the lowest or highest frequencies."
    elif operation=="Pattern recognition":
        H=low
        explanation="Periodic structures produce localized peaks in spatial-frequency space; their positions are useful fingerprints."
    else:
        H=np.ones_like(KR)
        explanation="Reconstruction with the full spectrum demonstrates that Fourier transformation is reversible when the spectrum is preserved."

    G=F*H
    rec=ifft2_image(G)
    if operation!="Image reconstruction":
        # Keep visualization range stable rather than hiding numerical effects through per-image rescaling.
        rec=np.clip(rec,0,1)
    mse=np.mean((img-rec)**2)
    psnr=10*np.log10(1/max(mse,1e-15))
    spectral_energy=np.sum(np.abs(G)**2)/(np.sum(np.abs(F)**2)+1e-15)

    section("1. Spatial domain → Fourier domain → processed image")
    c1,c2,c3=st.columns(3)
    c1.plotly_chart(heatmap(img,"Original spatial-domain image",scale="Gray",height=400),
                    use_container_width=True)
    c2.plotly_chart(heatmap(np.log1p(np.abs(F)),"Log Fourier magnitude |F(kx,ky)|",
                            x=kx,y=ky,scale="Turbo",height=400),
                    use_container_width=True)
    c3.plotly_chart(heatmap(rec,operation,scale="Gray",height=400),
                    use_container_width=True)

    section("2. The filter itself in Fourier space")
    st.plotly_chart(heatmap(H,"Transfer function H(kx,ky)",x=kx,y=ky,scale="Viridis",height=400),
                    use_container_width=True)
    explain(explanation)

    # Radial spectrum with physical frequency axis.
    bins=np.linspace(0,krmax,80)
    centers=(bins[:-1]+bins[1:])/2
    radial=np.array([np.mean(np.abs(F)[(KR>=bins[i])&(KR<bins[i+1])]) for i in range(len(bins)-1)])
    st.plotly_chart(fig_lines([(centers,radial,"radial mean |F|",{})],
                              "Radially averaged spatial-frequency spectrum",
                              "spatial frequency magnitude","mean Fourier magnitude",350),
                    use_container_width=True)

    section("3. Numerical analysis")
    metric_row([("Spectral energy retained",f"{100*spectral_energy:.2f}%"),
                ("MSE",f"{mse:.5e}"),
                ("PSNR",f"{psnr:.2f} dB"),
                ("Resolution",f"{nx} × {ny}")])

    math_section([
        ("2D Fourier transform", r"F(k_x,k_y)=\iint I(x,y)e^{-i2\pi(k_xx+k_yy)}\,dx\,dy", "Every point in the image contributes to every spatial-frequency component."),
        ("Inverse transform", r"I(x,y)=\iint F(k_x,k_y)e^{i2\pi(k_xx+k_yy)}\,dk_x\,dk_y", "The spatial image is reconstructed by superposing spatial-frequency components."),
        ("Frequency-domain filter", r"G(k_x,k_y)=H(k_x,k_y)F(k_x,k_y)", "The transfer function controls which spatial scales survive."),
        ("High-pass edge image", r"I_{\mathrm{edge}}=\mathcal F^{-1}\{[1-H_{\mathrm{LP}}]F\}", "Rapid spatial changes are emphasized by removing low spatial frequencies."),
        ("Convolution theorem", r"\mathcal F\{I*h\}=F(k_x,k_y)H(k_x,k_y)", "Spatial convolution is equivalent to multiplication in Fourier space."),
        ("Parseval energy relation", r"\iint |I|^2\,dxdy \propto \iint |F|^2\,dk_xdk_y", "Energy can be evaluated consistently in either domain.")
    ])

# ----------------------------- MRI -----------------------------
def mri_page():
    header("Fourier Transform in MRI",
           "MRI acquires spatial-frequency information called k-space. Upload an image or use the phantom, undersample k-space, and see exactly what the inverse transform reconstructs.",
           "05 · MEDICAL IMAGING — MRI")
    upload=st.file_uploader("Upload an MRI-like grayscale image",type=["png","jpg","jpeg","tif","tiff"],key="mri_file")
    if upload:
        raw=np.asarray(Image.open(upload).convert("L"),dtype=float)/255.
        img=np.asarray(Image.fromarray((raw*255).astype(np.uint8)).resize((256,256)),dtype=float)/255.
        source="Uploaded image"
    else:
        y,x=np.mgrid[-1:1:256j,-1:1:256j]
        img=((x*x+y*y)<.78**2).astype(float)
        img+=.35*((x+.24)**2+(y-.16)**2<.16**2)
        img+=.22*((x-.27)**2+(y+.20)**2<.10**2)
        img+=.12*np.exp(-((x+.05)**2+(y+.35)**2)/(.04))
        img=np.clip(img,0,1)
        source="Built-in anatomical phantom"
        st.info("No MRI sample uploaded — using a built-in anatomical phantom.")

    F=fft2_image(img)
    ny,nx=img.shape
    ky=np.fft.fftshift(np.fft.fftfreq(ny,d=1.0))
    kx=np.fft.fftshift(np.fft.fftfreq(nx,d=1.0))
    KY,KX=np.meshgrid(ky,kx,indexing="ij")
    KR=np.sqrt(KX*KX+KY*KY)

    acquisition=st.selectbox("K-space acquisition experiment",
                             ["Full k-space","Central k-space only",
                              "Every second phase-encode line",
                              "Central + sparse outer lines","Variable-density sampling"])
    if acquisition=="Full":
        M=np.ones_like(img)
    elif acquisition=="Central k-space only":
        radius=st.slider("Central radius",5,110,45,5)
        M=(KR<=radius/max(KR.max(),1e-12)).astype(float)
    elif acquisition=="Every second phase-encode line":
        M=np.zeros_like(img);M[::2,:]=1
    elif acquisition=="Central + sparse outer lines":
        radius=st.slider("Central fully sampled radius",5,110,40,5)
        M=(KR<=radius/max(KR.max(),1e-12)).astype(float)
        M[::4,:]=1
    else:
        sigma=st.slider("Variable-density width",.03,.35,.12,.01)
        rng=np.random.default_rng(7)
        probability=np.exp(-(KR/sigma)**2)
        M=(rng.random(img.shape)<probability).astype(float)
        M[ny//2-2:ny//2+2,:]=1

    acquired=F*M
    rec=np.abs(ifft2_image(acquired))
    mse=np.mean((img-rec)**2)
    corr=np.corrcoef(img.ravel(),rec.ravel())[0,1]
    psnr=10*np.log10(1/max(mse,1e-15))
    frac=np.mean(M)

    metric_row([("Source",source),("K-space acquired",f"{100*frac:.2f}%"),
                ("MSE",f"{mse:.5e}"),("Correlation",f"{corr:.5f}")])

    section("1. Anatomy → k-space → sampling → reconstruction")
    c1,c2,c3=st.columns(3)
    c1.plotly_chart(heatmap(img,"Input anatomy / phantom",scale="Gray",height=400),use_container_width=True)
    c2.plotly_chart(heatmap(np.log1p(np.abs(F)),"Measured k-space magnitude",x=kx,y=ky,scale="Turbo",height=400),use_container_width=True)
    c3.plotly_chart(heatmap(rec,"Reconstructed image",scale="Gray",height=400),use_container_width=True)

    section("2. Acquisition mask")
    st.plotly_chart(heatmap(M,"Sampling mask M(kx,ky)",x=kx,y=ky,scale="Viridis",height=360),use_container_width=True)

    section("3. What k-space contains")
    horizontal=np.abs(F[ny//2,:]);vertical=np.abs(F[:,nx//2])
    c1,c2=st.columns(2)
    c1.plotly_chart(fig_lines([(kx,np.log1p(horizontal),"central ky=0",{})],
                              "Horizontal k-space profile","kx","log(1+|S|)",320),use_container_width=True)
    c2.plotly_chart(fig_lines([(ky,np.log1p(vertical),"central kx=0",{})],
                              "Vertical k-space profile","ky","log(1+|S|)",320),use_container_width=True)
    explain("Central k-space is dominated by low spatial frequencies and therefore strongly influences overall contrast and gross anatomy. Outer k-space carries finer spatial detail and contributes strongly to edge sharpness and resolution.")

    section("4. Numerical reconstruction quality")
    metric_row([("Acquisition fraction",f"{100*frac:.2f}%"),("PSNR",f"{psnr:.2f} dB"),
                ("Correlation",f"{corr:.5f}"),("Reconstruction RMS",f"{np.sqrt(np.mean(rec**2)):.5f}")])

    math_section([
        ("MRI signal equation", r"S(k_x,k_y)=\iint \rho(x,y)e^{-i2\pi(k_xx+k_yy)}\,dx\,dy", "The measured signal is a Fourier-space sample of the spin-density distribution."),
        ("Inverse reconstruction", r"\rho(x,y)=\iint S(k_x,k_y)e^{i2\pi(k_xx+k_yy)}\,dk_x\,dk_y", "The image is recovered by inverse Fourier transformation."),
        ("Gradient encoding", r"k(t)=\gamma\int_0^t G(\tau)\,d\tau", "Magnetic-field gradients determine the trajectory through k-space."),
        ("Sampling model", r"S_{\mathrm{measured}}(k_x,k_y)=M(k_x,k_y)S(k_x,k_y)", "The acquisition mask determines which Fourier coefficients are actually measured."),
        ("Reconstruction from partial k-space", r"\hat\rho=\mathcal F^{-1}\{M S\}", "Undersampling produces a predictable loss or redistribution of spatial information."),
        ("Approximate resolution", r"\Delta x\approx\frac{1}{2k_{\max}}", "Maximum sampled spatial frequency sets the approximate resolution scale.")
    ])

# ----------------------------- CRYSTAL / RECIPROCAL SPACE -----------------------------
def primitive_vectors(a, crystal):
    if crystal=="Simple Cubic":
        return a*np.eye(3)
    if crystal=="BCC":
        return (a/2)*np.array([[-1,1,1],[1,-1,1],[1,1,-1]],float)
    return (a/2)*np.array([[0,1,1],[1,0,1],[1,1,0]],float)

def reciprocal_vectors(A):
    a1,a2,a3=A
    V=np.dot(a1,np.cross(a2,a3))
    b1=2*np.pi*np.cross(a2,a3)/V
    b2=2*np.pi*np.cross(a3,a1)/V
    b3=2*np.pi*np.cross(a1,a2)/V
    return np.array([b1,b2,b3])

def reciprocal_points(B,hmax):
    pts=[]; idx=[]
    for h in range(-hmax,hmax+1):
        for k in range(-hmax,hmax+1):
            for l in range(-hmax,hmax+1):
                pts.append(h*B[0]+k*B[1]+l*B[2]);idx.append((h,k,l))
    return np.asarray(pts),idx

def brillouin_zone(B,order=2):
    # Wigner-Seitz cell around Gamma in reciprocal space.
    Gpts,_=reciprocal_points(B,order)
    Gpts=Gpts[np.linalg.norm(Gpts,axis=1)>1e-10]
    halfspaces=[]
    for g in Gpts:
        # g·k <= |g|^2/2
        halfspaces.append(np.r_[g,-np.dot(g,g)/2])
    halfspaces=np.asarray(halfspaces)
    try:
        hs=HalfspaceIntersection(halfspaces,np.zeros(3))
        verts=hs.intersections
        hull=ConvexHull(verts)
        return verts,hull
    except Exception:
        return None,None

def crystal_page():
    header("Reciprocal Lattice, Diffraction & Crystal Structure",
           "Start with a real-space crystal, construct reciprocal vectors, build the reciprocal lattice and first Brillouin zone, then calculate structure factors and diffraction.",
           "06 · RECIPROCAL SPACE & CRYSTALS")

    c1,c2,c3=st.columns(3)
    crystal=c1.selectbox("Crystal",["Simple Cubic","BCC","FCC"])
    a=c2.number_input("Conventional lattice constant a (Å)",1.0,20.0,3.0,.1)
    hmax=c3.slider("Reciprocal lattice range",1,4,2)

    A=primitive_vectors(a,crystal)
    B=reciprocal_vectors(A)
    volume=abs(np.linalg.det(A))
    reciprocal_volume=abs(np.linalg.det(B))

    metric_row([("Real primitive-cell volume",f"{volume:.4f} Å³"),
                ("Reciprocal-cell volume",f"{reciprocal_volume:.4f} Å⁻³"),
                ("V·V* /(2π)^3",f"{volume*reciprocal_volume/(2*np.pi)**3:.6f}")])

    section("1. Reciprocal-vector construction")
    c1,c2=st.columns(2)
    with c1:
        st.markdown("**Direct primitive vectors**")
        st.latex(r"\mathbf a_1,\mathbf a_2,\mathbf a_3")
        st.write(A)
    with c2:
        st.markdown("**Reciprocal primitive vectors**")
        st.latex(r"\mathbf b_i\cdot\mathbf a_j=2\pi\delta_{ij}")
        st.write(B)

    pts,idx=reciprocal_points(B,hmax)
    fig=go.Figure(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode="markers",
                                marker=dict(size=4),
                                text=[f"({h},{k},{l})" for h,k,l in idx],
                                hovertemplate="%{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>"))
    fig.update_layout(template="plotly_white",height=520,title="3D reciprocal lattice",
                      scene=dict(xaxis_title="kx",yaxis_title="ky",zaxis_title="kz"))
    st.plotly_chart(fig,use_container_width=True)

    section("2. First Brillouin zone")
    verts,hull=brillouin_zone(B,2)
    if verts is not None:
        mesh=go.Mesh3d(x=verts[:,0],y=verts[:,1],z=verts[:,2],
                       i=hull.simplices[:,0],j=hull.simplices[:,1],k=hull.simplices[:,2],
                       opacity=.42,name="1st BZ")
        fig=go.Figure(mesh)
        fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[0],mode="markers",marker=dict(size=6),name="Γ"))
        fig.update_layout(template="plotly_white",height=520,title="First Brillouin zone = Wigner–Seitz cell of reciprocal lattice",
                          scene=dict(xaxis_title="kx",yaxis_title="ky",zaxis_title="kz"))
        st.plotly_chart(fig,use_container_width=True)
    else:
        st.warning("The Brillouin-zone construction could not be resolved for this parameter set.")

    explain("The first Brillouin zone is not an arbitrary plotting box. It is the Wigner–Seitz cell around Γ in reciprocal space: every point inside it is closer to Γ than to any other reciprocal-lattice point.")

    section("3. Structure factor and systematic absences")
    basis={
        "Simple Cubic":[(0,0,0)],
        "BCC":[(0,0,0),(.5,.5,.5)],
        "FCC":[(0,0,0),(0,.5,.5),(.5,0,.5),(.5,.5,0)]
    }[crystal]
    h,k,l=[st.number_input(q,0,8,1,key=f"hkl_{q}") for q in ["h","k","l"]]
    Fhkl=sum(np.exp(2j*np.pi*(h*x+k*y+l*z)) for x,y,z in basis)
    d= a/np.sqrt(h*h+k*k+l*l) if h*h+k*k+l*l>0 else np.inf
    metric_row([("|F(hkl)|",f"{abs(Fhkl):.6f}"),
                ("I ∝ |F|²",f"{abs(Fhkl)**2:.6f}"),
                ("d(hkl)",f"{d:.5f} Å" if np.isfinite(d) else "∞")])

    section("4. Diffraction experiment")
    lam=st.slider("X-ray wavelength λ (Å)",.5,3.0,1.5406,.001)
    peaks=[];ints=[];labels=[]
    maxidx=max(4,hmax+2)
    for H in range(0,maxidx+1):
        for K in range(0,maxidx+1):
            for L in range(0,maxidx+1):
                if H==K==L==0: continue
                q2=(H/a)**2+(K/a)**2+(L/a)**2
                if q2<=0: continue
                dhkl=1/np.sqrt(q2)
                arg=lam/(2*dhkl)
                if 0<arg<1:
                    FF=sum(np.exp(2j*np.pi*(H*x+K*y+L*z)) for x,y,z in basis)
                    inten=abs(FF)**2
                    if inten>1e-10:
                        peaks.append(2*np.degrees(np.arcsin(arg)));ints.append(inten);labels.append(f"({H}{K}{L})")
    if peaks:
        order=np.argsort(peaks)
        peaks=np.asarray(peaks)[order];ints=np.asarray(ints)[order];labels=np.asarray(labels)[order]
        ints=ints/(ints.max()+1e-15)
        fig=go.Figure(go.Scatter(x=peaks,y=ints,mode="markers+lines",
                                 customdata=labels,
                                 hovertemplate="hkl=%{customdata}<br>2θ=%{x:.3f}°<br>I/Imax=%{y:.3f}<extra></extra>"))
        fig.update_layout(template="plotly_white",height=430,title="Structure-factor-weighted powder diffraction",
                          xaxis_title="2θ (degrees)",yaxis_title="normalized intensity")
        st.plotly_chart(fig,use_container_width=True)

    section("5. Numerical physics")
    st.latex(r"\mathbf G=h\mathbf b_1+k\mathbf b_2+l\mathbf b_3")
    st.latex(r"|\mathbf G|=\frac{2\pi}{d_{hkl}}")
    st.latex(r"2d_{hkl}\sin\theta=n\lambda")

    math_section([
        ("Reciprocal basis", r"\mathbf b_1=2\pi\frac{\mathbf a_2\times\mathbf a_3}{\mathbf a_1\cdot(\mathbf a_2\times\mathbf a_3)},\quad \mathbf b_i\cdot\mathbf a_j=2\pi\delta_{ij}", "The reciprocal basis is defined by the direct-lattice primitive vectors."),
        ("Reciprocal-lattice vector", r"\mathbf G=h\mathbf b_1+k\mathbf b_2+l\mathbf b_3", "Every reciprocal-lattice point is indexed by three integers."),
        ("Structure factor", r"F(\mathbf G)=\sum_j f_j e^{i\mathbf G\cdot\mathbf r_j}", "The basis determines which reciprocal-lattice reflections are allowed or extinguished."),
        ("Diffraction intensity", r"I(\mathbf G)\propto |F(\mathbf G)|^2", "Measured diffraction intensity is related to the squared magnitude of the structure factor."),
        ("Bragg law", r"2d_{hkl}\sin\theta=n\lambda", "Equivalent to the Laue/reciprocal-space diffraction condition."),
        ("First Brillouin zone", r"\mathrm{BZ}_1=\mathrm{Wigner\!-\!Seitz\ cell\ of\ the\ reciprocal\ lattice}", "The fundamental primitive cell in reciprocal space used for band-structure physics.")
    ])

# ----------------------------- APP -----------------------------
MODULES=[
    "Our Voice",
    "Noise Cancellation",
    "Mobile Phone & Wi‑Fi",
    "Images",
    "Medical Imaging — MRI",
    "Reciprocal Lattice, Diffraction & Crystal Structure Analysis"
]
choice=st.sidebar.radio("Fourier Applications",MODULES)
st.sidebar.divider()
st.sidebar.markdown("### Fourier viewpoint")
st.sidebar.latex(r"X(k)=\int x(r)e^{-ikr}\,dr")
st.sidebar.caption("Six applications only. Each module follows input → Fourier space → visualization → numerical analysis → mathematics.")

if choice==MODULES[0]:
    voice_page()
elif choice==MODULES[1]:
    noise_page()
elif choice==MODULES[2]:
    communication_page()
elif choice==MODULES[3]:
    image_page()
elif choice==MODULES[4]:
    mri_page()
else:
    crystal_page()

st.markdown('<div class="fixed-footer">Developed with Love ❤️ by Aman Kumar Patel</div>',unsafe_allow_html=True)
