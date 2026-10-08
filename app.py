import streamlit as st
import numpy as np
import io
import wave
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from PIL import Image

st.set_page_config(page_title="Fourier Physics Research Workstation",page_icon="∿",layout="wide")
st.markdown("""<style>
.title{font-size:2.6rem;font-weight:850}.sub{font-size:1.05rem;opacity:.72}
.card{padding:1rem;border:1px solid rgba(120,120,120,.25);border-radius:10px;margin:.5rem 0}
.small{font-size:.88rem;opacity:.72}.eq{padding:.7rem;background:rgba(80,120,180,.08);border-radius:8px}
.stMarkdown p{font-size:1.02rem;line-height:1.72;color:#30343b;margin:.55rem 0}
.stMarkdown h3{font-size:1.35rem;margin-top:.35rem;margin-bottom:.65rem}
.stMarkdown ul{line-height:1.7}
.stTabs [data-baseweb="tab-list"]{
    display:grid !important;
    grid-template-columns:repeat(6,minmax(0,1fr));
    gap:10px;
    padding:8px;
    margin:18px 0 24px 0;
    background:rgba(120,130,150,.08);
    border:1px solid rgba(90,100,120,.14);
    border-radius:14px;
    box-shadow:0 3px 14px rgba(20,30,50,.06);
}
.stTabs [data-baseweb="tab"]{
    height:58px !important;
    min-width:0 !important;
    justify-content:center;
    border:1px solid transparent !important;
    border-radius:10px 10px 8px 8px !important;
    padding:8px 10px !important;
    font-weight:650 !important;
    font-size:.88rem !important;
    line-height:1.15 !important;
    color:#4b5563 !important;
    background:transparent !important;
    transition:all .18s ease;
}
.stTabs [data-baseweb="tab"]:hover{
    background:rgba(255,255,255,.8) !important;
    border-color:rgba(90,100,120,.16) !important;
    transform:translateY(-1px);
}
.stTabs [data-baseweb="tab"][aria-selected="true"]{
    color:#111827 !important;
    background:#ffffff !important;
    border:1px solid rgba(255,90,90,.38) !important;
    box-shadow:0 4px 12px rgba(30,40,60,.10);
}
.stTabs [data-baseweb="tab-highlight"]{
    height:3px !important;
    border-radius:3px !important;
    background:#ff4b4b !important;
}
.stTabs [data-baseweb="tab-border"]{display:none !important;}
.stTabs [data-baseweb="tab-panel"]{
    padding-top:4px !important;
}
@media (max-width:1100px){
    .stTabs [data-baseweb="tab-list"]{grid-template-columns:repeat(3,minmax(0,1fr));}
}
@media (max-width:650px){
    .stTabs [data-baseweb="tab-list"]{grid-template-columns:repeat(2,minmax(0,1fr));}
    .stTabs [data-baseweb="tab"]{font-size:.8rem !important;height:52px !important;}
}
[data-testid="stCaptionContainer"]{line-height:1.5}
</style>""",unsafe_allow_html=True)

def FT(x,dx): return np.fft.fftshift(np.fft.fft(np.asarray(x)))*dx
def IFT(X,dx): return np.fft.ifft(np.fft.ifftshift(X))/dx
def K(n,dx): return np.fft.fftshift(np.fft.fftfreq(n,d=dx))
def fig1(traces,title,xlabel,ylabel,height=440):
    f=go.Figure()
    for x,y,name,kw in traces:f.add_trace(go.Scatter(x=x,y=y,name=name,**kw))
    f.update_layout(template="plotly_white",title=title,height=height,xaxis_title=xlabel,yaxis_title=ylabel,legend=dict(orientation="h"))
    return f
def heat(z,title,x=None,y=None,scale="Viridis",height=500):
    f=go.Figure(go.Heatmap(z=z,x=x,y=y,colorscale=scale));f.update_layout(template="plotly_white",title=title,height=height);return f
def gaussian(x,s):return np.exp(-x*x/(2*s*s))
def render_academic(content):
    """Render academic prose and display mathematics safely."""
    lines = content.splitlines()
    prose = []
    math_lines = []
    in_math = False

    def flush_prose():
        if prose:
            block = "\n".join(prose).strip()
            if block:
                block = block.replace(r"\\(", "$").replace(r"\\)", "$")
                st.markdown(block)
            prose.clear()

    for line in lines:
        stripped = line.strip()

        if stripped == r"\[":
            flush_prose()
            in_math = True
            math_lines.clear()
            continue

        if stripped == r"\]" and in_math:
            equation = "\n".join(math_lines).strip()
            if equation:
                st.latex(equation)
            math_lines.clear()
            in_math = False
            continue

        if stripped == "$$" and not in_math:
            flush_prose()
            in_math = True
            math_lines.clear()
            continue

        if stripped == "$$" and in_math:
            equation = "\n".join(math_lines).strip()
            if equation:
                st.latex(equation)
            math_lines.clear()
            in_math = False
            continue

        if in_math:
            math_lines.append(line)
        else:
            prose.append(line)

    if in_math:
        equation = "\n".join(math_lines).strip()
        if equation:
            st.latex(equation)

    flush_prose()


def math_section(title, equations):
    st.markdown(f"### {title}")
    for label, eq in equations:
        st.markdown(f"**{label}**")
        st.latex(eq)

def wav_bytes(x,fs):
    buf=io.BytesIO()
    with wave.open(buf,"wb") as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(fs)
        w.writeframes((np.clip(x,-1,1)*32767).astype(np.int16).tobytes())
    return buf.getvalue()

def audio_wav(uploaded):
    if uploaded is None:
        return None, None
    raw=uploaded.getvalue()
    with wave.open(io.BytesIO(raw),"rb") as w:
        fs=w.getframerate(); n=w.getnframes(); ch=w.getnchannels(); sw=w.getsampwidth()
        data=np.frombuffer(w.readframes(n),dtype=np.int16).astype(float)
    if ch>1:
        data=data.reshape(-1,ch).mean(axis=1)
    return data/(32768.0 if sw==2 else np.max(np.abs(data))+1e-12),fs

def page_header(title,subtitle,stage="INPUT → FOURIER → VISUALIZE → ANALYZE → MATHEMATICS"):
    st.markdown(
        f'<div class="hero"><div class="kicker">FOURIER APPLICATION LABORATORY</div>'
        f'<div class="title">{title}</div><div class="sub">{subtitle}</div>'
        f'<div class="labflow">'
        f'<span>01 · INPUT</span><b>→</b><span>02 · FOURIER SPACE</span><b>→</b>'
        f'<span>03 · VISUALIZE</span><b>→</b><span>04 · NUMERICAL ANALYSIS</span><b>→</b>'
        f'<span>05 · MATHEMATICS</span></div></div>',unsafe_allow_html=True)

def voice_page():
    page_header("Our Voice","Uploaded audio → waveform → Fourier spectrum → spectrogram → harmonics → complete mathematics")
    up=st.file_uploader("Upload a WAV voice sample",type=["wav"],key="voice_upload")
    if up:
        x,fs=audio_wav(up)
    else:
        fs=8000;t=np.arange(0,3,1/fs);f0=140
        x=sum(np.sin(2*np.pi*f0*h*t)/h for h in range(1,18));x*=np.exp(-((t-1.5)/1.2)**8);x/=np.max(np.abs(x))
        st.info("No sample uploaded — using a built-in synthetic voice sample.")
    duration=len(x)/fs
    n=st.slider("Analysis samples",1024,min(16384,len(x)),min(8192,len(x)),1024)
    x=x[:n];t=np.arange(n)/fs;X=np.fft.rfft(x);f=np.fft.rfftfreq(n,1/fs)
    c=st.columns(2)
    c[0].plotly_chart(fig1([(t,x,"voice",{})],"Time-domain waveform","time (s)","amplitude",390),use_container_width=True)
    c[1].plotly_chart(fig1([(f,np.abs(X)/n,"|X(f)|",{})],"Fourier spectrum","frequency (Hz)","magnitude",390),use_container_width=True)
    st.markdown("### Time–frequency visualization")
    fsp, tsp, Z=signal.stft(x,fs=fs,nperseg=min(512,n),noverlap=min(384,max(0,n//2)))
    st.plotly_chart(heat(20*np.log10(np.maximum(np.abs(Z),1e-6)),"Voice spectrogram",tsp,fsp,"Turbo",500),use_container_width=True)
    st.audio(up.getvalue() if up else None,format="audio/wav") if up else None
    peak=f[np.argmax(np.abs(X[1:]))+1]
    st.metric("Dominant frequency",f"{peak:.2f} Hz")
    math_section("Complete Mathematics",[
        ("Continuous Fourier transform",r"X(f)=\int_{-\infty}^{\infty}x(t)e^{-i2\pi ft}\,dt"),
        ("Inverse Fourier transform",r"x(t)=\int_{-\infty}^{\infty}X(f)e^{i2\pi ft}\,df"),
        ("Speech as excitation filtered by vocal tract",r"S(f)=E(f)H(f)"),
        ("Short-time Fourier transform",r"X(\tau,f)=\int x(t)w(t-\tau)e^{-i2\pi ft}\,dt"),
        ("Harmonic voice model",r"x(t)\approx\sum_{m=1}^{M}A_m\sin(2\pi m f_0t+\phi_m)")
    ])

def noise_page():
    page_header("Noise Cancellation","Uploaded noisy audio → spectrum → frequency-domain filter → reconstructed audio → quantitative analysis")
    up=st.file_uploader("Upload a noisy WAV sample",type=["wav"],key="noise_upload")
    if up: x,fs=audio_wav(up)
    else:
        fs=4000;t=np.arange(0,3,1/fs);clean=np.sin(2*np.pi*350*t)+.35*np.sin(2*np.pi*700*t);x=clean+.55*np.sin(2*np.pi*1100*t)+.2*np.random.default_rng(2).normal(size=len(t));st.info("No sample uploaded — using a synthetic noisy signal.")
    n=min(len(x),st.slider("Analysis samples",2048,min(32768,len(x)),min(12000,len(x)),1024));x=x[:n];t=np.arange(n)/fs
    X=np.fft.rfft(x);f=np.fft.rfftfreq(n,1/fs)
    cutoff=st.slider("Low-pass cutoff (Hz)",50.,min(fs/2-10,fs*0.45),700.,10.)
    H=(f<=cutoff).astype(float);Y=X*H;y=np.fft.irfft(Y,n=n)
    c=st.columns(2)
    c[0].plotly_chart(fig1([(t,x,"noisy",{}),(t,y,"filtered",{"line":dict(dash="dash")})],"Waveform before and after filtering","time (s)","amplitude",390),use_container_width=True)
    c[1].plotly_chart(fig1([(f,np.abs(X)/n,"input",{}),(f,np.abs(Y)/n,"filtered",{})],"Frequency-domain noise removal","Hz","magnitude",390),use_container_width=True)
    st.audio(wav_bytes(y,fs),format="audio/wav")
    removed=np.mean((x-y)**2);total=np.mean(x**2)
    st.metric("Removed spectral-energy fraction",f"{100*removed/(total+1e-15):.2f}%")
    math_section("Complete Mathematics",[
        ("Additive noise model",r"y(t)=s(t)+n(t)"),
        ("Fourier transform of the mixture",r"Y(f)=S(f)+N(f)"),
        ("Frequency-domain filter",r"\hat S(f)=H(f)Y(f)"),
        ("Inverse reconstruction",r"\hat s(t)=\mathcal F^{-1}\{H(f)Y(f)\}"),
        ("Signal-to-noise ratio",r"\mathrm{SNR}=10\log_{10}\left(\frac{P_s}{P_n}\right)")
    ])

def comm_page():
    page_header("Mobile Phone & Wi‑Fi","Communication waveform → carrier spectrum → bandwidth → OFDM subcarriers → complete mathematics")
    mode=st.selectbox("Signal model",["AM carrier","OFDM subcarriers"])
    fs=20000;t=np.arange(0,0.08,1/fs)
    if mode=="AM carrier":
        fm=st.slider("Message frequency (Hz)",50,1000,300,10);fc=st.slider("Carrier frequency (Hz)",2000,8000,5000,100)
        m=1+.6*np.cos(2*np.pi*fm*t);x=m*np.cos(2*np.pi*fc*t);title="AM waveform and spectrum"
        equations=[("AM signal",r"x(t)=[1+\mu m(t)]\cos(2\pi f_ct)"),("Modulation property",r"\mathcal F\{m(t)\cos(2\pi f_ct)\}=\frac12[M(f-f_c)+M(f+f_c)]"),("Bandwidth",r"B_{\rm AM}=2f_{m,\max}")]
    else:
        N=st.slider("Number of OFDM subcarriers",4,64,16,4);df=st.slider("Subcarrier spacing (Hz)",50,500,100,10);fc=5000
        carriers=fc+(np.arange(N)-N/2)*df
        x=sum(np.cos(2*np.pi*q*t+2*np.pi*i/N) for i,q in enumerate(carriers))/N;title="OFDM waveform and spectrum"
        equations=[("OFDM signal",r"x(t)=\sum_{k=0}^{N-1}X_k e^{i2\pi k\Delta f t}"),("Orthogonality",r"\int_0^T e^{i2\pi(k-l)\Delta f t}\,dt=0,\quad k\ne l"),("Subcarrier spacing",r"\Delta f=\frac1T"),("Occupied bandwidth",r"B\approx N\Delta f")]
    X=np.fft.rfft(x);f=np.fft.rfftfreq(len(x),1/fs)
    c=st.columns(2);c[0].plotly_chart(fig1([(t[:1200],x[:1200],"communication waveform",{})],"Time-domain signal","time (s)","amplitude",390),use_container_width=True);c[1].plotly_chart(fig1([(f,np.abs(X),"spectrum",{})],title,"frequency (Hz)","magnitude",390),use_container_width=True)
    if mode=="OFDM":
        st.plotly_chart(fig1([(carriers,np.ones_like(carriers),"subcarriers",{"mode":"markers"})],"OFDM subcarrier grid","frequency (Hz)","relative amplitude",330),use_container_width=True)
    st.metric("Sampling rate",f"{fs/1000:.1f} kHz")
    math_section("Complete Mathematics",equations)

def image_page():
    page_header("Images","Upload an image → 2D Fourier transform → filtering → sharpening / denoising / edges / microscopy / pattern recognition / reconstruction")
    up=st.file_uploader("Upload image",type=["png","jpg","jpeg","bmp"],key="image_upload")
    if up: img=np.array(Image.open(up).convert("L"),dtype=float)/255.
    else:
        N=256;y,x=np.mgrid[0:N,0:N];img=.5+.25*np.sin(2*np.pi*x/18)+.25*np.sin(2*np.pi*y/31);img+=.08*np.random.default_rng(3).normal(size=(N,N));img=np.clip(img,0,1);st.info("No image uploaded — using a built-in microscopy-like test pattern.")
    maxdim=st.slider("Maximum image dimension",128,512,min(256,max(img.shape)),64)
    if max(img.shape)>maxdim:
        im=Image.fromarray((img*255).astype(np.uint8)).resize((maxdim,maxdim));img=np.array(im)/255.
    F=np.fft.fftshift(np.fft.fft2(img));Y,X=np.indices(img.shape);cx=(img.shape[0]-1)/2;cy=(img.shape[1]-1)/2;r=np.sqrt((X-cy)**2+(Y-cx)**2);R=max(r.max(),1)
    operation=st.selectbox("Fourier operation",["Denoising","Sharpening","Edge detection","Microscopy / texture","Pattern recognition","Reconstruction"])
    cutoff=st.slider("Normalized radial cutoff",.02,.95,.25,.01)
    low=r<cutoff*R
    if operation=="Denoising":mask=low
    elif operation=="Sharpening":mask=1+2*(~low)
    elif operation=="Edge detection":mask=(~low)*1.
    elif operation=="Microscopy / texture":mask=(r<.7*R)
    elif operation=="Pattern recognition":mask=(r<.45*R)
    else:mask=np.ones_like(r)
    G=F*mask;rec=np.real(np.fft.ifft2(np.fft.ifftshift(G)));rec=(rec-rec.min())/(rec.max()-rec.min()+1e-12)
    c=st.columns(3);c[0].plotly_chart(heat(img,"Original image",height=370),use_container_width=True);c[1].plotly_chart(heat(np.log1p(np.abs(F)),"2D Fourier magnitude",height=370),use_container_width=True);c[2].plotly_chart(heat(rec,operation+" result",height=370),use_container_width=True)
    mse=np.mean((img-rec)**2); psnr=10*np.log10(1/max(mse,1e-15)); energy=100*np.sum(np.abs(G)**2)/(np.sum(np.abs(F)**2)+1e-15)
    c=st.columns(4);c[0].metric("Spectral energy retained",f"{energy:.2f}%");c[1].metric("MSE",f"{mse:.5e}");c[2].metric("PSNR",f"{psnr:.2f} dB");c[3].metric("Cutoff radius",f"{cutoff:.2f} R")
    ri=np.floor(r).astype(int);mr=int(min(R,min(img.shape)/2));rad=np.array([np.mean(np.abs(F)[ri==q]) if np.any(ri==q) else 0 for q in range(mr+1)])
    st.plotly_chart(fig1([(np.arange(mr+1),rad,"radial spectrum",{})],"Radially averaged spatial-frequency spectrum","radial frequency index","mean |F|",340),use_container_width=True)
    math_section("Complete Mathematics",[
        ("2D Fourier transform",r"F(k_x,k_y)=\iint I(x,y)e^{-i(k_xx+k_yy)}\,dx\,dy"),
        ("Inverse transform",r"I(x,y)=\frac{1}{(2\pi)^2}\iint F(k_x,k_y)e^{i(k_xx+k_yy)}\,dk_x\,dk_y"),
        ("Frequency-domain filtering",r"G(k_x,k_y)=H(k_x,k_y)F(k_x,k_y)"),
        ("High-pass edge extraction",r"I_{\rm edge}=\mathcal F^{-1}\{[1-H_{\rm LP}]F\}"),
        ("Convolution theorem",r"\mathcal F\{I*h\}=F(k_x,k_y)H(k_x,k_y)")
    ])

def mri_page():
    page_header("Medical Imaging — MRI","Uploaded image / built-in phantom → k-space → sampling mask → Fourier reconstruction → complete MRI mathematics")
    up=st.file_uploader("Upload a grayscale MRI-like image",type=["png","jpg","jpeg"],key="mri_upload")
    if up: img=np.array(Image.open(up).convert("L").resize((256,256)),dtype=float)/255.
    else:
        y,x=np.mgrid[-1:1:256j,-1:1:256j];img=((x*x+y*y)<.72**2).astype(float);img+=.35*((x+.22)**2+(y-.15)**2<.18**2);img+=.18*((x-.28)**2+(y+.22)**2<.11**2);img=np.clip(img,0,1);st.info("No MRI sample uploaded — using a built-in phantom.")
    F=np.fft.fftshift(np.fft.fft2(img));y,x=np.indices(img.shape);r=np.sqrt((x-128)**2+(y-128)**2)
    sampling=st.selectbox("k-space sampling",["Full","Central k-space","Every 2nd phase-encode line","Central + sparse outer k-space"])
    if sampling=="Full":mask=np.ones_like(img)
    elif sampling=="Central k-space":mask=(r<55)
    elif sampling=="Every 2nd phase-encode line":mask=np.zeros_like(img);mask[::2,:]=1
    else:mask=(r<55);mask[::4,:]=1
    rec=np.abs(np.fft.ifft2(np.fft.ifftshift(F*mask)));frac=np.mean(mask);mse=np.mean((img-rec)**2);corr=np.corrcoef(img.ravel(),rec.ravel())[0,1]
    c=st.columns(3);c[0].plotly_chart(heat(img,"Input anatomy / phantom",height=380),use_container_width=True);c[1].plotly_chart(heat(np.log1p(np.abs(F)),"MRI k-space",height=380),use_container_width=True);c[2].plotly_chart(heat(rec,"Fourier reconstructed image",height=380),use_container_width=True)
    psnr=10*np.log10(1/max(mse,1e-15))
    c=st.columns(4);c[0].metric("Acquisition fraction",f"{100*frac:.2f}%");c[1].metric("MSE",f"{mse:.4e}");c[2].metric("Correlation",f"{corr:.5f}");c[3].metric("PSNR",f"{psnr:.2f} dB")
    center=np.abs(F[F.shape[0]//2,:]); phase=np.abs(F[:,F.shape[1]//2])
    c=st.columns(2)
    c[0].plotly_chart(fig1([(np.arange(len(center)),np.log1p(center),"kx center line",{})],"k-space horizontal profile","kx sample","log(1+|S|)",320),use_container_width=True)
    c[1].plotly_chart(fig1([(np.arange(len(phase)),np.log1p(phase),"ky center line",{})],"k-space vertical profile","ky sample","log(1+|S|)",320),use_container_width=True)
    math_section("Complete MRI Mathematics",[
        ("Spatial encoding / k-space signal",r"S(k_x,k_y)=\iint \rho(x,y)e^{-i2\pi(k_xx+k_yy)}\,dx\,dy"),
        ("Image reconstruction",r"\rho(x,y)=\iint S(k_x,k_y)e^{i2\pi(k_xx+k_yy)}\,dk_x\,dk_y"),
        ("Gradient-defined spatial frequency",r"k(t)=\gamma\int_0^t G(\tau)\,d\tau"),
        ("Resolution",r"\Delta x\approx\frac{1}{2k_{\max}}"),
        ("Partial k-space reconstruction",r"\hat\rho=\mathcal F^{-1}\{M(k_x,k_y)S(k_x,k_y)\}")
    ])

def crystal_page():
    page_header("Reciprocal Lattice, Diffraction & Crystal Structure Analysis","3D reciprocal vectors → reciprocal lattice → Brillouin zone → structure factor → diffraction")
    a=st.slider("Lattice constant a",1.,6.,3.,.1);basis=st.selectbox("Crystal structure",["Simple Cubic","BCC","FCC"])
    hmax=st.slider("Reciprocal-index range",1,5,3)
    pts=[]
    for h in range(-hmax,hmax+1):
        for k in range(-hmax,hmax+1):
            for l in range(-hmax,hmax+1):
                if h*h+k*k+l*l<=hmax*hmax:pts.append((2*np.pi*h/a,2*np.pi*k/a,2*np.pi*l/a))
    pts=np.array(pts);fig=go.Figure(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode="markers",marker=dict(size=4)));fig.update_layout(template="plotly_white",height=500,title="3D reciprocal lattice",scene=dict(xaxis_title="b₁ direction",yaxis_title="b₂ direction",zaxis_title="b₃ direction"))
    st.plotly_chart(fig,use_container_width=True)
    c=st.columns(2)
    c[0].markdown("### Reciprocal-vector geometry")
    c[0].latex(r"\mathbf b_1=2\pi\frac{\mathbf a_2\times\mathbf a_3}{\mathbf a_1\cdot(\mathbf a_2\times\mathbf a_3)},\quad \mathbf b_i\cdot\mathbf a_j=2\pi\delta_{ij}")
    c[0].latex(r"\mathbf G=h\mathbf b_1+k\mathbf b_2+l\mathbf b_3")
    c[1].markdown("### Bragg condition")
    c[1].latex(r"2d_{hkl}\sin\theta=n\lambda")
    c[1].latex(r"d_{hkl}=\frac{a}{\sqrt{h^2+k^2+l^2}}\quad\text{(cubic crystal)}")
    h=np.arange(1,2*hmax+1);d=a/np.sqrt(h*h);lam=1.;theta=np.arcsin(np.minimum(.99,lam/(2*d)));I=np.ones_like(h,dtype=float)
    if basis=="BCC":I=(1+(-1)**h)**2
    if basis=="FCC":I=np.where((h%2==0),4.,0.)
    st.plotly_chart(fig1([(2*np.degrees(theta),I,"Bragg intensity",{"mode":"markers+lines"})],"Simulated powder diffraction","2θ (degrees)","relative intensity",430),use_container_width=True)
    h0=st.number_input("h",0,5,1);k0=st.number_input("k",0,5,1);l0=st.number_input("l",0,5,1)
    pos={"Simple Cubic":[(0,0,0)],"BCC":[(0,0,0),(.5,.5,.5)],"FCC":[(0,0,0),(0,.5,.5),(.5,0,.5),(.5,.5,0)]}[basis]
    F=sum(np.exp(2j*np.pi*(h0*x+k0*y+l0*z)) for x,y,z in pos)
    c=st.columns(2);c[0].metric("|F(hkl)|",f"{abs(F):.6f}");c[1].metric("I ∝ |F|²",f"{abs(F)**2:.6f}")
    st.markdown("### First Brillouin zone")
    q=np.linspace(-np.pi/a,np.pi/a,500);st.plotly_chart(fig1([(q,q*q,"E(k)",{})],"1D first Brillouin zone","k","E(k)",360),use_container_width=True)
    math_section("Complete Mathematics",[
        ("Reciprocal vectors",r"\mathbf a_i\cdot\mathbf b_j=2\pi\delta_{ij}"),
        ("Reciprocal-lattice vector",r"\mathbf G=h\mathbf b_1+k\mathbf b_2+l\mathbf b_3"),
        ("Structure factor",r"F(\mathbf G)=\sum_j f_j e^{i\mathbf G\cdot\mathbf r_j}"),
        ("Diffraction intensity",r"I(\mathbf G)\propto |F(\mathbf G)|^2"),
        ("Bragg law",r"2d_{hkl}\sin\theta=n\lambda"),
        ("Cubic spacing",r"d_{hkl}=\frac{a}{\sqrt{h^2+k^2+l^2}}"),
        ("Brillouin zone",r"\text{First BZ}=\text{Wigner-Seitz cell of the reciprocal lattice}")
    ])

modules=["Our Voice","Noise Cancellation","Mobile Phone & Wi-Fi","Images","Medical Imaging — MRI","Reciprocal Lattice, Diffraction & Crystal Structure Analysis"]
choice=st.sidebar.radio("Fourier Applications",modules)
st.sidebar.divider()
st.sidebar.latex(r"X(k)=\int x(r)e^{-ikr}\,dr")
st.sidebar.caption("Six focused Fourier applications")
if choice=="Our Voice": voice_page()
elif choice=="Noise Cancellation": noise_page()
elif choice=="Mobile Phone & Wi-Fi": comm_page()
elif choice=="Images": image_page()
elif choice=="Medical Imaging — MRI": mri_page()
else: crystal_page()

st.markdown("""
<div class="fixed-footer">
  <span>Developed with Love ❤️ by Aman Kumar Patel</span>
</div>
<style>
.fixed-footer{
position:fixed;
left:0;
bottom:0;
width:100%;
height:42px;
z-index:999999;
display:flex;
align-items:center;
justify-content:center;
gap:18px;
padding:0 18px;
box-sizing:border-box;
background:rgba(20,20,24,.96);
color:#f5f5f5;
border-top:1px solid rgba(255,255,255,.18);
font-size:13px;
letter-spacing:.15px;
box-shadow:0 -4px 18px rgba(0,0,0,.18);
}
.main .block-container{padding-bottom:70px;}
</style>
""",unsafe_allow_html=True)
def verify(name,a,b,tol=1e-5):
    err=np.linalg.norm(a-b)/(np.linalg.norm(a)+1e-15)
    c=st.columns(2);c[0].metric("Relative numerical error",f"{err:.3e}");c[1].metric("Status","PASS" if err<tol else "CHECK")
    st.progress(float(max(0,min(1,1-err/max(tol,1e-15)))),text=name)
    return err

