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
    """Render academic prose and LaTeX using Streamlit's native Markdown math renderer."""
    rendered = content.replace(r"\[", "$$").replace(r"\]", "$$")
    st.markdown(rendered)

def shell(title,theory,derivation,application,warning=None):
    st.markdown(f'<div class="title">{title}</div>',unsafe_allow_html=True)
    st.markdown('<div class="sub">Theory → Derivation → Interactive Experiment → Numerical Verification → Physical Interpretation → Research Application</div>',unsafe_allow_html=True)
    tabs=st.tabs(["01 · Theory","02 · Derivation","03 · Experiment","04 · Numerical Verification","05 · Physical Interpretation","06 · Research Application"])
    with tabs[0]:
        render_academic(theory)
    with tabs[1]:
        render_academic(derivation)
    with tabs[4]:
        st.markdown("### Physical interpretation")
        st.markdown(f"**{title}** is interpreted through spectral content, phase structure, localization and the response of the Fourier basis.")
        n=256
        xx=np.linspace(-5,5,n,endpoint=False); dx=xx[1]-xx[0]
        uu=np.exp(-xx**2/(2*0.65**2))
        UU=FT(uu,dx); kk=K(n,dx)
        c=st.columns(3)
        c[0].plotly_chart(fig1([(xx,uu,"field",{})],"Representative physical-space field","x","amplitude",320),use_container_width=True)
        c[1].plotly_chart(fig1([(kk,np.abs(UU),"spectrum",{})],"Reciprocal-space content","k","|X(k)|",320),use_container_width=True)
        phase=np.unwrap(np.angle(UU))
        c[2].plotly_chart(fig1([(kk,phase,"phase",{})],"Spectral phase","k","phase",320),use_container_width=True)
        st.info("Interpretation: concentrated real-space structure produces broad reciprocal-space content; phase carries spatial alignment information.")
    with tabs[5]:
        render_academic(application)
        st.markdown("### Visual research application")
        app_mode=st.selectbox("Choose a physical application",["Spectroscopy","Fraunhofer diffraction","Optical imaging / MTF","Quantum position ↔ momentum"],key=f"app_mode_{title}")
        n=320
        xx=np.linspace(-6,6,n,endpoint=False); dx=xx[1]-xx[0]
        if app_mode=="Spectroscopy":
            sig=np.exp(-xx**2/(2*.45**2))+0.65*np.exp(-(xx-1.6)**2/(2*.22**2))
            spec=np.abs(FT(sig,dx)); kk=K(n,dx)
            f=make_subplots(rows=1,cols=2,subplot_titles=("Measured / model field","Spectral signature"))
            f.add_trace(go.Scatter(x=xx,y=sig,name="signal"),row=1,col=1)
            f.add_trace(go.Scatter(x=kk,y=spec,name="spectrum"),row=1,col=2)
            f.update_layout(template="plotly_white",height=430,xaxis_title="coordinate",xaxis2_title="wave number",yaxis_title="amplitude")
            st.plotly_chart(f,use_container_width=True)
            st.metric("Dominant spectral component",f"{kk[np.argmax(spec)]:.3f}")
        elif app_mode=="Fraunhofer diffraction":
            X,Y=np.meshgrid(xx,xx); aperture=(X**2+Y**2<1.0**2).astype(float)
            F=np.fft.fftshift(np.fft.fft2(aperture)); I=np.abs(F)**2; I/=I.max()
            c=st.columns(2)
            c[0].plotly_chart(heat(aperture,"Circular aperture",xx,xx,"Gray",390),use_container_width=True)
            c[1].plotly_chart(heat(np.log1p(80*I),"Far-field diffraction intensity",xx,xx,"Magma",390),use_container_width=True)
            st.metric("Central intensity",f"{I[n//2,n//2]:.6f}")
        elif app_mode=="Optical imaging / MTF":
            r=np.linspace(0,1,300)
            mtf=np.exp(-3.2*r**2)
            c=st.columns(2)
            c[0].plotly_chart(fig1([(r,mtf,"MTF",{})],"Contrast transfer","normalized spatial frequency","MTF",390),use_container_width=True)
            c[1].plotly_chart(fig1([(r,20*np.log10(np.maximum(mtf,1e-8)),"MTF (dB)",{})],"Resolution bandwidth","normalized spatial frequency","dB",390),use_container_width=True)
            cutoff=r[np.argmin(np.abs(mtf-.1))]
            st.metric("10% MTF frequency",f"{cutoff:.3f} × normalized cutoff")
        else:
            sigma=0.8
            psi=np.exp(-xx**2/(4*sigma**2))
            psi/=np.sqrt(np.trapz(abs(psi)**2,xx))
            P=np.abs(FT(psi,dx))**2
            kk=K(n,dx)
            c=st.columns(2)
            c[0].plotly_chart(fig1([(xx,np.abs(psi)**2,"|ψ(x)|²",{})],"Position probability density","x","probability density",390),use_container_width=True)
            c[1].plotly_chart(fig1([(kk,P/P.max(),"|ψ̃(k)|²",{})],"Momentum-space spectrum","k","normalized probability",390),use_container_width=True)
            st.metric("Position-space width σ",f"{sigma:.3f}")
    with tabs[3]:
        st.markdown("### Numerical verification laboratory")
        n0=st.select_slider("Base sample count",[64,128,256,512,1024],256,key=f"nv_n_{title}")
        sigma=st.slider("Test-field width",0.25,1.5,0.65,0.05,key=f"nv_sigma_{title}")
        orders=[64,128,256,512,1024]
        errors=[];energies=[];bandwidths=[]
        for nn in orders:
            q=np.linspace(-5,5,nn,endpoint=False); dd=q[1]-q[0]
            u=np.exp(-q*q/(2*sigma*sigma)); U=FT(u,dd); rec=np.real(IFT(U,dd))
            err=np.linalg.norm(u-rec)/(np.linalg.norm(u)+1e-15)
            energy_x=np.sum(np.abs(u)**2)*dd
            energy_k=np.sum(np.abs(U)**2)*(K(nn,dd)[1]-K(nn,dd)[0])/(2*np.pi)
            p=np.abs(U); threshold=.01*p.max()
            active=np.abs(K(nn,dd))[p>threshold]
            bw=(active.max()-active.min()) if active.size else 0
            errors.append(err);energies.append(energy_k/energy_x);bandwidths.append(bw)
        c=st.columns(3)
        c[0].metric("Reconstruction error",f"{errors[orders.index(n0)]:.3e}")
        c[1].metric("Parseval energy ratio",f"{energies[orders.index(n0)]:.8f}")
        c[2].metric("1% spectral bandwidth",f"{bandwidths[orders.index(n0)]:.4f}")
        f=make_subplots(rows=1,cols=3,subplot_titles=("Convergence","Energy consistency","Spectral bandwidth"))
        f.add_trace(go.Scatter(x=orders,y=errors,mode="lines+markers",name="error"),row=1,col=1)
        f.add_trace(go.Scatter(x=orders,y=energies,mode="lines+markers",name="energy ratio"),row=1,col=2)
        f.add_trace(go.Scatter(x=orders,y=bandwidths,mode="lines+markers",name="bandwidth"),row=1,col=3)
        f.update_xaxes(type="log",row=1,col=1); f.update_yaxes(type="log",row=1,col=1)
        f.update_layout(template="plotly_white",height=470,showlegend=False)
        st.plotly_chart(f,use_container_width=True)
        st.markdown("#### Independent reconstruction at the selected resolution")
        q=np.linspace(-5,5,n0,endpoint=False); dd=q[1]-q[0]; u=np.exp(-q*q/(2*sigma*sigma)); U=FT(u,dd); rec=np.real(IFT(U,dd))
        st.plotly_chart(fig1([(q,u,"original",{}),(q,rec,"inverse transform",{"line":dict(dash="dash")})],"Forward → inverse reconstruction","x","field"),use_container_width=True)
        st.caption("A trustworthy numerical transform should converge with increasing resolution, conserve the appropriate quadratic norm, and reproduce the original field under inverse transformation.")
    return tabs

def verify(name,a,b,tol=1e-5):
    err=np.linalg.norm(a-b)/(np.linalg.norm(a)+1e-15)
    c=st.columns(2);c[0].metric("Relative numerical error",f"{err:.3e}");c[1].metric("Status","PASS" if err<tol else "CHECK")
    st.progress(float(max(0,min(1,1-err/max(tol,1e-15)))),text=name)
    return err

modules=[
"Our Voice","Noise Cancellation","Mobile Phone & Wi-Fi","Images",
"Medical Imaging — MRI","Reciprocal Lattice, Diffraction & Crystal Structure Analysis"
]
choice=st.sidebar.radio("Fourier Applications",modules)
st.sidebar.divider()
st.sidebar.latex(r"X(k)=\int x(r)e^{-ikr}\,dr")
st.sidebar.caption("Fourier Physics • Real-world applications • Computational laboratory")
st.sidebar.divider()
st.sidebar.markdown("**Applications**")
st.sidebar.caption("Voice → Noise → Communication → Images → MRI → Crystals")

if choice=="Our Voice":
    tabs=shell("Fourier Transform of Our Voice",
    """### Why does your voice have a spectrum?
    Human speech is a time-varying acoustic pressure wave. The Fourier transform decomposes it into frequency components, revealing the fundamental frequency, harmonics and formants.
    \[
    X(f)=\int x(t)e^{-i2\pi ft}\,dt
    \]
    For speech, the spectrum changes with time, so a short-time Fourier representation is especially useful.""",
    """### From vocal vibration to spectrum
    Vocal-fold vibration produces a quasi-periodic excitation. The vocal tract acts approximately as a frequency-selective filter:
    \[
    \text{speech}(t)=\text{excitation}(t)*h_{\rm vocal\ tract}(t).
    \]
    Therefore,
    \[
    X(f)=E(f)H(f).
    \]
    Fourier analysis separates excitation frequency from resonant formant structure.""",
    """Voice Fourier analysis is used in speech recognition, speaker analysis, acoustic measurement, hearing science and communication systems.""")
    with tabs[2]:
        f0=st.slider("Fundamental frequency (Hz)",80,300,140,1)
        vowel=st.selectbox("Vowel-like timbre",["A","E","I","O","U"])
        duration=st.slider("Signal duration (s)",1.0,4.0,2.0,.1)
        fs=8000;t=np.arange(0,duration,1/fs)
        formants={"A":[800,1200,2500],"E":[500,1900,2500],"I":[300,2200,3000],"O":[500,900,2500],"U":[350,700,2500]}[vowel]
        env=np.minimum(1,np.minimum(t*25,(duration-t)*25))
        voice=sum((1/(h))*np.sin(2*np.pi*f0*h*t) for h in range(1,25))
        filt=sum(np.exp(-((h*f0-fm)/90)**2)*np.sin(2*np.pi*h*f0*t) for h in range(1,25) for fm in formants)
        x=env*(.65*voice+.35*filt);x/=np.max(np.abs(x))+1e-12
        X=FT(x,1/fs);freq=K(len(x),1/fs)
        pos=freq>=0
        c=st.columns(2)
        c[0].plotly_chart(fig1([(t[:min(len(t),1600)],x[:min(len(t),1600)],"voice",{})],"Acoustic waveform","time (s)","pressure",420),use_container_width=True)
        c[1].plotly_chart(fig1([(freq[pos],np.abs(X[pos]),"spectrum",{})],"Voice spectrum","frequency (Hz)","magnitude",420),use_container_width=True)
        audio_buf=io.BytesIO()
        with wave.open(audio_buf,"wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(fs)
            wav.writeframes((x*32767).astype(np.int16).tobytes())
        st.audio(audio_buf.getvalue(),format="audio/wav")
        st.metric("Fundamental",f"{f0} Hz")
        st.metric("Highest modeled formant",f"{max(formants)} Hz")
    with tabs[3]:
        st.markdown("### Spectral verification")
        X=FT(x,1/fs);xr=np.real(IFT(X,1/fs))
        verify("Voice reconstruction",xr,x,1e-8)
        err=np.abs(X-np.fft.fftshift(np.fft.fft(x))/fs)
        st.metric("Independent spectrum agreement",f"{np.max(err):.3e}")
        st.plotly_chart(fig1([(t[:1600],x[:1600],"original",{}),(t[:1600],xr[:1600],"reconstructed",{"line":dict(dash="dash")})],"Forward → inverse voice reconstruction","time (s)","amplitude"),use_container_width=True)
    with tabs[4]:
        st.markdown("### What the spectrum tells us")
        st.info("The harmonic spacing reveals the fundamental pitch, while groups of enhanced harmonics reveal vocal-tract resonances (formants). A voice is therefore a physical example of a time-domain waveform carrying structured frequency information.")
    with tabs[5]:
        st.markdown("### Research application: speech spectroscopy")
        st.plotly_chart(fig1([(freq[pos],20*np.log10(np.maximum(np.abs(X[pos])/np.max(np.abs(X[pos])),1e-5)),"voice spectrum",{})],"Voice spectral envelope","frequency (Hz)","relative level (dB)",430),use_container_width=True)
        st.markdown("The same analysis pipeline underlies acoustic characterization and speech-recognition front ends.")

elif choice=="Noise Cancellation":
    tabs=shell("Fourier Transform for Noise Cancellation",
    """### Noise cancellation is spectral separation
    A recorded signal can be represented as
    \[
    y(t)=s(t)+n(t).
    \]
    Fourier transformation converts addition in time into addition in frequency:
    \[
    Y(f)=S(f)+N(f).
    \]
    Filtering can suppress frequency regions dominated by noise.""",
    """### Spectral filtering
    A linear frequency-domain filter obeys
    \[
    \tilde S(f)=H(f)Y(f).
    \]
    An idealized low-pass, band-pass or notch response selectively attenuates unwanted spectral components. Real systems must balance noise suppression against distortion.""",
    """Fourier-domain noise reduction is used in microphones, headphones, industrial sensing, seismic measurements, astronomy and biomedical instrumentation.""")
    with tabs[2]:
        fs=4000;duration=2.;t=np.arange(0,duration,1/fs)
        signal_freq=st.slider("Signal frequency (Hz)",100,800,350,10)
        noise_strength=st.slider("Noise strength",0.,1.5,.65,.05)
        noise_freq=st.slider("Interference frequency (Hz)",50,1800,1100,10)
        s0=np.sin(2*np.pi*signal_freq*t)+.45*np.sin(2*np.pi*2*signal_freq*t)
        n0=noise_strength*np.sin(2*np.pi*noise_freq*t)+.25*noise_strength*np.random.default_rng(4).normal(size=len(t))
        y=s0+n0;Y=FT(y,1/fs);f=K(len(y),1/fs);cut=st.slider("Low-pass cutoff (Hz)",100,1800,700,10);H=(np.abs(f)<cut).astype(float);clean=np.real(IFT(Y*H,1/fs))
        c=st.columns(2)
        c[0].plotly_chart(fig1([(t[:2000],y[:2000],"noisy",{}),(t[:2000],clean[:2000],"filtered",{})],"Noisy → filtered waveform","time (s)","amplitude",420),use_container_width=True)
        pos=f>=0;c[1].plotly_chart(fig1([(f[pos],np.abs(Y[pos]),"noisy spectrum",{}),(f[pos],np.abs(Y[pos]*H[pos]),"filtered spectrum",{})],"Frequency-domain filtering","Hz","magnitude",420),use_container_width=True)
        st.metric("Cutoff",f"{cut} Hz");st.metric("Noise RMS before",f"{np.std(y-s0):.4f}");st.metric("Residual RMS after",f"{np.std(clean-s0):.4f}")
    with tabs[3]:
        snr_before=10*np.log10(np.mean(s0**2)/np.mean((y-s0)**2))
        snr_after=10*np.log10(np.mean(s0**2)/np.mean((clean-s0)**2))
        c=st.columns(3);c[0].metric("SNR before",f"{snr_before:.2f} dB");c[1].metric("SNR after",f"{snr_after:.2f} dB");c[2].metric("Improvement",f"{snr_after-snr_before:.2f} dB")
        verify("Filter reconstruction consistency",np.real(IFT(FT(clean,1/fs),1/fs)),clean,1e-8)
        st.plotly_chart(fig1([(t[:2000],s0[:2000],"clean reference",{}),(t[:2000],clean[:2000],"estimated",{"line":dict(dash="dash")})],"Cancellation quality","time (s)","amplitude"),use_container_width=True)
    with tabs[4]:
        st.write("The filter succeeds when the useful signal and unwanted interference occupy sufficiently different spectral regions. If they overlap, aggressive filtering also removes information.")
    with tabs[5]:
        st.markdown("### Research application: adaptive spectral suppression")
        st.plotly_chart(fig1([(f[pos],20*np.log10(np.maximum(np.abs(Y[pos])/np.max(np.abs(Y[pos])),1e-6)),"input",{}),(f[pos],20*np.log10(np.maximum(np.abs(Y[pos]*H[pos])/np.max(np.abs(Y[pos])),1e-6)),"filtered",{})],"Noise suppression in spectral dB","frequency (Hz)","relative dB",430),use_container_width=True)

elif choice=="Mobile Phone & Wi-Fi":
    tabs=shell("Fourier Transform in Mobile Communication & Wi-Fi",
    """### Communication is frequency engineering
    Information is carried by electromagnetic fields whose spectra occupy finite bandwidths. Fourier analysis reveals carriers, sidebands, channel spacing and occupied bandwidth.
    \[
    X(f)=\mathcal F\{x(t)\}.
    \]""",
    """### Modulation and spectral occupancy
    For amplitude modulation,
    \[
    x(t)=m(t)\cos(2\pi f_ct)
    \]
    produces shifted copies of the message spectrum around \(\\pm f_c\).
    Modern OFDM systems instead place information on many closely spaced orthogonal subcarriers.""",
    """Fourier analysis is fundamental to RF spectrum monitoring, cellular communication, Wi-Fi channel planning, OFDM, filtering and interference analysis.""")
    with tabs[2]:
        fc=st.slider("Carrier frequency (normalized MHz)",1.,100.,20.,1.)
        bw=st.slider("Message bandwidth (MHz)",1.,15.,5.,.5)
        ncar=st.slider("OFDM subcarriers",4,64,16,4)
        df=bw/ncar
        freqs=fc+(np.arange(ncar)-(ncar-1)/2)*df
        amps=np.ones(ncar);amps[::5]*=.55
        f=np.linspace(fc-bw*1.4,fc+bw*1.4,2400)
        spec=np.zeros_like(f)
        for q,a0 in zip(freqs,amps):spec+=a0*np.exp(-((f-q)/(df*.18))**2)
        c=st.columns(2)
        c[0].plotly_chart(fig1([(f,spec,"OFDM spectrum",{})],"Subcarrier spectrum","frequency (MHz)","relative amplitude",420),use_container_width=True)
        c[1].plotly_chart(fig1([(freqs,amps,"subcarriers",{"mode":"markers+lines"})],"Orthogonal subcarrier grid","frequency (MHz)","relative power",420),use_container_width=True)
        st.metric("Channel bandwidth",f"{bw:.2f} MHz");st.metric("Subcarrier spacing",f"{df:.3f} MHz")
    with tabs[3]:
        occupied=freqs.max()-freqs.min()+df
        orthogonality=np.mean(np.cos(2*np.pi*np.arange(ncar)[:,None]*np.arange(ncar)[None,:]/ncar),axis=1)
        st.metric("Occupied bandwidth",f"{occupied:.3f} MHz");st.metric("Orthogonality residual",f"{np.max(np.abs(orthogonality[1:])):.3e}")
        st.plotly_chart(fig1([(freqs,amps,"active carriers",{"mode":"markers+lines"})],"Numerical channel occupancy","frequency (MHz)","power",420),use_container_width=True)
    with tabs[4]:
        st.markdown("### Why Fourier analysis matters")
        st.write("A receiver can separate channels because different information streams occupy distinguishable spectral regions. Fourier analysis therefore turns an electromagnetic waveform into a map of where information and interference live in frequency.")
    with tabs[5]:
        st.markdown("### Research application: spectrum and channel planning")
        ch=st.slider("Channel index",1,11,6)
        centers=2.4+(np.arange(11)-(ch-1))*0.02
        st.plotly_chart(fig1([(centers,np.ones(11),"channel centers",{"mode":"markers"})],"Example channel plan","frequency (GHz)","relative channel power",430),use_container_width=True)
        st.info("In practical RF engineering, Fourier-domain measurements are used to identify occupied bandwidth, adjacent-channel interference and spectral leakage.")

elif choice=="Images":
    tabs=shell("Fourier Transform of Images",
    """### Images contain spatial frequencies
    A 2D image \(I(x,y)\) contains slowly varying structures, fine texture and edges. Its Fourier transform is
    \[
    F(k_x,k_y)=\iint I(x,y)e^{-i(k_xx+k_yy)}dxdy.
    \]
    Low spatial frequencies describe broad structure; high spatial frequencies describe fine detail and sharp transitions.""",
    """### Filtering in the Fourier plane
    A frequency-domain filter satisfies
    \[
    G(k_x,k_y)=H(k_x,k_y)F(k_x,k_y),
    \]
    followed by
    \[
    g(x,y)=\mathcal F^{-1}\{G\}.
    \]
    Therefore sharpening, denoising and edge enhancement can be designed as spatial-frequency operations.""",
    """2D Fourier methods are used in image sharpening, denoising, edge detection, microscopy, pattern recognition, diffraction imaging and image reconstruction.""")
    with tabs[2]:
        pattern=st.selectbox("Image / object",["Cells-like microscopy texture","Resolution chart","Crystal-like lattice pattern"])
        N=256;y,x=np.mgrid[-1:1:complex(N),-1:1:complex(N)]
        if pattern=="Cells-like microscopy texture":
            rng=np.random.default_rng(3);img=np.zeros((N,N))
            for _ in range(18):
                cx,cy=rng.uniform(-.8,.8,2);r=.035+rng.uniform(0,.06);img+=np.exp(-((x-cx)**2+(y-cy)**2)/(2*r*r))
            img+=.08*rng.normal(size=(N,N))
        elif pattern=="Resolution chart":
            img=(np.sin(2*np.pi*(8*x+20*x*x))+np.sin(2*np.pi*12*y)>0).astype(float)
        else:
            img=(np.cos(2*np.pi*10*x)+np.cos(2*np.pi*10*y)>1).astype(float)
        F=np.fft.fftshift(np.fft.fft2(img));mag=np.log1p(np.abs(F));radius=np.sqrt(x*x+y*y)
        cutoff=st.slider("Low-pass cutoff",.02,.8,.25,.01)
        H=(radius<cutoff).astype(float);rec=np.real(np.fft.ifft2(np.fft.ifftshift(F*H)))
        c=st.columns(3);c[0].plotly_chart(heat(img,"Original image",height=360),use_container_width=True);c[1].plotly_chart(heat(mag,"2D Fourier magnitude",height=360),use_container_width=True);c[2].plotly_chart(heat(rec,"Filtered reconstruction",height=360),use_container_width=True)
        st.metric("Retained Fourier area",f"{100*np.mean(H):.2f}%")
    with tabs[3]:
        mse=np.mean((img-rec)**2);snr=10*np.log10(np.mean(img**2)/(mse+1e-15))
        c=st.columns(3);c[0].metric("MSE",f"{mse:.5e}");c[1].metric("Reconstruction SNR",f"{snr:.2f} dB");c[2].metric("Fourier energy retained",f"{100*np.sum(np.abs(F*H)**2)/np.sum(np.abs(F)**2):.2f}%")
        st.plotly_chart(heat(np.abs(F*H),"Retained spatial frequencies",height=430),use_container_width=True)
    with tabs[4]:
        st.write("Edges and fine structures occupy high spatial frequencies. Low-pass filtering suppresses fine detail and noise; high-pass or band-pass filtering can emphasize edges and texture.")
    with tabs[5]:
        st.markdown("### Research application: microscopy and pattern recognition")
        hp=F*(1-H);edges=np.real(np.fft.ifft2(np.fft.ifftshift(hp)))
        c=st.columns(2);c[0].plotly_chart(heat(edges,"High-frequency / edge component",height=430),use_container_width=True);c[1].plotly_chart(heat(np.abs(F),"Microscopy spatial-frequency map",height=430),use_container_width=True)

elif choice=="Medical Imaging — MRI":
    tabs=shell("Fourier Transform in MRI",
    """### MRI measures Fourier-space information
    MRI does not directly measure an image at each pixel. Gradient fields encode spatial position into frequency, and the scanner samples **k-space**.
    \[
    S(k_x,k_y)=\iint \rho(x,y)e^{-i2\pi(k_xx+k_yy)}dxdy.
    \]
    The image is recovered by an inverse Fourier transform.""",
    """### k-space to image
    The measured signal is a Fourier-space representation of spin density:
    \[
    \rho(x,y)=\mathcal F^{-1}\{S(k_x,k_y)\}.
    \]
    The centre of k-space strongly influences contrast and overall structure, while outer k-space contains high spatial-frequency detail and edge information.""",
    """Fourier reconstruction is central to MRI image formation, accelerated acquisition, partial Fourier methods, compressed sensing and k-space trajectory analysis.""")
    with tabs[2]:
        N=256;y,x=np.mgrid[-1:1:complex(N),-1:1:complex(N)]
        phantom=((x/.72)**2+(y/.72)**2<1).astype(float)
        phantom+=.35*((x+.22)**2+(y-.15)**2<.18**2)
        phantom+=.18*((x-.28)**2+(y+.22)**2<.11**2)
        F=np.fft.fftshift(np.fft.fft2(phantom));kx=np.fft.fftshift(np.fft.fftfreq(N));ky=kx
        mask_mode=st.selectbox("k-space acquisition",["Full","Central 40% only","Every 2nd phase-encode line","Radial-like spokes"])
        if mask_mode=="Full":mask=np.ones_like(F,dtype=bool)
        elif mask_mode=="Central 40% only":mask=(x*x+y*y)<.4**2
        elif mask_mode=="Every 2nd phase-encode line":mask=np.zeros_like(F,dtype=bool);mask[::2,:]=True
        else:
            ang=np.linspace(0,np.pi,24,endpoint=False);mask=np.zeros_like(F,dtype=bool)
            for a0 in ang:
                d=np.abs(x*np.sin(a0)-y*np.cos(a0));mask|=d<.012
        recon=np.abs(np.fft.ifft2(np.fft.ifftshift(F*mask)))
        c=st.columns(3);c[0].plotly_chart(heat(phantom,"Ground-truth phantom",height=360),use_container_width=True);c[1].plotly_chart(heat(np.log1p(np.abs(F)),"MRI k-space",height=360),use_container_width=True);c[2].plotly_chart(heat(recon,"Reconstructed MRI",height=360),use_container_width=True)
        st.metric("Acquisition fraction",f"{100*np.mean(mask):.2f}%")
    with tabs[3]:
        mse=np.mean((phantom-recon)**2);corr=np.corrcoef(phantom.ravel(),recon.ravel())[0,1]
        st.metric("Reconstruction MSE",f"{mse:.5e}");st.metric("Image correlation",f"{corr:.5f}");st.metric("Sampling reduction",f"{100*(1-np.mean(mask)):.1f}%")
        st.plotly_chart(heat(np.abs(F)*mask,"Acquired k-space",height=430),use_container_width=True)
    with tabs[4]:
        st.info("MRI makes Fourier analysis physically tangible: changing which part of k-space is measured changes contrast, resolution and artefacts in the reconstructed image.")
    with tabs[5]:
        st.markdown("### Research application: accelerated MRI")
        frac=np.linspace(.1,1.,10);quality=[]
        for q in frac:
            mm=(x*x+y*y)<(0.2+0.8*q)**2
            rr=np.abs(np.fft.ifft2(np.fft.ifftshift(F*mm)))
            quality.append(np.corrcoef(phantom.ravel(),rr.ravel())[0,1])
        st.plotly_chart(fig1([(frac,quality,"correlation",{"mode":"lines+markers"})],"Image quality vs k-space coverage","fraction of k-space","correlation",430),use_container_width=True)
        st.write("This is the computational logic behind studying accelerated acquisition: fewer measurements can reduce scan burden, but the reconstruction quality depends strongly on which Fourier-space information is retained.")

elif choice=="Reciprocal Lattice, Diffraction & Crystal Structure Analysis":
    tabs=shell("Reciprocal Lattice, Diffraction & Crystal Structure Analysis",
    """### Crystals are naturally described in reciprocal space
    Real-space lattice vectors \(\mathbf a_1,\mathbf a_2,\mathbf a_3\) generate reciprocal vectors
    \[
    \mathbf b_i\cdot\mathbf a_j=2\pi\delta_{ij}.
    \]
    Periodicity in real space produces discrete reciprocal-lattice points. Diffraction occurs when scattering vectors connect reciprocal-lattice points.""",
    """### Structure factor and diffraction
    For atoms at positions \(\mathbf r_j\),
    \[
    F(\mathbf G)=\sum_j f_j e^{i\mathbf G\cdot\mathbf r_j}.
    \]
    The measured intensity is approximately
    \[
    I(\mathbf G)\propto |F(\mathbf G)|^2.
    \]
    Thus the Fourier transform connects atomic arrangement to diffraction intensity and crystal-structure information.""",
    """Reciprocal-lattice analysis underpins X-ray, electron and neutron diffraction, Brillouin zones, band-structure calculations, crystal identification and structure refinement.""")
    with tabs[2]:
        a=st.slider("Lattice constant a",1.,5.,2.,.1)
        basis=st.selectbox("Basis",["Simple cubic","BCC","FCC","Two-atom basis"])
        hmax=st.slider("Reciprocal index range",2,6,3)
        pts=[]
        for h in range(-hmax,hmax+1):
            for k0 in range(-hmax,hmax+1):
                for l in range(-hmax,hmax+1):
                    if h*h+k0*k0+l*l<=hmax*hmax:pts.append((2*np.pi*h/a,2*np.pi*k0/a,2*np.pi*l/a))
        pts=np.array(pts);c=st.columns(2)
        c[0].plotly_chart(go.Figure(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode="markers",marker=dict(size=4))).update_layout(template="plotly_white",height=480,title="Reciprocal lattice"),use_container_width=True)
        hkl=np.arange(1,2*hmax+1);I=np.ones_like(hkl,dtype=float)
        if basis=="BCC": I=(1+(-1)**hkl)**2
        elif basis=="FCC": I=((1+(-1)**hkl)**2)*(1+(-1)**hkl)**2/4
        elif basis=="Two-atom basis": I=(1+np.cos(np.pi*hkl))**2
        d=a/np.sqrt(hkl*hkl);theta=np.arcsin(np.minimum(0.95,1/(2*d)));twotheta=2*np.degrees(theta)
        c[1].plotly_chart(fig1([(twotheta,I,"Bragg peaks",{"mode":"markers+lines"})],"Simulated powder diffraction","2θ (degrees)","relative intensity",480),use_container_width=True)
        st.metric("Number of reciprocal points",len(pts))
    with tabs[3]:
        st.markdown("### Structure-factor verification")
        h,k0,l=st.number_input("h",0,6,1),st.number_input("k",0,6,1),st.number_input("l",0,6,1)
        basis_pos={"Simple cubic":[(0,0,0)],"BCC":[(0,0,0),(.5,.5,.5)],"FCC":[(0,0,0),(0,.5,.5),(.5,0,.5),(.5,.5,0)],"Two-atom basis":[(0,0,0),(.5,.5,.5)]}[basis]
        amp=sum(np.exp(2j*np.pi*(h*rx+k0*ry+l*rz)) for rx,ry,rz in basis_pos)
        st.metric("Structure-factor amplitude |F(hkl)|",f"{abs(amp):.6f}");st.metric("Intensity |F|²",f"{abs(amp)**2:.6f}")
        st.write("Extinction occurs when symmetry-related contributions cancel exactly in reciprocal space.")
    with tabs[4]:
        st.markdown("### Brillouin-zone interpretation")
        q=np.linspace(-np.pi/a,np.pi/a,600)
        E=q*q
        st.plotly_chart(fig1([(q,E,"free-electron parabola",{})],"First Brillouin zone","k","E(k)",430),use_container_width=True)
        st.info("The first Brillouin zone is the Wigner–Seitz cell of the reciprocal lattice. Crystal periodicity makes reciprocal space the natural language of diffraction and band physics.")
    with tabs[5]:
        st.markdown("### Research application: diffraction → structure")
        st.plotly_chart(fig1([(twotheta,I,"calculated diffraction",{"mode":"markers+lines"})],"Structure fingerprint","2θ","relative intensity",430),use_container_width=True)
        st.markdown("Changing lattice constant shifts the diffraction pattern; changing the basis changes systematic intensities and extinctions. This is the computational core of crystal-structure analysis.")

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

