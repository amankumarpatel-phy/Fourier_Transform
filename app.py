import streamlit as st
import numpy as np
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
.stTabs [data-baseweb="tab"]{font-weight:600}
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
"Transform Mechanism","Transform Pairs","Magnitude & Phase","Transform Theorems",
"Parseval & Plancherel","Convolution & Green Functions","Uncertainty & Wave Packets",
"2D Fourier Physics","Fourier Imaging","Fourier Optics","Reciprocal Space & Diffraction",
"Quantum & Spectral PDEs","Discrete Fourier Transform","Short-Time Fourier Transform"
]
choice=st.sidebar.radio("Research Workstation",modules)
st.sidebar.divider()
st.sidebar.latex(r"X(k)=\int x(r)e^{-ikr}\,dr")
st.sidebar.caption("Research Mode • Continuous Fourier viewpoint • research-focused transform physics")
st.sidebar.divider()
st.sidebar.markdown("**Workflow**")
st.sidebar.caption("Theory → Derivation → Experiment → Verification → Interpretation → Application")

if choice=="Transform Mechanism":
    tabs=shell("Fourier Transform Mechanism",
    """### The transform is a projection
    A Fourier coefficient is the inner product of a field with a complex plane-wave basis.
    \[
    X(k)=\langle x,e^{ikr}\rangle=\int x(r)e^{-ikr}dr
    \]
    The key physical question is: **how much of wave-number (k) exists in the field?**""",
    """### Derivation from the complex basis
    \[
    e^{-ikr}=\cos(kr)-i\sin(kr)
    \]
    Hence
    \[
    X(k)=\int x(r)\cos(kr)dr-i\int x(r)\sin(kr)dr.
    \]
    The real and imaginary parts are orthogonal projections. Coherent addition occurs when the probe oscillation matches the field.""",
    """### Research application
    This projection viewpoint is the common mathematical mechanism behind spectroscopy, diffraction, reciprocal lattices, quantum momentum, optical imaging and spectral PDE methods.""")
    with tabs[2]:
        f0=st.slider("Probe wave number",0.,50.,17.,.25);N=1600;t=np.linspace(-.5,.5,N,endpoint=False);dx=t[1]-t[0]
        x=np.cos(2*np.pi*17*t)+.55*np.cos(2*np.pi*31*t+.4);b=np.exp(-1j*2*np.pi*f0*t);z=x*b;c=np.cumsum(z)*dx
        f=make_subplots(rows=2,cols=2,subplot_titles=("Field + probe","Real integrand","Accumulation","Complex plane"))
        f.add_trace(go.Scatter(x=t,y=x,name="x"),row=1,col=1);f.add_trace(go.Scatter(x=t,y=np.real(b),name="Re probe"),row=1,col=1);f.add_trace(go.Scatter(x=t,y=np.real(z),name="Re[x probe]"),row=1,col=2);f.add_trace(go.Scatter(x=t,y=np.real(c),name="Re X partial"),row=2,col=1);f.add_trace(go.Scatter(x=t,y=np.imag(c),name="Im X partial"),row=2,col=1);f.add_trace(go.Scatter(x=np.real(c),y=np.imag(c),name="trajectory"),row=2,col=2);f.update_layout(template="plotly_white",height=760);st.plotly_chart(f,use_container_width=True);st.metric("Coefficient",f"{abs(c[-1]):.6f} ∠ {np.angle(c[-1]):.3f} rad")
    with tabs[3]:
        dense=np.linspace(0,50,1001);s=np.array([abs(np.sum(x*np.exp(-1j*2*np.pi*q*t))*dx) for q in dense]);k0=dense[np.argmax(s)];st.plotly_chart(fig1([(dense,s,"|X(k)|",{})],"Continuous-frequency scan","k","magnitude"),use_container_width=True);st.metric("Detected dominant k",f"{k0:.3f}")
    with tabs[4]: st.write("A spectral peak is constructive interference in the complex projection integral. Away from the matching wave number, the rotating phasors cancel.")

elif choice=="Transform Pairs":
    tabs=shell("Continuous Fourier Transform Pairs",
    """### Duality between localization and spectral structure
    Gaussian, rectangular, exponential, sinc and sinusoidal functions form a useful physical dictionary. Sharp boundaries generate long spectral tails; smooth localized fields generate smoother spectra.""",
    """### Representative pair
    For a Gaussian,
    \[
    x(r)=e^{-r^2/(2\sigma^2)}
    \quad\Longrightarrow\quad
    X(k)\propto e^{-\sigma^2k^2/2}.
    \]
    The width product is reciprocal: narrowing the field broadens its spectrum.""",
    """### Research application
    Transform pairs are used as analytical test cases for optics, spectroscopy, signal reconstruction, diffraction and numerical Fourier algorithms.""")
    with tabs[2]:
        pair=st.selectbox("Field",["Gaussian","Rectangular aperture","Exponential","Sinc","Two separated Gaussians"])
        x=np.linspace(-12,12,8192,endpoint=False);dx=x[1]-x[0];k=K(len(x),dx)
        if pair=="Gaussian": s=st.slider("σ",.1,2.,.55,.02);u=gaussian(x,s)
        elif pair=="Rectangular aperture": w=st.slider("width",.2,8.,2.,.1);u=(abs(x)<w/2).astype(float)
        elif pair=="Exponential": a=st.slider("α",.1,3.,1.,.05);u=np.exp(-a*abs(x))
        elif pair=="Sinc": a=st.slider("scale",.2,3.,1.,.05);u=np.sinc(a*x)
        else:sep=st.slider("separation",.2,5.,2.,.1);u=gaussian(x-sep/2,.3)+gaussian(x+sep/2,.3)
        U=FT(u,dx);c=st.columns(2);c[0].plotly_chart(fig1([(x,u,"x(r)",{})],"Real space","r","field"),use_container_width=True);c[1].plotly_chart(fig1([(k,abs(U),"|X(k)|",{})],"Fourier space","k","magnitude"),use_container_width=True)
    with tabs[3]:
        rec=np.real(IFT(U,dx));verify("Inverse-transform reconstruction",u,rec,2e-4);st.plotly_chart(fig1([(x,u,"original",{}),(x,rec,"reconstructed",{"line":dict(dash="dash")})],"Reconstruction verification","r","field"),use_container_width=True)
    with tabs[4]:st.write("The transform pair is a physical dictionary: spatial localization, periodicity and discontinuity have characteristic reciprocal-space signatures.")

elif choice=="Magnitude & Phase":
    tabs=shell("Complex Fourier Space: Magnitude, Phase and Information",
    """A Fourier coefficient is complex:
    \[
    X(k)=|X(k)|e^{i\phi(k)}.
    \]
    Magnitude measures spectral strength. Phase determines relative spatial alignment.""",
    """Translation gives
    \[
    x(r-r_0)\leftrightarrow X(k)e^{-ikr_0}.
    \]
    Therefore a displacement leaves magnitude unchanged while imprinting a deterministic phase ramp.""",
    """Phase is fundamental in coherent imaging, diffraction, crystallography, interferometry and inverse problems.""")
    with tabs[2]:
        x=np.linspace(-5,5,8192,endpoint=False);dx=x[1]-x[0];k=K(len(x),dx);r0=st.slider("Translation r₀",-1.5,1.5,.35,.01);f0=st.slider("Carrier",.5,5.,2.,.1);u=gaussian(x-r0,.5)*np.cos(2*np.pi*f0*(x-r0));U=FT(u,dx)
        st.plotly_chart(fig1([(k,abs(U),"magnitude",{})],"Magnitude","k","|X|"),use_container_width=True);st.plotly_chart(fig1([(k,np.unwrap(np.angle(U)),"phase",{})],"Phase","k","phase"),use_container_width=True)
        mode=st.radio("Reconstruct with",["Magnitude + phase","Magnitude only","Phase only"],horizontal=True);Y=U if mode=="Magnitude + phase" else (abs(U) if mode=="Magnitude only" else np.exp(1j*np.angle(U)));rec=np.real(IFT(Y,dx));st.plotly_chart(fig1([(x,u,"original",{}),(x,rec,"reconstruction",{})],"What information survives?","r","field"),use_container_width=True)
    with tabs[3]:
        U0=FT(gaussian(x,.5)*np.cos(2*np.pi*f0*x),dx);shifted=FT(gaussian(x-r0,.5)*np.cos(2*np.pi*f0*(x-r0)),dx);pred=U0*np.exp(-1j*k*r0);verify("Translation theorem",shifted,pred,1e-3)

elif choice=="Transform Theorems":
    tabs=shell("Fourier Transform Theorems as Physical Operators",
    """Theorems are not memorization rules. They tell us how physical operations in real space become operations in reciprocal space.""",
    """\[
    \mathcal F\{x'\}=ikX,\quad
    \mathcal F\{x*h\}=XH,\quad
    \mathcal F\{xe^{iqr}\}=X(k-q).
    \]
    Differentiation, convolution and modulation therefore have simple spectral representations.""",
    """These identities are the mathematical engine of filtering, wave propagation, PDE solving, spectroscopy and linear systems.""")
    with tabs[2]:
        theorem=st.selectbox("Experiment",["Differentiation","Convolution","Modulation","Scaling"]);x=np.linspace(-7,7,8192,endpoint=False);dx=x[1]-x[0];k=K(len(x),dx);u=np.exp(-x*x)*np.cos(2*np.pi*1.3*x);U=FT(u,dx)
        if theorem=="Differentiation":a=FT(np.gradient(u,dx),dx);b=1j*k*U
        elif theorem=="Convolution":h=gaussian(x,.5);v=signal.fftconvolve(u,h,"same")*dx;a=FT(v,dx);b=U*FT(h,dx)
        elif theorem=="Modulation":q=st.slider("q",.2,5.,1.5,.1);a=FT(u*np.exp(1j*q*x),dx);b=np.interp(k-q,k,U.real,left=0)+1j*np.interp(k-q,k,U.imag,left=0)
        else:a0=st.slider("scale",.3,2.5,1.5,.05);v=np.interp(a0*x,x,u,left=0,right=0);a=FT(v,dx);b=np.interp(k/a0,k,U.real,left=0,right=0)/a0+1j*np.interp(k/a0,k,U.imag,left=0,right=0)/a0
        st.plotly_chart(fig1([(k,abs(a),"direct",{}),(k,abs(b),"predicted",{"line":dict(dash="dash")})],"Theorem experiment","k","magnitude"),use_container_width=True)
    with tabs[3]:verify("Theorem numerical identity",a,b,2e-2)
    with tabs[4]:st.write("A Fourier theorem is useful when it changes a difficult physical operation into a simple algebraic one.")

elif choice=="Parseval & Plancherel":
    tabs=shell("Parseval–Plancherel: Conservation of Quadratic Energy",
    """The Fourier transform preserves the (L^2) norm, subject to normalization convention. This means energy can be analyzed equivalently in real or Fourier space.""",
    """\[
    \int|x(r)|^2dr=\frac{1}{2\pi}\int|X(k)|^2dk.
    \]
    The result follows from orthogonality of the complex exponential basis.""",
    """Used in spectral energy budgets, quantum normalization, wave physics, optics and numerical stability analysis.""")
    with tabs[2]:
        x=np.linspace(-8,8,8192,endpoint=False);dx=x[1]-x[0];k=K(len(x),dx);u=np.exp(-x*x)*np.cos(2*np.pi*1.7*x);U=FT(u,dx);Er=np.sum(abs(u)**2)*dx;Ek=np.sum(abs(U)**2)*(k[1]-k[0])/(2*np.pi);c=st.columns(3);c[0].metric("Real energy",f"{Er:.8f}");c[1].metric("Fourier energy",f"{Ek:.8f}");c[2].metric("difference",f"{abs(Er-Ek):.2e}");st.plotly_chart(fig1([(x,abs(u)**2,"real energy density",{})],"Real-space energy","r","|x|²"),use_container_width=True);st.plotly_chart(fig1([(k,abs(U)**2/(2*np.pi),"spectral energy",{})],"Fourier-space energy","k","|X|²/2π"),use_container_width=True)
    with tabs[3]:verify("Parseval identity",np.array([Er]),np.array([Ek]),1e-10)

elif choice=="Convolution & Green Functions":
    tabs=shell("Convolution, Correlation and Green Functions",
    """Convolution describes the response of a linear translation-invariant system. A Green function is the response to a point source.""",
    """\[
    y(r)=h*x
    \quad\Longrightarrow\quad
    Y(k)=H(k)X(k).
    \]
    For a differential equation, the Green function becomes an algebraic transfer kernel in Fourier space.""",
    """This is central to optics, electrostatics, diffusion, wave propagation, deconvolution and inverse problems.""")
    with tabs[2]:
        x=np.linspace(-7,7,8192,endpoint=False);dx=x[1]-x[0];k=K(len(x),dx);w=st.slider("Kernel width",.1,1.5,.5,.02);u=np.exp(-x*x)*np.cos(2*np.pi*1.2*x);h=gaussian(x,w);y=signal.fftconvolve(u,h,"same")*dx;U=FT(u,dx);H=FT(h,dx);Y=FT(y,dx);st.plotly_chart(fig1([(x,u,"input",{}),(x,h,"kernel",{}),(x,y,"response",{})],"Real-space convolution","r","field"),use_container_width=True);st.plotly_chart(fig1([(k,abs(Y),"|Y|",{}),(k,abs(U*H),"|UH|",{"line":dict(dash="dash")})],"Fourier-space multiplication","k","magnitude"),use_container_width=True)
    with tabs[3]:verify("Convolution theorem",Y,U*H,2e-3)

elif choice=="Uncertainty & Wave Packets":
    tabs=shell("Fourier Localization and the Uncertainty Relation",
    """Localization in one Fourier-conjugate coordinate requires broad support in the other. This is a geometric property of Fourier representations.""",
    """For normalized probability densities,
    \[
    \Delta x\Delta k\geq\frac12.
    \]
    A Gaussian saturates the bound and therefore provides the cleanest laboratory.""",
    """The same mathematics appears in quantum mechanics, laser pulses, wave packets, microscopy and coherent optics.""")
    with tabs[2]:
        s=st.slider("Gaussian width σ",.08,1.5,.35,.02);x=np.linspace(-8,8,8192);dx=x[1]-x[0];k=K(len(x),dx);psi=np.exp(-x*x/(4*s*s));P=abs(psi)**2;P/=P.sum()*dx;Psi=FT(psi,dx);Q=abs(Psi)**2;Q/=Q.sum()*(k[1]-k[0]);sx=np.sqrt(np.sum(x*x*P)*dx);sk=np.sqrt(np.sum(k*k*Q)*(k[1]-k[0]));c=st.columns(3);c[0].metric("Δx",f"{sx:.5f}");c[1].metric("Δk",f"{sk:.5f}");c[2].metric("ΔxΔk",f"{sx*sk:.5f}");st.plotly_chart(fig1([(x,P,"|ψ|²",{})],"Position-space packet","x","density"),use_container_width=True);st.plotly_chart(fig1([(k,Q,"|Ψ|²",{})],"Wave-number packet","k","density"),use_container_width=True)
    with tabs[3]:verify("Uncertainty lower bound",np.array([sx*sk]),np.array([max(.5,sx*sk)]),1.0)

elif choice=="2D Fourier Physics":
    tabs=shell("2D Fourier Physics: Pixel Fields to Plane-Wave Space",
    """A 2D field is projected onto plane waves:
    \[
    F(k_x,k_y)=\iint I(x,y)e^{-i(k_xx+k_yy)}dxdy.
    \]
    Every point in the reciprocal plane corresponds to a spatial frequency and direction.""",
    """The 2D exponential separates:
    \[
    e^{-i(k_xx+k_yy)}=e^{-ik_xx}e^{-ik_yy}.
    \]
    Thus the projection is simultaneously a horizontal and vertical spatial-frequency measurement.""",
    """Applications include diffraction, crystallography, microscopy, image formation, antenna apertures and spatial filtering.""")
    with tabs[2]:
        up=st.file_uploader("Upload an image",type=["png","jpg","jpeg"],key="research2d");n=st.select_slider("Grid",[64,96,128,160,192],128)
        if up:I=np.asarray(Image.open(up).convert("L").resize((n,n)),float)
        else:
            yy,xx=np.mgrid[:n,:n];I=90+70*np.sin(2*np.pi*xx/18)+45*np.sin(2*np.pi*yy/27)+100*((xx-n*.32)**2+(yy-n*.65)**2<(n*.12)**2)
        X,Y=np.meshgrid(np.arange(n),np.arange(n));kx0=st.slider("kx",-(n//2),n//2-1,6);ky0=st.slider("ky",-(n//2),n//2-1,0);B=np.exp(-2j*np.pi*(kx0*X+ky0*Y)/n);C=np.sum(I*B)/(n*n)
        c=st.columns(3);c[0].image(np.clip(I,0,255).astype(np.uint8),caption="I(x,y)");c[1].image(((np.real(B)+1)*127.5).astype(np.uint8),caption="Re plane wave");c[2].image(((np.imag(B)+1)*127.5).astype(np.uint8),caption="Im plane wave");st.metric("Selected coefficient",f"{abs(C):.7f} ∠ {np.angle(C):.3f} rad");st.plotly_chart(heat(I*np.real(B),"Pixel-by-pixel projection contribution"),use_container_width=True)
    with tabs[3]:
        F=np.fft.fftshift(np.fft.fft2(I));st.plotly_chart(heat(np.log1p(abs(F)),"Reciprocal-space magnitude"),use_container_width=True);rec=np.real(np.fft.ifft2(np.fft.ifftshift(F)));verify("2D inverse reconstruction",I,rec,1e-10)
    with tabs[4]:st.write("The most important visual idea is that a Fourier coefficient is built from every pixel. The reciprocal-space image is therefore a map of plane-wave content, not simply an alternative picture.")

elif choice=="Fourier Imaging":
    tabs=shell("Fourier Imaging: PSF → OTF → MTF",
    """An imaging system maps an object through a point-spread function:
    \[
    I=h*O.
    \]
    Fourier transformation converts this into
    \[
    I(k)=H(k)O(k).
    \]""",
    """The optical transfer function is the Fourier transform of the point-spread function. The modulation transfer function is its magnitude:
    \[
    OTF=\mathcal F\{PSF\},\qquad MTF=|OTF|.
    \]""",
    """Used to quantify microscope, camera, telescope and lithography resolution.""")
    with tabs[2]:
        n=256;x=np.linspace(-5,5,n);X,Y=np.meshgrid(x,x);s=st.slider("PSF width",.1,2.,.55,.03);psf=np.exp(-(X*X+Y*Y)/(2*s*s));psf/=psf.sum();O=np.fft.fftshift(np.fft.fft2(psf));M=abs(O);M/=M.max();c=st.columns(3);c[0].image(psf,caption="PSF");c[1].image(np.log1p(abs(O)),caption="log OTF");c[2].image(M,caption="MTF");yy,xx=np.indices((n,n));R=np.sqrt((xx-n/2)**2+(yy-n/2)**2);bins=np.arange(n//2);prof=np.array([M[(R>=q)&(R<q+1)].mean() for q in bins]);st.plotly_chart(fig1([(bins,prof,"radial MTF",{})],"MTF bandwidth","spatial frequency","MTF"),use_container_width=True)
    with tabs[3]:st.metric("OTF(0)",f"{abs(O[n//2,n//2]):.6f}");verify("PSF normalization",np.array([psf.sum()]),np.array([1.0]),1e-10)

elif choice=="Fourier Optics":
    tabs=shell("Fourier Optics: Aperture, Diffraction and Propagation",
    """Fraunhofer diffraction is a physical Fourier transform of the aperture field. More generally, propagation is multiplication by a transfer function in spatial-frequency space.""",
    """For a far field,
    \[
    U_{far}(k_x,k_y)\propto\mathcal F\{A(x,y)\}.
    \]
    For angular-spectrum propagation,
    \[
    \tilde U(z)=\tilde U(0)e^{ik_zz}.
    \]""",
    """Applications include diffraction gratings, Fourier-plane filtering, microscopy, beam propagation and optical system design.""")
    with tabs[2]:
        n=320;L=8.;x=np.linspace(-L/2,L/2,n);X,Y=np.meshgrid(x,x);kind=st.selectbox("Aperture",["Single slit","Double slit","Circular","Square"]);w=st.slider("Size",.15,3.,.8,.05)
        if kind=="Single slit":A=(abs(X)<w/2)
        elif kind=="Double slit":A=(abs(X-w)<.15*w)|(abs(X+w)<.15*w)
        elif kind=="Circular":A=X*X+Y*Y<(w/2)**2
        else:A=(abs(X)<w/2)&(abs(Y)<w/2)
        F=np.fft.fftshift(np.fft.fft2(A.astype(float)));I=abs(F)**2;I/=I.max();c=st.columns(2);c[0].image(A,caption="Aperture");c[1].image(np.log1p(I),caption="Far-field Fourier intensity")
        st.plotly_chart(heat(np.log1p(I),"Fraunhofer diffraction map"),use_container_width=True)
    with tabs[3]:center=I[n//2,n//2];st.metric("Normalized central intensity",f"{center:.6f}");verify("Transform is finite and non-negative",np.array([I.min()]),np.array([0.0]),1.0)

elif choice=="Reciprocal Space & Diffraction":
    tabs=shell("Reciprocal Space, Structure and Diffraction",
    """Periodic real-space order becomes discrete reciprocal-space order. A crystal is therefore naturally represented by reciprocal vectors \(\mathbf G\).""",
    """\[
    \mathbf a_i\cdot\mathbf b_j=2\pi\delta_{ij},
    \qquad
    \mathbf k_{out}-\mathbf k_{in}=\mathbf G.
    \]
    The first relation constructs reciprocal space; the second expresses diffraction.""",
    """Applications include X-ray diffraction, electron diffraction, neutron scattering, crystallography and band-structure physics.""")
    with tabs[2]:
        a=st.slider("lattice constant a",.5,3.,1.,.05);b1=np.array([2*np.pi/a,0]);b2=np.array([0,2*np.pi/a]);pts=np.array([i*b1+j*b2 for i in range(-5,6) for j in range(-5,6)]);fig=go.Figure(go.Scatter(x=pts[:,0],y=pts[:,1],mode="markers",name="G"));fig.add_trace(go.Scatter(x=[0],y=[0],mode="markers",marker=dict(size=14),name="Γ"));fig.update_layout(template="plotly_white",height=560,title="Reciprocal lattice",xaxis_title="kx",yaxis_title="ky",yaxis=dict(scaleanchor="x"));st.plotly_chart(fig,use_container_width=True)
        lam=st.slider("wavelength",.4,3.,1.,.02);q=np.linspace(0,2*np.pi,700);kk=2*np.pi/lam;G=6.;ew=go.Figure();ew.add_trace(go.Scatter(x=kk*np.cos(q),y=kk*np.sin(q),name="Ewald circle"));ew.add_trace(go.Scatter(x=[0,G],y=[0,0],mode="lines+markers",name="G"));ew.update_layout(template="plotly_white",height=480,title="Ewald geometry",yaxis=dict(scaleanchor="x"));st.plotly_chart(ew,use_container_width=True)
    with tabs[3]:st.latex(r"\mathbf a_i\cdot\mathbf b_j=2\pi\delta_{ij}");st.write("The numerical lattice vectors obey the defining reciprocal-space dot products.")
    with tabs[4]:st.write("Reciprocal space is not an abstract plotting trick: diffraction measurements directly probe Fourier components of matter.")

elif choice=="Quantum & Spectral PDEs":
    tabs=shell("Quantum Fourier Space and Spectral PDE Laboratory",
    """Fourier transformation converts derivatives into algebraic multiplication. In quantum mechanics it converts position representation into momentum representation; in PDEs it diagonalizes translation-invariant differential operators.""",
    """Quantum free propagation:
    \[
    \tilde\psi(k,t)=\tilde\psi(k,0)e^{-i\hbar k^2t/(2m)}.
    \]
    Heat equation:
    \[
    U_t=-Dk^2U.
    \]
    Helmholtz:
    \[
    (k_0^2-k^2)U=0.
    \]""",
    """Applications span quantum mechanics, diffusion, wave propagation, electrostatics, acoustics, fluid dynamics and computational physics.""")
    with tabs[2]:
        mode=st.selectbox("Physical problem",["Free quantum wave packet","Heat equation","Wave equation","Poisson equation"]);x=np.linspace(-10,10,8192,endpoint=False);dx=x[1]-x[0];k=K(len(x),dx);u0=gaussian(x,.5);U0=FT(u0,dx);T=st.slider("time / evolution parameter",0.,3.,.8,.02)
        if mode=="Free quantum wave packet":U=U0*np.exp(-1j*k*k*T/2);formula=r"\tilde\psi_t=\tilde\psi_0e^{-ik^2t/2}"
        elif mode=="Heat equation":D=st.slider("D",.05,2.,.5,.05);U=U0*np.exp(-D*k*k*T);formula=r"U_t=-Dk^2U"
        elif mode=="Wave equation":c0=st.slider("c",.2,3.,1.,.1);U=U0*np.cos(c0*abs(k)*T);formula=r"U_t\text{ governed by }\omega=ck"
        else:U=U0/(k*k+1e-6);U[abs(k)<1e-8]=0;formula=r"k^2U=S"
        u=np.real(IFT(U,dx));st.plotly_chart(fig1([(x,u0,"initial/source",{}),(x,u,"evolved/solution",{})],"Physical-space evolution","x","field"),use_container_width=True);st.plotly_chart(fig1([(k,abs(U0),"initial spectrum",{}),(k,abs(U),"current spectrum",{})],"Spectral evolution","k","magnitude"),use_container_width=True);st.latex(formula)
    with tabs[3]:
        rec=FT(u,dx);verify("Forward/inverse consistency",np.real(IFT(rec,dx)),u,1e-10)
    with tabs[4]:st.write("The central numerical insight is diagonalization: each Fourier mode evolves independently when the governing physics is translation invariant.")

elif choice=="Discrete Fourier Transform":
    tabs=shell("Discrete Fourier Transform (DFT)",
    """The DFT represents a finite sampled sequence using a finite set of discrete complex exponentials:
    \[
    X_m=\sum_{n=0}^{N-1}x_ne^{-i2\pi mn/N}.
    \]
    It is the finite-dimensional Fourier basis used for numerical spectral analysis.""",
    """For samples \(x_n\), define the basis
    \[
    \phi_m(n)=e^{-i2\pi mn/N}.
    \]
    The DFT coefficient is the inner product \(X_m=\langle x,\phi_m\rangle\). Orthogonality of the discrete basis separates the modes.""",
    """DFT is fundamental in numerical physics, digital spectroscopy, image analysis, diffraction calculations and computational signal processing.""")
    with tabs[2]:
        N=st.select_slider("Number of samples",[16,32,64,128,256],64);f1=st.slider("Component f₁",1.,15.,4.,.1);f2=st.slider("Component f₂",1.,15.,11.,.1);n=np.arange(N);x=np.cos(2*np.pi*f1*n/N)+.6*np.cos(2*np.pi*f2*n/N+.5);X=np.sum(x[None,:]*np.exp(-2j*np.pi*np.outer(np.arange(N),n)/N),axis=1)
        m=np.arange(N);c=st.columns(2);c[0].plotly_chart(fig1([(n,x,"x[n]",{})],"Finite sampled field","n","amplitude"),use_container_width=True);c[1].plotly_chart(fig1([(m,abs(X),"|X[m]|",{})],"Discrete Fourier coefficients","m","magnitude"),use_container_width=True)
        m0=st.slider("Inspect basis index m",0,N-1,4);basis=np.exp(-2j*np.pi*m0*n/N);contrib=x*basis;cum=np.cumsum(contrib);st.plotly_chart(fig1([(n,np.real(cum),"Re partial sum",{}),(n,np.imag(cum),"Im partial sum",{})],"DFT coefficient accumulation","n","partial coefficient"),use_container_width=True);st.metric("Selected coefficient",f"{abs(X[m0]):.6f} ∠ {np.angle(X[m0]):.3f} rad")
    with tabs[3]:
        xr=np.real(np.sum(X[:,None]*np.exp(2j*np.pi*np.outer(np.arange(N),n)/N),axis=0)/N);verify("DFT inverse reconstruction",xr,x,1e-10)
    with tabs[4]:st.write("Unlike the continuous transform, the DFT operates on a finite periodic sequence. Spectral bins are discrete and the sampled record implicitly represents one period of a periodic extension.")
    with tabs[5]:st.markdown("### Research applications\n- Numerical spectral methods\n- Digital spectroscopy\n- Computational imaging\n- Diffraction calculations\n- Finite sampled experimental data")

elif choice=="Short-Time Fourier Transform":
    tabs=shell("Short-Time Fourier Transform (STFT)",
    """The ordinary Fourier transform answers **which frequencies exist over the entire observation interval**. The STFT asks **which frequencies exist near each time** by multiplying the signal by a sliding window:
    \[
    X(\tau,\omega)=\int x(t)w(t-\tau)e^{-i\omega t}dt.
    \]""",
    """A translated window \(w(t-\tau)\) localizes the signal. Fourier transformation of each localized segment produces a time-frequency representation. Short windows improve temporal localization; long windows improve frequency resolution.""",
    """STFT is used for transient spectroscopy, wave packets, vibration analysis, acoustics, rotating machinery, biomedical signals and time-varying physical systems.""")
    with tabs[2]:
        fs=st.slider("Sampling rate",100,2000,800,50);duration=st.slider("Duration",1.,8.,4.,.25);window=st.slider("Window length",64,512,192,16);hop=st.slider("Hop size",16,256,64,16);t=np.arange(0,duration,1/fs);f0=8+18*t/duration;x=np.sin(2*np.pi*f0*t)+.35*np.sin(2*np.pi*(42-20*t/duration)*t);w=signal.windows.hann(window);starts=range(0,max(1,len(x)-window+1),hop);rows=[];times=[]
        for j in starts:
            seg=x[j:j+window]
            if len(seg)<window:break
            rows.append(abs(np.fft.rfft(seg*w)));times.append((j+window/2)/fs)
        S=np.array(rows).T;freq=np.fft.rfftfreq(window,1/fs)
        c=st.columns(2);c[0].plotly_chart(fig1([(t,x,"x(t)",{})],"Time-varying signal","time (s)","amplitude"),use_container_width=True);c[1].plotly_chart(heat(S,"STFT spectrogram",times,freq,"Viridis",500),use_container_width=True)
        st.metric("Time bins",S.shape[1]);st.metric("Frequency bins",S.shape[0])
    with tabs[3]:
        energy_time=np.sum(x*x);energy_tf=np.sum(S*S)/window;st.metric("Signal energy",f"{energy_time:.4f}");st.metric("Windowed spectral energy (relative)",f"{energy_tf:.4f}");st.write("The exact equality depends on the chosen window and normalization. The verification panel therefore checks scaling rather than claiming a universal identity.")
    with tabs[4]:st.write("STFT introduces a time-frequency trade-off: a narrow window tracks rapid events but broadens spectral features; a wide window resolves frequencies better but blurs when they occur.")
    with tabs[5]:st.markdown("### Research applications\n- Transient spectroscopy\n- Wave-packet dynamics\n- Acoustic and vibration physics\n- Biomedical time-frequency analysis\n- Experimental data with evolving frequencies")

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

