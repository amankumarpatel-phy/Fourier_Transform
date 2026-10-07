import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal, ndimage
from PIL import Image

st.set_page_config(page_title="Advanced Fourier Physics Lab",page_icon="∿",layout="wide")
st.markdown("""<style>
.main-title{font-size:2.45rem;font-weight:800}.subtitle{opacity:.72}
.box{padding:1rem;border-left:4px solid #4c78a8;background:rgba(76,120,168,.08);border-radius:6px}
</style>""",unsafe_allow_html=True)

def lines(traces,title,xlabel,ylabel,height=430):
    f=go.Figure()
    for x,y,n,kw in traces:f.add_trace(go.Scatter(x=x,y=y,name=n,**kw))
    f.update_layout(template="plotly_white",title=title,height=height,xaxis_title=xlabel,yaxis_title=ylabel,legend=dict(orientation="h"))
    return f

def FT(x,dx):
    return np.fft.fftshift(np.fft.fft(np.asarray(x)))*dx
def IFT(X,dx):
    return np.fft.ifft(np.fft.ifftshift(X))/dx
def Kaxis(n,dx): return np.fft.fftshift(np.fft.fftfreq(n,d=dx))
def gaussian(x,s): return np.exp(-x*x/(2*s*s))/(s*np.sqrt(2*np.pi))

pages=[
"Home","1 · What the Transform Does","2 · Fourier Transform Pairs",
"3 · Magnitude, Phase & Complex Plane","4 · Transform Theorems",
"5 · Convolution Theorem","6 · Correlation & Translation",
"7 · Uncertainty & Wave Packets","8 · 2D Fourier Transform",
"9 · 2D Frequency-Space Filtering","10 · Fourier Optics",
"11 · Reciprocal Lattice","12 · Crystal Structure Factor",
"13 · Quantum Position–Momentum","14 · Fourier Methods for PDEs","15 · Graduate Fourier Physics","16 · Fourier Imaging Theory","17 · Reciprocal-Space Geometry","18 · Fourier Operators & Dispersion","19 · Wigner Phase Space","20 · Coherent Diffraction","21 · Sampling & Aliasing Physics","22 · 2D Green Functions","23 · Bloch Waves & Band Formation","24 · Helmholtz & Wave-Vector Shells","25 · Fourier Inverse Problems"
]
page=st.sidebar.radio("Advanced Fourier Laboratory",pages)
st.sidebar.divider()
st.sidebar.latex(r"X(k)=\int x(r)e^{-ikr}\,dr")
st.sidebar.caption("Continuous Fourier analysis • spatial frequency • reciprocal space")

def mechanism(key):
    st.markdown("### How the transform works")
    st.write("A Fourier coefficient is a complex projection. Choose a basis wave, multiply it by the field, and integrate. Matching oscillations add coherently; mismatched oscillations cancel.")
    with st.expander("Open the projection laboratory"):
        f0=st.slider("Probe frequency",0.,80.,17.,.5,key=key+"f")
        t=np.linspace(-.5,.5,1000,endpoint=False); dx=t[1]-t[0]
        x=np.cos(2*np.pi*17*t)+.5*np.cos(2*np.pi*35*t+.5)
        basis=np.exp(-1j*2*np.pi*f0*t); z=x*basis; c=np.cumsum(z)*dx; C=c[-1]
        a,b=st.columns(2)
        with a: st.plotly_chart(lines([(t,x,"x(t)",{}),(t,np.real(basis),"Re basis",{"line":dict(dash="dash")})],"Field and complex probe","t","amplitude",360),use_container_width=True)
        with b: st.plotly_chart(lines([(t,np.real(c),"Re accumulation",{}),(t,np.imag(c),"Im accumulation",{})],"Running complex integral","t","partial X(f)",360),use_container_width=True)
        st.latex(r"X(f)=\int x(t)e^{-i2\pi ft}dt")
        q=st.columns(3);q[0].metric("f",f"{f0:.1f} Hz");q[1].metric("|X(f)|",f"{abs(C):.5f}");q[2].metric("phase",f"{np.angle(C):.3f} rad")
        st.write("The complex exponential is a rotating basis vector. At the matching frequency its rotation is cancelled by the signal's oscillation, producing a large resultant.")

if page=="Home":
    st.markdown('<div class="main-title">Advanced Fourier Physics Laboratory</div>',unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Fourier transformation as a change of basis: from real-space fields to frequency, wave-vector and reciprocal space.</div>',unsafe_allow_html=True)
    st.markdown('<div class="box"><b>Central idea:</b> Fourier analysis is not merely spectrum plotting. It is the projection of a physical field onto complex exponential basis functions.</div>',unsafe_allow_html=True)
    st.latex(r"x(r)\xleftrightarrow{\mathcal F}X(k)")
    st.markdown("### Physics pathway")
    st.write("Projection → magnitude/phase → transform theorems → convolution → correlation → uncertainty → 2D spatial frequencies → diffraction → reciprocal lattice → structure factor → quantum momentum → PDE Green functions.")
    t=np.linspace(-4,4,2048);dx=t[1]-t[0];x=gaussian(t,.55);X=FT(x,dx);k=Kaxis(len(t),dx)
    a,b=st.columns(2)
    with a:st.plotly_chart(lines([(t,x,"x(r)",{})],"Real space","r","x(r)"),use_container_width=True)
    with b:st.plotly_chart(lines([(k,np.abs(X),"|X(k)|",{})],"Fourier space","k","magnitude"),use_container_width=True)

elif page=="1 · What the Transform Does":
    st.markdown('<div class="main-title">1 · What the Fourier Transform Actually Does</div>',unsafe_allow_html=True)
    st.latex(r"X(k)=\int_{-\infty}^{\infty}x(r)e^{-ikr}dr")
    mechanism("m1")
    t=np.linspace(-1,1,1000);f0=st.slider("Complex-plane probe",1.,60.,17.,.5);x=np.cos(2*np.pi*17*t)+.5*np.cos(2*np.pi*31*t)
    z=x*np.exp(-1j*2*np.pi*f0*t);c=np.cumsum(z)/len(z)
    fig=go.Figure(go.Scatter(x=np.real(c),y=np.imag(c),mode="lines",name="partial integral"))
    fig.add_trace(go.Scatter(x=[0,np.real(c[-1])],y=[0,np.imag(c[-1])],mode="lines+markers",name="final coefficient"))
    fig.update_layout(template="plotly_white",height=500,title="Complex-plane accumulation",xaxis_title="Real",yaxis_title="Imaginary",yaxis=dict(scaleanchor="x"))
    st.plotly_chart(fig,use_container_width=True)
    st.write("This is the core mechanism behind every other module in this laboratory.")

elif page=="2 · Fourier Transform Pairs":
    st.markdown('<div class="main-title">2 · Continuous Fourier Transform Pairs</div>',unsafe_allow_html=True)
    pair=st.selectbox("Physical function",["Gaussian","Rectangular aperture","Exponential","Sinc","Cosine"])
    x=np.linspace(-10,10,4096);dx=x[1]-x[0];k=Kaxis(len(x),dx)
    if pair=="Gaussian":s=st.slider("σ",.1,2.,.5,.05);u=gaussian(x,s)
    elif pair=="Rectangular aperture":w=st.slider("Width",.2,6.,2.,.1);u=(abs(x)<w/2).astype(float)
    elif pair=="Exponential":a=st.slider("α",.1,3.,1.,.1);u=np.exp(-a*abs(x))
    elif pair=="Sinc":a=st.slider("Scale",.2,3.,1.,.1);u=np.sinc(a*x)
    else:f0=st.slider("f0",.5,5.,2.,.1);u=np.cos(2*np.pi*f0*x)
    U=FT(u,dx)
    a,b=st.columns(2)
    with a:st.plotly_chart(lines([(x,u,"x(r)",{})],"Real-space function","r","x(r)"),use_container_width=True)
    with b:st.plotly_chart(lines([(k,np.real(U),"Re X",{}),(k,np.abs(U),"|X|",{"line":dict(dash="dash")})],"Transform","k","X(k)"),use_container_width=True)
    st.info("The transform pair is a statement about how localization, periodicity and smoothness in one domain appear in the conjugate domain.")

elif page=="3 · Magnitude, Phase & Complex Plane":
    st.markdown('<div class="main-title">3 · Magnitude, Phase & Complex Fourier Space</div>',unsafe_allow_html=True)
    t=np.linspace(-3,3,4096);dx=t[1]-t[0];k=Kaxis(len(t),dx);shift=st.slider("Translation",-.9,.9,.25,.01);f0=st.slider("Carrier",.5,8.,3.,.1)
    x=np.exp(-((t-shift)/.45)**2)*np.cos(2*np.pi*f0*(t-shift));X=FT(x,dx)
    a,b=st.columns(2)
    with a:st.plotly_chart(lines([(k,np.abs(X),"|X|",{})],"Magnitude","k","magnitude"),use_container_width=True)
    with b:st.plotly_chart(lines([(k,np.unwrap(np.angle(X)),"phase",{})],"Phase","k","phase"),use_container_width=True)
    st.latex(r"x(r-r_0)\leftrightarrow X(k)e^{-ikr_0}")
    mode=st.radio("Reconstruct from",["Magnitude + phase","Magnitude only","Phase only"],horizontal=True)
    mag=np.abs(X);ph=np.angle(X)
    Y=X if mode=="Magnitude + phase" else (mag if mode=="Magnitude only" else np.exp(1j*ph))
    xr=np.real(IFT(Y,dx))
    st.plotly_chart(lines([(t,x,"original",{}),(t,xr,"reconstruction",{"line":dict(dash="dash")})],"Information carried by magnitude and phase","r","amplitude"),use_container_width=True)

elif page=="4 · Transform Theorems":
    st.markdown('<div class="main-title">4 · Fourier Transform Theorems</div>',unsafe_allow_html=True)
    theorem=st.selectbox("Explore",["Linearity","Translation","Scaling","Differentiation","Modulation"])
    x=np.linspace(-6,6,4096);dx=x[1]-x[0];k=Kaxis(len(x),dx);g=np.exp(-x*x)*np.cos(2*np.pi*1.2*x)
    if theorem=="Linearity":
        a=st.slider("a",.1,2.,1.2,.1);b=st.slider("b",.1,2.,.7,.1);h=np.sin(2*np.pi*.7*x);A=FT(a*g+b*h,dx);B=a*FT(g,dx)+b*FT(h,dx)
        st.plotly_chart(lines([(k,np.abs(A),"direct",{}),(k,np.abs(B),"aX+bH",{"line":dict(dash="dash")})],"Linearity verification","k","magnitude"),use_container_width=True);st.latex(r"\mathcal F\{ax+bh\}=aX+bH")
    elif theorem=="Translation":
        r0=st.slider("r0",-.9,.9,.3,.01);y=np.exp(-((x-r0))**2)*np.cos(2*np.pi*1.2*(x-r0));A=FT(g,dx);B=FT(y,dx)
        st.plotly_chart(lines([(k,np.abs(A),"original |X|",{}),(k,np.abs(B),"shifted |X|",{"line":dict(dash="dash")})],"Translation preserves magnitude","k","magnitude"),use_container_width=True);st.latex(r"x(r-r_0)\leftrightarrow X(k)e^{-ikr_0}")
    elif theorem=="Scaling":
        a=st.slider("scale a",.25,3.,1.7,.05);y=np.interp(a*x,x,g,left=0,right=0);A=FT(g,dx);B=FT(y,dx)
        st.plotly_chart(lines([(k,np.abs(A),"|X(k)|",{}),(k,np.abs(B),"scaled",{"line":dict(dash="dash")})],"Scaling theorem","k","magnitude"),use_container_width=True);st.latex(r"x(ar)\leftrightarrow |a|^{-1}X(k/a)")
    elif theorem=="Differentiation":
        dg=np.gradient(g,dx);A=FT(dg,dx);B=1j*k*FT(g,dx)
        st.plotly_chart(lines([(k,np.abs(A),"F{dg/dr}",{}),(k,np.abs(B),"ikX",{"line":dict(dash="dash")})],"Differentiation becomes multiplication","k","magnitude"),use_container_width=True);st.latex(r"\mathcal F\{x'\}=ikX")
    else:
        q=st.slider("modulation q",.2,5.,1.5,.1);y=g*np.cos(2*np.pi*q*x);A=FT(g,dx);B=FT(y,dx)
        st.plotly_chart(lines([(k,np.abs(A),"original",{}),(k,np.abs(B),"modulated",{})],"Modulation shifts spectral content","k","magnitude"),use_container_width=True);st.latex(r"\mathcal F\{x(r)e^{iqr}\}=X(k-q)")

elif page=="5 · Convolution Theorem":
    st.markdown('<div class="main-title">5 · Convolution Theorem</div>',unsafe_allow_html=True)
    t=np.linspace(-6,6,4096);dx=t[1]-t[0];k=Kaxis(len(t),dx);w=st.slider("Kernel width",.15,1.5,.45,.05)
    x=np.exp(-t*t)*np.cos(2*np.pi*1.1*t);h=gaussian(t,w);y=signal.fftconvolve(x,h,mode="same")*dx;X=FT(x,dx);H=FT(h,dx);Y=FT(y,dx)
    fig=make_subplots(rows=2,cols=2,subplot_titles=("Functions","Convolution","Fourier factors","Product"))
    fig.add_trace(go.Scatter(x=t,y=x,name="x"),row=1,col=1);fig.add_trace(go.Scatter(x=t,y=h,name="h"),row=1,col=1);fig.add_trace(go.Scatter(x=t,y=y,name="x*h"),row=1,col=2)
    fig.add_trace(go.Scatter(x=k,y=np.abs(X),name="|X|"),row=2,col=1);fig.add_trace(go.Scatter(x=k,y=np.abs(H),name="|H|"),row=2,col=1);fig.add_trace(go.Scatter(x=k,y=np.abs(Y),name="|F{x*h}|"),row=2,col=2);fig.add_trace(go.Scatter(x=k,y=np.abs(X*H),name="|XH|"),row=2,col=2)
    fig.update_layout(template="plotly_white",height=760);st.plotly_chart(fig,use_container_width=True)
    st.latex(r"\boxed{\mathcal F\{x*h\}=XH}")
    st.write("Translation-and-integration in real space becomes multiplication in Fourier space. This is the mathematical heart of linear filtering.")

elif page=="6 · Correlation & Translation":
    st.markdown('<div class="main-title">6 · Correlation as Fourier-Domain Matching</div>',unsafe_allow_html=True)
    t=np.linspace(-5,5,4096);dx=t[1]-t[0];shift=st.slider("Unknown displacement",-.4,.4,.08,.005);template=np.exp(-(t/.35)**2);obs=np.exp(-((t-shift)/.35)**2)+.08*np.random.default_rng(4).normal(size=len(t))
    corr=signal.correlate(obs,template,mode="same")*dx;lag=t;peak=lag[np.argmax(corr)]
    X=FT(obs,dx);H=FT(template,dx);corr2=np.real(IFT(X*np.conjugate(H),dx))
    st.plotly_chart(lines([(t,obs,"observation",{}),(t,template,"template",{})],"Pattern matching","t","amplitude"),use_container_width=True)
    st.plotly_chart(lines([(lag,corr,"direct correlation",{}),(lag,corr2,"Fourier-domain correlation",{"line":dict(dash="dash")})],"Correlation","lag","R"),use_container_width=True)
    st.metric("Recovered displacement",f"{peak:.4f}")
    st.latex(r"R_{xy}(\tau)=\mathcal F^{-1}\{X(k)Y^*(k)\}")

elif page=="7 · Uncertainty & Wave Packets":
    st.markdown('<div class="main-title">7 · Fourier Localization and Uncertainty</div>',unsafe_allow_html=True)
    s=st.slider("Gaussian width σ",.08,1.5,.35,.02);x=np.linspace(-7,7,8192);dx=x[1]-x[0];k=Kaxis(len(x),dx);psi=np.exp(-x*x/(4*s*s));P=np.abs(psi)**2;P/=P.sum()*dx;Psi=FT(psi,dx);Q=np.abs(Psi)**2;Q/=Q.sum()*(k[1]-k[0])
    xm=np.sum(x*P)*dx;km=np.sum(k*Q)*(k[1]-k[0]);dxs=np.sqrt(np.sum((x-xm)**2*P)*dx);dks=np.sqrt(np.sum((k-km)**2*Q)*(k[1]-k[0]))
    a,b=st.columns(2)
    with a:st.plotly_chart(lines([(x,P,"|ψ(x)|²",{})],"Position-space localization","x","density"),use_container_width=True)
    with b:st.plotly_chart(lines([(k,Q,"|Ψ(k)|²",{})],"Wave-number distribution","k","density"),use_container_width=True)
    c=st.columns(3);c[0].metric("Δx",f"{dxs:.4f}");c[1].metric("Δk",f"{dks:.4f}");c[2].metric("ΔxΔk",f"{dxs*dks:.4f}")
    st.latex(r"\Delta x\Delta k\geq\frac12")
    st.write("Localization in real space requires a broad superposition of Fourier modes. The uncertainty principle is therefore deeply connected to Fourier duality.")

elif page=="8 · 2D Fourier Transform":
    st.markdown('<div class="main-title">8 · 2D Fourier Transform — Pixel-by-Pixel Projection</div>',unsafe_allow_html=True)
    up=st.file_uploader("Upload image",type=["png","jpg","jpeg"],key="imgft")
    if up:img=Image.open(up).convert("L")
    else:
        N=128;yy,xx=np.mgrid[:N,:N];a=110+60*np.sin(2*np.pi*xx/18)+35*np.sin(2*np.pi*yy/27)+100*((xx-40)**2+(yy-85)**2<16**2);img=Image.fromarray(np.clip(a,0,255).astype(np.uint8))
    N=st.select_slider("Analysis resolution",[64,96,128,160,192],128);I=np.asarray(img.resize((N,N)),float);X,Y=np.meshgrid(np.arange(N),np.arange(N))
    kx=st.slider("kx",-(N//2),N//2-1,6);ky=st.slider("ky",-(N//2),N//2-1,0);basis=np.exp(-2j*np.pi*(kx*X+ky*Y)/N);C=np.sum(I*basis)/(N*N)
    c=st.columns(3);c[0].image(I.astype(np.uint8),caption="I(x,y)");c[1].image(((np.real(basis)+1)*127.5).astype(np.uint8),caption="Re basis");c[2].image(((np.imag(basis)+1)*127.5).astype(np.uint8),caption="Im basis")
    st.latex(r"F(k_x,k_y)=\frac1{N^2}\sum_{x,y}I(x,y)e^{-i2\pi(k_xx+k_yy)/N}")
    st.metric("Selected coefficient",f"|F|={abs(C):.6f}, phase={np.angle(C):.3f} rad")
    st.plotly_chart(go.Figure(go.Heatmap(z=I*np.real(basis),colorscale="RdBu",zmid=0)).update_layout(template="plotly_white",height=500,title="Pixel contributions to one Fourier coefficient"),use_container_width=True)
    F=np.fft.fftshift(np.fft.fft2(I));kk=np.fft.fftshift(np.fft.fftfreq(N));mag=np.abs(F);phase=np.angle(F)
    a,b=st.columns(2)
    with a:st.plotly_chart(go.Figure(go.Heatmap(x=kk,y=kk,z=np.log1p(mag),colorscale="Viridis")).update_layout(template="plotly_white",height=500,title="log magnitude"),use_container_width=True)
    with b:st.plotly_chart(go.Figure(go.Heatmap(x=kk,y=kk,z=phase,colorscale="Twilight",zmin=-np.pi,zmax=np.pi)).update_layout(template="plotly_white",height=500,title="phase"),use_container_width=True)
    R=np.sqrt((X-N/2)**2+(Y-N/2)**2);rad=st.slider("Reconstruction radius",2,N//2,18);mask=R<rad;rec=np.real(np.fft.ifft2(np.fft.ifftshift(F*mask)))
    c=st.columns(2);c[0].image(np.clip(rec,0,255).astype(np.uint8),caption="Low-frequency reconstruction");c[1].image(mask.astype(float),caption="Selected Fourier coefficients")
    st.info("A 2D transform is the same projection idea with plane waves exp[-i(kx x+ky y)].")

elif page=="9 · 2D Frequency-Space Filtering":
    st.markdown('<div class="main-title">9 · Filtering Directly in Spatial-Frequency Space</div>',unsafe_allow_html=True)
    up=st.file_uploader("Image",type=["png","jpg","jpeg"],key="imgfilter")
    if up:I=np.asarray(Image.open(up).convert("L").resize((256,256)),float)
    else:
        yy,xx=np.mgrid[:256,:256];I=100+60*np.sin(xx/6)+35*np.sin(yy/15)+25*np.random.default_rng(1).normal(size=(256,256));I=ndimage.gaussian_filter(I,1)
    F=np.fft.fftshift(np.fft.fft2(I));N=I.shape[0];yy,xx=np.mgrid[:N,:N];R=np.sqrt((xx-N/2)**2+(yy-N/2)**2)
    kind=st.selectbox("Transfer function H(kx,ky)",["Ideal low-pass","Ideal high-pass","Band-pass","Gaussian low-pass","Directional filter"]);c0=st.slider("Cutoff",2,120,35)
    if kind=="Ideal low-pass":H=R<c0
    elif kind=="Ideal high-pass":H=R>c0
    elif kind=="Band-pass":H=(R>c0*.6)&(R<c0*1.4)
    elif kind=="Gaussian low-pass":H=np.exp(-R**2/(2*c0*c0))
    else:
        ang=np.arctan2(yy-N/2,xx-N/2);theta=st.slider("Direction",-np.pi,np.pi,0.,.05);width=st.slider("Angular width",.05,1.5,.3,.05);d=np.angle(np.exp(1j*(ang-theta)));H=(abs(d)<width)&(R<c0)
    G=F*H;out=np.real(np.fft.ifft2(np.fft.ifftshift(G)));c=st.columns(4)
    c[0].image(np.clip(I,0,255).astype(np.uint8),caption="Input");c[1].image(H.astype(float),caption="H(kx,ky)");c[2].image(np.log1p(abs(G)),caption="Filtered spectrum");c[3].image(np.clip(out,0,255).astype(np.uint8),caption="Reconstructed field")
    st.latex(r"G(k_x,k_y)=H(k_x,k_y)F(k_x,k_y)")
    st.write("The filter acts on Fourier coefficients, not directly on pixels. The inverse transform converts the modified spectral field back into a spatial image.")

elif page=="10 · Fourier Optics":
    st.markdown('<div class="main-title">10 · Fourier Optics and Fraunhofer Diffraction</div>',unsafe_allow_html=True)
    N=384;L=8;x=np.linspace(-L/2,L/2,N);X,Y=np.meshgrid(x,x);typ=st.selectbox("Aperture",["Single slit","Double slit","Circular","Square","2D grating"]);w=st.slider("Size",.15,3.,.8,.05)
    if typ=="Single slit":A=(abs(X)<w/2)
    elif typ=="Double slit":A=(abs(X-w)<.15*w)|(abs(X+w)<.15*w)
    elif typ=="Circular":A=(X*X+Y*Y<(w/2)**2)
    elif typ=="Square":A=(abs(X)<w/2)&(abs(Y)<w/2)
    else:A=(np.sin(2*np.pi*X/w)>0)&(np.sin(2*np.pi*Y/w)>0)
    U=np.fft.fftshift(np.fft.fft2(A.astype(float)));I=abs(U)**2;I/=I.max()
    f=make_subplots(rows=1,cols=2,subplot_titles=("Aperture A(x,y)","Fraunhofer |F{A}|²"));f.add_trace(go.Heatmap(x=x,y=x,z=A,colorscale="Gray",showscale=False),row=1,col=1);f.add_trace(go.Heatmap(x=x,y=x,z=np.log10(I+1e-9),colorscale="Viridis"),row=1,col=2);f.update_layout(template="plotly_white",height=560);st.plotly_chart(f,use_container_width=True)
    st.latex(r"U_{far}(k_x,k_y)\propto\mathcal F\{A(x,y)\}")
    st.write("Far-field diffraction is a physical Fourier transform. Narrow apertures generate broad angular spectra; periodic apertures generate discrete diffraction orders.")

elif page=="11 · Reciprocal Lattice":
    st.markdown('<div class="main-title">11 · Reciprocal Lattice from Fourier Space</div>',unsafe_allow_html=True)
    typ=st.selectbox("Direct lattice",["1D chain","Square","Rectangular","Triangular"]);a=st.slider("a",.5,2.,1.,.1);b=st.slider("b",.5,2.,1.4,.1);M=12
    if typ=="1D chain":
        r=np.arange(-M,M+1)*a;q=np.linspace(-12,12,2500);S=abs(np.sum(np.exp(-1j*np.outer(q,r)),axis=1))**2;S/=S.max();f=make_subplots(rows=1,cols=2,subplot_titles=("Real lattice","Fourier intensity"));f.add_trace(go.Scatter(x=r,y=np.ones_like(r),mode="markers"),row=1,col=1);f.add_trace(go.Scatter(x=q,y=S),row=1,col=2);f.update_layout(template="plotly_white",height=470);st.plotly_chart(f,use_container_width=True)
    else:
        pts=[]
        for i in range(-M,M+1):
            for j in range(-M,M+1):
                p=(i*a,j*a) if typ=="Square" else ((i*a,j*b) if typ=="Rectangular" else (i*a+j*a/2,j*np.sqrt(3)*a/2));pts.append(p)
        pts=np.array(pts);q=np.linspace(-10,10,260);QX,QY=np.meshgrid(q,q);S=np.zeros_like(QX,dtype=complex)
        for p in pts[::max(1,len(pts)//700)]:S+=np.exp(-1j*(QX*p[0]+QY*p[1]))
        I=abs(S)**2;I/=I.max();f=make_subplots(rows=1,cols=2,subplot_titles=("Real lattice","Reciprocal-space intensity"));f.add_trace(go.Scatter(x=pts[:,0],y=pts[:,1],mode="markers"),row=1,col=1);f.add_trace(go.Heatmap(x=q,y=q,z=np.log10(I+1e-8),colorscale="Viridis"),row=1,col=2);f.update_layout(template="plotly_white",height=550);st.plotly_chart(f,use_container_width=True)
    st.latex(r"\mathbf a_i\cdot\mathbf b_j=2\pi\delta_{ij}")
    st.write("Periodic real-space order transforms into concentrated reciprocal-space order. Reciprocal lattice vectors are Fourier-space signatures of translational symmetry.")

elif page=="12 · Crystal Structure Factor":
    st.markdown('<div class="main-title">12 · Crystal Structure Factor</div>',unsafe_allow_html=True)
    basis=st.selectbox("Basis",["Monatomic","Diatomic","Four-site square","Custom three-site"]);H=st.slider("h",-8,8,1);K=st.slider("k",-8,8,1)
    B={"Monatomic":np.array([[0,0]]),"Diatomic":np.array([[0,0],[.5,.5]]),"Four-site square":np.array([[0,0],[.5,0],[0,.5],[.5,.5]]),"Custom three-site":np.array([[0,0],[.25,.5],[.7,.2]])}[basis]
    Fb=np.sum(np.exp(-2j*np.pi*(H*B[:,0]+K*B[:,1])));R=8;Fl=np.sum([np.exp(-2j*np.pi*(H*i+K*j)) for i in range(-R,R+1) for j in range(-R,R+1)]);Ftot=Fb*Fl
    c=st.columns(3);c[0].metric("Basis |F_b|",f"{abs(Fb):.4f}");c[1].metric("Basis phase",f"{np.angle(Fb):.3f}");c[2].metric("Total |F|",f"{abs(Ftot):.4f}")
    st.latex(r"F(\mathbf G)=\sum_{\mathbf R}e^{-i\mathbf G\cdot\mathbf R}\sum_jf_je^{-i\mathbf G\cdot\mathbf r_j}")
    st.write("The lattice determines where reciprocal peaks can occur; the basis determines their intensity and systematic absences. This is the Fourier origin of diffraction selection rules.")

elif page=="13 · Quantum Position–Momentum":
    st.markdown('<div class="main-title">13 · Quantum Position ↔ Momentum</div>',unsafe_allow_html=True)
    s=st.slider("Wave-packet width",.15,1.5,.45,.03);p0=st.slider("Mean k",0.,8.,3.,.1);x=np.linspace(-8,8,4096);dx=x[1]-x[0];k=Kaxis(len(x),dx);psi=np.exp(-x*x/(4*s*s))*np.exp(1j*p0*x);Psi=FT(psi,dx)
    a,b=st.columns(2)
    with a:st.plotly_chart(lines([(x,np.real(psi),"Re ψ",{}),(x,np.imag(psi),"Im ψ",{})],"Position-space state","x","ψ"),use_container_width=True)
    with b:st.plotly_chart(lines([(k,abs(Psi)**2,"|ψ̃(k)|²",{})],"Momentum-space probability","k","density"),use_container_width=True)
    st.latex(r"\tilde\psi(k)=\frac1{\sqrt{2\pi}}\int\psi(x)e^{-ikx}dx")
    st.write("Momentum representation is the Fourier representation of the same state. Changing localization changes the momentum-space width.")

elif page=="14 · Fourier Methods for PDEs":
    st.markdown('<div class="main-title">14 · Fourier Transform as a PDE Solver</div>',unsafe_allow_html=True)
    s=st.slider("Source width",.2,1.5,.6,.05);x=np.linspace(-10,10,4096);dx=x[1]-x[0];k=Kaxis(len(x),dx);source=gaussian(x,s);S=FT(source,dx);eps=1e-7;U=S/(k*k+eps);U[abs(k)<1e-10]=0;u=np.real(IFT(U,dx))
    f=make_subplots(rows=2,cols=1,subplot_titles=("Real-space source and solution","Fourier-space amplitudes"));f.add_trace(go.Scatter(x=x,y=source,name="source"),row=1,col=1);f.add_trace(go.Scatter(x=x,y=u,name="u"),row=1,col=1);f.add_trace(go.Scatter(x=k,y=abs(S),name="|S(k)|"),row=2,col=1);f.add_trace(go.Scatter(x=k,y=abs(U),name="|U(k)|"),row=2,col=1);f.update_layout(template="plotly_white",height=700);st.plotly_chart(f,use_container_width=True)
    st.latex(r"-u''(x)=s(x)\xrightarrow{\mathcal F}k^2U(k)=S(k)")
    st.write("Spatial derivatives become multiplication by ik. A differential equation therefore becomes algebraic in Fourier space, after which the inverse transform returns the physical solution.")
\n
elif page=="15 · Graduate Fourier Physics":
    st.markdown('<div class="main-title">15 · Graduate Fourier Physics</div>',unsafe_allow_html=True)
    tab1,tab2,tab3,tab4,tab5=st.tabs(["Parseval / Plancherel","Duality","Uncertainty","Green Functions","Complex-plane projection"])
    x=np.linspace(-8,8,8192,endpoint=False);dx=x[1]-x[0];k=kaxis(len(x),dx);u=np.exp(-x*x)*np.cos(2*np.pi*1.7*x);U=FT(u,dx)
    with tab1:
        Er=np.sum(abs(u)**2)*dx;Ek=np.sum(abs(U)**2)*(k[1]-k[0])/(2*np.pi)
        c=st.columns(3);c[0].metric("∫|x|²dr",f"{Er:.7f}");c[1].metric("∫|X|²dk/2π",f"{Ek:.7f}");c[2].metric("relative error",f"{abs(Er-Ek)/Er:.2e}")
        st.latex(r"\int |x(r)|^2dr=\frac{1}{2\pi}\int |X(k)|^2dk")
        st.plotly_chart(linefig([(k,abs(U)**2/(2*np.pi),"spectral energy",{})],"Energy distribution in Fourier space","k","|X(k)|²/2π"),use_container_width=True)
    with tab2:
        st.latex(r"\mathcal F\{\mathcal F\{x(r)\}\}=2\pi x(-r)")
        st.write("Applying the transform twice returns the original function with reversal and the normalization dictated by the chosen convention.")
        X2=FT(U, k[1]-k[0]);st.plotly_chart(linefig([(x,u,"original",{}),(x,np.real(X2/(2*np.pi)),"double transform / 2π",{"line":dict(dash="dash")})],"Fourier duality check","r","amplitude"),use_container_width=True)
    with tab3:
        sig=st.slider("Gaussian σ",.08,1.5,.35,.02,key="grad_unc");psi=np.exp(-x*x/(4*sig*sig));P=abs(psi)**2;P/=P.sum()*dx;Psi=FT(psi,dx);Q=abs(Psi)**2;Q/=Q.sum()*(k[1]-k[0]);xm=np.sum(x*P)*dx;km=np.sum(k*Q)*(k[1]-k[0]);sx=np.sqrt(np.sum((x-xm)**2*P)*dx);sk=np.sqrt(np.sum((k-km)**2*Q)*(k[1]-k[0]))
        c=st.columns(3);c[0].metric("Δx",f"{sx:.5f}");c[1].metric("Δk",f"{sk:.5f}");c[2].metric("ΔxΔk",f"{sx*sk:.5f}")
        st.latex(r"\Delta x\Delta k\geq\frac12")
        st.plotly_chart(linefig([(x,P,"|ψ(x)|²",{})],"Localization ↔ spectral width","x","density"),use_container_width=True)
    with tab4:
        alpha=st.slider("Green-kernel scale",.1,2.,.5,.05,key="green");source=gaussian(x,.5);S=FT(source,dx);G=1/(k*k+alpha*alpha);response=np.real(IFT(S*G,dx))
        st.latex(r"u=G*s\quad\Longleftrightarrow\quad U(k)=G(k)S(k)")
        st.plotly_chart(linefig([(x,source,"source",{}),(x,response,"response",{})],"Green-function response","x","field"),use_container_width=True)
    with tab5:
        probe=st.slider("Probe k",0.,50.,17.,.5,key="complexprobe");t=np.linspace(-.5,.5,1600,endpoint=False);dx2=t[1]-t[0];sig=np.cos(2*np.pi*17*t)+.5*np.cos(2*np.pi*31*t);z=sig*np.exp(-1j*2*np.pi*probe*t);c=np.cumsum(z)*dx2
        fig=go.Figure(go.Scatter(x=np.real(c),y=np.imag(c),mode="lines",name="partial projection"));fig.add_trace(go.Scatter(x=[0,c[-1].real],y=[0,c[-1].imag],mode="lines+markers",name="result"));fig.update_layout(template="plotly_white",height=500,title="Complex-plane accumulation",xaxis_title="Re X",yaxis_title="Im X",yaxis=dict(scaleanchor="x"));st.plotly_chart(fig,use_container_width=True)
        st.metric("Final coefficient magnitude",f"{abs(c[-1]):.6f}")

elif page=="16 · Fourier Imaging Theory":
    st.markdown('<div class="main-title">16 · Fourier Imaging Theory: PSF → OTF → MTF</div>',unsafe_allow_html=True)
    n=256;L=8.;x=np.linspace(-L/2,L/2,n);X,Y=np.meshgrid(x,x);sigma=st.slider("PSF width",.1,2.,.55,.03);psf=np.exp(-(X*X+Y*Y)/(2*sigma*sigma));psf/=psf.sum();otf=np.fft.fftshift(np.fft.fft2(psf));mtf=abs(otf);mtf/=mtf.max()
    c=st.columns(3);c[0].image(psf,caption="PSF h(x,y)");c[1].image(np.log1p(abs(otf)),caption="log |OTF|");c[2].image(mtf,caption="MTF")
    yy,xx=np.indices((n,n));R=np.sqrt((xx-n/2)**2+(yy-n/2)**2);bins=np.arange(n//2);profile=np.array([mtf[(R>=q)&(R<q+1)].mean() for q in bins])
    st.plotly_chart(linefig([(bins,profile,"MTF",{})],"Radial modulation transfer function","spatial-frequency radius","MTF"),use_container_width=True)
    st.latex(r"OTF=\mathcal F\{PSF\},\qquad MTF=|OTF|")
    st.write("This connects Fourier analysis directly to microscope, camera and optical-system resolution: an imaging system is a spatial-frequency transfer function.")

elif page=="17 · Reciprocal-Space Geometry":
    st.markdown('<div class="main-title">17 · Reciprocal-Space Geometry: Ewald Sphere & Brillouin Zone</div>',unsafe_allow_html=True)
    tab1,tab2=st.tabs(["Ewald construction","Reciprocal lattice"])
    with tab1:
        lam=st.slider("Wavelength λ",.4,3.,1.,.02);kk=2*np.pi/lam;G=st.slider("|G|",.2,12.,6.,.1);theta=st.slider("G angle",0.,2*np.pi,0.,.02);q=np.linspace(0,2*np.pi,700);fig=go.Figure();fig.add_trace(go.Scatter(x=kk*np.cos(q),y=kk*np.sin(q),name="Ewald circle"));fig.add_trace(go.Scatter(x=[0,G*np.cos(theta)],y=[0,G*np.sin(theta)],mode="lines+markers",name="G"));fig.update_layout(template="plotly_white",height=520,title="2D Ewald construction",xaxis_title="kx",yaxis_title="ky",yaxis=dict(scaleanchor="x"));st.plotly_chart(fig,use_container_width=True);st.latex(r"\mathbf k_{out}-\mathbf k_{in}=\mathbf G")
    with tab2:
        a=st.slider("a",.5,3.,1.,.05);b1=np.array([2*np.pi/a,0.]);b2=np.array([0.,2*np.pi/a]);pts=np.array([i*b1+j*b2 for i in range(-4,5) for j in range(-4,5)]);fig=go.Figure(go.Scatter(x=pts[:,0],y=pts[:,1],mode="markers",name="G"));fig.add_trace(go.Scatter(x=[0],y=[0],mode="markers",marker=dict(size=14),name="Γ"));fig.update_layout(template="plotly_white",height=520,title="Square reciprocal lattice",xaxis_title="kx",yaxis_title="ky",yaxis=dict(scaleanchor="x"));st.plotly_chart(fig,use_container_width=True);st.latex(r"\mathbf a_i\cdot\mathbf b_j=2\pi\delta_{ij}");st.write("The first Brillouin zone is the Wigner–Seitz cell of this reciprocal lattice; crystal momentum is therefore intrinsically a Fourier-space coordinate.")


elif page=="18 · Fourier Operators & Dispersion":
    st.markdown('<div class="main-title">18 · Fourier Operators & Dispersion Relations</div>',unsafe_allow_html=True)
    st.write("Translation-invariant differential operators become multiplication by their Fourier symbols. This is the operator-theoretic reason Fourier space is so powerful.")
    op=st.selectbox("Operator",["First derivative","Second derivative","Laplacian","Helmholtz operator","Custom polynomial symbol"])
    k=np.linspace(-12,12,1800)
    if op=="First derivative": symbol=1j*k; formula=r"\partial_x\rightarrow ik"
    elif op=="Second derivative": symbol=-k*k; formula=r"\partial_x^2\rightarrow-k^2"
    elif op=="Laplacian": symbol=-k*k; formula=r"\nabla^2\rightarrow-|\mathbf k|^2"
    elif op=="Helmholtz operator":
        q=st.slider("k0",.5,8.,3.,.1);symbol=q*q-k*k;formula=r"(\partial_x^2+k_0^2)\rightarrow(k_0^2-k^2)"
    else:
        a=st.slider("a",-.5,.5,.1,.05);b=st.slider("b",-.5,.5,.2,.05);symbol=a*(1j*k)**3+b*(1j*k)**2+1j*k;formula=r"p(\partial_x)\rightarrow p(ik)"
    fig=go.Figure();fig.add_trace(go.Scatter(x=k,y=np.real(symbol),name="Re symbol"));fig.add_trace(go.Scatter(x=k,y=np.imag(symbol),name="Im symbol"));fig.update_layout(template="plotly_white",height=500,title="Fourier symbol of the operator",xaxis_title="k",yaxis_title="operator symbol")
    st.plotly_chart(fig,use_container_width=True);st.latex(formula)
    st.write("A dispersion relation is essentially the spectrum of the governing operator. Zeros, poles and curvature of the symbol encode propagation, resonance and stability.")

elif page=="19 · Wigner Phase Space":
    st.markdown('<div class="main-title">19 · Wigner Phase Space: Fourier Structure Beyond Probability</div>',unsafe_allow_html=True)
    st.write("The Wigner distribution combines position and wave-number information and exposes interference between Fourier components.")
    x=np.linspace(-6,6,600);dx=x[1]-x[0];k=np.linspace(-8,8,500);sigma=st.slider("Packet width",.25,1.2,.5,.03);q0=st.slider("Carrier k",0.,5.,2.,.1)
    psi=np.exp(-x*x/(4*sigma*sigma))*np.exp(1j*q0*x)
    W=np.zeros((len(k),len(x)))
    for j,x0 in enumerate(x):
        s=np.linspace(-3,3,241)
        vals=np.interp(x0+s,x,psi,left=0,right=0)*np.conjugate(np.interp(x0-s,x,psi,left=0,right=0))
        W[:,j]=np.real(np.array([np.trapz(vals*np.exp(-1j*kk*s),s) for kk in k]))
    st.plotly_chart(heat(W,"Wigner quasi-probability W(x,k)",x,k,"RdBu",560),use_container_width=True)
    st.latex(r"W(x,k)=\frac{1}{2\pi}\int\psi^*(x+\xi/2)\psi(x-\xi/2)e^{-ik\xi}d\xi")
    st.write("Unlike an ordinary probability density, W can become negative. Those negative regions encode quantum interference and have no classical probability interpretation.")

elif page=="20 · Coherent Diffraction":
    st.markdown('<div class="main-title">20 · Coherent Diffraction & Interference in Fourier Space</div>',unsafe_allow_html=True)
    n=384;L=8.;x=np.linspace(-L/2,L/2,n);X,Y=np.meshgrid(x,x);sep=st.slider("Source separation",.2,4.,1.2,.05);phase=st.slider("Relative phase",-np.pi,np.pi,0.,.05);sig=st.slider("Spot width",.08,1.,.25,.02)
    A=np.exp(-((X-sep/2)**2+Y**2)/(2*sig**2))+np.exp(1j*phase)*np.exp(-((X+sep/2)**2+Y**2)/(2*sig**2))
    F=np.fft.fftshift(np.fft.fft2(A));I=abs(F)**2;I/=I.max()
    c=st.columns(3);c[0].image(abs(A),caption="Coherent source amplitude");c[1].image(np.angle(A),caption="Source phase");c[2].image(np.log1p(I),caption="Fourier intensity")
    st.latex(r"I(\mathbf k)=|A_1(\mathbf k)+e^{i\phi}A_2(\mathbf k)|^2")
    st.write("The Fourier transform preserves complex amplitude, so relative phase survives into the interference pattern. This is the mathematical core of coherent diffraction.")

elif page=="21 · Sampling & Aliasing Physics":
    st.markdown('<div class="main-title">21 · Sampling, Spectral Replication & Aliasing</div>',unsafe_allow_html=True)
    f0=st.slider("Signal frequency",.5,20.,6.,.1);fs=st.slider("Sampling frequency",4.,40.,10.,.1);duration=2.;t=np.linspace(0,duration,1800);continuous=np.cos(2*np.pi*f0*t);ts=np.arange(0,duration,1/fs);samples=np.cos(2*np.pi*f0*ts)
    st.plotly_chart(linefig([(t,continuous,"continuous signal",{}),(ts,samples,"samples",{"mode":"markers"})],"Sampling in real time","t","amplitude"),use_container_width=True)
    alias=abs(((f0+fs/2)%fs)-fs/2)
    c=st.columns(3);c[0].metric("f0",f"{f0:.2f} Hz");c[1].metric("fs/2",f"{fs/2:.2f} Hz");c[2].metric("observed alias",f"{alias:.2f} Hz")
    st.latex(r"x_s(t)=x(t)\sum_n\delta(t-nT)")
    st.latex(r"X_s(f)=\frac1T\sum_m X(f-mf_s)")
    st.write("Sampling creates periodic replicas of the spectrum. If replicas overlap, different physical frequencies become indistinguishable: aliasing is a loss of information caused by insufficient sampling.")

elif page=="22 · 2D Green Functions":
    st.markdown('<div class="main-title">22 · 2D Green Functions in Fourier Space</div>',unsafe_allow_html=True)
    n=192;L=12.;x=np.linspace(-L/2,L/2,n);dx=x[1]-x[0];X,Y=np.meshgrid(x,x);sigma=st.slider("Source width",.1,1.5,.5,.03);source=np.exp(-(X*X+Y*Y)/(2*sigma**2));S=np.fft.fftshift(np.fft.fft2(source));kx=2*np.pi*np.fft.fftshift(np.fft.fftfreq(n,d=dx));KX,KY=np.meshgrid(kx,kx);K2=KX*KX+KY*KY;G=1/(K2+0.15**2);U=S*G;u=np.real(np.fft.ifft2(np.fft.ifftshift(U)))
    c=st.columns(3);c[0].image(source,caption="Source s(x,y)");c[1].image(np.log1p(abs(G)),caption="Green kernel |G(k)|");c[2].image(u,caption="Response u(x,y)")
    st.latex(r"-\nabla^2u+\mu^2u=s\quad\Longrightarrow\quad U(\mathbf k)=\frac{S(\mathbf k)}{|\mathbf k|^2+\mu^2}")
    st.write("The inverse-square operator becomes an algebraic denominator in reciprocal space. This is the computational structure behind electrostatics, screened potentials and many field theories.")

elif page=="23 · Bloch Waves & Band Formation":
    st.markdown('<div class="main-title">23 · Bloch Waves: Fourier Components of a Periodic Potential</div>',unsafe_allow_html=True)
    N=256;x=np.linspace(-8,8,N);V0=st.slider("Potential amplitude",0.,8.,3.,.1);a=st.slider("Lattice period",.8,3.,2.,.05);modes=st.slider("Fourier harmonics",1,5,3)
    V=np.zeros_like(x)
    for n0 in range(1,modes+1):V+=V0/n0**2*np.cos(2*np.pi*n0*x/a)
    st.plotly_chart(linefig([(x,V,"V(x)",{})],"Periodic potential","x","V(x)"),use_container_width=True)
    coeff=[]
    for n0 in range(-6,7):
        coeff.append(np.trapz(V*np.exp(-1j*2*np.pi*n0*x/a),x)/a)
    ns=np.arange(-6,7)
    st.plotly_chart(linefig([(ns,np.abs(coeff),"|V_G|",{})],"Potential Fourier components","reciprocal index","magnitude"),use_container_width=True)
    st.latex(r"V(x)=\sum_G V_Ge^{iGx},\qquad \psi_k(x)=e^{ikx}u_k(x)")
    st.write("A periodic crystal is naturally represented by reciprocal vectors G. The Fourier components of the potential couple plane waves whose momenta differ by reciprocal-lattice vectors; this is the origin of band formation.")

elif page=="24 · Helmholtz & Wave-Vector Shells":
    st.markdown('<div class="main-title">24 · Helmholtz Equation & Wave-Vector Geometry</div>',unsafe_allow_html=True)
    k0=st.slider("Helmholtz wave number",.5,12.,4.,.1);q=np.linspace(-8,8,500);QX,QY=np.meshgrid(q,q);shell=np.abs(QX*QX+QY*QY-k0*k0);tol=st.slider("Shell thickness",.03,.5,.12,.01);mask=shell<tol
    st.plotly_chart(heat(mask.astype(float),"Allowed wave-vector shell",q,q,"Viridis",550),use_container_width=True)
    st.latex(r"(\nabla^2+k_0^2)u=0\quad\Longrightarrow\quad(|\mathbf k|^2-k_0^2)U(\mathbf k)=0")
    st.write("A homogeneous Helmholtz field occupies a shell in Fourier space: every allowed plane wave has |k|=k0. This connects Fourier analysis directly to dispersion surfaces and wave propagation.")

elif page=="25 · Fourier Inverse Problems":
    st.markdown('<div class="main-title">25 · Fourier Inverse Problems & Regularization</div>',unsafe_allow_html=True)
    st.write("Many experiments measure incomplete or noisy Fourier information. Recovering the field becomes an inverse problem.")
    x=np.linspace(-6,6,2048);dx=x[1]-x[0];k=kaxis(len(x),dx);true=np.exp(-x*x)*np.cos(2*np.pi*1.5*x);H=np.exp(-0.025*k*k);Y=FT(true,dx)*H;noise=st.slider("Noise level",0.,.25,.05,.005);rng=np.random.default_rng(3);Yn=Y+noise*(rng.normal(size=len(Y))+1j*rng.normal(size=len(Y)))
    lam=st.slider("Tikhonov regularization",0.,2.,.15,.01);rec=np.real(IFT(np.conjugate(H)*Yn/(abs(H)**2+lam),dx))
    st.plotly_chart(linefig([(x,true,"true field",{}),(x,rec,"regularized reconstruction",{})],"Inverse Fourier problem","x","field"),use_container_width=True)
    st.latex(r"\hat X=\frac{H^*Y}{|H|^2+\lambda}")
    st.write("When the transfer function suppresses high-frequency information, naive inversion amplifies noise. Regularization trades exact inversion for stability—an essential idea in imaging, spectroscopy and inverse scattering.")

st.divider()
st.caption("Aman Edge Physics · Advanced Fourier Physics Laboratory")
