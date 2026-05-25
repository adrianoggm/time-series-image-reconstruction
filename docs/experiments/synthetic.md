# Synthetic time-series experiment

This is the LaTeX fragment that describes the first validation experiment on a
hand-crafted 3-channel signal (cosine, sine, sine−cosine). It belongs to the
paper draft; it lives here as a historical reference and to keep the README
focused on setup / how-to-run.

---

\section{Reconstructions from the Artificial Time Series}

For our first experiment, we construct a synthetic time series \( \mathbf{w} \in \mathbb{R}^{N \times 3} \) of length \(N=129\) by sampling an underlying parameter \(t\) uniformly over \([0,\pi]\).  Concretely, define
\[
t_i = \frac{i}{N-1}\,\pi,\qquad i=0,1,\dots,N-1,
\]
and let each vector \(\mathbf{w}_i = [w_{i,0},\,w_{i,1},\,w_{i,2}]^\top\) be given by
\[
w_{i,0} = \cos(t_i), 
\qquad
w_{i,1} = \sin(t_i),
\qquad
w_{i,2} = \sin(t_i)\;-\;\cos(t_i).
\]
Thus the full data matrix is
\[
\mathbf{w} = 
\begin{bmatrix}
\cos(t_0) & \sin(t_0) & \sin(t_0)-\cos(t_0) \\
\cos(t_1) & \sin(t_1) & \sin(t_1)-\cos(t_1) \\
\vdots    & \vdots    & \vdots             \\
\cos(t_{N-1}) & \sin(t_{N-1}) & \sin(t_{N-1})-\cos(t_{N-1})
\end{bmatrix}.
\]
This three‐dimensional signal—composed of a pure cosine channel, a pure sine channel, and their difference serves as the input for our RP, GAF and MTF reconstructions.
Figure \ref{fig:artificial_encodings} shows the image encodings of the three  channels under each technique:

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.32\linewidth}
        \centering
        \includegraphics[width=\linewidth,height=0.13\textheight]{images/COSRP.png}
        \caption{ RP}
        \label{fig:cos_rp}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\linewidth}
        \centering
        \includegraphics[width=\linewidth,height=0.13\textheight]{images/COSGasf.png}
        \caption{GASF}
        \label{fig:cos_gasf}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\linewidth}
        \centering
        \includegraphics[width=\linewidth,height=0.13\textheight]{images/COSmtf.png}
        \caption{MTF}
        \label{fig:cos_mtf}
    \end{subfigure}
    \caption{Image encodings of the cosine channel from the artificial time series.}
    \label{fig:artificial_encodings}
\end{figure}
Once created the images we calculate the reconstruction through the decoding of the images:let's see the graphic comparison. 
\begin{figure}[H]
    \centering
    \includegraphics[width=\linewidth,height=0.31\textheight]{images/Comparison_all_wideRP.png}
    \caption{Comparasion RP reconstruction with Artifical time series.}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=\linewidth,height=0.31\textheight]{images/Comparison_all_wideGAF.png}
    \caption{Comparasion from GASF reconstruction with Artifical time series.}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=\linewidth,height=0.31\textheight]{images/Comparison_all_widemtf.png}
    \caption{Comparasion from MTF reconstruction with Artifical time series.}
\end{figure}

