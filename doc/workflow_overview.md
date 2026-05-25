# Workflow Overview — Diagrama Mermaid

> Diagrama de flujo del pipeline propuesto: codificación de cada ventana de serie temporal como imagen, reconstrucción con un método específico para cada codificación, y evaluación mediante comparación visual, métricas puntuales, métricas de tendencia y desempeño de clasificación de actividad.

---

## Versión 1 — Diagrama completo agrupado por etapas (recomendado para el paper)

```mermaid
flowchart TD
    %% ============ INPUT ============
    A["<b>Input</b><br/>Time-series window<br/>x ∈ ℝ<sup>N×3</sup> (acc x,y,z)<br/>N = 128"]:::input

    %% ============ PREPROCESS ============
    A --> B["<b>Per-axis normalization</b><br/>x̃<sub>t</sub> = (x<sub>t</sub> − min)/(max − min)<br/>store (min, max) as metadata"]:::preprocess

    %% ============ ENCODING SUBGRAPH ============
    subgraph ENC ["<b>① ENCODING</b> &nbsp;&nbsp; 𝒯 : ℝ<sup>N</sup> → ℝ<sup>N×N</sup>"]
        direction TB
        C{"<b>Encoding method</b>"}:::router
        C -->|Geometric| RP["<b>RP</b> · Recurrence Plot<br/>R<sub>i,j</sub> = σ·‖s<sub>i</sub>−s<sub>j</sub>‖<br/>Phase-space (m=2, τ=1)"]:::enc
        C -->|Polar / sum| GASF["<b>GASF</b><br/>G<sup>sum</sup><sub>i,j</sub> = cos(φ<sub>i</sub>+φ<sub>j</sub>)<br/>Symmetric · diag = 2x̃²−1"]:::enc
        C -->|Polar / diff| GADF["<b>GADF</b><br/>G<sup>diff</sup><sub>i,j</sub> = sin(φ<sub>i</sub>−φ<sub>j</sub>)<br/>Antisymmetric · scale [-0.7, 0.7]"]:::enc
        C -->|Markovian| MTF["<b>MTF</b> · Markov Transition Field<br/>M<sub>i,j</sub> = W<sub>s<sub>i</sub>,s<sub>j</sub></sub><br/>Q-quantile discretization (Q=4)"]:::enc
        RP --> IMG["<b>RGB image</b><br/>(M<sub>x</sub>, M<sub>y</sub>, M<sub>z</sub>) → N×N×3<br/>8-bit PNG"]:::img
        GASF --> IMG
        GADF --> IMG
        MTF --> IMG
    end

    B --> C

    %% ============ RECONSTRUCTION SUBGRAPH ============
    subgraph REC ["<b>② RECONSTRUCTION</b> &nbsp;&nbsp; 𝒯⁻¹ : ℝ<sup>N×N</sup> → ℝ<sup>N</sup> &nbsp; (tailored to encoding)"]
        direction TB
        D{"<b>Inverse mapping</b>"}:::router
        D -->|"RP path"| RecRP["<b>RP⁻¹</b><br/>Threshold ε → Jaccard graph<br/>Dijkstra → MDS → PCA<br/>Sign-flip disambiguation"]:::rec
        D -->|"GASF path"| RecGASF["<b>GASF⁻¹</b> · closed-form<br/>φ<sub>i</sub> = ½ arccos(G<sup>sum</sup><sub>i,i</sub>)<br/>x̃<sub>i</sub> = cos(φ<sub>i</sub>)"]:::rec
        D -->|"GADF path"| RecGADF["<b>GADF⁻¹</b> · closed-form<br/>pivot i* = argmin |G̅<sub>i</sub>|<br/>φ<sub>j</sub> = arcsin(G<sub>i*,j</sub>) + φ<sub>i*</sub>"]:::rec
        D -->|"MTF path"| RecMTF["<b>MTF⁻¹</b><br/>Cosine sim. on rows<br/>Greedy clustering → Q states<br/>Pivot-based state ordering"]:::rec
    end

    IMG --> D

    %% ============ DENORMALIZATION ============
    RecRP --> E["<b>Denormalization</b><br/>x<sub>j</sub> = min + x̃<sub>j</sub>·(max − min)<br/>using stored (min, max)"]:::preprocess
    RecGASF --> E
    RecGADF --> E
    RecMTF --> E

    E --> F["<b>Reconstructed window</b><br/>x̂ ∈ ℝ<sup>N×3</sup>"]:::output

    %% ============ EVALUATION SUBGRAPH ============
    subgraph EVAL ["<b>③ EVALUATION</b>"]
        direction LR
        V1["<b>Visual comparison</b><br/>Side-by-side TS overlays<br/>Original vs reconstructed"]:::eval
        V2["<b>Pointwise metrics</b><br/>MAE · MRE · RMSE<br/>per-axis & aggregated"]:::eval
        V3["<b>Trend-level metrics</b><br/>Pearson correlation<br/>DTW distance"]:::eval
        V4["<b>Activity classification</b><br/>CNN · GRU · LSTM<br/>KNN · SVC · RandomForest<br/>LOSO / LOTO CV"]:::eval
    end

    F --> V1
    F --> V2
    F --> V3
    F --> V4
    A -.->|"ground truth"| V1
    A -.->|"ground truth"| V2
    A -.->|"ground truth"| V3

    %% ============ NODE STYLES ============
    classDef input        fill:#312e81,stroke:#1e1b4b,color:#ffffff,stroke-width:2.5px,font-weight:bold
    classDef preprocess   fill:#e0e7ff,stroke:#4f46e5,color:#1e1b4b,stroke-width:2px
    classDef router       fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:2.5px,font-weight:bold
    classDef enc          fill:#d1fae5,stroke:#059669,color:#064e3b,stroke-width:2px
    classDef img          fill:#fce7f3,stroke:#db2777,color:#831843,stroke-width:2.5px,font-weight:bold
    classDef rec          fill:#dbeafe,stroke:#2563eb,color:#1e3a8a,stroke-width:2px
    classDef output       fill:#7c3aed,stroke:#5b21b6,color:#ffffff,stroke-width:2.5px,font-weight:bold
    classDef eval         fill:#ffe4e6,stroke:#e11d48,color:#881337,stroke-width:2px

    %% ============ SUBGRAPH STYLES ============
    style ENC  fill:#ecfdf5,stroke:#10b981,stroke-width:3px,color:#064e3b
    style REC  fill:#eff6ff,stroke:#3b82f6,stroke-width:3px,color:#1e3a8a
    style EVAL fill:#fff1f2,stroke:#f43f5e,stroke-width:3px,color:#881337

    %% ============ LINK STYLES ============
    linkStyle default stroke:#475569,stroke-width:1.8px
```

---

## Versión 2 — Compacta para figura de paper (left-right, una sola codificación a modo de ejemplo)

```mermaid
flowchart LR
    A["<b>TS window</b><br/>x ∈ ℝ<sup>N×3</sup>"]:::input
    A --> B["<b>Normalize</b><br/>store (min, max)"]:::pre
    B --> C["<b>Encode</b><br/>RP / GASF / GADF / MTF<br/>→ N×N×3 image"]:::enc
    C --> D["<b>Reconstruct</b><br/>method-tailored 𝒯⁻¹"]:::rec
    D --> E["<b>Denormalize</b>"]:::pre
    E --> F["<b>x̂ ∈ ℝ<sup>N×3</sup></b>"]:::out

    F --> G1["<b>Visual</b><br/>plots"]:::eval
    F --> G2["<b>Pointwise</b><br/>MAE/RMSE"]:::eval
    F --> G3["<b>Trend</b><br/>corr/DTW"]:::eval
    F --> G4["<b>Classification</b><br/>CNN/GRU/LSTM/KNN/SVC/RF"]:::eval

    A -.->|ground truth| G1
    A -.->|ground truth| G2
    A -.->|ground truth| G3

    classDef input fill:#1e3a8a,stroke:#1e3a8a,color:#fff,stroke-width:2px
    classDef pre fill:#e0e7ff,stroke:#3730a3,color:#1e1b4b
    classDef enc fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef rec fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e
    classDef out fill:#7c3aed,stroke:#5b21b6,color:#fff,stroke-width:2px
    classDef eval fill:#fee2e2,stroke:#b91c1c,color:#7f1d1d
```

---

## Versión 3 — Subgrupos (para enfatizar las 3 etapas del pipeline)

```mermaid
flowchart TB
    A["<b>Input</b><br/>TS window<br/>x ∈ ℝ<sup>N×3</sup>"]:::input

    subgraph S1 ["① ENCODING"]
        direction TB
        E0["Per-axis normalization<br/>store (min,max) metadata"]:::pre
        E0 --> E1["<b>RP</b><br/>σ·‖s<sub>i</sub>−s<sub>j</sub>‖"]:::enc
        E0 --> E2["<b>GASF</b><br/>cos(φ<sub>i</sub>+φ<sub>j</sub>)"]:::enc
        E0 --> E3["<b>GADF</b><br/>sin(φ<sub>i</sub>−φ<sub>j</sub>)"]:::enc
        E0 --> E4["<b>MTF</b><br/>W<sub>s<sub>i</sub>,s<sub>j</sub></sub>"]:::enc
        E1 --> EI["RGB image<br/>N×N×3"]:::img
        E2 --> EI
        E3 --> EI
        E4 --> EI
    end

    subgraph S2 ["② RECONSTRUCTION (tailored to encoding)"]
        direction TB
        R1["<b>RP⁻¹:</b> Jaccard graph → Dijkstra → MDS → PCA"]:::rec
        R2["<b>GASF⁻¹:</b> diagonal inversion (closed-form)"]:::rec
        R3["<b>GADF⁻¹:</b> pivot-row inversion (closed-form)"]:::rec
        R4["<b>MTF⁻¹:</b> cosine clustering + state ordering"]:::rec
        RO["Denormalize → x̂ ∈ ℝ<sup>N×3</sup>"]:::out
        R1 --> RO
        R2 --> RO
        R3 --> RO
        R4 --> RO
    end

    subgraph S3 ["③ EVALUATION"]
        direction LR
        V1["<b>Visual</b><br/>TS overlay plots"]:::eval
        V2["<b>Pointwise</b><br/>MAE · MRE · RMSE"]:::eval
        V3["<b>Trend</b><br/>Pearson · DTW"]:::eval
        V4["<b>Classification</b><br/>CNN·GRU·LSTM<br/>KNN·SVC·RF"]:::eval
    end

    A --> S1
    EI --> R1
    EI --> R2
    EI --> R3
    EI --> R4
    RO --> S3
    A -.->|ground truth| S3

    classDef input fill:#1e3a8a,stroke:#1e3a8a,color:#fff,stroke-width:2px
    classDef pre fill:#e0e7ff,stroke:#3730a3,color:#1e1b4b
    classDef enc fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef img fill:#fce7f3,stroke:#be185d,color:#831843,stroke-width:2px
    classDef rec fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e
    classDef out fill:#7c3aed,stroke:#5b21b6,color:#fff,stroke-width:2px
    classDef eval fill:#fee2e2,stroke:#b91c1c,color:#7f1d1d
```

---

## Cómo exportar a PNG para LaTeX

Para incluir el diagrama en el paper como `images/workflow_overview.png`:

**Opción A — Mermaid CLI (recomendado):**
```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i workflow_overview.md -o images/workflow_overview.png -w 1920 -H 1080 -b transparent
```

**Opción B — Online:** copiar el bloque mermaid a https://mermaid.live, exportar como PNG/SVG.

**Opción C — VS Code:** instalar la extensión *Markdown Preview Mermaid Support* y hacer captura del preview.

Para la figura del paper se recomienda la **Versión 3 (con subgrupos)** porque explicita las tres etapas del pipeline anunciadas en el caption y permite que cada método de codificación y reconstrucción quede visualmente vinculado a su etapa correspondiente.
