"""
generate_pdf_guide.py
Generates a comprehensive, professional PDF technical report for:
"Comparing Recommendation Algorithms on MovieLens 100K"
Includes full mathematical formulas, step-by-step evaluation metrics,
algorithm working, pipeline architecture, and examiner demonstration guide.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

PDF_FILENAME = "Comparing_Recommendation_Algorithms_Guide.pdf"


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and render total page count:
    'Page X of Y' in the running footer, along with a top running header.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#475569"))

        # Top running header on pages 2+
        if self._pageNumber > 1:
            self.drawString(36, 756, "Comparing Recommendation Algorithms (MovieLens 100K) — Technical & Mathematical Guide")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.6)
            self.line(36, 748, 576, 748)

        # Running footer on all pages
        self.setFont("Helvetica", 8)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 22, page_str)
        self.drawString(36, 22, "MovieLens 100K Benchmark | Academic Examination Technical Documentation")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(36, 32, 576, 32)

        self.restoreState()


def create_pdf(output_path):
    # Printable area: 8.5 x 11 in (612 x 792 pt).
    # Margins: 36 pt (0.5 in). Usable width = 540 pt. Usable height = 700 pt.
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=42,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    # Color definitions
    PRIMARY = colors.HexColor("#1E3A8A")       # Deep Navy Blue
    SECONDARY = colors.HexColor("#0D9488")     # Teal
    DARK_TEXT = colors.HexColor("#0F172A")     # Slate 900
    MUTED_TEXT = colors.HexColor("#475569")    # Slate 600
    BG_LIGHT = colors.HexColor("#F8FAFC")      # Light slate
    BG_CALLOUT = colors.HexColor("#F1F5F9")    # Slate 100
    ACCENT_BORDER = colors.HexColor("#93C5FD") # Light Blue
    CODE_BG = colors.HexColor("#F8FAFC")

    # Typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=PRIMARY,
        spaceAfter=2
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        textColor=MUTED_TEXT,
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=PRIMARY,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=SECONDARY,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=DARK_TEXT,
        spaceAfter=4
    )

    formula_style = ParagraphStyle(
        'FormulaStyle',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1E293B"),
        alignment=1, # Centered
        spaceBefore=2,
        spaceAfter=2
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.2,
        textColor=DARK_TEXT
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell_style,
        fontName='Helvetica-Bold'
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, SUMMARY & SECTION 1 ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("COMPARING RECOMMENDATION ALGORITHMS", title_style))
    story.append(Paragraph("Complete Technical Architecture, Algorithm Mechanics & Mathematical Metrics Guide", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=8))

    # Executive Box
    summary_html = (
        "<b>Executive Project Summary:</b> This project implements a clean, production-grade, 7-stage machine "
        "learning recommendation pipeline evaluating three distinct paradigm approaches (Bayesian Popularity Baseline, "
        "Item-based Collaborative Filtering with Cosine Similarity, and Truncated SVD Matrix Factorization) on the "
        "standard MovieLens 100K benchmark (943 users, 1,682 movies, 100,000 ratings). Evaluation follows rigorous "
        "temporal splitting (80% past ratings per user for training, 20% future for testing) and tests both error accuracy "
        "(RMSE) and top-10 recommendation utility (Precision@10, Recall@10, and Catalog Coverage %)."
    )
    story.append(Table([[Paragraph(summary_html, body_style)]],
                       colWidths=[540],
                       style=TableStyle([
                           ('BACKGROUND', (0,0), (-1,-1), BG_CALLOUT),
                           ('BOX', (0,0), (-1,-1), 1, ACCENT_BORDER),
                           ('TOPPADDING', (0,0), (-1,-1), 6),
                           ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                           ('LEFTPADDING', (0,0), (-1,-1), 8),
                           ('RIGHTPADDING', (0,0), (-1,-1), 8),
                       ])))
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. System Architecture & 7-Stage Pipeline", h1_style))
    story.append(Paragraph(
        "The project is structured strictly into single-responsibility modules under <code>src/</code>, "
        "orchestrated in a narrative sequence via <code>run_experiments.py</code> and inspected visually in <code>app.py</code>.",
        body_style
    ))

    arch_data = [
        [Paragraph("Stage", table_header_style),
         Paragraph("Module File", table_header_style),
         Paragraph("Core Functions", table_header_style),
         Paragraph("Engineering Responsibility", table_header_style)],

        [Paragraph("<b>1. Loading</b>", table_cell_style),
         Paragraph("<code>src/data_loading.py</code>", table_cell_style),
         Paragraph("<code>load_data()</code><br/><code>download_ml100k()</code>", table_cell_style),
         Paragraph("Fetches dataset with resilient SSL verification fallback; converts user/item IDs into 0-based indices.", table_cell_style)],

        [Paragraph("<b>2. Preprocess</b>", table_cell_style),
         Paragraph("<code>src/preprocessing.py</code>", table_cell_style),
         Paragraph("<code>validate()</code><br/><code>detect_outliers()</code><br/><code>data_fingerprint()</code>", table_cell_style),
         Paragraph("Integrity checks, IQR outlier detection (auditing only without arbitrary deletion), SHA-256 fingerprinting.", table_cell_style)],

        [Paragraph("<b>3. Splitting</b>", table_cell_style),
         Paragraph("<code>src/splitting.py</code>", table_cell_style),
         Paragraph("<code>temporal_split_per_user()</code><br/><code>sanity_check_splits()</code>", table_cell_style),
         Paragraph("Sorts each user's ratings chronologically; allocates first 80% to train and remaining 20% to test. Builds dense R_train matrix.", table_cell_style)],

        [Paragraph("<b>4. Models</b>", table_cell_style),
         Paragraph("<code>src/models/*.py</code>", table_cell_style),
         Paragraph("<code>PopularityBaseline</code><br/><code>ItemKNN</code><br/><code>SVDRecommender</code>", table_cell_style),
         Paragraph("Object-oriented algorithm implementations providing standardized <code>fit(R_train)</code> and <code>predict_matrix()</code>.", table_cell_style)],

        [Paragraph("<b>5. Evaluate</b>", table_cell_style),
         Paragraph("<code>src/evaluation/*.py</code>", table_cell_style),
         Paragraph("<code>build_top10()</code><br/><code>compute_metrics()</code>", table_cell_style),
         Paragraph("Top-10 candidate generation with -inf train masking; computes RMSE, Precision@10, Recall@10, and Catalog Coverage %.", table_cell_style)],

        [Paragraph("<b>6. Storage & Plots</b>", table_cell_style),
         Paragraph("<code>run_experiments.py</code><br/><code>src/plots.py</code>", table_cell_style),
         Paragraph("<code>stage_save()</code><br/><code>plot_model_comparison()</code>", table_cell_style),
         Paragraph("Persists results artifacts (<code>metrics.csv</code>, <code>top10_recommendations.pkl</code>) and generates 4 Seaborn plots.", table_cell_style)],

        [Paragraph("<b>7. Sync Check</b>", table_cell_style),
         Paragraph("<code>run_experiments.py</code>", table_cell_style),
         Paragraph("<code>stage_check()</code>", table_cell_style),
         Paragraph("Automated parity verification; detects dataset changes dynamically and reports delta against baseline.", table_cell_style)]
    ]

    arch_table = Table(arch_data, colWidths=[65, 115, 120, 240])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 8))

    # Clean transition to Section 2 on Page 2
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: SECTION 2 RECOMMENDATION ALGORITHMS
    # =========================================================================
    story.append(Paragraph("2. Deep Dive: Recommendation Algorithms", h1_style))
    story.append(Paragraph(
        "Each model implements identical interfaces: <code>fit(R_train)</code> consuming the dense 943x1682 "
        "user-item rating matrix (0 = unobserved) and <code>predict_matrix()</code> returning a continuous 943x1682 "
        "estimated rating score matrix clipped to the range [1.0, 5.0].",
        body_style
    ))

    # Model 1
    story.append(Paragraph("Algorithm 1: PopularityBaseline (Bayesian Weighted Rating)", h2_style))
    p_desc = (
        "A non-personalized heuristic inspired by the IMDB Top 250 formula. Simple average ratings fail when a movie has only "
        "one 5-star rating from an obscure user. Bayesian shrinkage pulls low-vote movies towards the global prior mean <i>C</i>."
    )
    story.append(Paragraph(p_desc, body_style))
    story.append(Table([[Paragraph("Weighted Rating (WR) = (v / (v + m)) * R + (m / (v + m)) * C", formula_style)]],
                       colWidths=[540], style=TableStyle([
                           ('BACKGROUND', (0,0), (-1,-1), CODE_BG),
                           ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
                           ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3)
                       ])))
    p_params = (
        "• <b>v</b>: Number of training ratings received by the movie.<br/>"
        "• <b>m</b>: 75th percentile of vote counts across all movies (confidence threshold; <i>m = 48</i> on ML-100K).<br/>"
        "• <b>R</b>: Arithmetic mean rating for that specific movie across training entries.<br/>"
        "• <b>C</b>: Global average rating across all ratings in the training set (<i>C ≈ 3.53</i>). Movies with no votes receive <i>C</i>.<br/>"
        "• <b>Outcome</b>: Outputs an identical preference vector for all users. High precision on aggregate blockbusters, "
        "but zero personalization and near-zero catalog coverage (3.86%)."
    )
    story.append(Paragraph(p_params, body_style))
    story.append(Spacer(1, 4))

    # Model 2
    story.append(Paragraph("Algorithm 2: ItemKNN (Item-Based Collaborative Filtering)", h2_style))
    knn_desc = (
        "Personalized memory-based model exploiting item-item co-occurrence. Computes item cosine similarity over column vectors "
        "of <i>R<sub>train</sub></i> (with missing entries treated as 0). The similarity matrix diagonal is set to 0. "
        "For each target item <i>i</i>, only its <b>top-30</b> nearest neighbors are retained; all others are set to 0."
    )
    story.append(Paragraph(knn_desc, body_style))
    story.append(Table([[Paragraph("r̂(u, i) = [ Σ_j (sim(i, j) * R(u, j)) ] / [ Σ_j (|sim(i, j)| * 1[R(u, j) > 0]) ]", formula_style)]],
                       colWidths=[540], style=TableStyle([
                           ('BACKGROUND', (0,0), (-1,-1), CODE_BG),
                           ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
                           ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3)
                       ])))
    knn_params = (
        "• <b>Vectorized Implementation</b>: Computed across all users and items via matrix multiplications:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<code>num = R_train @ S.T</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>den = (R_train > 0).astype(float) @ np.abs(S).T</code><br/>"
        "• <b>Zero-Denominator Fallback</b>: When a user hasn't rated any of item <i>i</i>'s top-30 neighbors (<i>den = 0</i>), "
        "the model safely falls back to that user's historical personal mean rating (rated items only), avoiding NaN artifacts.<br/>"
        "• <b>Outcome</b>: Outstanding catalog coverage (80.50%), surfacing niche items, but suffers from lower precision on the test set."
    )
    story.append(Paragraph(knn_params, body_style))
    story.append(Spacer(1, 4))

    # Model 3
    story.append(Paragraph("Algorithm 3: SVDRecommender (Matrix Factorization via Truncated SVD)", h2_style))
    svd_desc = (
        "Latent factor model mapping users and items into a shared <i>k = 20</i> dimensional embedding space. "
        "Because raw missing entries (0s) would falsely bias the decomposition towards low ratings, user-mean centering is essential."
    )
    story.append(Paragraph(svd_desc, body_style))
    story.append(Table([[Paragraph("Centered: R_c(u, i) = R(u, i) - μ_u  [for observed cells];  0  [for unobserved cells]<br/>Decomposition: R_c ≈ U_k · Σ_k · V_k^T    ==>    Score: R̂(u, i) = (U_k · Σ_k · V_k^T)(u, i) + μ_u", formula_style)]],
                       colWidths=[540], style=TableStyle([
                           ('BACKGROUND', (0,0), (-1,-1), CODE_BG),
                           ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
                           ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3)
                       ])))
    svd_params = (
        "• <b>μ<sub>u</sub></b>: Arithmetic mean of ratings provided by user <i>u</i> across their training history.<br/>"
        "• <b>Dimensionality Reduction</b>: <code>scipy.sparse.linalg.svds(centered, k=20)</code> extracts top-20 singular vectors.<br/>"
        "• <b>Reconstruction & Clipping</b>: Adding back <i>μ<sub>u</sub></i> restores personal rating scales. Predictions are clipped to [1.0, 5.0].<br/>"
        "• <b>Outcome</b>: The winning model. Achieves lowest RMSE (1.1022), highest Precision@10 (9.14%), highest Recall@10 (8.85%), "
        "and balanced Catalog Coverage (20.87%)."
    )
    story.append(Paragraph(svd_params, body_style))

    # Clean transition to Section 3 on Page 3
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: SECTION 3 MATHEMATICAL BREAKDOWN OF EVALUATION METRICS
    # =========================================================================
    story.append(Paragraph("3. Detailed Mathematical Breakdown of Evaluation Metrics", h1_style))
    story.append(Paragraph(
        "Evaluation strictly separates rating prediction accuracy from top-10 ranking utility. "
        "Below is the exact step-by-step mathematical definition of every metric.",
        body_style
    ))

    # 3.1 RMSE
    story.append(Paragraph("3.1 Root Mean Squared Error (RMSE)", h2_style))
    rmse_desc = (
        "<b>Definition:</b> Measures the square root of the average squared difference between predicted ratings "
        "and true ground-truth ratings in the <b>held-out test set only</b> (20,000 ratings across 943 users)."
    )
    story.append(Paragraph(rmse_desc, body_style))
    story.append(Table([[Paragraph("RMSE = sqrt( (1 / |D_test|) · Σ_{(u, i) ∈ D_test} ( y_{u, i} - ŷ_{u, i} )² )", formula_style)]],
                       colWidths=[540], style=TableStyle([
                           ('BACKGROUND', (0,0), (-1,-1), CODE_BG),
                           ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
                           ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3)
                       ])))
    rmse_work = (
        "• <b>Step 1:</b> For all <i>N = 20,000</i> rows in <code>test_df</code>, extract true rating <i>y<sub>u,i</sub></i> and coordinates <i>(u, i)</i>.<br/>"
        "• <b>Step 2:</b> Retrieve predicted score: <i>ŷ<sub>u,i</sub> = predict_matrix()[u, i]</i>.<br/>"
        "• <b>Step 3:</b> Residual: <i>e<sub>u,i</sub> = y<sub>u,i</sub> - ŷ<sub>u,i</sub></i>. Square each error: <i>e²<sub>u,i</sub></i> (quadratically penalizes large misses).<br/>"
        "• <b>Step 4:</b> Mean Squared Error: <i>MSE = (1 / N) Σ e²<sub>u,i</sub></i>. Root: <i>RMSE = sqrt(MSE)</i>.<br/>"
        "• <b>Interpretation:</b> Scale in stars [0.0 to 4.0]. Lower is better. Baseline Popularity gets 1.1254, while SVD reaches 1.1022."
    )
    story.append(Paragraph(rmse_work, body_style))
    story.append(Spacer(1, 4))

    # 3.2 Top-10 Ranking Methodology & Train-Masking
    story.append(Paragraph("3.2 Top-10 Candidate Generation & Train Item Masking (-inf)", h2_style))
    mask_desc = (
        "In practical recommender systems, recommending movies the user has already watched in training is unhelpful. "
        "Therefore, candidate generation strictly masks all training items:"
    )
    story.append(Paragraph(mask_desc, body_style))
    story.append(Table([[Paragraph("scores_u[i] = -inf  FOR ALL i ∈ Train_u    ==>    Top-10_u = argsort(scores_u)[-10:]", formula_style)]],
                       colWidths=[540], style=TableStyle([
                           ('BACKGROUND', (0,0), (-1,-1), CODE_BG),
                           ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
                           ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3)
                       ])))
    mask_detail = (
        "• <b>Mechanism:</b> For each user <i>u</i>, indices where <i>R<sub>train</sub>[u, i] > 0</i> are overwritten with <i>-∞</i>. "
        "<code>np.argpartition</code> followed by descending sort isolates the top 10 highest-scoring unseen items.<br/>"
        "• <b>Relevance Criterion:</b> An item in a user's test set is defined as <b>relevant</b> if and only if: <b>Test Rating ≥ 4.0 Stars</b>."
    )
    story.append(Paragraph(mask_detail, body_style))
    story.append(Spacer(1, 4))

    # 3.3 Precision@10
    story.append(Paragraph("3.3 Precision@10 (Recommendation Accuracy)", h2_style))
    p10_desc = "Proportion of recommended items in the Top-10 list that the user actually liked (rated ≥ 4.0 in their test set)."
    story.append(Paragraph(p10_desc, body_style))
    story.append(Table([[Paragraph("Precision@10(u) = Hits_u / 10 = | Top-10_u ∩ Relevant_u | / 10", formula_style)]],
                       colWidths=[540], style=TableStyle([
                           ('BACKGROUND', (0,0), (-1,-1), CODE_BG),
                           ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
                           ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3)
                       ])))
    p10_work = "• <b>Worked Example:</b> User 42 has 3 hits in their Top-10 list: <i>Precision@10(42) = 3 / 10 = 0.30 (30%)</i>."
    story.append(Paragraph(p10_work, body_style))
    story.append(Spacer(1, 4))

    # 3.4 Recall@10
    story.append(Paragraph("3.4 Recall@10 (Coverage of User Desires)", h2_style))
    r10_desc = "Proportion of all relevant items in the user's test set that were captured in their Top-10 list."
    story.append(Paragraph(r10_desc, body_style))
    story.append(Table([[Paragraph("Recall@10(u) = Hits_u / | Relevant_u | = | Top-10_u ∩ Relevant_u | / | Relevant_u |", formula_style)]],
                       colWidths=[540], style=TableStyle([
                           ('BACKGROUND', (0,0), (-1,-1), CODE_BG),
                           ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
                           ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3)
                       ])))
    r10_work = "• <b>Worked Example:</b> User 42 has 12 relevant items in test set (<i>|Relevant<sub>42</sub>| = 12</i>): <i>Recall@10(42) = 3 / 12 = 0.25 (25%)</i>."
    story.append(Paragraph(r10_work, body_style))
    story.append(Spacer(1, 4))

    # 3.5 The 907-User Cohort Rule
    story.append(Paragraph("3.5 The 907-User Cohort Rule (Division-by-Zero Prevention)", h2_style))
    rule_desc = (
        "<b>Vital Mathematical Rigor:</b> Out of 943 total users, <b>36 users have zero ratings ≥ 4.0 in their test set</b> "
        "(they either gave only 1–3 stars, or had few test items). For these users, <i>|Relevant<sub>u</sub>| = 0</i>. "
        "Evaluating Recall@10 on them results in <b>0 / 0 (undefined division by zero)</b>.<br/>"
        "• <b>The Rule:</b> We filter evaluation strictly to the <b>907 users</b> who have at least one relevant test item (<i>|Relevant<sub>u</sub>| ≥ 1</i>).<br/>"
        "• <b>Fairness Guarantee:</b> Both Precision@10 and Recall@10 are computed on this <b>exact same 907-user cohort</b>. "
        "This ensures statistical consistency and avoids skewing metrics."
    )
    story.append(Paragraph(rule_desc, body_style))
    story.append(Spacer(1, 4))

    # 3.6 Catalog Coverage
    story.append(Paragraph("3.6 Catalog Coverage (%) (Ecosystem Diversity)", h2_style))
    cov_desc = "Percentage of distinct catalog items recommended to at least one user in their Top-10 list."
    story.append(Paragraph(cov_desc, body_style))
    story.append(Table([[Paragraph("Catalog Coverage (%) = [ | ∪_{u=1}^U Top-10_u | / | Total Catalog | ] × 100 = [ | Unique Recommended Items | / 1,682 ] × 100", formula_style)]],
                       colWidths=[540], style=TableStyle([
                           ('BACKGROUND', (0,0), (-1,-1), CODE_BG),
                           ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
                           ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3)
                       ])))

    # Clean transition to Section 4 on Page 4
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: RESULTS, OUTLIERS, DYNAMIC DEMO & CHEAT SHEET
    # =========================================================================
    story.append(Paragraph("4. Benchmark Experimental Results & Empirical Insights", h1_style))
    story.append(Paragraph(
        "The complete experimental pipeline was executed under a fixed random seed. "
        "Below is the exact benchmark table produced across all 943 users and 20,000 test ratings:",
        body_style
    ))

    bench_data = [
        [Paragraph("Model Algorithm", table_header_style),
         Paragraph("RMSE (↓)", table_header_style),
         Paragraph("Precision@10 (↑)", table_header_style),
         Paragraph("Recall@10 (↑)", table_header_style),
         Paragraph("Coverage (↑)", table_header_style),
         Paragraph("Overall Architectural Character", table_header_style)],

        [Paragraph("<b>PopularityBaseline</b><br/>(Bayesian Shrinkage)", table_cell_style),
         Paragraph("1.1254", table_cell_bold),
         Paragraph("0.0607 (6.07%)", table_cell_style),
         Paragraph("0.0532 (5.32%)", table_cell_style),
         Paragraph("3.86% (65)", table_cell_style),
         Paragraph("Recommends identical blockbuster movies. Decent hit-rate due to popularity bias, but zero personalization.", table_cell_style)],

        [Paragraph("<b>ItemKNN</b><br/>(Cosine Top-30)", table_cell_style),
         Paragraph("1.1057", table_cell_bold),
         Paragraph("0.0056 (0.56%)", table_cell_style),
         Paragraph("0.0060 (0.60%)", table_cell_style),
         Paragraph("<b>80.50%</b> (1,354)", table_cell_bold),
         Paragraph("Massive catalog exploration. Recommends niche items, but heavily penalized on sparse temporal test sets.", table_cell_style)],

        [Paragraph("<b>SVDRecommender</b><br/>(Truncated SVD k=20)", table_cell_style),
         Paragraph("<b>1.1022</b>", table_cell_bold),
         Paragraph("<b>0.0914</b> (9.14%)", table_cell_bold),
         Paragraph("<b>0.0885</b> (8.85%)", table_cell_bold),
         Paragraph("20.87% (351)", table_cell_style),
         Paragraph("<b>Best Overall Model.</b> Lowest rating error, highest precision and recall, healthy balance of diversity.", table_cell_style)],
    ]

    bench_table = Table(bench_data, colWidths=[110, 55, 80, 75, 75, 145])
    bench_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(bench_table)
    story.append(Spacer(1, 6))

    takeaways = (
        "<b>Three Core Examiner Insights:</b><br/>"
        "1. <b>The Popularity Bias Trap:</b> Popularity achieves 6.07% precision because blockbusters dominate user histories. But its 3.86% coverage makes it commercially sterile.<br/>"
        "2. <b>The Coverage vs. Relevance Trade-off in ItemKNN:</b> ItemKNN activates 80.5% of the catalog, but because users in the test set only rated a tiny fraction of movies, niche recommendations rarely register as 'hits', deflating Precision to 0.56%.<br/>"
        "3. <b>The Latent Factor Advantage of SVD:</b> SVD wins across all quantitative metrics (RMSE 1.1022, Precision 9.14%, Recall 8.85%). "
        "Decomposing the rating matrix into 20 latent concepts captures deep user preferences while maintaining solid diversity."
    )
    story.append(Paragraph(takeaways, body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("5. Data Integrity, Outliers & Examiner Live Demo", h1_style))
    iqr_text = (
        "<b>Outlier Handling (<code>src/preprocessing.py</code> -> <code>detect_outliers()</code>):</b><br/>"
        "• Uses Interquartile Range (IQR) on user rating counts: <i>IQR = Q3 - Q1</i>. Users with ratings > <i>Q3 + 1.5 · IQR</i> are flagged.<br/>"
        "• <b>Key Talking Point:</b> We <b>DO NOT delete outlier rows</b>. In recommender systems, hyper-active power users and critical 1-star reviewers "
        "represent genuine user behavior; deleting them corrupts collaborative filtering signals. We audit and log them for transparency."
    )
    story.append(Paragraph(iqr_text, body_style))
    story.append(Spacer(1, 4))

    # Dynamic Live Demo Walkthrough
    demo_box = (
        "<b>LIVE EXAMINER DEMONSTRATION INSTRUCTIONS:</b><br/>"
        "If the examiner asks: <i>'What happens if I delete a row of data right now?'</i>, follow these 3 steps:<br/>"
        "<b>1. Open raw file:</b> Open <code>data/ml-100k/u.data</code> and delete row 1.<br/>"
        "<b>2. Streamlit UI Alert:</b> Switch to the Streamlit dashboard (<code>http://localhost:8501</code>). The app computes the SHA-256 hash, "
        "detects the mismatch, and displays: <i>'Raw data file u.data changed on disk (hash mismatch). Click Recompute Pipeline.'</i><br/>"
        "Click the <b>'Recompute Pipeline Now'</b> button — the entire pipeline re-trains live in ~3 seconds and updates all charts!<br/>"
        "<b>3. CLI Runner Dynamic Mode:</b> Run <code>python run_experiments.py</code> in terminal. "
        "Stage 7 logs: <code>[DYNAMIC MODE] Baseline metrics differ because raw dataset was modified</code>, computes the exact delta table, "
        "and updates the baseline safely without throwing false assertion errors."
    )
    story.append(Table([[Paragraph(demo_box, body_style)]],
                       colWidths=[540], style=TableStyle([
                           ('BACKGROUND', (0,0), (-1,-1), BG_CALLOUT),
                           ('BOX', (0,0), (-1,-1), 1, SECONDARY),
                           ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                           ('LEFTPADDING', (0,0), (-1,-1), 8), ('RIGHTPADDING', (0,0), (-1,-1), 8)
                       ])))
    story.append(Spacer(1, 6))

    # Reference table
    story.append(Paragraph("6. Project Commands & Quick Reference", h1_style))
    ref_data = [
        [Paragraph("Task", table_header_style), Paragraph("Command", table_header_style), Paragraph("Expected Result", table_header_style)],
        [Paragraph("Run Full Pipeline", table_cell_bold), Paragraph("<code>python run_experiments.py</code>", table_cell_style), Paragraph("Executes 7 stages in order, prints metrics, saves artifacts.", table_cell_style)],
        [Paragraph("Launch Streamlit App", table_cell_bold), Paragraph("<code>streamlit run app.py</code>", table_cell_style), Paragraph("Launches interactive UI on <code>http://localhost:8501</code>.", table_cell_style)],
        [Paragraph("Run Unit Tests", table_cell_bold), Paragraph("<code>pytest -v</code>", table_cell_style), Paragraph("Runs 7 comprehensive tests in <code>tests/test_pipeline.py</code>.", table_cell_style)],
        [Paragraph("GitHub Repo", table_cell_bold), Paragraph("<code>git status && git log -1</code>", table_cell_style), Paragraph("Synced with <code>https://github.com/ragerock780/Da-project</code>.", table_cell_style)],
    ]
    ref_table = Table(ref_data, colWidths=[110, 180, 250])
    ref_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5), ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(ref_table)

    # Build the PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated: {output_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(current_dir, PDF_FILENAME)
    create_pdf(target_path)
