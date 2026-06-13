"""Generate docs/report.pdf — BIU DRL Ex03 submission report."""

from __future__ import annotations

import pickle
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ------------------------------------------------------------------
# Load saved training results
# ------------------------------------------------------------------
with open("results/training_results.pkl", "rb") as fh:
    R = pickle.load(fh)

rf_first = sum(R["rf_returns"][:50]) / 50
rf_last = sum(R["rf_returns"][-50:]) / 50
a2c_first = sum(R["a2c_returns"][:50]) / 50
a2c_last = sum(R["a2c_returns"][-50:]) / 50
# Absolute return gain (first-50 → last-50). Percentage improvement is avoided
# because the early-training return is near zero/negative under the variety-
# penalised reward, which makes ratio-based percentages meaningless.
rf_gain = rf_last - rf_first
a2c_gain = a2c_last - a2c_first
lstm_epochs = len(R["lstm_train_losses"])

# Variance (std of episodic return, first vs last 100 episodes) — computed live
rf_std0, rf_std1 = R["rf_std"]
a2c_std0, a2c_std1 = R["a2c_std"]
rf_var_red = (rf_std0 - rf_std1) / rf_std0 * 100
a2c_var_red = (a2c_std0 - a2c_std1) / a2c_std0 * 100

# Learned-policy action distributions over a greedy 28-day episode
rf_dist = R["rf_dist"]
a2c_dist = R["a2c_dist"]
rf_distinct = R["rf_distinct"]
a2c_distinct = R["a2c_distinct"]
a2c_used = ", ".join(f"{k} ({v})" for k, v in a2c_dist.items() if v > 0)
rf_used = ", ".join(f"{k} ({v})" for k, v in rf_dist.items() if v > 0)

# ------------------------------------------------------------------
# Document setup
# ------------------------------------------------------------------
OUT = Path("docs/report.pdf")
doc = SimpleDocTemplate(
    str(OUT),
    pagesize=A4,
    leftMargin=2.5 * cm,
    rightMargin=2.5 * cm,
    topMargin=2.5 * cm,
    bottomMargin=2.5 * cm,
    title="BIU DRL Ex03 — Fitness RL Report",
    author="Hodaya Kashkash",
)

styles = getSampleStyleSheet()

H1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontSize=16, spaceAfter=8, textColor=colors.HexColor("#1a3a5c")
)
H2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontSize=13, spaceAfter=6, textColor=colors.HexColor("#2b5797")
)
H3 = ParagraphStyle(
    "H3", parent=styles["Heading3"], fontSize=11, spaceAfter=4, textColor=colors.HexColor("#4472c4")
)
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=15, spaceAfter=8)
CAPTION = ParagraphStyle(
    "Caption",
    parent=styles["Normal"],
    fontSize=8,
    textColor=colors.grey,
    alignment=1,
    spaceAfter=12,
)
TITLE_STYLE = ParagraphStyle(
    "Title", parent=styles["Title"], fontSize=22, textColor=colors.HexColor("#1a3a5c"), alignment=1
)
SUBTITLE = ParagraphStyle(
    "Sub", parent=styles["Normal"], fontSize=12, alignment=1, textColor=colors.HexColor("#555555")
)

PLOTS = Path("results/plots")


def fig(name: str, width: float = 14 * cm) -> Image:
    p = PLOTS / name
    img = Image(str(p))
    img.drawWidth = width
    img.drawHeight = width * 0.5
    return img


def hr() -> HRFlowable:
    return HRFlowable(
        width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=8, spaceBefore=4
    )


# ------------------------------------------------------------------
# Story
# ------------------------------------------------------------------
story: list = []

# --- Title page ---
story += [
    Spacer(1, 3 * cm),
    Paragraph("BIU Deep Reinforcement Learning", SUBTITLE),
    Spacer(1, 0.4 * cm),
    Paragraph("Exercise 03", TITLE_STYLE),
    Spacer(1, 0.3 * cm),
    Paragraph("LSTM + REINFORCE + A2C for Personal Fitness Planning", TITLE_STYLE),
    Spacer(1, 1.2 * cm),
    Paragraph("Hodaya Kashkash", SUBTITLE),
    Paragraph("Bar-Ilan University — Dr. Yoram Segal", SUBTITLE),
    Spacer(1, 0.8 * cm),
    hr(),
    Spacer(1, 0.5 * cm),
    Paragraph(
        "<b>Abstract.</b> This report presents a reinforcement-learning pipeline for "
        "personal workout planning. An LSTM transition model (val MSE = "
        f"{R['lstm_val_losses'][-1]:.4f}) is trained on {R['n_days']} days of real "
        "Hevy workout-log data to serve as a learned environment dynamics model. "
        "Two policy-gradient algorithms — REINFORCE and Advantage Actor-Critic (A2C) "
        "— are evaluated over 500 episodes each. A2C achieves a higher final average "
        f"return ({a2c_last:.2f} vs {rf_last:.2f}) and converges more reliably, "
        "consistent with its lower-variance TD advantage.",
        BODY,
    ),
    PageBreak(),
]

# --- 1. Dataset ---
story += [
    Paragraph("1. Dataset and Representation", H1),
    hr(),
    Paragraph(
        "The Hevy App Workout Dataset (Kaggle: "
        "<i>tejaswinimukesh/hevy-app-workout-dataset-from-dumbbells-to-data</i>) "
        "provides 100 weeks of real exercise-log data covering 7,882 exercise-set records "
        f"across {R['n_days']} distinct workout days. Each row contains: exercise name, "
        "muscle group, sets index, reps, weight (lbs), and session timestamps.",
        BODY,
    ),
    Paragraph(
        "<b>Feature engineering.</b> Raw exercise rows are aggregated to daily "
        "workout summaries. The state vector <i>s<sub>t</sub></i> (dim=5) is:",
        BODY,
    ),
    Table(
        [
            ["Index", "Feature", "Encoding"],
            ["0", "rolling_7day_load", "MinMaxScaler → [0, 1]"],
            ["1", "muscle_balance_score", "Shannon entropy of muscle distribution"],
            ["2", "session_duration_avg", "MinMaxScaler → [0, 1]"],
            ["3", "day_sin", "sin(2pi * t / 28)"],
            ["4", "day_cos", "cos(2pi * t / 28)"],
        ],
        colWidths=[1.5 * cm, 5 * cm, 8 * cm],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b5797")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4fa"), colors.white]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        ),
    ),
    Spacer(1, 0.4 * cm),
    Paragraph(
        "Sinusoidal encoding for <i>day_in_cycle</i> is essential: a scalar fraction "
        "<i>t/27</i> treats day 0 and day 27 as maximally distant, breaking the "
        "cyclical structure. sin/cos place day 0 and day 28 at the same point on the "
        "unit circle. The MinMaxScaler is fitted exclusively on the training split "
        "(80%) and applied without refitting to validation — preventing data leakage.",
        BODY,
    ),
    Paragraph(
        "<b>Action space.</b> K-Means (k=6) clusters daily summaries into six "
        "workout archetypes: Rest, Upper Push, Upper Pull, Lower Body, Full Body, "
        "and Core/Cardio. This converts the continuous workout space into a "
        "discrete MDP action set.",
        BODY,
    ),
]

# --- 2. LSTM ---
story += [
    Spacer(1, 0.5 * cm),
    Paragraph("2. LSTM Transition Model", H1),
    hr(),
    Paragraph(
        "The LSTM serves as a learned world model: "
        "<i>s<sub>t+1</sub> = f(s<sub>t</sub>, a<sub>t</sub>, h<sub>t</sub>)</i>. "
        "It is trained with supervised MSE loss on consecutive 7-day windows "
        f"({R['n_train_windows']} train / {R['n_val_windows']} val), "
        f"using Adam (lr=0.001) for {lstm_epochs} epochs.",
        BODY,
    ),
    Paragraph(
        "This distinction is fundamental to the assignment: the LSTM answers "
        "'Given the recent training history and the next workout type, what trainee "
        "state is likely tomorrow?' It is a predictive model, not a decision-maker. "
        "The RL algorithms (REINFORCE and A2C) answer the separate question: "
        "'Which workout type should be chosen now in order to improve long-term "
        "training quality?' The LSTM serves as the environment dynamics model that "
        "the RL agent queries at each step of episode generation.",
        BODY,
    ),
    Paragraph(
        "<b>Architecture.</b> Embedding(6, 8) maps discrete actions to dense vectors, "
        "concatenated with the state at each timestep (input size = 13). Two LSTM "
        "layers (hidden=64, dropout=0.2) capture temporal dependencies; a linear "
        "head projects to state_dim=5.",
        BODY,
    ),
    fig("lstm_loss.png"),
    Paragraph(
        f"Figure 1. LSTM training and validation MSE loss over {lstm_epochs} epochs. "
        f"Train loss: {R['lstm_train_losses'][0]:.4f} → {R['lstm_train_losses'][-1]:.4f}. "
        f"Val loss: {R['lstm_val_losses'][0]:.4f} → {R['lstm_val_losses'][-1]:.4f}.",
        CAPTION,
    ),
    Paragraph(
        "Both curves decrease monotonically and converge to similar values. "
        "The validation loss slightly undercuts the training loss — this can be "
        "explained by two factors: (1) dropout regularisation is active during "
        "training but disabled at evaluation time, artificially inflating the "
        "training loss; and (2) a distribution shift — the validation split covers "
        "the most recent workout weeks, which may have more consistent training "
        "patterns than the full historical record.",
        BODY,
    ),
    PageBreak(),
]

# --- 3. REINFORCE ---
story += [
    Paragraph("3. REINFORCE", H1),
    hr(),
    Paragraph(
        "REINFORCE trains a stochastic policy <i>pi(a|s)</i> using episodic Monte Carlo returns:",
        BODY,
    ),
    Paragraph(
        "theta &larr; theta + alpha * sum_t [ grad log pi(a_t | s_t) * (G_t - mean(G)) ]",
        ParagraphStyle(
            "Math",
            parent=BODY,
            fontName="Courier",
            fontSize=9,
            leftIndent=1.5 * cm,
            textColor=colors.HexColor("#333333"),
        ),
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "Subtracting the episode mean <i>mean(G)</i> as a baseline reduces gradient "
        "variance without introducing bias. Episodes are 28 days long; "
        "500 episodes were trained with Adam (lr=3e-4).",
        BODY,
    ),
    fig("reinforce_return.png"),
    Paragraph(
        f"Figure 2. REINFORCE episodic return over 500 episodes. "
        f"First-50 average: {rf_first:.2f}. Last-50 average: {rf_last:.2f} "
        f"(absolute gain +{rf_gain:.2f}).",
        CAPTION,
    ),
    Paragraph(
        "The return trend is positive but noisy. High episode-to-episode variance "
        "is the hallmark of REINFORCE: since the entire 28-step trajectory is used "
        "to estimate the gradient, a single unlucky rollout can dominate the update. "
        "The rolling mean reveals a steady upward trend despite this variance.",
        BODY,
    ),
    PageBreak(),
]

# --- 4. A2C ---
story += [
    Paragraph("4. A2C — Advantage Actor-Critic", H1),
    hr(),
    Paragraph(
        "A2C replaces Monte Carlo returns with per-step TD advantages:",
        BODY,
    ),
    Paragraph(
        "delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)",
        ParagraphStyle(
            "Math",
            parent=BODY,
            fontName="Courier",
            fontSize=9,
            leftIndent=1.5 * cm,
            textColor=colors.HexColor("#333333"),
        ),
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "The Actor is updated with <i>delta_t.detach()</i> as the advantage signal; "
        "the Critic minimises <i>value_coeff * delta_t^2</i>. An entropy bonus "
        "(coeff=0.01) prevents premature action-space collapse. Separate Adam "
        "optimisers are used (actor lr=3e-4, critic lr=1e-3).",
        BODY,
    ),
    fig("a2c_return.png"),
    Paragraph(
        f"Figure 3. A2C episodic return over 500 episodes. "
        f"First-50 average: {a2c_first:.2f}. Last-50 average: {a2c_last:.2f} "
        f"(absolute gain +{a2c_gain:.2f}).",
        CAPTION,
    ),
    Paragraph(
        "A2C converges faster and more stably than REINFORCE. Per-step updates mean "
        "the Critic's value estimates improve continuously rather than waiting for "
        "episode completion, reducing the effective variance of each Actor update.",
        BODY,
    ),
    PageBreak(),
]

# --- 5. Comparison ---
story += [
    Paragraph("5. REINFORCE vs A2C Comparison", H1),
    hr(),
    fig("comparison.png"),
    Paragraph(
        "Figure 4. Rolling mean return (window=20) for REINFORCE and A2C on the same axes.",
        CAPTION,
    ),
    Table(
        [
            ["Metric", "REINFORCE", "A2C"],
            ["First-50 avg return", f"{rf_first:.2f}", f"{a2c_first:.2f}"],
            ["Last-50 avg return", f"{rf_last:.2f}", f"{a2c_last:.2f}"],
            ["Return gain (first→last 50)", f"+{rf_gain:.2f}", f"+{a2c_gain:.2f}"],
            ["Return std (first 100 eps)", f"{rf_std0:.2f}", f"{a2c_std0:.2f}"],
            ["Return std (last 100 eps)", f"{rf_std1:.2f}", f"{a2c_std1:.2f}"],
            ["Variance reduction", f"{rf_var_red:.0f}%", f"{a2c_var_red:.0f}%"],
            ["Distinct actions in episode", f"{rf_distinct} / 6", f"{a2c_distinct} / 6"],
            ["Update frequency", "Per episode", "Per step"],
            ["Variance reduction method", "Mean baseline", "TD advantage + Critic"],
            ["Extra parameters", "0", "CriticNetwork (~1k)"],
        ],
        colWidths=[6 * cm, 4.5 * cm, 4.5 * cm],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b5797")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#f0f4fa")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        ),
    ),
    Spacer(1, 0.4 * cm),
    Paragraph(
        f"A2C reduces return standard deviation by {a2c_var_red:.0f}% over training "
        f"(from {a2c_std0:.2f} to {a2c_std1:.2f}), compared to {rf_var_red:.0f}% for "
        f"REINFORCE (from {rf_std0:.2f} to {rf_std1:.2f}). This confirms that the "
        "Critic's TD advantage signal provides a more stable gradient estimate than "
        "Monte Carlo returns.",
        BODY,
    ),
    Paragraph(
        "A2C outperforms REINFORCE on both final return and stability. The key "
        "mechanism is the Critic's ability to assign credit at each step rather "
        "than propagating a single episode-level gradient backwards. This is "
        "especially beneficial in long episodes (T=28) where credit assignment "
        "over many steps accumulates variance for Monte Carlo methods.",
        BODY,
    ),
    PageBreak(),
]

# --- 6. Practical Interpretation (data-driven) ---
story += [
    Paragraph("6. Practical Interpretation — What Did the Policy Learn?", H1),
    hr(),
    Paragraph(
        "To answer the professor's central question — <i>did the policy avoid "
        "excessive concentration on one muscle group?</i> — we ran each trained "
        "policy greedily for one 28-day episode and counted how many days were "
        "assigned to each workout archetype.",
        BODY,
    ),
    fig("action_distribution.png"),
    Paragraph(
        "Figure 5. Action distribution of the learned REINFORCE and A2C policies "
        "over a greedy 28-day episode.",
        CAPTION,
    ),
    Paragraph(
        f"<b>Key finding.</b> A2C learned a substantially more diverse policy than "
        f"REINFORCE. Over the 28-day episode, A2C used <b>{a2c_distinct} of 6</b> "
        f"workout archetypes ({a2c_used}), whereas REINFORCE used only "
        f"<b>{rf_distinct} of 6</b> ({rf_used}). This is direct evidence that A2C's "
        "lower-variance gradient — combined with the entropy bonus and the "
        "action-variety reward term — escapes the single-action mode collapse that "
        "a naive reward design produces.",
        BODY,
    ),
    Paragraph(
        "<b>Reward design note.</b> An earlier reward formulation that derived the "
        "imbalance penalty solely from the LSTM-predicted state caused both policies "
        "to collapse onto a single action (the same archetype every day). The LSTM "
        "world model cannot reliably differentiate the muscle-group consequences of "
        "different actions, so the penalty never reacted to the agent's real choices. "
        "We fixed this by adding a repetition penalty (weight lambda_repetition, in "
        "config) computed from the Shannon entropy of the agent's OWN recent action "
        "history — a direct, model-independent incentive for variety.",
        BODY,
    ),
    PageBreak(),
    fig("episode_trajectory.png"),
    Paragraph(
        "Figure 6. A2C policy over one 28-day episode: chosen action per day (top) "
        "and the resulting normalised rolling load (bottom). The dashed line marks "
        "the optimal load (0.5) that the bell-shaped gain rewards.",
        CAPTION,
    ),
    Paragraph(
        "The trajectory shows the A2C agent interleaving higher-load training days "
        "with recovery/rest days rather than maximising volume every day — the "
        "alternating high-load / recovery pattern consistent with evidence-based "
        "periodisation. The rolling load oscillates around the optimal band rather "
        "than saturating at the overload threshold.",
        BODY,
    ),
    fig("state_analysis.png"),
    Paragraph(
        "Figure 7. Normalised rolling load over the same A2C-generated episode "
        "(state component 0), confirming the policy keeps cumulative load near the "
        "rewarded optimum rather than driving it to the overload region.",
        CAPTION,
    ),
]

# --- 7. Limitations ---
story += [
    Spacer(1, 0.5 * cm),
    Paragraph("7. Limitations and Future Improvements", H1),
    hr(),
    Paragraph("<b>Model limitations:</b>", H3),
    Paragraph(
        "The LSTM is trained on a single trainee's log. Out-of-distribution "
        "states (e.g., injury, travel) are not represented. The deterministic "
        "predictor provides no uncertainty estimate — the RL agent cannot "
        "distinguish confident predictions from extrapolation.",
        BODY,
    ),
    Paragraph(
        "The muscle_balance_score in the state vector captures global entropy but "
        "the LSTM action embedding carries no per-muscle information — it maps "
        "discrete cluster labels (0–5) to dense vectors without encoding which "
        "muscle groups each cluster targets. As a result, the imbalance_penalty in "
        "the reward function may not create the intended incentive for muscle-group "
        "variety: the LSTM cannot predict different muscle_balance outcomes for "
        "Upper Push vs Upper Pull actions.",
        BODY,
    ),
    Paragraph("<b>Reward design:</b>", H3),
    Paragraph(
        "The bell-shaped gain and linear penalties are heuristic. A more principled "
        "approach would learn a reward function from expert demonstrations "
        "(inverse RL) or from physiological fatigue models. The lambda parameters "
        "(0.4 and 0.3) were not systematically tuned.",
        BODY,
    ),
    Paragraph("<b>Action space:</b>", H3),
    Paragraph(
        "K-Means clustering of daily summaries loses intra-session structure "
        "(exercise selection, set ordering). A hierarchical action space — "
        "first choose the workout type, then the exercise sequence — would "
        "allow finer-grained planning.",
        BODY,
    ),
    Paragraph("<b>Algorithms:</b>", H3),
    Paragraph(
        "Both REINFORCE and A2C are on-policy and sample-inefficient. "
        "Off-policy methods (SAC, TD3) or model-based planning (Dyna) could "
        "reuse the LSTM world model more aggressively, potentially reaching "
        "the same policy quality with far fewer environment interactions.",
        BODY,
    ),
]

# --- 8. Conclusion ---
story += [
    Spacer(1, 0.5 * cm),
    Paragraph("8. Conclusion", H1),
    hr(),
    Paragraph(
        f"This project demonstrates a complete pipeline from raw workout-log data "
        f"to a trained RL policy for personalised fitness planning. The LSTM "
        f"transition model (val MSE = {R['lstm_val_losses'][-1]:.4f}) provides a "
        f"data-driven world model that replaces a hand-coded simulator. "
        f"A2C outperforms REINFORCE on every axis: a higher final return "
        f"({a2c_last:.2f} vs {rf_last:.2f}), a larger end-of-training variance "
        f"reduction ({a2c_var_red:.0f}% vs {rf_var_red:.0f}%), and — most importantly "
        f"for this task — a far more balanced workout policy that uses "
        f"{a2c_distinct} of 6 muscle archetypes versus only {rf_distinct} for "
        f"REINFORCE. This confirms that critic-reduced variance matters for episodic "
        f"fitness tasks with T=28 steps. The modular SDK architecture ensures every "
        f"result is reproducible by running a single command.",
        BODY,
    ),
]

# ------------------------------------------------------------------
# Build
# ------------------------------------------------------------------
doc.build(story)
print(f"Report written to {OUT}")
