# AfriScore
**Privacy-Preserving AI Credit & Portable Financial Identity Infrastructure for Africa**

## Hackathon Context 
- **Track**: Hackathon
- **Primary Challenge Area**: Financial Inclusion
- **Secondary**: Digital Transformation
- **Theme Alignment**: Powering Possibilities — unlocking economic potential for the underbanked through intelligent, ethical AI infrastructure.

AfriScore is building the missing **credit decisioning and identity layer** for Africa. We create user-controlled, explainable, privacy-first credit scores and financial identities using consented alternative data. This enables banks (like Wema/ALAT), fintechs, and lenders to responsibly serve youth, MSMEs, women, gig workers, and smallholders at scale.

### Why AfriScore Wins
- Solves a **foundational problem**: Only ~6-9% formal credit access in Nigeria/Africa. Massive economic unlock.
- **Direct value to Wema**: Real-time, explainable credit scoring for ALAT products, embedded lending, responsible expansion into thin-file segments.
- **Demo-ready in 3 days**: Compelling Gradio prototype showing scoring + explanations + lender workflow + Wema integration mock.
- **Global potential**: Infrastructure play with network effects and data moat. Attractive to top investors in emerging markets fintech/AI.

### Project Vision (Investor/Judge View)
AfriScore is not another lending app. It is **foundational infrastructure** — the "FICO + modern credit bureau layer" for the next billion users in emerging markets. Privacy-first, AI-native, Africa-built.

**Key Differentiators**:
- User sovereignty + consent-first design.
- Explainable AI (SHAP/feature importance) — critical for trust and regulation.
- Specialized models for African contexts (youth, informal economy, smallholder signals).
- API-first for easy integration with banks like Wema.
- Path to federated learning / on-device components for scale and privacy.

### MVP Scope for Hackatholics 7.0 (Uyo Pitch Centre)
**Goal**: A polished, live demo that judges can interact with and that clearly communicates the infrastructure vision.

**Core Features**:
1. **Synthetic Alt-Data Users** — Realistic profiles with mobile money, utility, education, behavioral, and (mock) satellite signals.
2. **Explainable Credit Scoring Engine** — Train simple but effective model (XGBoost). Output score (0-1000), risk tier, and top contributing factors with explanations.
3. **User-Facing Demo** — Simulate consent, view personal score + breakdown, "build my financial identity".
4. **Lender Dashboard Mock** — Batch scoring, approval simulation, risk distribution (what Wema would see).
5. **Wema Integration Simulation** — One-click "Connect to ALAT" that shows how scores power instant loan decisions or credit-building products.
6. **Pitch-Ready Narrative** — Clear problem → solution → impact metrics → Wema synergy → scalability.

**Tech Stack (Lightweight & Hackathon-Friendly)**:
- Python 3.10+
- pandas, scikit-learn or XGBoost, SHAP (for explanations)
- Gradio (beautiful, interactive web demo — perfect for judges)
- Optional: FastAPI for backend simulation
- Synthetic data generation (no real user data needed for prototype)

**Out of Scope for MVP**:
- Real production model training on live data
- Full federated learning / ZK proofs
- Mobile app or USSD
- Actual bank API integration

### How to Run the Demo (After Setup)
```bash
cd afri_score
pip install -r requirements.txt
python backend/generate_synthetic_data.py
python backend/train_model.py
gradio frontend/app.py
```

Then open the local Gradio URL.

### Project Structure
```
afri_score/
├── README.md
├── requirements.txt
├── backend/
│   ├── generate_synthetic_data.py
│   ├── train_model.py
│   └── scoring_engine.py
├── frontend/
│   └── app.py          # Gradio demo
├── data/
│   └── synthetic_users.csv
├── docs/
│   └── one_pager.md
└── pitch/
    └── pitch_deck_outline.md
```

### Next Immediate Steps (We Build Together)
1. Create requirements.txt and synthetic data generator.
2. Build the scoring model + SHAP explanations.
3. Create the full Gradio demo app.
4. Write the one-pager and pitch materials.
5. Polish visuals and prepare for Uyo presentation.

**This is infrastructure-grade thinking in a hackathon package.** Let's build something the judges remember and that serious capital would want to back.

Ready? I'll start generating the core files now. Tell me if you want to adjust the project name (current: AfriScore) or any scope details.
