# Scaling-Law DC Strategy | AI in Finance Assignment

This project implements and evaluates a **Directional-Change (DC) trading strategy** on tick-level financial data, with AI-driven enhancements for parameter optimization and robustness testing.  
The work was developed as part of the *AI in Finance Assignment*.

---

## 📌 Project Overview
- **Strategy**: Directional-Change (DC) sampling with overshoot scaling.  
- **Core Idea**:  
  - Open long on DC↑ confirmation.  
  - Close on DC↓ with adaptive threshold θ′ = θ × Y × exp(−maxOSV), with Y = 0.5.  
- **AI Enhancement**: Bayesian optimization and regime detection to improve parameter selection and execution quality.  
- **Deployment**: Back-tested with historical tick data and validated in a paper-trading environment (cTrader).
