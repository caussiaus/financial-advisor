# IPS Config‑Space Toolkit – **Advisor README**

> *A practical guide for portfolio managers who want to turn "What‑if" life planning conversations into quantitative cash‑flow & investment strategy maps.*

---

## 1  What this toolkit does

- **Enumerates** every meaningful configuration of a household's life choices (education paths, work decisions, gifting style, risk appetite, etc.).
- **Projects** a 40‑year cash‑flow timeline for *each* configuration.
- **Flags** infeasible scenarios (logical conflicts or funding shortfalls).
- **Ranks** viable configurations on metrics you choose (probability of success, CVaR, quality‑of‑life score).
- **Exports** clean CSV / Excel files you can hand your client—or slice further in Power BI.

---

## 2  Toolkit components

| Folder/File            | What it is                                                                                | You touch it?                |
| ---------------------- | ----------------------------------------------------------------------------------------- | ---------------------------- |
| `setup_env.sh`         | One‑liner that creates/activates a Python **venv** on macOS and installs pandas/openpyxl. | **Run once** per machine.    |
| `ips_model.py`         | Main script – builds factor grid, calculates cash flows, writes outputs.                  | Edit parameters & rules.     |
| `configurations.csv`   | One row per permutation + headline metrics.                                               | Filter/sort for insight.     |
| `cashflows_CFG_##.csv` | 40‑year waterfall for that specific config.                                               | Feed into dashboards.        |
| `ips_output.xlsx`      | Same as above, Excel format.                                                              | Client‑friendly deliverable. |

---

## 3  End‑to‑end workflow (15‑minute sprint)

### STEP 1  Client discovery

Ask the household **what choices are on the table** *today* (and a few plausible tomorrow's). Example prompts:

- "Public university vs U.S. private?"
- "Do you see yourself going full‑time next year?"
- "Lump‑sum charity now or annual pledge?"

Record each choice as a **single factor with clear states**. *If two decisions are inseparable, merge them into one factor.*

### STEP 2  Encode the factor space

Open `** → **` and type the state lists.\
*(Everything else updates automatically.)*

### STEP 3  Run the model

```bash
source ~/ips_env/bin/activate   # or your venv path
python ips_model.py              # <10 s for ~600 configs
```

Outputs drop into `ips_output/`.

### STEP 4  Review feasibility & core metrics

Open \`\`.  Columns you'll see out‑of‑the‑box:

- `PV_10yr_Surplus` – aggregate wiggle‑room in first decade
- `Mean_Annual_CF`   – lifestyle affordability proxy
- (†) *Your own Monte‑Carlo or funded‑ratio columns once you bolt them in*

Red‑flagged configs (negative surplus, logical conflicts) are already filtered out.

### STEP 5  Layer a quality‑of‑life (QoL) score

Financial metrics often miss *felt* satisfaction.  Options:

| Approach                | How to implement                                                             | When to use                                  |
| ----------------------- | ---------------------------------------------------------------------------- | -------------------------------------------- |
| **Hard‑code weights**   | In `ips_model.py`, add a dict like `QoL = 0.4*Lifestyle + 0.3*Education + …` | When client gives explicit priorities.       |
| **Post‑hoc discussion** | Print top‑N configs, mark pros/cons live with the client.                    | Great for exploratory planning meetings.     |
| **Fuzzy‑set QCA**       | Convert metrics to 0‑1 membership scores and minimise conditions.            | When you need academic‑style causal insight. |

*(Tip → Leave the QoL layer as a separate function so you can swap in different weight sets without re‑running cash‑flows.)*

### STEP 6  Select the corresponding investment strategy

Every configuration carries a `RISK_BAND`.  Map that to a model portfolio (lookup table in \`\`).

> **If life changes**—e.g. daughter accepts Johns Hopkins—**flip the ****\`\`**** toggle**, re‑run, and the optimal weight set for *that* reality pops out.

---

## 4  Handling mutually‑exclusive or infeasible combos

- Static conflicts live in \`\` – one line per business rule.\
  Example: if `DON_STYLE = 2` (gift now) you *cannot* also set `MOM_GIFT_USE = 1` (invest gift).
- Dynamic insolvency is caught after the first deterministic cash‑flow run – tweak the `passes_financial_tests()` threshold.

The script skips infeasible rows; they won't clutter your output.

---

## 5  Updating after life events

1. Adjust the relevant factor state (or add a new factor).
2. Re‑run the script – new outputs appear in seconds.
3. Compare the *delta* row‑by‑row or paste multiple `configurations.csv` files into a Power Query folder import.

---

## 6  Common questions

**Q:** *Do I need to model quality‑of‑life inside Python?*\
**A:** Not mandatory. Many advisors prefer using the numeric outputs as talking points and layer subjective weighting *with* the client during the review.

**Q:** *We want stochastic returns & funded‑ratio.*\
**A:** Plug your Monte‑Carlo engine in `simulate_portfolio()` and append the CVaR / success‑prob columns before writing the configuration table.

**Q:** *Can I change time‑horizon or discounting?*\
**A:** Update `YEARS` and the discount factor line in `passes_financial_tests()`.

---

## 7  Hardcoded Parameters & Limitations

### ⚠️  **Current Hardcoded Scores**

The enhanced Monte Carlo analysis contains **numerous hardcoded parameters** that should be configurable:

#### **Monte Carlo Economic Assumptions**
- Equity returns: 8% ± 16% (mean ± std)
- Bond returns: 4% ± 8%
- Recession probability: 15% annually
- Income/expense volatility: 10%/8%

#### **Quality of Life Scoring Weights**
- Financial security: 35%
- Income stability: 25%
- Lifestyle quality: 20%
- Generosity fulfillment: 10%
- Cushion comfort: 10%

#### **Financial Stress Ranking**
- Shortfall probability: 40% weight
- Cash flow volatility: 30% weight
- Insolvency probability: 30% weight

#### **FSQCA Fuzzy Membership Thresholds**
- "High" bonus threshold: 20%
- Risk tolerance mapping: 1.0/0.5/0.0
- Various lifestyle scoring adjustments

**💡 Future Enhancement:** Move these to `ips_config.json` for client‑specific calibration.

---

## 8  What's Missing: Portfolio Management Perspective

The current toolkit provides **excellent cash‑flow stress analysis** but lacks core **portfolio management functionality**:

### 🎯 **Missing: Actual Portfolio Optimization**

**Current State:** Uses static `RISK_BAND` lookup table
**Missing:**
- **Mean‑variance optimization** for each risk band
- **Black‑Litterman** or other return forecasting
- **Dynamic rebalancing** strategies
- **Tax‑loss harvesting** opportunities
- **Asset‑liability matching** (duration/timing)

### 📊 **Missing: Return‑Risk Trade‑off Analysis**

**Current:** Only cash‑flow stress, no portfolio performance modeling
**Missing:**
- **Monte Carlo portfolio returns** integrated with cash flows
- **Efficient frontier** for each life configuration
- **Probability of meeting goals** given portfolio allocation
- **CVaR/VaR** of portfolio outcomes, not just cash flows
- **Funded ratio** analysis (assets vs. liability PV)

### ⚡ **Missing: Dynamic Strategy Framework**

**Current:** Static 40‑year projections
**Missing:**
- **Glide path optimization** (age‑based allocation changes)
- **Trigger‑based rebalancing** (market conditions, life events)
- **Options strategies** for downside protection
- **Tactical allocation** overlays
- **De‑risking schedules** approaching major expenses

### 💰 **Missing: Tax Optimization**

**Current:** Pre‑tax cash flow analysis only
**Missing:**
- **Asset location** (tax‑advantaged vs. taxable accounts)
- **Tax‑efficient fund selection** within asset classes
- **Roth conversion** optimization timing
- **Municipal vs. corporate bonds** for tax brackets
- **Capital gains management** strategies

### 🔄 **Missing: Liquidity Management**

**Current:** Binary cash‑flow shortfall analysis
**Missing:**
- **Cash flow timing optimization** (when to sell what)
- **Liquidity ladder** construction
- **Emergency fund sizing** integrated with portfolio
- **Credit line utilization** vs. portfolio liquidation
- **Sequence of returns** risk mitigation

### 📈 **Missing: Performance Attribution**

**Current:** Stress ranking only
**Missing:**
- **Factor attribution** (market, style, alpha sources)
- **Cost attribution** (fees, taxes, trading costs)
- **Decision attribution** (allocation vs. selection vs. timing)
- **Behavioral attribution** (client behavior impact)

### 🎛️ **Missing: Risk Budgeting**

**Current:** Single risk tolerance score
**Missing:**
- **Risk allocation** across factors (equity, credit, duration, FX)
- **Concentration limits** by asset, sector, geography
- **Correlation stress testing** (what if diversification fails?)
- **Tail risk budgeting** (allocation to hedge strategies)

### 🌱 **Missing: ESG & Investment Constraints**

**Current:** No investment preferences
**Missing:**
- **ESG scoring** integration with allocations
- **Values‑based screening** (negative/positive screens)
- **Impact investing** allocation targets
- **Regulatory constraints** (pension fund rules, etc.)
- **Behavioral constraints** (loss aversion, home bias)

---

## 9  Next Implementation Priority

### 🔧 **Phase 1: Portfolio Integration**
1. **Portfolio simulation engine** → Integrate actual returns with cash flows
2. **Efficient frontier generator** → For each life configuration
3. **Monte Carlo portfolio performance** → Integrate with existing stress analysis

### 🔧 **Phase 2: Dynamic Strategies**
4. **Dynamic rebalancing** → Age/market‑based allocation shifts
5. **Tax‑aware optimization** → After‑tax wealth maximization
6. **Liquidity‑constrained optimization** → When/what to sell

### 🔧 **Phase 3: Advanced Features**
7. **REST API** wrapper so junior advisors can run configs via web form
8. **Power‑BI template** already wired to the output folder
9. **Optimizer hook** (cvxpy or `Solver`) → for each config, solve for *minimum‑risk weights* that still hit required return

---

### 🚀 Ready to play?

1. Clone the repo / drop the two files in a folder.
2. `bash setup_env.sh`
3. Edit two dictionaries (`FACTOR_SPACE`, `PARAM`).
4. `python ips_model.py`
5. Open `ips_output/ips_output.xlsx`.
6. Book your client review!  🎉

**⚠️ Note:** Current analysis provides **cash‑flow stress insights** but requires portfolio management integration for complete investment advisory solution.

