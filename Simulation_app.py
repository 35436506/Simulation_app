import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

st.set_page_config(
    page_title="Ch12 · Monte Carlo Simulation",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; }
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] input[type="text"] { color: #111827 !important; background: #f8fafc !important; }
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stTextInput input { color: #111827 !important; }

.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-label { font-size: 12px; color: #94a3b8; margin-bottom: 4px; }
.metric-value { font-size: 26px; font-weight: 700; color: #38bdf8; }
.metric-sub   { font-size: 12px; color: #64748b; }

.section-header {
    background: linear-gradient(90deg,#0ea5e9,#6366f1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 22px; font-weight: 700; margin-bottom: 4px;
}
.info-box {
    background: #1e293b; border-left: 3px solid #0ea5e9;
    border-radius: 6px; padding: .75rem 1rem;
    font-size: 13px; color: #94a3b8; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ───────────────────────────────────────────────────────
MODULES = {
    "🏠 Giới thiệu": "home",
    "🏥 Health Insurance": "health",
    "✈️ Airline Overbooking": "airline",
    "📦 Inventory Control": "inventory",
    "📂 Project Selection": "project",
    "📈 Portfolio Optimization": "portfolio",
    "🏘️ Real Estate NPV": "realestate",
    "🎰 RNG Playground": "rng",
}

with st.sidebar:
    st.markdown("## 🎲 Ch12 Simulation")
    st.markdown("---")
    choice = st.radio("", list(MODULES.keys()), label_visibility="collapsed")
    page = MODULES[choice]
    st.markdown("---")
    st.caption("Spreadsheet Modeling & Decision Analysis · Chapter 12")

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def metric_card(label, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def plot_histogram(data, title, xlabel, color="#38bdf8", percentile_lines=None, target=None):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data, nbinsx=50, name="Frequency",
        marker_color=color, opacity=0.8,
        histnorm="probability density",
    ))
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
    fig.add_trace(go.Scatter(
        x=x, y=stats.norm.pdf(x, mu, sigma),
        name="Normal fit", line=dict(color="#f59e0b", width=2, dash="dash")
    ))
    if percentile_lines:
        # Stagger annotation y positions to avoid overlap
        y_offsets = [0.98, 0.82, 0.66]
        for i, (p, label) in enumerate(percentile_lines):
            val = np.percentile(data, p)
            yref_pos = y_offsets[i] if i < len(y_offsets) else 0.98 - i * 0.16
            fig.add_vline(
                x=val, line_width=1.5, line_dash="dot", line_color="#a78bfa",
                annotation=dict(
                    text=f"P{p}: {val:,.0f}",
                    yref="paper", y=yref_pos,
                    font=dict(size=11, color="#a78bfa"),
                    bgcolor="rgba(15,23,42,0.7)",
                    borderpad=3,
                    xanchor="left",
                )
            )
    if target is not None:
        fig.add_vline(x=target, line_width=2, line_color="#ef4444",
                      annotation=dict(
                          text=f"Target: {target:,.0f}",
                          yref="paper", y=0.98,
                          font=dict(size=11, color="#ef4444"),
                          bgcolor="rgba(15,23,42,0.7)",
                          borderpad=3,
                          xanchor="right",
                      ))
    fig.update_layout(
        title=dict(text=title, y=0.97, yanchor="top"),
        xaxis_title=xlabel, yaxis_title="Density",
        template="plotly_dark", height=400,
        margin=dict(l=40, r=40, t=60, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
    )
    return fig


def plot_cdf(data, title, xlabel, target=None):
    sorted_d = np.sort(data)
    cdf = np.arange(1, len(sorted_d)+1) / len(sorted_d)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sorted_d, y=cdf*100, name="CDF",
                             line=dict(color="#38bdf8", width=2)))
    if target is not None:
        prob = np.mean(data <= target) * 100
        fig.add_vline(x=target, line_width=1.5, line_dash="dot", line_color="#ef4444",
                      annotation_text=f"P(≤{target:,.0f})={prob:.1f}%")
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="Cumulative Probability (%)",
                      template="plotly_dark", height=350, margin=dict(l=40, r=40, t=50, b=40))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════
if page == "home":
    st.markdown('<div class="section-header">Monte Carlo Simulation · Chapter 12</div>', unsafe_allow_html=True)
    st.markdown("**Spreadsheet Modeling & Decision Analysis – Ragsdale**")
    st.markdown("---")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("### Tại sao mô phỏng?")
        st.markdown("""
Khi đầu vào của mô hình **không chắc chắn** (chi phí, nhu cầu, lãi suất…),
chỉ chạy một kịch bản "trung bình" là chưa đủ.

Monte Carlo Simulation giúp ta hiểu **toàn bộ phân phối** của kết quả —
xác suất thua lỗ, 90th percentile, VaR…
""")
    with cols[1]:
        st.markdown("### Quy trình 6 bước")
        st.markdown("""
1. Xây mô hình bảng tính cơ sở
2. Xác định biến ngẫu nhiên (RNG)
3. Đánh dấu ô đầu ra
4. Chọn số lần lặp (replications)
5. Chạy mô phỏng
6. Phân tích phân phối đầu ra
""")
    with cols[2]:
        st.markdown("### 5 bài toán trong app")
        st.markdown("""
| Module | Kỹ thuật |
|--------|----------|
| 🏥 Health Insurance | Normal RNG, VaR |
| ✈️ Airline Overbooking | Discrete RNG, Binomial |
| 📦 Inventory Control | (s,Q) policy optimization |
| 📂 Project Selection | Binary + simulation |
| 📈 Portfolio | Correlation matrix |
""")
    st.markdown("---")
    st.info("👈 Chọn module từ sidebar để bắt đầu.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: HEALTH INSURANCE  (Fig 12-2, 12-6, 12-17)
# ═══════════════════════════════════════════════════════════════════════════
elif page == "health":
    st.markdown('<div class="section-header">🏥 Corporate Health Insurance</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Dự báo tổng chi phí bảo hiểm y tế mà công ty phải trả trong 12 tháng. '
                'Hai biến ngẫu nhiên: tốc độ tăng số nhân viên (Uniform) và mức tăng chi phí bồi thường (Normal).</div>',
                unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Parameters")
        n_employees = st.number_input("Số NV ban đầu", value=18533, step=100)
        avg_claim   = st.number_input("Chi phí BT trung bình/NV ($)", value=250, step=10)
        contribution = st.number_input("Đóng góp NV/tháng ($)", value=125, step=5)
        st.markdown("**Growth assumptions:**")
        emp_min_change = st.slider("Decrease tối đa NV/tháng", -0.10, 0.0, -0.03, 0.005, format="%.3f")
        emp_max_change = st.slider("Increase tối đa NV/tháng", 0.0, 0.15, 0.07, 0.005, format="%.3f")
        claim_mean_growth = st.slider("Tăng trưởng chi phí BT/tháng (mean)", 0.0, 0.05, 0.01, 0.001, format="%.3f")
        claim_std_growth  = st.slider("Std dev tăng trưởng chi phí", 0.0, 0.02, 0.003, 0.001, format="%.3f")
        n_sims = st.selectbox("Số lần mô phỏng", [1000, 2000, 5000, 10000], index=1)
        target_cost = st.number_input("Target (VaR threshold) $", value=39_000_000, step=500_000)
        run = st.button("▶ Chạy mô phỏng", use_container_width=True)

    if run or "health_results" not in st.session_state:
        rng = np.random.default_rng(42)
        total_costs = []
        for _ in range(n_sims):
            emp = n_employees
            claim = avg_claim
            total = 0.0
            for _ in range(12):
                emp_change = rng.uniform(emp_min_change, emp_max_change)
                emp = max(1, emp * (1 + emp_change))
                claim_change = rng.normal(claim_mean_growth, claim_std_growth)
                claim = max(0, claim * (1 + claim_change))
                total_claims = emp * claim
                total += max(0, total_claims - contribution * emp)
            total_costs.append(total)
        st.session_state.health_results = np.array(total_costs)

    data = st.session_state.health_results
    prob_exceed = np.mean(data > target_cost) * 100
    p10, p50, p90 = np.percentile(data, [10, 50, 90])

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Mean cost", f"${np.mean(data)/1e6:.2f}M")
    with c2: metric_card("Std Dev", f"${np.std(data)/1e6:.2f}M")
    with c3: metric_card("P(> Target)", f"{prob_exceed:.1f}%", f"Target: ${target_cost/1e6:.0f}M")
    with c4: metric_card("90th percentile", f"${p90/1e6:.2f}M")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_histogram(
            data, "Phân phối tổng chi phí công ty (12 tháng)",
            "Total Company Cost ($)",
            percentile_lines=[(10, "P10"), (50, "P50"), (90, "P90")],
            target=target_cost
        ), use_container_width=True)
    with col2:
        st.plotly_chart(plot_cdf(data, "CDF – Xác suất tích lũy",
                                 "Total Company Cost ($)", target=target_cost),
                        use_container_width=True)

    with st.expander("📊 Confidence Interval cho Mean"):
        n = len(data)
        se = np.std(data, ddof=1) / np.sqrt(n)
        ci_lo, ci_hi = stats.t.interval(0.95, df=n-1, loc=np.mean(data), scale=se)
        st.markdown(f"""
| Thống kê | Giá trị |
|----------|---------|
| Sample mean | ${np.mean(data):,.0f} |
| Sample std dev | ${np.std(data, ddof=1):,.0f} |
| 95% CI Lower | ${ci_lo:,.0f} |
| 95% CI Upper | ${ci_hi:,.0f} |
| Margin of error | ${(ci_hi - ci_lo)/2:,.0f} |
""")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: AIRLINE OVERBOOKING  (Fig 12-18)
# ═══════════════════════════════════════════════════════════════════════════
elif page == "airline":
    st.markdown('<div class="section-header">✈️ Airline Overbooking Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Tìm số lượng đặt chỗ tối ưu. Nhu cầu là biến rời rạc (Discrete RNG), '
                'số khách show-up theo phân phối Binomial.</div>', unsafe_allow_html=True)

    DEMAND_DIST = {14: 0.03, 15: 0.05, 16: 0.07, 17: 0.09, 18: 0.11,
                   19: 0.15, 20: 0.18, 21: 0.14, 22: 0.08, 23: 0.05, 24: 0.03, 25: 0.02}

    with st.sidebar:
        st.markdown("### ⚙️ Parameters")
        seats = st.number_input("Ghế trên máy bay", value=19, min_value=5, max_value=100)
        ticket_price = st.number_input("Giá vé ($)", value=150, step=10)
        prob_noshow = st.slider("Xác suất no-show", 0.0, 0.5, 0.10, 0.01)
        bump_cost = st.number_input("Chi phí bù đắp khách bị đẩy ($)", value=325, step=25)
        res_min = st.number_input("Đặt chỗ tối thiểu thử", value=int(seats), min_value=int(seats))
        res_max = st.number_input("Đặt chỗ tối đa thử", value=int(seats)+6, min_value=int(seats))
        n_sims = st.selectbox("Số lần mô phỏng", [1000, 2000, 5000], index=1)
        run = st.button("▶ Chạy mô phỏng", use_container_width=True)

    demand_values = list(DEMAND_DIST.keys())
    demand_probs = list(DEMAND_DIST.values())

    def simulate_overbooking(reservations, n_sims, rng):
        profits = []
        for _ in range(n_sims):
            demand = rng.choice(demand_values, p=demand_probs)
            tickets_sold = min(demand, reservations)
            showups = rng.binomial(tickets_sold, 1 - prob_noshow)
            bumped = max(0, showups - seats)
            revenue = tickets_sold * ticket_price
            cost = bumped * bump_cost
            profits.append(revenue - cost)
        return np.array(profits)

    if run or "airline_results" not in st.session_state:
        rng = np.random.default_rng(42)
        results = {}
        for r in range(int(res_min), int(res_max)+1):
            profits = simulate_overbooking(r, n_sims, rng)
            results[r] = {"mean": np.mean(profits), "p20": np.percentile(profits, 20),
                          "p80": np.percentile(profits, 80), "min": np.min(profits),
                          "max": np.max(profits), "data": profits}
        st.session_state.airline_results = results

    results = st.session_state.airline_results
    summary = pd.DataFrame({r: {"Mean Profit": v["mean"], "P20": v["p20"],
                                 "P80": v["p80"], "Min": v["min"], "Max": v["max"]}
                             for r, v in results.items()}).T.reset_index()
    summary.columns = ["Reservations", "Mean Profit", "P20", "P80", "Min", "Max"]
    best_r = int(summary.loc[summary["Mean Profit"].idxmax(), "Reservations"])

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Optimal Reservations", str(best_r), "max mean profit")
    with c2: metric_card("Mean Profit (optimal)", f"${results[best_r]['mean']:,.0f}")
    with c3: metric_card("Seats available", str(seats), f"no-show={prob_noshow:.0%}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=summary["Reservations"], y=summary["Mean Profit"],
                                 name="Mean", line=dict(color="#38bdf8", width=2), mode="lines+markers"))
        fig.add_trace(go.Scatter(x=summary["Reservations"], y=summary["P80"],
                                 name="80th pct", line=dict(color="#a78bfa", dash="dash")))
        fig.add_trace(go.Scatter(x=summary["Reservations"], y=summary["P20"],
                                 name="20th pct", line=dict(color="#f87171", dash="dash"),
                                 fill="tonexty", fillcolor="rgba(167,139,250,0.1)"))
        fig.add_vline(x=best_r, line_dash="dot", line_color="#fbbf24",
                      annotation_text=f"Optimal={best_r}")
        fig.update_layout(title="Profit vs. Reservations Accepted",
                          xaxis_title="Reservations", yaxis_title="Profit ($)",
                          template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        opt_data = results[best_r]["data"]
        st.plotly_chart(plot_histogram(opt_data, f"Phân phối lợi nhuận (R={best_r})",
                                       "Profit ($)", color="#a78bfa"), use_container_width=True)

    st.dataframe(summary.style.format({c: "${:,.0f}" for c in
                                        ["Mean Profit","P20","P80","Min","Max"]}), use_container_width=True)

    with st.expander("📊 Xem phân phối nhu cầu gốc (từ sách)"):
        df_demand = pd.DataFrame({"Demand": list(DEMAND_DIST.keys()),
                                   "Probability": list(DEMAND_DIST.values())})
        fig2 = px.bar(df_demand, x="Demand", y="Probability", template="plotly_dark",
                      color_discrete_sequence=["#38bdf8"], title="Discrete Demand Distribution")
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: INVENTORY CONTROL  (Fig 12-23, 12-27, 12-36, 12-38)
# ═══════════════════════════════════════════════════════════════════════════
elif page == "inventory":
    st.markdown('<div class="section-header">📦 Inventory Control Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">(s, Q) policy: đặt hàng Q đơn vị khi tồn kho ≤ s (reorder point). '
                'Nhu cầu hàng ngày và lead time đều là biến ngẫu nhiên rời rạc.</div>', unsafe_allow_html=True)

    DEMAND_DIST_INV = {0:0.01,1:0.02,2:0.04,3:0.06,4:0.09,5:0.14,
                       6:0.18,7:0.22,8:0.16,9:0.06,10:0.02}
    LEAD_DIST = {3:0.2, 4:0.6, 5:0.2}

    with st.sidebar:
        st.markdown("### ⚙️ Parameters")
        init_inv = st.number_input("Tồn kho ban đầu", value=50, step=5)
        reorder_pt = st.slider("Reorder Point (s)", 1, 80, 28)
        order_qty = st.slider("Order Quantity (Q)", 5, 200, 50, step=5)
        n_days = st.number_input("Số ngày mô phỏng", value=30, min_value=10, max_value=365)
        holding_cost = st.number_input("Chi phí tồn kho/đơn vị/ngày ($)", value=0.1, step=0.05, format="%.2f")
        order_cost = st.number_input("Chi phí mỗi lần đặt hàng ($)", value=50.0, step=10.0)
        shortage_cost = st.number_input("Chi phí thiếu hàng/đơn vị ($)", value=5.0, step=0.5)
        n_sims = st.selectbox("Số lần mô phỏng", [500, 1000, 2000, 5000], index=2)
        st.markdown("---")
        st.markdown("**Parameterized scan:**")
        do_scan = st.checkbox("So sánh nhiều (s, Q)")
        if do_scan:
            s_list_str = st.text_input("Các reorder point", "20,25,28,30,35,40")
            q_list_str = st.text_input("Các order qty", "40,50,60")
        run = st.button("▶ Chạy mô phỏng", use_container_width=True)

    demand_vals = list(DEMAND_DIST_INV.keys())
    demand_probs_inv = list(DEMAND_DIST_INV.values())
    lead_vals = list(LEAD_DIST.keys())
    lead_probs = list(LEAD_DIST.values())

    def simulate_inventory(s, Q, n_days, n_sims, init_inv, rng):
        service_levels, avg_invs, total_costs = [], [], []
        for _ in range(n_sims):
            inv = init_inv
            inv_pos = init_inv
            orders = {}  # day_arrive -> qty
            total_demand = 0
            satisfied = 0
            inv_sum = 0
            n_orders = 0
            shortage_total = 0
            for day in range(1, int(n_days)+1):
                received = orders.pop(day, 0)
                inv += received
                inv_pos = inv + sum(orders.values())
                demand_d = rng.choice(demand_vals, p=demand_probs_inv)
                sat_d = min(inv, demand_d)
                short_d = demand_d - sat_d
                inv -= sat_d
                total_demand += demand_d
                satisfied += sat_d
                inv_sum += inv
                shortage_total += short_d
                if inv_pos <= s:
                    lead = rng.choice(lead_vals, p=lead_probs)
                    arrive_day = day + lead
                    orders[arrive_day] = orders.get(arrive_day, 0) + Q
                    inv_pos += Q
                    n_orders += 1
            service_levels.append(satisfied / total_demand if total_demand > 0 else 1.0)
            avg_invs.append(inv_sum / n_days)
            total_cost = (holding_cost * inv_sum + order_cost * n_orders +
                          shortage_cost * shortage_total)
            total_costs.append(total_cost)
        return (np.array(service_levels), np.array(avg_invs), np.array(total_costs))

    if run or "inv_results" not in st.session_state:
        rng = np.random.default_rng(42)
        sl, ai, tc = simulate_inventory(reorder_pt, order_qty, n_days, n_sims, init_inv, rng)
        st.session_state.inv_results = (sl, ai, tc)
        if "do_scan" in st.session_state and st.session_state.do_scan:
            scan_res = []
            s_list = [int(x) for x in s_list_str.split(",")]
            q_list = [int(x) for x in q_list_str.split(",")]
            rng2 = np.random.default_rng(42)
            for sv in s_list:
                for qv in q_list:
                    sl2, ai2, tc2 = simulate_inventory(sv, qv, n_days, n_sims, init_inv, rng2)
                    scan_res.append({"s": sv, "Q": qv, "Avg SL": np.mean(sl2),
                                     "Avg Inv": np.mean(ai2), "Avg Cost": np.mean(tc2)})
            st.session_state.inv_scan = pd.DataFrame(scan_res)

    sl, ai, tc = st.session_state.inv_results
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Service Level", f"{np.mean(sl):.1%}", "avg across sims")
    with c2: metric_card("Avg Inventory", f"{np.mean(ai):.1f}", "units/day")
    with c3: metric_card("Avg Total Cost", f"${np.mean(tc):,.0f}")
    with c4: metric_card("P(SL ≥ 95%)", f"{np.mean(sl >= 0.95):.1%}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_histogram(tc, "Phân phối tổng chi phí tồn kho",
                                       "Total Cost ($)", color="#34d399"), use_container_width=True)
    with col2:
        fig_sl = go.Figure()
        fig_sl.add_trace(go.Histogram(x=sl*100, nbinsx=30, marker_color="#f59e0b",
                                       opacity=0.8, name="Service Level",
                                       histnorm="probability density"))
        fig_sl.add_vline(x=95, line_dash="dot", line_color="#ef4444",
                         annotation_text="95% target")
        fig_sl.update_layout(title="Phân phối Service Level", xaxis_title="Service Level (%)",
                              template="plotly_dark", height=380)
        st.plotly_chart(fig_sl, use_container_width=True)

    if "inv_scan" in st.session_state:
        st.markdown("### 📊 Parameterized Scan: So sánh (s, Q)")
        scan_df = st.session_state.inv_scan
        st.dataframe(scan_df.style.format({"Avg SL": "{:.1%}", "Avg Inv": "{:.1f}",
                                            "Avg Cost": "${:,.0f}"}), use_container_width=True)
        fig_scan = px.scatter(scan_df, x="Avg SL", y="Avg Cost", color="Q", size="Avg Inv",
                              text="s", template="plotly_dark",
                              title="Trade-off: Service Level vs Cost",
                              labels={"Avg SL": "Service Level", "Avg Cost": "Total Cost ($)"})
        fig_scan.update_traces(textposition="top center")
        st.plotly_chart(fig_scan, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: PROJECT SELECTION  (Fig 12-39)
# ═══════════════════════════════════════════════════════════════════════════
elif page == "project":
    st.markdown('<div class="section-header">📂 Project Selection Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Chọn tập hợp dự án (0/1) trong giới hạn ngân sách để tối đa hoá lợi nhuận. '
                'Revenue mỗi dự án theo phân phối Triangular, thành công theo Binomial.</div>', unsafe_allow_html=True)

    DEFAULT_PROJECTS = pd.DataFrame({
        "Project": [1, 2, 3, 4, 5, 6, 7, 8],
        "Investment": [250, 650, 250, 500, 700, 30, 350, 70],
        "P_Success": [0.9, 0.7, 0.6, 0.4, 0.8, 0.6, 0.7, 0.9],
        "Rev_Min": [600, 1250, 500, 1600, 1150, 150, 750, 220],
        "Rev_MLikely": [750, 1500, 600, 1800, 1200, 180, 900, 250],
        "Rev_Max": [900, 1600, 750, 1900, 1400, 250, 1000, 320],
        "Selected": [True, True, False, True, False, True, True, True],
    })

    with st.sidebar:
        st.markdown("### ⚙️ Parameters")
        budget = st.number_input("Ngân sách tổng ($K)", value=2000, step=100)
        n_sims = st.selectbox("Số lần mô phỏng", [1000, 2000, 5000], index=1)
        run = st.button("▶ Chạy mô phỏng", use_container_width=True)

    st.markdown("### Chọn dự án (có thể chỉnh)")
    edited = st.data_editor(DEFAULT_PROJECTS, use_container_width=True,
                             column_config={
                                 "Selected": st.column_config.CheckboxColumn("Chọn?"),
                                 "P_Success": st.column_config.NumberColumn("P(Success)", format="%.2f"),
                             })
    selected = edited[edited["Selected"]]
    total_inv = selected["Investment"].sum()

    col1, col2 = st.columns(2)
    with col1: metric_card("Tổng đầu tư", f"${total_inv:,}K", f"Budget: ${budget:,}K")
    with col2: metric_card("Số dự án chọn", str(len(selected)))

    if total_inv > budget:
        st.error(f"⚠️ Tổng đầu tư ${total_inv}K vượt ngân sách ${budget}K!")

    if run or "proj_results" not in st.session_state:
        rng = np.random.default_rng(42)
        profits_all = []
        for _ in range(n_sims):
            total_profit = 0
            for _, row in selected.iterrows():
                success = rng.random() < row["P_Success"]
                if success:
                    lo, mid, hi = row["Rev_Min"], row["Rev_MLikely"], row["Rev_Max"]
                    rev = rng.triangular(lo, mid, hi)
                    total_profit += rev - row["Investment"]
                else:
                    total_profit -= row["Investment"]
            profits_all.append(total_profit)
        st.session_state.proj_results = np.array(profits_all)

    data = st.session_state.proj_results
    p_loss = np.mean(data < 0) * 100
    p_lt1000 = np.mean(data < 1000) * 100

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Mean Profit", f"${np.mean(data):,.0f}K")
    with c2: metric_card("P(Loss)", f"{p_loss:.1f}%")
    with c3: metric_card("P(< $1000K)", f"{p_lt1000:.1f}%")
    with c4: metric_card("90th Percentile", f"${np.percentile(data, 90):,.0f}K")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_histogram(data, "Phân phối lợi nhuận danh mục dự án",
                                       "Total Profit ($K)", color="#f59e0b",
                                       target=0), use_container_width=True)
    with col2:
        st.plotly_chart(plot_cdf(data, "CDF Lợi nhuận", "Total Profit ($K)", target=0),
                        use_container_width=True)

    with st.expander("📊 Phân tích từng dự án (sensitivity)"):
        proj_stats = []
        rng2 = np.random.default_rng(42)
        for _, row in selected.iterrows():
            profits = []
            for _ in range(n_sims):
                success = rng2.random() < row["P_Success"]
                if success:
                    rev = rng2.triangular(row["Rev_Min"], row["Rev_MLikely"], row["Rev_Max"])
                    profits.append(rev - row["Investment"])
                else:
                    profits.append(-row["Investment"])
            proj_stats.append({"Project": int(row["Project"]), "Mean Profit": np.mean(profits),
                                "P(Loss)": np.mean(np.array(profits) < 0)})
        df_ps = pd.DataFrame(proj_stats)
        fig_ps = px.bar(df_ps, x="Project", y="Mean Profit", color="P(Loss)",
                        color_continuous_scale="RdYlGn_r", template="plotly_dark",
                        title="Expected Profit by Project")
        st.plotly_chart(fig_ps, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: PORTFOLIO OPTIMIZATION  (Fig 12-43)
# ═══════════════════════════════════════════════════════════════════════════
elif page == "portfolio":
    st.markdown('<div class="section-header">📈 Portfolio Optimization & Efficient Frontier</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Phân bổ $1B vào 5 loại nhà máy điện. '
                'Return từng loại có tương quan (correlation matrix). '
                'Xây dựng efficient frontier: max E[return] với ràng buộc σ ≤ σ_max.</div>',
                unsafe_allow_html=True)

    ASSETS = ["Gas", "Coal", "Oil", "Nuclear", "Wind"]
    MEANS  = [0.16, 0.12, 0.10, 0.09, 0.08]
    STDS   = [0.12, 0.06, 0.04, 0.03, 0.01]
    CORR   = np.array([
        [1.00, -0.49, -0.31,  0.16,  0.12],
        [-0.49,  1.00, -0.41,  0.11,  0.07],
        [-0.31, -0.41,  1.00,  0.13,  0.09],
        [0.16,  0.11,  0.13,  1.00,  0.04],
        [0.12,  0.07,  0.09,  0.04,  1.00],
    ])
    MW_PER_M = [2.0, 1.2, 3.5, 1.0, 0.5]

    with st.sidebar:
        st.markdown("### ⚙️ Allocation ($M)")
        allocs = []
        for a in ASSETS:
            allocs.append(st.slider(a, 0, 1000, 200, 50))
        total_alloc = sum(allocs)
        n_sims = st.selectbox("Số lần mô phỏng", [2000, 5000, 10000], index=0)
        n_frontier = st.slider("Số điểm efficient frontier", 10, 50, 20)
        run = st.button("▶ Chạy mô phỏng", use_container_width=True)
        st.markdown(f"**Total invested: ${total_alloc}M** {'✅' if total_alloc==1000 else '⚠️ (target $1000M)'}")

    # Cholesky decomposition for correlated returns
    cov = np.outer(STDS, STDS) * CORR
    L = np.linalg.cholesky(cov)

    weights = np.array(allocs) / 1000.0

    if run or "port_results" not in st.session_state:
        rng = np.random.default_rng(42)
        z = rng.standard_normal((n_sims, 5))
        corr_returns = z @ L.T + np.array(MEANS)
        weighted_returns = (corr_returns * weights).sum(axis=1)
        st.session_state.port_results = (corr_returns, weighted_returns)

    corr_returns, weighted_returns = st.session_state.port_results
    port_mean = np.mean(weighted_returns)
    port_std  = np.std(weighted_returns)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Mean Return", f"{port_mean:.2%}")
    with c2: metric_card("Std Dev (Risk)", f"{port_std:.2%}")
    with c3: metric_card("Sharpe-like", f"{port_mean/port_std:.2f}", "(return/risk)")
    with c4: metric_card("P(Return < 0)", f"{np.mean(weighted_returns < 0):.1%}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_histogram(weighted_returns * 100,
                                       "Phân phối Return danh mục", "Return (%)",
                                       color="#818cf8", target=0), use_container_width=True)
    with col2:
        # Efficient frontier (Monte Carlo random weights)
        rng2 = np.random.default_rng(99)
        ef_means, ef_stds = [], []
        for _ in range(5000):
            w = rng2.dirichlet(np.ones(5))
            z2 = rng2.standard_normal((500, 5))
            ret = (z2 @ L.T + np.array(MEANS)) @ w
            ef_means.append(np.mean(ret))
            ef_stds.append(np.std(ret))
        w_current = [np.mean(corr_returns[:,i]) for i in range(5)]
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(x=ef_stds, y=ef_means, mode="markers",
                                    marker=dict(size=3, color="#334155", opacity=0.5), name="Random portfolios"))
        fig_ef.add_trace(go.Scatter(x=[port_std], y=[port_mean], mode="markers",
                                    marker=dict(size=14, color="#f59e0b", symbol="star"),
                                    name="Current portfolio"))
        fig_ef.update_layout(title="Efficient Frontier", xaxis_title="Risk (Std Dev)",
                              yaxis_title="Expected Return", template="plotly_dark", height=380)
        st.plotly_chart(fig_ef, use_container_width=True)

    # Asset contribution chart
    asset_df = pd.DataFrame({"Asset": ASSETS, "Weight": weights*100,
                              "Mean Return": [m*100 for m in MEANS], "Std Dev": [s*100 for s in STDS]})
    fig_bar = px.bar(asset_df, x="Asset", y="Weight", color="Mean Return",
                     color_continuous_scale="Blues", template="plotly_dark",
                     title="Portfolio Allocation (%)", text="Weight")
    fig_bar.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("📊 Correlation Matrix"):
        df_corr = pd.DataFrame(CORR, index=ASSETS, columns=ASSETS)
        fig_hm = px.imshow(df_corr, template="plotly_dark", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, title="Asset Return Correlations")
        st.plotly_chart(fig_hm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: REAL ESTATE NPV  (Fig 12-47, 12-48)
# ═══════════════════════════════════════════════════════════════════════════
elif page == "realestate":
    st.markdown('<div class="section-header">🏘️ Real Estate Investment NPV</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Tính NPV của bất động sản cho thuê trong 5 năm. '
                'Các biến bất định: tốc độ tăng doanh thu, chi phí vận hành, giá trị tài sản tăng.</div>',
                unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Acquisition")
        land_val    = st.number_input("Land value ($)", value=30000, step=1000)
        building    = st.number_input("Buildings ($)", value=140000, step=1000)
        down        = st.number_input("Down payment ($)", value=40000, step=1000)
        apr         = st.slider("APR (%)", 5.0, 20.0, 11.0, 0.5) / 100
        term        = st.slider("Loan term (years)", 5, 30, 20)
        st.markdown("### 📊 Economic Assumptions")
        gross_rent  = st.number_input("Annual gross rent ($)", value=35000, step=500)
        rent_growth_mean = st.slider("Rent growth rate (mean %)", 0.0, 15.0, 4.0, 0.5) / 100
        rent_growth_std  = st.slider("Rent growth rate (std %)", 0.0, 5.0, 1.0, 0.25) / 100
        vc_rate     = st.slider("Vacancy & Credit rate (%)", 0.0, 10.0, 3.0, 0.5) / 100
        opex_rate   = st.slider("Operating expense rate (%)", 30.0, 60.0, 45.0, 1.0) / 100
        tax_rate    = st.slider("Tax rate (%)", 15.0, 40.0, 28.0, 1.0) / 100
        prop_growth_mean = st.slider("Property growth (mean %)", 0.0, 10.0, 2.5, 0.5) / 100
        prop_growth_std  = st.slider("Property growth (std %)", 0.0, 5.0, 1.5, 0.25) / 100
        discount    = st.slider("Discount rate (%)", 5.0, 20.0, 12.0, 0.5) / 100
        sale_comm   = st.slider("Sales commission (%)", 3.0, 8.0, 5.0, 0.5) / 100
        n_sims      = st.selectbox("Số lần mô phỏng", [1000, 2000, 5000], index=1)
        n_years     = 5
        run = st.button("▶ Chạy mô phỏng", use_container_width=True)

    purchase = land_val + building
    loan = purchase - down
    annual_payment = loan * apr / (1 - (1+apr)**(-term))

    def simulate_realestate(n_sims, rng):
        npvs = []
        tax_basis_land = land_val
        depreciable = building
        dep_life = 27.5
        annual_dep = depreciable / dep_life
        for _ in range(n_sims):
            rent = gross_rent
            prop_val = purchase
            cf_list = []
            remaining_balance = loan
            for yr in range(1, n_years+1):
                rent_gr = rng.normal(rent_growth_mean, rent_growth_std)
                if yr > 1:
                    rent *= (1 + rent_gr)
                vc = rent * vc_rate
                opex = rent * opex_rate
                noi = rent - vc - opex
                interest = remaining_balance * apr
                principal = annual_payment - interest
                remaining_balance -= principal
                taxable = noi - annual_dep - interest
                taxes = taxable * tax_rate if taxable > 0 else taxable * tax_rate
                ncf = noi - taxes - principal
                cf_list.append(ncf)
            # Sale at end of year 5
            prop_gr = rng.normal(prop_growth_mean, prop_growth_std)
            for _ in range(n_years):
                prop_val *= (1 + prop_gr)
            sale_exp = prop_val * sale_comm
            tax_basis = tax_basis_land + depreciable - annual_dep * n_years
            taxable_gain = prop_val - sale_exp - tax_basis
            gain_tax = taxable_gain * tax_rate if taxable_gain > 0 else 0
            net_sale = prop_val - sale_exp - gain_tax - remaining_balance
            # NPV
            cf_list[-1] += net_sale
            npv = -down + sum(cf / (1+discount)**t for t, cf in enumerate(cf_list, 1))
            npvs.append(npv)
        return np.array(npvs)

    if run or "re_results" not in st.session_state:
        rng = np.random.default_rng(42)
        st.session_state.re_results = simulate_realestate(n_sims, rng)

    data = st.session_state.re_results
    p_positive = np.mean(data > 0) * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Mean NPV", f"${np.mean(data):,.0f}")
    with c2: metric_card("P(NPV > 0)", f"{p_positive:.1f}%")
    with c3: metric_card("Annual payment", f"${annual_payment:,.0f}")
    with c4: metric_card("10th Percentile NPV", f"${np.percentile(data,10):,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_histogram(data, "Phân phối NPV bất động sản",
                                       "NPV ($)", color="#10b981", target=0), use_container_width=True)
    with col2:
        st.plotly_chart(plot_cdf(data, "CDF – NPV", "NPV ($)", target=0), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: RNG PLAYGROUND
# ═══════════════════════════════════════════════════════════════════════════
elif page == "rng":
    st.markdown('<div class="section-header">🎰 RNG Playground – Thám hiểm các phân phối</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Khám phá 6 phân phối xác suất phổ biến trong Analytic Solver. '
                'Tương đương các hàm PsiNormal, PsiUniform, PsiTriangular, PsiDiscrete, PsiBinomial, PsiLognormal.</div>',
                unsafe_allow_html=True)

    dist_choice = st.selectbox("Chọn phân phối", [
        "Normal (PsiNormal)", "Uniform (PsiUniform)", "Triangular (PsiTriangular)",
        "Discrete (PsiDiscrete)", "Binomial (PsiBinomial)", "Lognormal"
    ])

    col_params, col_chart = st.columns([1, 2])
    n_sims = 10000
    rng = np.random.default_rng(99)

    with col_params:
        st.markdown("#### Parameters")
        if dist_choice.startswith("Normal"):
            mu = st.number_input("Mean (μ)", value=0.0)
            sigma = st.number_input("Std Dev (σ)", value=1.0, min_value=0.01)
            samples = rng.normal(mu, sigma, n_sims)
            st.code(f"=PsiNormal({mu}, {sigma})", language="excel")
        elif dist_choice.startswith("Uniform"):
            lo = st.number_input("Min (a)", value=0.0)
            hi = st.number_input("Max (b)", value=1.0)
            if hi <= lo: hi = lo + 1
            samples = rng.uniform(lo, hi, n_sims)
            st.code(f"=PsiUniform({lo}, {hi})", language="excel")
        elif dist_choice.startswith("Triangular"):
            lo = st.number_input("Min (a)", value=0.0)
            mid = st.number_input("Most Likely (c)", value=5.0)
            hi = st.number_input("Max (b)", value=10.0)
            if not (lo <= mid <= hi): mid = (lo+hi)/2
            samples = rng.triangular(lo, mid, hi, n_sims)
            st.code(f"=PsiTriangular({lo}, {mid}, {hi})", language="excel")
        elif dist_choice.startswith("Discrete"):
            st.markdown("Nhập giá trị và xác suất (cách nhau bởi dấu phẩy):")
            vals_str = st.text_input("Values", "10,20,30")
            probs_str = st.text_input("Probabilities", "0.3,0.5,0.2")
            try:
                vals = [float(x) for x in vals_str.split(",")]
                probs_d = [float(x) for x in probs_str.split(",")]
                s = sum(probs_d)
                probs_d = [p/s for p in probs_d]
                samples = rng.choice(vals, p=probs_d, size=n_sims)
                st.code(f"=PsiDiscrete({{{vals_str}}},{{{probs_str}}})", language="excel")
            except:
                st.error("Lỗi nhập liệu"); samples = np.zeros(n_sims)
        elif dist_choice.startswith("Binomial"):
            n_trials = st.number_input("n (trials)", value=20, min_value=1)
            p = st.slider("p (success prob)", 0.0, 1.0, 0.3, 0.01)
            samples = rng.binomial(int(n_trials), p, n_sims)
            st.code(f"=PsiBinomial({int(n_trials)}, {p})", language="excel")
        else:
            mu_ln = st.number_input("μ (log-scale)", value=0.0)
            sigma_ln = st.number_input("σ (log-scale)", value=0.5, min_value=0.01)
            samples = rng.lognormal(mu_ln, sigma_ln, n_sims)
            st.code(f"=PsiLognormal({mu_ln}, {sigma_ln})", language="excel")

        st.markdown("---")
        st.markdown(f"**Mean:** {np.mean(samples):.3f}")
        st.markdown(f"**Std Dev:** {np.std(samples):.3f}")
        st.markdown(f"**Min / Max:** {np.min(samples):.3f} / {np.max(samples):.3f}")
        st.markdown(f"**P10 / P50 / P90:** {np.percentile(samples,10):.3f} / {np.percentile(samples,50):.3f} / {np.percentile(samples,90):.3f}")

    with col_chart:
        nbins = 20 if dist_choice.startswith("Discrete") or dist_choice.startswith("Binomial") else 60
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=samples, nbinsx=nbins, marker_color="#818cf8",
                                   opacity=0.85, histnorm="probability density", name="Sample"))
        if not dist_choice.startswith("Discrete") and not dist_choice.startswith("Binomial"):
            mu2, sig2 = np.mean(samples), np.std(samples)
            xr = np.linspace(np.min(samples), np.max(samples), 300)
            fig.add_trace(go.Scatter(x=xr, y=stats.norm.pdf(xr, mu2, sig2),
                                     line=dict(color="#fbbf24", dash="dash"), name="Normal fit"))
        fig.update_layout(title=f"{dist_choice} – {n_sims:,} samples",
                          template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

        fig_box = go.Figure()
        fig_box.add_trace(go.Box(x=samples, name=dist_choice, marker_color="#38bdf8",
                                  boxmean=True))
        fig_box.update_layout(template="plotly_dark", height=160, margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig_box, use_container_width=True)
