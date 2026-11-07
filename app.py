import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Grind & Crumb - Ops Command Center", layout="wide")

st.title("☕ Grind & Crumb - Executive & Ops Command Center")
st.markdown("### Real-time visibility into store performance, delivery margins, and operational leaks.")

# Sidebar navigation
menu = st.sidebar.radio("Navigate Tool", ["Store Profitability & Leases", "Delivery Margin Calculator", "App v3.0 & Oat Milk Impact", "£1.5m Allocation Model"])

# Data setup based on hackathon findings
store_data = pd.DataFrame({
    "Store": ["York", "Sheffield Div St", "Northern Qtr", "Hull Paragon", "Trinity (Leeds)", "Piccadilly (Mcr)"],
    "Annual Sales (£)": [890000, 750000, 720000, 610000, 500000, 1100000],
    "Annual Profit (£)": [194000, 159000, 153000, 50000, 27000, -48000],
    "Margin %": [21.8, 21.2, 21.25, 8.2, 5.4, -4.36],
    "Rent & Rates (£)": [85000, 70000, 75000, 60000, 95000, 446000],
    "Status": ["Top Performer", "Healthy", "Healthy", "High Waste (20.6%)", "Cannibalising Briggate", "CRITICAL: Excessive Lease"]
})

if menu == "Store Profitability & Leases":
    st.header("🏪 Store-by-Store Health Dashboard")
    st.info("Manchester Piccadilly generates the highest sales (£1.1m) but loses £48k/yr due to a £446k rent/rates burden.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(store_data, x="Store", y="Annual Profit (£)", color="Margin %", title="Annual Profit by Store (£)")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        selected_store = st.selectbox("Select Store for Deep Dive:", store_data["Store"])
        details = store_data[store_data["Store"] == selected_store].iloc[0]
        st.metric("Annual Sales", f"£{details['Annual Sales (£)']:,}")
        st.metric("Net Profit", f"£{details['Annual Profit (£)']:,}")
        st.metric("Rent & Rates Burden", f"£{details['Rent & Rates (£)']:,}")
        st.warning(f"Status: {details['Status']}")

elif menu == "Delivery Margin Calculator":
    st.header("🚚 Delivery Platform Unit Economics Calculator")
    st.markdown("Calculate true profit per order after platform commissions, packaging, and **4 minutes of packing labor**.")
    
    col1, col2 = st.columns(2)
    with col1:
        order_val = st.number_input("Basket Value (£)", value=7.50, step=0.50)
        channel = st.selectbox("Channel", ["In-Store / Owned App", "Dashly (Delivery)", "RushEats (Delivery)"])
        hourly_wage = st.number_input("Hourly Staff Wage (£)", value=11.44)
        
    # Calculations based on brief rules
    labor_cost_per_order = (hourly_wage * 1.13) * (4 / 60) if "Delivery" in channel else 0.0
    comm_rate = 0.30 if "RushEats" in channel else (0.25 if "Dashly" in channel else 0.0)
    comm_fee = order_val * comm_rate
    packaging_cost = 0.45 if "Delivery" in channel else 0.10
    cogs = order_val * 0.25 # 25% food cost
    
    net_profit = order_val - cogs - comm_fee - packaging_cost - labor_cost_per_order
    margin_pct = (net_profit / order_val) * 100 if order_val > 0 else 0
    
    with col2:
        st.subheader("Order Breakdown")
        st.write(f"**Commission Fee:** £{comm_fee:.2f}")
        st.write(f"**Packaging Cost:** £{packaging_cost:.2f}")
        st.write(f"**Packing Labor (4 mins):** £{labor_cost_per_order:.2f}")
        st.markdown("---")
        if net_profit > 1.0:
            st.success(f"**Net Profit:** £{net_profit:.2f} ({margin_pct:.1f}% margin)")
        elif net_profit > 0:
            st.warning(f"**Net Profit:** £{net_profit:.2f} ({margin_pct:.1f}% margin) - Thin Margin!")
        else:
            st.error(f"**Net Profit:** £{net_profit:.2f} ({margin_pct:.1f}% margin) - UNPROFITABLE ORDER!")

elif menu == "App v3.0 & Oat Milk Impact":
    st.header("📉 Self-Inflicted Revenue Drag")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("App v3.0 Failure")
        st.metric("Daily Order Drop", "-23%", delta="-115 orders/day")
        st.metric("Annual Revenue Lost", "~£240,000")
        st.write("43% of app users stopped ordering post-update due to crashes & lost stamps.")
    with col2:
        st.subheader("Oat Milk Surcharge Impact")
        st.metric("Oat Drink Share", "24%", delta="-7% after Jan surcharge")
        st.write("53% of oat milk drinkers state they visit less frequently.")

elif menu == "£1.5m Allocation Model":
    st.header("💰 Recommended £1.5m Investment Strategy")
    st.markdown("### Where the money should go (and what to STOP doing):")
    
    st.subheader("❌ WHAT TO STOP DOING")
    st.write("1. **Stop Blind Store Expansion:** Pause Wellington Place (Leeds Trinity is already cannibalising Briggate by 19%).")
    st.write("2. **Stop Subsidising Small Delivery Orders:** Implement a minimum basket size (£8) or small-order surcharge (£1.50).")
    st.write("3. **Scrap the Oat Milk Charge:** Bring back lost customer footfall.")
    
    st.subheader("✅ £1.5M CAPITAL ALLOCATION")
    alloc_df = pd.DataFrame({
        "Category": ["Store Lease & Ops Fixes (Mcr/Hull)", "App Re-build & Tech Stability", "Delivery Surcharge Offset & Ops Tooling", "Working Capital Buffer"],
        "Allocation (£)": [500000, 400000, 300000, 300000]
    })
    fig_pie = px.pie(alloc_df, values="Allocation (£)", names="Category", title="£1.5m Strategy Breakdown")
    st.plotly_chart(fig_pie, use_container_width=True)