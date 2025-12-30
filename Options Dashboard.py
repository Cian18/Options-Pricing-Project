import numpy as np
import polars as pl
import streamlit as st
import scipy
import altair as alt




###############################################################################################################################
###############################################################################################################################
#### 1. start by creating a simple simulation of a bond and making an interactive web app to see how price varies with YTM ####
###############################################################################################################################
###############################################################################################################################

class Bond:
    def __init__(self, face_value: float, coupon_rate: float, maturity: int, frequency: int = 1):
        self.face_value = face_value
        self.coupon_rate = coupon_rate  # as a decimal, e.g., 0.05
        self.maturity = maturity        # in years
        self.frequency = frequency      # 1 for annual, 2 for semi-annual

    def get_cashflow(self):
        # Calculate total number of payments
        total_payments = self.maturity * self.frequency
        
        # Create the Time column (e.g., [1, 2, 3...] / frequency)
        # np.arange creates a sequence from 1 to total_payments
        periods = np.arange(1, total_payments + 1)
        times = periods / self.frequency
        
        # Create the Cashflow column
        # Start with just the coupon amount for every period
        coupon_payment = (self.coupon_rate * self.face_value) / self.frequency
        cashflows = np.full(total_payments, coupon_payment)
        
        # Add the Face Value to the last payment
        cashflows[-1] += self.face_value
        
        # Wrap it all in a Polars DataFrame
        df = pl.DataFrame({
            "period": periods,
            "time": times,
            "cashflow": cashflows
        })
        
        return df
    
    # define method to get the current price given an array of spot rates
    def get_price(self, spot_rates: np.ndarray):
        # Get the cashflow dataframe
        df = self.get_cashflow()

        # Add the spot rates as a column so we can do row-by-row math
        df = df.with_columns(pl.Series("spot rate", spot_rates))

        # Calculate the Present Value (PV) using the time column
        # Formula: CF / (1 + r)^t
        df = df.with_columns(
            (pl.col("cashflow") / (1 + pl.col("spot rate"))**pl.col("time")).alias("pv")
        )

        # Return the sum of the PV column
        return df["pv"].sum(), df
    
    # create a method to calculate the YTM using numerical optimisation
    def get_YTM(self, spot_rates):
        price, df = self.get_price(spot_rates)

        def opt_func(y):
            # Wrap the full expression in parentheses, THEN call .sum()
            price_guess = df.select(
                (pl.col("cashflow") / (1 + y/self.frequency)**(pl.col("time")*self.frequency)).sum()
            ).item()

            return price_guess - price
        
        ytm = scipy.optimize.newton(opt_func, x0=self.coupon_rate)
        
        # return ytm as a percentage
        return ytm * 100
    
    # the following method outputs macaulay duration, modified duration, and DV01
    def get_duration(self, spot_rates):
        # calculate the total present value and the cashflow with the get_price method
        PV_tot, df = self.get_price(spot_rates)

        # calculate the YTM (need it to calculate the modified duration)
        y = self.get_YTM(spot_rates)

        # calculate the Macaulay duration
        dur = df.select((pl.col("time")*pl.col("pv")).sum()/PV_tot).item()

        # calculate the modified duration
        mod_dur = dur/(1 + y/(100*self.frequency))

        # calculate the DV01
        dv01 = mod_dur * PV_tot * 0.0001

        return dur, mod_dur, dv01
    
    # create method to give the data to plot YTM - Price curve
    def ytm_price_curve(self, min_ytm: float = -0.02, max_ytm: float = 0.2, num_points: int = 50):
        # Initialize the YTM range
        ytm_df = pl.DataFrame({"YTM (%)": np.linspace(min_ytm, max_ytm, num_points)*100})
        
        # get the cashflows
        df_cf = self.get_cashflow()
        
        # Use cross join to map the cashflows to every possible value of YTM (allows for quick calculation of PV with polars)
        curve_df = ytm_df.join(df_cf, how='cross')

        #calculate the present value of every row (remembering to take the frequency into account)
        curve_df = curve_df.with_columns((pl.col("cashflow") / (1 + pl.col("YTM (%)")/(100*self.frequency))**(pl.col("time")*self.frequency)).alias("PV"))

        # group and sum present values corresponding to the same YTM to find the price
        curve_df = curve_df.group_by("YTM (%)").agg(pl.col("PV").sum().alias("Price")).sort("YTM (%)")

        return curve_df
    



######################################################################################################################################
######################################################################################################################################
#### 2. Create a class to represent the risk free interest rate as a function of time. (The spot rates will be derived from this) ####
######################################################################################################################################
######################################################################################################################################

class RiskFreeCurve:
    def __init__(self, magnitude: float, curvature: float):
        self.magnitude = magnitude
        self.curvature = curvature

    # define the interest rate as a logarithmic function of time 
    def rate(self, t: float):
        return self.magnitude + self.curvature * np.log(t + 1)

    # calculate the spot rates
    def spot_rates(self, time_to_maturity, frequency):
        interval = 1/frequency
        times = np.arange(interval, time_to_maturity + interval, interval)
        return times, self.rate(times)
    
    # acquire a dataframe for plotting the interest rate against time
    def plot(self, n_points: int, max_t, min_t = 0):
        time = np.linspace(min_t, max_t, n_points)

        # create the polars dataframe
        df = pl.DataFrame({"Time (years)": time, "Interest Rate": self.rate(time)})

        return df
    



###############################
###############################
#### 3. Create the web app ####
###############################
###############################


#################################################
## Create the sidebar to contain all variables ##
#################################################

# start with the risk free interest rate
st.sidebar.markdown("""<h2 style='text-align: center;  font-size: 30px;; color: #ffff; margin-bottom: 0px;'>Interest Rate Variables</h2>""", unsafe_allow_html=True)

# Magnitude: -2% to 8%, starting at 0.5%
int_mag = st.sidebar.slider("Interest Rate Plot Magnitude", -0.02, 0.08, 0.005, 0.001)

# Curvature: 0% to 4%, starting at 0.5% (with a smaller step for smooth plotting)
int_curv = st.sidebar.slider("Interest Rate Plot Curvature", 0.00, 0.04, 0.005, 0.001)

rate = RiskFreeCurve(int_mag, int_curv)

st.sidebar.divider()

# now define variables for the YTM - Price
st.sidebar.markdown("""<h2 style='text-align: center;  font-size: 30px; color: #ffff; margin-bottom: 0px;'>Bond Variables</h2>""", unsafe_allow_html=True)

face_val = st.sidebar.number_input("Face Value (£)", min_value=0, max_value=1000000, value=1000)
coupon_rate = st.sidebar.number_input("Coupon Rate (%)", min_value=0, max_value=12, value=5)
maturity = st.sidebar.number_input("Time to Maturity (years)", min_value=1, max_value=30, value=5)
freq = st.sidebar.selectbox("Coupon Frequency (per year)", options=[1, 2], index=0)
bond = Bond(face_val, coupon_rate/100, maturity, freq)

# now calculate the current Price and YTM given the spot rates (derived from the risk free interest rate curve)
time, spot_rates = rate.spot_rates(maturity, freq)
cur_price, _ = bond.get_price(spot_rates)
cur_ytm = bond.get_YTM(spot_rates)




########################################################################
## Create a function that creates a coloured box surrounding the text ##
########################################################################
def custom_box(text, bg_color="#2e2e2e", border_color="#57A4FF"):
    st.markdown(f"""
        <div style="
            background-color: {bg_color};
            border: 1px solid {border_color};
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin-bottom: 20px;">
            {text}
        </div>
    """, unsafe_allow_html=True)




##################################################################################
## Create section introducing Interest Rates, Bond Price, and Yield to Maturity ##
##################################################################################

st.set_page_config(layout="wide")

st.divider()

st.markdown("""<h1 style='text-align: center; font-size: 60px; color: #ffff;'>Bond Price and Yield to Maturity</h1>""", unsafe_allow_html=True)

st.divider()

# create three columns
col1, col2, col3 = st.columns(3)


# Column 1: introduce the price
with col1:
    col1.markdown("""<h4 style='text-align: center; color: #ffff;'>Risk Free Interest Rate (simulated)</h4>""", unsafe_allow_html=True)
    custom_box("""
    - In reality, the <b>Risk Free Interest Rates</b> are usually derived from highly secure government securities.
    - For the purposes of learning, the <b>Risk Free Interest Rate</b> have been simulated with a <b>logarithmic</b> curve.
    """, bg_color="#1e1b4b", border_color="#6366f1") # Professional Deep Navy

    st.latex(r"\Large r(t)=M + C \cdot \ln(t + 1)")

    custom_box("""
    - **Variable Definitions:**
        - $r(t)$: Risk Free Interest Rate
        - $M$: The magnitude variable
        - $C$: The curvature variable
        - $t$: Time in years
    - The interest rate curve can be changed by varying the <b>Magnitude</b> and <b>Curvature</b> in the sidebar.
    """, bg_color="#1e1b4b", border_color="#6366f1")


# Column 2: introduce the Price
with col2:
    col2.markdown("""<h4 style='text-align: center; color: #ffff;'>Bond Price</h4>""", unsafe_allow_html=True)
    custom_box("""
    - **Present Value Sum:** The price is the sum of all future cash flows discounted by their specific spot rates, adjusted for compounding frequency.
    """, bg_color="#064e3b", border_color="#10b981") # Professional Forest Emerald
    
    # display the equation
    st.latex(r"\Large P = \sum_{i=1}^{n \cdot k} \frac{CF_i}{\left(1 + \frac{r_i}{k}\right)^{t_i \cdot k}}")

    custom_box("""
    - **Variable Definitions:**
        - $CF_i$: Cash flow at period $i$
        - $r_i$: Annualized spot rate for time $t_i$
        - $t_i$: Time in years until payment
        - $k$: Payment Frequency
        - $n$: Time to Maturity (years)
    """, bg_color="#064e3b", border_color="#10b981")


# Column 3: introduce YTM
with col3:
    col3.markdown("""<h4 style='text-align: center; color: #ffff;'>Yield to Maturity</h4>""", unsafe_allow_html=True)
    custom_box("""
    While spot rates vary by year, the YTM provides a **single annualized value** to compare different bonds.
                 
    - YTM is the single rate that equates the bond's price to the sum of its cash flows.
    """, bg_color="#1e293b", border_color="#94a3b8") # Professional Dark Slate

    # display the equation
    st.latex(r"\Large P = \sum_{i=1}^{n \cdot k} \frac{CF_i}{\left(1 + \frac{\text{YTM}}{k}\right)^{t_i \cdot k}}")

    custom_box("""
    - **Solver Method:** Calculated numerically (Newton-Raphson) to find the root where Price equals Present Value.
    - **Note:** The variable definitions are the same as shown in the **Bond Price** section.
    """, bg_color="#1e293b", border_color="#94a3b8")


st.divider()



##########################################################################
## Create section introducing the Duration, Modified Duration, and DV01 ##
##########################################################################

st.markdown("""<h1 style='text-align: center; font-size: 60px; color: #ffff;'>Duration and Volatility</h1>""", unsafe_allow_html=True)
st.divider()

col1, col2, col3 = st.columns(3)

# Column 1: Macaulay Duration (Deep Plum Theme)
with col1:
    col1.markdown("<h4 style='text-align: center; color: #ffff;'>Macaulay Duration</h4>", unsafe_allow_html=True)
    custom_box("""
    - <b>Weighted Average</b> of the times at which every payment is made.
    - The weights are the fraction of the total present value of the bond (Price) that is paid at the current date.
    """, bg_color="#312e81", border_color="#818cf8") # Deep Plum/Indigo

    # put in the equation
    st.latex(r"\Large Duration=\sum_{i=1}^{n \cdot k} \frac{t_i \cdot PV(C_i)}{PV}")

    custom_box("""
    - **Variable Definitions:**
        - $Duration$: Macaulay Duration
        - $PV(C_i)$: Present value of payment at time $i$
        - $PV$: Total Present Value of Bond (Bond Price)
    - **Note:** all other variables are the same as defined in the <b>Bond Price</b> section.
    """, bg_color="#312e81", border_color="#818cf8")

# Column 2: Modified Duration (Burnt Amber Theme)
with col2:
    col2.markdown("<h4 style='text-align: center; color: #ffff;'>Modified Duration</h4>", unsafe_allow_html=True)
    custom_box("""
    - <b>Price Sensitivity:</b> how much the <b>Bond Price</b> changes given a unit change in <b>YTM</b> (used as a risk metric).
    """, bg_color="#451a03", border_color="#f59e0b") # Burnt Amber

    # write the two equations with custom line spacing
    st.latex(r"\Large ModD = \frac{Duration}{1+\frac{YTM}{k}}")
    st.latex(r"\Large ModD = -\frac{1}{P} \cdot \frac{dP}{dy}")

    custom_box("""
    - Modified duration (ModD) is the negative slope of the Price-Yield curve normalized by price.
    """, bg_color="#451a03", border_color="#f59e0b")


# Column 3: DV01 (Midnight Charcoal Theme)
with col3:
    col3.markdown("<h4 style='text-align: center; color: #ffff;'>DV01</h4>", unsafe_allow_html=True)
    custom_box("""
    - The actual currency change in price for a 1 basis point (0.01%) move in <b>YTM</b>.
    """, bg_color="#0f172a", border_color="#64748b") # Midnight Charcoal

    st.latex(r"\Large DV01=ModD \cdot PV \cdot 0.0001")

    custom_box("""
    - <b>Risk Metric:</b> Used by traders to understand PnL (Profit and Loss) risk per tick move in yields.
    """, bg_color="#0f172a", border_color="#64748b")


st.divider()

col1, col2 = st.columns(2)




# --- 1. Simulated Risk-Free Interest Rate Plot --- Overlayed by spot rates.

# create the logarithmic curve
rate_df = rate.plot(50, maturity)
rate_chart = alt.Chart(rate_df).mark_line(strokeWidth=3, color="#57A4FF").encode(
    x=alt.X('Time (years):Q', title="Time (years)"),
    y=alt.Y('Interest Rate:Q', 
            title="Interest Rate (%)", 
            scale=alt.Scale(zero=False),
            axis=alt.Axis(format='%') # This converts 0.05 to 5%
           ),
).properties(height=400)

# create the scatter spot rates
spot_df = pl.DataFrame({"Time (years)": time, "Interest Rate": spot_rates})
spot_scatter = alt.Chart(spot_df).mark_point(size=150, color="#9DE825", filled=True).encode(
    x='Time (years):Q',
    y=alt.Y('Interest Rate:Q', axis=alt.Axis(format='%')) # Keep format consistent
)

col1.markdown("""<h4 style='text-align: center; color: #ffff;'>Simulated Risk-Free Interest Rate</h4>""", unsafe_allow_html=True)
col1.altair_chart(rate_chart + spot_scatter, use_container_width=True)



# --- 2. Show dataframe with cashflow, spot rates, and discounted cashflow ---
_, df_cf = bond.get_price(spot_rates)

display_df = df_cf.select([
    pl.col("time").alias("Time (Yrs)"),
    pl.col("cashflow").alias("Cash Flow (£)"),
    (pl.col("spot rate") * 100).alias("Spot Rate (%)"),
    pl.col("pv").alias("Present Value (£)")
])

col2.markdown("""<h4 style='text-align: center; color: #ffff;'>Bond Valuation Breakdown</h4>""", unsafe_allow_html=True)
col2.dataframe(
    display_df, 
    use_container_width=True, 
    hide_index=True,
    column_config={
        "Spot Rate (%)": st.column_config.NumberColumn(format="%.2f%%"),
        "Cash Flow (£)": st.column_config.NumberColumn(format="£%.2f"),
        "Present Value (£)": st.column_config.NumberColumn(format="£%.2f")
    }
)


st.divider()

# --- 2. Discounted Cash Flow Distribution & Duration Plot ---

col1, col2 = st.columns(2)
# Prepare the data

macaulay_dur, mod_dur, DV01 = bond.get_duration(spot_rates)

# 1. Define the Bar Chart for Present Value (pv)
bars = alt.Chart(df_cf).mark_bar(size=25).encode(
    x=alt.X('time:Q', title="Time (Years)"),
    y=alt.Y('pv:Q', title="Present Value (£)"),
    color=alt.condition(
        alt.datum.time == maturity,
        alt.value('#ff7f0e'), # Highlight principal + final coupon
        alt.value('#57A4FF')  # Standard coupons
    ),
    tooltip=['time', 'cashflow', 'pv']
)

# 2. Define the Duration Vertical Line
dur_line = alt.Chart(pl.DataFrame({"dur": [macaulay_dur]})).mark_rule(
    color='red', 
    strokeDash=[5, 5], 
    strokeWidth=3
).encode(
    x='dur:Q'
)

# 3. FIXED: Label showing the word "Duration"
# We add a constant 'y' value so the text knows where to sit vertically
text_label = alt.Chart(pl.DataFrame({"dur": [macaulay_dur]})).mark_text(
    align='left', 
    dx=5, 
    dy=-5, # Fine-tune vertical offset from the y-point
    color='red', 
    fontSize=14, 
    fontWeight='bold'
).encode(
    x='dur:Q',
    y=alt.value(0), # Positions the text at the top (0 pixels from top)
    text=alt.value("Duration") 
)

# 4. Combine and set Height to match the Interest Rate plot
final_cf_plot = (bars + dur_line + text_label).properties(
    height=400,
    width='container'
)

col1.markdown("""<h4 style='text-align: center; color: #ffff;'>Discounted Cash Flow Distribution</h4>""", unsafe_allow_html=True)
col1.altair_chart(final_cf_plot, use_container_width=True)


# --- 3. Display current price, YTM, Duration, Modified Duration, and DV01
#create a gap to place the values in a better location
col2.write("####")
col2.write("####")

col3, col4 = col2.columns(2)
col3.metric("Current Price", f"£{cur_price:.2f}", border=True) 
col4.metric("Current YTM", f"{cur_ytm:.2f}%", border=True)

col3, col4, col5 = col2.columns(3)

col3.metric("Macaulay Duration", f"{macaulay_dur:.2f}", border=True)
col4.metric("Modified Duration", f"{mod_dur:.2f}", border=True)
col5.metric("DV01", f"{DV01:.2f}", border=True)


# --- 3. Bond Price vs Yield to Maturity Plot ---
st.divider()
curve_df = bond.ytm_price_curve()

line = alt.Chart(curve_df).mark_line(strokeWidth=3, color="#1f77b4").encode(
    x=alt.X('YTM (%):Q', title='Yield to Maturity (%)'),
    y=alt.Y('Price:Q', title='Bond Price (£)', scale=alt.Scale(zero=False))
)

point_df = pl.DataFrame({"YTM": [cur_ytm], "Price": [cur_price]})
point = alt.Chart(point_df).mark_point(size=200, color='#ff7f0e', filled=True).encode(
    x='YTM:Q',
    y='Price:Q'
)

st.markdown("""<h4 style='text-align: center; color: #ffff;'>Bond Price vs Yield to Maturity</h4>""", unsafe_allow_html=True)
st.altair_chart((line + point).properties(height=600), use_container_width=True)


st.divider()